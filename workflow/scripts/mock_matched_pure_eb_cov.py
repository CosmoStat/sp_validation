"""Build the mock-matched pure-E/B Monte-Carlo covariance.

This is the paper's ``precompute_pure_eb_chunk.py`` plus
``gather_pure_eb_chunks.py`` path in one deterministic invocation.  The
processed 1000-bin ξ covariance is sampled in batches with seeds
``seed + batch_id``; each sample is rebinned to the 20-bin reporting grid and
passed through Schneider (2022).  The output has the same keys and six-block
ordering as the paper covariance:

    [ξ+^E, ξ−^E, ξ+^B, ξ−^B, ξ+^amb, ξ−^amb]

The script intentionally does not add the checkout's ``src/`` directory to
``sys.path``.  The caller should provide the compatible cs_util checkout via
``PYTHONPATH`` in the container invocation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse


def _build_cosmology(cosmo_params):
    """Build the same CCL cosmology used by the paper chunk script."""

    import pyccl as ccl

    return ccl.Cosmology(
        Omega_c=cosmo_params["Omega_m"] - cosmo_params["Omega_b"],
        Omega_b=cosmo_params["Omega_b"],
        h=cosmo_params["h"],
        sigma8=cosmo_params["sigma_8"],
        n_s=cosmo_params["n_s"],
    )


def _load_xi(path, min_sep, max_sep, nbins):
    """Load a paper TreeCorr text dump and reconstruct its bin edges."""

    data = np.loadtxt(path, comments="#", max_rows=nbins)
    if data.shape[0] != nbins:
        raise ValueError(f"Expected {nbins} ξ rows in {path}, found {data.shape[0]}")
    edges = np.logspace(np.log10(min_sep), np.log10(max_sep), nbins + 1)
    return {
        "meanr": np.asarray(data[:, 1], dtype=float),
        "xip": np.asarray(data[:, 3], dtype=float),
        "xim": np.asarray(data[:, 4], dtype=float),
        "left_edges": edges[:-1],
        "right_edges": edges[1:],
    }


def _load_nz(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected two-column n(z) file, found {data.shape}: {path}")
    return np.asarray(data[:, :2], dtype=float)


def _make_binning_matrix(reporting, integration):
    reporting_edges = np.concatenate(
        [reporting["left_edges"], [reporting["right_edges"][-1]]]
    )
    bin_indices = np.digitize(integration["meanr"], reporting_edges) - 1
    valid = (bin_indices >= 0) & (bin_indices < len(reporting["meanr"]))
    row_indices = bin_indices[valid]
    col_indices = np.where(valid)[0]
    matrix = sparse.csr_matrix(
        (np.ones(len(row_indices)), (row_indices, col_indices)),
        shape=(len(reporting["meanr"]), len(integration["meanr"])),
    )
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(row_sums == 0):
        raise ValueError("At least one reporting bin has no integration samples")
    return sparse.diags(1.0 / row_sums) @ matrix


def _sample_bounds(n_samples, n_batches, batch_id):
    samples_per_batch = n_samples // n_batches
    start = batch_id * samples_per_batch
    stop = start + samples_per_batch if batch_id < n_batches - 1 else n_samples
    return start, stop


def build_covariance(
    *,
    covariance_path,
    xi_reporting_path,
    xi_integration_path,
    nz_path,
    output_path,
    n_samples=2000,
    n_batches=16,
    seed=42,
    min_sep=1.0,
    max_sep=250.0,
    nbins=20,
    min_sep_int=0.5,
    max_sep_int=300.0,
    nbins_int=1000,
    pad_xim=True,
    interp_order=5,
    diagonal_covariance=False,
):
    """Generate and save the 120×120 mock-matched covariance."""

    if n_samples < 2 or n_batches < 1 or n_batches > n_samples:
        raise ValueError("Require 2 <= n_samples and 1 <= n_batches <= n_samples")

    from cosmo_numba.B_modes.schneider2022 import get_pure_EB_modes
    from cs_util.cosmo import PLANCK18, get_theo_xi

    reporting = _load_xi(xi_reporting_path, min_sep, max_sep, nbins)
    integration = _load_xi(xi_integration_path, min_sep_int, max_sep_int, nbins_int)
    covariance = np.asarray(np.loadtxt(covariance_path), dtype=float)
    expected_shape = (2 * nbins_int, 2 * nbins_int)
    if covariance.shape != expected_shape:
        raise ValueError(
            f"Expected processed ξ covariance shape {expected_shape}, "
            f"found {covariance.shape}"
        )
    if diagonal_covariance:
        # Diagnostic only: retain the CosmoCov diagonal while removing all
        # fine-bin and xi+/xi- correlations.  The default is the full stored
        # covariance used for the paper product.
        covariance = np.diag(np.diag(covariance))

    z_dist = _load_nz(nz_path)
    cosmo = _build_cosmology(dict(PLANCK18))
    mean_int = np.concatenate(
        get_theo_xi(
            theta=integration["meanr"],
            z=z_dist[:, 0],
            nz=z_dist[:, 1],
            backend="ccl",
            cosmo=cosmo,
        )
    )
    binning_matrix = _make_binning_matrix(reporting, integration)

    all_samples = []
    for batch_id in range(n_batches):
        start, stop = _sample_bounds(n_samples, n_batches, batch_id)
        n_batch = stop - start
        print(
            f"Batch {batch_id + 1}/{n_batches}: samples {start}:{stop} "
            f"({n_batch}), seed={seed + batch_id}",
            flush=True,
        )
        rng = np.random.default_rng(seed=seed + batch_id)
        samples_int = rng.multivariate_normal(mean_int, covariance, size=n_batch)
        samples_xip_int = samples_int[:, :nbins_int]
        samples_xim_int = samples_int[:, nbins_int:]
        samples_xip_rep = (binning_matrix @ samples_xip_int.T).T
        samples_xim_rep = (binning_matrix @ samples_xim_int.T).T

        transformed = []
        for index in range(n_batch):
            modes = get_pure_EB_modes(
                theta=reporting["meanr"],
                theta_int=integration["meanr"],
                xip=samples_xip_rep[index],
                xim=samples_xim_rep[index],
                xip_int=samples_xip_int[index],
                xim_int=samples_xim_int[index],
                tmin=min_sep,
                tmax=max_sep,
                pad_xim=pad_xim,
                interp_order=interp_order,
                parallel=True,
            )
            transformed.append(np.concatenate(modes))
        all_samples.append(np.asarray(transformed, dtype=float))

    eb_samples = np.vstack(all_samples)
    cov_pure_eb = np.cov(eb_samples.T)
    if cov_pure_eb.shape != (120, 120):
        raise ValueError(
            f"Expected 120x120 output covariance, found {cov_pure_eb.shape}"
        )

    # Match gather_pure_eb_chunks.py's package exactly, with provenance extras.
    modes_data = get_pure_EB_modes(
        theta=reporting["meanr"],
        xip=reporting["xip"],
        xim=reporting["xim"],
        theta_int=integration["meanr"],
        xip_int=integration["xip"],
        xim_int=integration["xim"],
        tmin=min_sep,
        tmax=max_sep,
        pad_xim=pad_xim,
        interp_order=interp_order,
        parallel=True,
    )
    xip_E, xim_E, xip_B, xim_B, xip_amb, xim_amb = modes_data

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        theta=reporting["meanr"],
        theta_int=integration["meanr"],
        xip_total=reporting["xip"],
        xim_total=reporting["xim"],
        xip_E=xip_E,
        xim_E=xim_E,
        xip_B=xip_B,
        xim_B=xim_B,
        xip_amb=xip_amb,
        xim_amb=xim_amb,
        cov_pure_eb=cov_pure_eb,
        n_samples=n_samples,
        n_batches=n_batches,
        seed=seed,
        covariance_path=str(covariance_path),
        nz_path=str(nz_path),
        pad_xim=pad_xim,
        interp_order=interp_order,
        diagonal_covariance=diagonal_covariance,
    )
    print(f"Saved {output_path} with cov_pure_eb shape {cov_pure_eb.shape}", flush=True)
    return output_path


def _parser(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cov-integration", required=True)
    parser.add_argument("--xi-reporting", required=True)
    parser.add_argument("--xi-integration", required=True)
    parser.add_argument("--nz", required=True)
    parser.add_argument("--out", required=True, help="Final .npz output path")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument(
        "--n-batches",
        "--n-chunks",
        dest="n_batches",
        type=int,
        default=16,
        help="Deterministic RNG batches, not separate output files",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-sep", type=float, default=1.0)
    parser.add_argument("--max-sep", type=float, default=250.0)
    parser.add_argument("--nbins", type=int, default=20)
    parser.add_argument("--min-sep-int", type=float, default=0.5)
    parser.add_argument("--max-sep-int", type=float, default=300.0)
    parser.add_argument("--nbins-int", type=int, default=1000)
    parser.add_argument(
        "--no-pad-xim",
        action="store_true",
        help="Diagnostic: disable cosmo_numba's default xi- padding.",
    )
    parser.add_argument(
        "--interp-order",
        type=int,
        default=5,
        choices=(1, 3, 5, 7, 9),
        help="Interpolation order passed to get_pure_EB_modes.",
    )
    parser.add_argument(
        "--diagonal-covariance",
        action="store_true",
        help="Diagnostic: zero all off-diagonal CosmoCov elements.",
    )
    return parser.parse_args(argv)


def _main(argv=None):
    args = _parser(argv)
    build_covariance(
        covariance_path=args.cov_integration,
        xi_reporting_path=args.xi_reporting,
        xi_integration_path=args.xi_integration,
        nz_path=args.nz,
        output_path=args.out,
        n_samples=args.n_samples,
        n_batches=args.n_batches,
        seed=args.seed,
        min_sep=args.min_sep,
        max_sep=args.max_sep,
        nbins=args.nbins,
        min_sep_int=args.min_sep_int,
        max_sep_int=args.max_sep_int,
        nbins_int=args.nbins_int,
        pad_xim=not args.no_pad_xim,
        interp_order=args.interp_order,
        diagonal_covariance=args.diagonal_covariance,
    )


if __name__ == "__main__":
    _main()
