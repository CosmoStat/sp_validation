"""Compute one chunk of MC samples for the pure E/B covariance.

CLI refactor of the former Snakemake ``script:`` rule. The compute is
unchanged: draw ``n_samples // n_chunks`` Gaussian realisations of ξ±(θ) from
the 1000-bin integration-grid CosmoCov covariance (deterministic seed
``42 + chunk_id``), rebin to the reporting grid, and push each draw through the
Schneider-2022 pure-mode integral transforms (``cosmo_numba``). The per-chunk
E/B/amb sample block is written to ``{out}/pure_eb_chunk_{chunk_id}.npz`` for
the gather stage. Each chunk is independent (fresh RNG per chunk_id), so the
20 chunks reproduce the paper's 2000-sample covariance bit-for-bit whether run
in parallel or looped in one process.

    python precompute_pure_eb_chunk.py \
        --chunk-id 0 --n-chunks 20 --n-samples 2000 \
        --version SP_v1.4.6.3_leak_corr --blind A \
        --cat-config /path/cosmo_val/cat_config.yaml \
        --xi-reporting  <xi 20-bin .txt> \
        --xi-integration <xi 1000-bin .txt> \
        --cov-integration <cov ..._processed.txt> \
        --min-sep 1.0 --max-sep 250.0 --nbins 20 \
        --min-sep-int 0.5 --max-sep-int 300.0 --nbins-int 1000 \
        --npatch 1 --out <output_dir>
"""

import argparse
import os

import numpy as np
import tqdm
from pure_eb_operator import (
    fine_support,
    load_xi,
    make_binning_matrix,
    print_drop_counts,
    pure_eb_modes,
)


def _build_cosmology(cosmo_params):
    """Build a CCL cosmology from a PLANCK18-style params dict."""
    import pyccl as ccl

    return ccl.Cosmology(
        Omega_c=cosmo_params["Omega_m"] - cosmo_params["Omega_b"],
        Omega_b=cosmo_params["Omega_b"],
        h=cosmo_params["h"],
        sigma8=cosmo_params["sigma_8"],
        n_s=cosmo_params["n_s"],
    )


def compute_chunk(
    chunk_id,
    n_chunks,
    n_samples_total,
    version,
    blind,
    cat_config,
    xi_reporting,
    xi_integration,
    cov_integration,
    min_sep,
    max_sep,
    nbins,
    min_sep_int,
    max_sep_int,
    nbins_int,
    output_dir,
    cosmo_params=None,
    transform_averaged=True,
):
    from cs_util.cosmo import PLANCK18, get_theo_xi

    from sp_validation.cosmo_val import CosmologyValidation

    if cosmo_params is None:
        cosmo_params = dict(PLANCK18)

    samples_per_chunk = n_samples_total // n_chunks
    start_idx = chunk_id * samples_per_chunk
    end_idx = (
        start_idx + samples_per_chunk if chunk_id < n_chunks - 1 else n_samples_total
    )
    n_samples_chunk = end_idx - start_idx
    print(
        f"Chunk {chunk_id}/{n_chunks}: samples {start_idx}-{end_idx} "
        f"({n_samples_chunk} samples)"
    )

    gg = load_xi(xi_reporting, min_sep, max_sep, nbins)
    gg_int = load_xi(
        xi_integration,
        min_sep_int,
        max_sep_int,
        nbins_int,
        require_npairs=transform_averaged,
    )

    cv = CosmologyValidation(
        versions=[version],
        catalog_config=cat_config,
        output_dir=output_dir,
    )
    cv.blind = blind
    z, nz = cv.get_redshift(version)
    z_dist = np.column_stack([z, nz])
    print(f"Using n(z) for blind {blind}")

    cosmo_cov = _build_cosmology(cosmo_params)

    cov_int = np.loadtxt(cov_integration)

    theta_int = gg_int["meanr"]
    # Pair-count weights for the averaged operator; uniform for the legacy
    # pointwise path, where the matrix only rebins samples to the reporting grid.
    binning_matrix = make_binning_matrix(
        gg, gg_int, gg_int["npairs"] if transform_averaged else None
    )
    fine_indices = fine_support(binning_matrix) if transform_averaged else None

    # One n(z) gives one tracer pair: get_theo_xi's single (xi+, xi-) entry.
    (xi_pm,) = get_theo_xi(
        theta=theta_int,
        z=z_dist[:, 0],
        nz=z_dist[:, 1],
        backend="ccl",
        cosmo=cosmo_cov,
    ).values()
    mean_int = np.concatenate(xi_pm)

    rng = np.random.default_rng(seed=42 + chunk_id)

    samples_int = rng.multivariate_normal(mean_int, cov_int, size=n_samples_chunk)
    samples_int_xip = samples_int[:, :nbins_int]
    samples_int_xim = samples_int[:, nbins_int:]
    samples_rep_xip = (binning_matrix @ samples_int_xip.T).T
    samples_rep_xim = (binning_matrix @ samples_int_xim.T).T

    transformed_samples = []
    dropped_total = np.zeros(6, dtype=int)
    for i in tqdm.tqdm(range(n_samples_chunk), desc=f"Chunk {chunk_id}"):
        modes, dropped = pure_eb_modes(
            theta_rep=gg["meanr"],
            xip_rep=samples_rep_xip[i],
            xim_rep=samples_rep_xim[i],
            theta_int=theta_int,
            xip_int=samples_int_xip[i],
            xim_int=samples_int_xim[i],
            tmin=min_sep,
            tmax=max_sep,
            transform_averaged=transform_averaged,
            binning_matrix=binning_matrix,
            fine_indices=fine_indices,
        )
        dropped_total += dropped
        transformed_samples.append(np.concatenate(modes))

    if transform_averaged:
        print_drop_counts("MC outputs", dropped_total)

    eb_samples = np.array(transformed_samples)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"pure_eb_chunk_{chunk_id}.npz")
    np.savez(out_path, eb_samples=eb_samples, chunk_id=chunk_id)
    print(f"Saved {n_samples_chunk} samples to {out_path}")
    return out_path


def _from_cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chunk-id", type=int, required=True)
    ap.add_argument("--n-chunks", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--version", required=True)
    ap.add_argument("--blind", default="A")
    ap.add_argument("--cat-config", required=True)
    ap.add_argument("--xi-reporting", required=True)
    ap.add_argument("--xi-integration", required=True)
    ap.add_argument("--cov-integration", required=True)
    ap.add_argument("--min-sep", type=float, default=1.0)
    ap.add_argument("--max-sep", type=float, default=250.0)
    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--min-sep-int", type=float, default=0.5)
    ap.add_argument("--max-sep-int", type=float, default=300.0)
    ap.add_argument("--nbins-int", type=int, default=1000)
    ap.add_argument("--npatch", type=int, default=1)
    ap.add_argument("--out", required=True, help="Output directory (lc {output})")
    ap.add_argument(
        "--transform-averaged",
        dest="transform_averaged",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Transform on the fine grid, then npairs-average each output block.",
    )
    a = ap.parse_args(argv)
    compute_chunk(
        chunk_id=a.chunk_id,
        n_chunks=a.n_chunks,
        n_samples_total=a.n_samples,
        version=a.version,
        blind=a.blind,
        cat_config=a.cat_config,
        xi_reporting=a.xi_reporting,
        xi_integration=a.xi_integration,
        cov_integration=a.cov_integration,
        min_sep=a.min_sep,
        max_sep=a.max_sep,
        nbins=a.nbins,
        min_sep_int=a.min_sep_int,
        max_sep_int=a.max_sep_int,
        nbins_int=a.nbins_int,
        output_dir=a.out,
        transform_averaged=a.transform_averaged,
    )


if __name__ == "__main__":
    _from_cli()
