"""Compute a chunk of MC samples for pure E/B covariance."""

import os
import sys
from pathlib import Path

import numpy as np
import pyccl as ccl
import tqdm
from IPython import get_ipython
from scipy import sparse

ipython = get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
else:
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)


def _load_snakemake():
    if hasattr(sys, "ps1"):
        from snakemake_helpers import snakemake_interactive

        return snakemake_interactive(
            "results/paper_plots/intermediate/chunks/SP_v1.4.6_leak_corr_pure_eb_chunk_0.npz",
            str(Path.cwd()),
        )
    from snakemake.script import snakemake

    return snakemake


snakemake = _load_snakemake()
params = snakemake.params


def _build_cosmology(cosmo_params):
    """Build CCL cosmology from PLANCK18 params dict."""
    return ccl.Cosmology(
        Omega_c=cosmo_params["Omega_m"] - cosmo_params["Omega_b"],
        Omega_b=cosmo_params["Omega_b"],
        h=cosmo_params["h"],
        sigma8=cosmo_params["sigma_8"],
        n_s=cosmo_params["n_s"],
    )


def _load_xi(path, min_sep, max_sep, nbins):
    """Load 2PCF from treecorr output file and compute bin edges."""
    # Treecorr files have header comments (##) and column names (#), then data,
    # then ## cov followed by covariance matrix
    data = np.loadtxt(path, comments="#", max_rows=nbins)
    meanr = data[:, 1]
    xip = data[:, 3]
    xim = data[:, 4]
    # Treecorr uses log-spaced bins
    bin_edges = np.logspace(np.log10(min_sep), np.log10(max_sep), nbins + 1)
    return {
        "meanr": meanr,
        "xip": xip,
        "xim": xim,
        "left_edges": bin_edges[:-1],
        "right_edges": bin_edges[1:],
    }


def main():
    from cosmo_numba.B_modes.schneider2022 import get_pure_EB_modes

    from sp_validation.cosmo_val import CosmologyValidation
    from sp_validation.cosmology import get_theo_xi

    chunk_id = int(params["chunk_id"])
    n_chunks = int(params["n_chunks"])
    n_samples_total = int(params["n_samples"])

    # Compute this chunk's sample range
    samples_per_chunk = n_samples_total // n_chunks
    start_idx = chunk_id * samples_per_chunk
    end_idx = (
        start_idx + samples_per_chunk if chunk_id < n_chunks - 1 else n_samples_total
    )
    n_samples_chunk = end_idx - start_idx

    print(
        f"Chunk {chunk_id}/{n_chunks}: samples {start_idx}-{end_idx} ({n_samples_chunk} samples)"
    )

    numeric_params = {
        "min_sep": float(params["min_sep"]),
        "max_sep": float(params["max_sep"]),
        "nbins": int(params["nbins"]),
        "min_sep_int": float(params["min_sep_int"]),
        "max_sep_int": float(params["max_sep_int"]),
        "nbins_int": int(params["nbins_int"]),
        "npatch": int(params["npatch"]),
    }

    # Load precomputed correlation functions from Snakemake inputs
    gg = _load_xi(
        snakemake.input["xi_reporting"],
        numeric_params["min_sep"],
        numeric_params["max_sep"],
        numeric_params["nbins"],
    )
    gg_int = _load_xi(
        snakemake.input["xi_integration"],
        numeric_params["min_sep_int"],
        numeric_params["max_sep_int"],
        numeric_params["nbins_int"],
    )

    # Get redshift distribution for this blind
    blind = params.get("blind", "A")
    cv = CosmologyValidation(
        versions=[params["version"]],
        catalog_config="/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_val/cat_config.yaml",
        output_dir="/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_val/output",
    )
    cv.blind = blind
    z, nz = cv.get_redshift(params["version"])
    z_dist = np.column_stack([z, nz])
    print(f"Using n(z) for blind {blind}")

    # Build cosmology for theoretical predictions
    cosmo_cov = _build_cosmology(params["cosmo_params"])

    # Load integration covariance
    cov_int = np.loadtxt(snakemake.input["cov_integration"])
    nbins_int = numeric_params["nbins_int"]

    # Build binning matrix (same as b_modes.py)
    theta_int = gg_int["meanr"]
    reporting_bin_edges = np.concatenate([gg["left_edges"], [gg["right_edges"][-1]]])
    bin_indices = np.digitize(theta_int, reporting_bin_edges) - 1
    valid_mask = (bin_indices >= 0) & (bin_indices < len(gg["meanr"]))
    row_indices, col_indices = bin_indices[valid_mask], np.where(valid_mask)[0]

    binning_matrix = sparse.csr_matrix(
        (np.ones(len(row_indices)), (row_indices, col_indices)),
        shape=(len(gg["meanr"]), nbins_int),
    )
    row_sums = np.array(binning_matrix.sum(axis=1)).flatten()
    binning_matrix = sparse.diags(1 / row_sums) @ binning_matrix

    # Get theoretical mean (same as b_modes.py)
    mean_int = np.concatenate(
        get_theo_xi(
            theta=theta_int,
            z=z_dist[:, 0],
            nz=z_dist[:, 1],
            backend="ccl",
            cosmo=cosmo_cov,
        )
    )

    # Set seed based on chunk_id for reproducibility
    rng = np.random.default_rng(seed=42 + chunk_id)

    # Draw this chunk's samples
    samples_int = rng.multivariate_normal(mean_int, cov_int, size=n_samples_chunk)
    samples_int_xip = samples_int[:, :nbins_int]
    samples_int_xim = samples_int[:, nbins_int:]
    samples_rep_xip = (binning_matrix @ samples_int_xip.T).T
    samples_rep_xim = (binning_matrix @ samples_int_xim.T).T

    # Transform samples
    min_sep = numeric_params["min_sep"]
    max_sep = numeric_params["max_sep"]

    transformed_samples = [
        np.concatenate(
            get_pure_EB_modes(
                theta=gg["meanr"],
                theta_int=gg_int["meanr"],
                xip=samples_rep_xip[i],
                xim=samples_rep_xim[i],
                xip_int=samples_int_xip[i],
                xim_int=samples_int_xim[i],
                tmin=min_sep,
                tmax=max_sep,
            )
        )
        for i in tqdm.tqdm(range(n_samples_chunk), desc=f"Chunk {chunk_id}")
    ]

    eb_samples = np.array(transformed_samples)

    # Save chunk results (only samples needed for covariance)
    np.savez(
        snakemake.output[0],
        eb_samples=eb_samples,
        chunk_id=chunk_id,
    )
    print(f"Saved {n_samples_chunk} samples to {snakemake.output[0]}")


if __name__ == "__main__":
    main()
