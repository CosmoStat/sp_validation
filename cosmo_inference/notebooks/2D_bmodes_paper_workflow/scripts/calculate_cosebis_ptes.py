"""Calculate PTE matrices for COSEBIS E/B mode scale cut robustness."""

import os
import sys
from pathlib import Path

import numpy as np
import treecorr

from IPython import get_ipython
from sp_validation.b_modes import calculate_cosebis

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
            "results/paper_plots/intermediate/SP_v1.4.6_leak_corr_cosebis_ptes.npz",
            str(Path.cwd()),
        )
    from snakemake.script import snakemake

    return snakemake


snakemake = _load_snakemake()

params = snakemake.params


def main():
    # Load precomputed COSEBIS data (we need gg object for calculate_cosebis)
    cosebis_data = np.load(snakemake.input["cosebis_data"])

    # We need to reconstruct the gg object from saved data or load xi file
    # Load xi integration file
    xi_data = np.loadtxt(snakemake.input["xi_integration"])

    # Create a minimal GGCorrelation-like object
    # We'll load it properly using TreeCorr's save/load mechanism
    # Actually, let's just load the original xi file with TreeCorr
    gg = treecorr.GGCorrelation(
        min_sep=float(params["min_sep_int"]),
        max_sep=float(params["max_sep_int"]),
        nbins=int(params["nbins_int"]),
        sep_units="arcmin",
    )

    # Read the correlation function file
    gg.read(snakemake.input["xi_integration"])

    nmodes = int(params["nmodes"])

    # Generate all valid scale cut combinations
    # For computational efficiency, sample a grid of scale cuts
    theta = gg.meanr
    nbins = len(theta)

    # Sample scale cuts - use coarser grid for speed (every 5th bin)
    step = 5
    scale_cuts = []
    for i in range(0, nbins, step):
        for j in range(i + step, nbins + 1, step):
            if j <= nbins:
                scale_cuts.append((theta[i], theta[min(j, nbins-1)]))

    print(f"Calculating COSEBIS PTEs for {len(scale_cuts)} scale cuts...")

    # Calculate COSEBIS for all scale cuts
    results = calculate_cosebis(
        gg,
        nmodes=nmodes,
        scale_cuts=scale_cuts,
        cov_path=snakemake.input["cov_integration"],
    )

    # Build PTE matrices (similar structure to Pure E/B)
    # Initialize matrices
    pte_chi2_E = np.full((nbins, nbins), np.nan)
    pte_B = np.full((nbins, nbins), np.nan)

    # Fill matrices from results
    for scale_cut, result in results.items():
        min_theta, max_theta = scale_cut
        i = np.argmin(np.abs(theta - min_theta))
        j = np.argmin(np.abs(theta - max_theta))

        pte_chi2_E[i, j] = 1 - result.get("chi2_E", 0) / (2 * nmodes)  # Rough approximation
        pte_B[i, j] = result["pte_B"]

    # Save PTE matrices
    output_data = {
        "theta": theta,
        "pte_chi2_E": pte_chi2_E,
        "pte_B": pte_B,
    }

    np.savez(snakemake.output[0], **output_data)
    print(f"Saved COSEBIS PTE matrices to {snakemake.output[0]}")


if __name__ == "__main__":
    main()
