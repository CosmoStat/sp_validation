"""Calculate PTE matrices for Pure E/B mode scale cut robustness."""

import os
import sys
from pathlib import Path

import numpy as np

from sp_validation.b_modes import calculate_eb_statistics

# Unbuffered output for snakemake logging
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)


def _load_snakemake():
    if hasattr(sys, "ps1"):
        from snakemake_helpers import snakemake_interactive

        return snakemake_interactive(
            "results/paper_plots/intermediate/SP_v1.4.6_leak_corr_pure_eb_ptes.npz",
            str(Path.cwd()),
        )
    from snakemake.script import snakemake

    return snakemake


snakemake = _load_snakemake()

params = snakemake.params


class FakeGG:
    """Minimal GGCorrelation-like object for calculate_eb_statistics."""
    def __init__(self, nbins, npatch):
        self.nbins = nbins
        self.npatch1 = npatch
        self.npatch2 = npatch


def main():
    # Load precomputed Pure E/B data
    dataset = np.load(snakemake.input["pure_eb_data"])

    theta = dataset["theta"]
    nbins = len(theta)

    # Get covariance path from input if semi-analytic was used
    cov_path_int = snakemake.input.get("cov_integration")

    # Reconstruct results dict for calculate_eb_statistics
    results = {
        "gg": FakeGG(nbins, int(params["npatch"])),
        "xip_E": dataset["xip_E"],
        "xim_E": dataset["xim_E"],
        "xip_B": dataset["xip_B"],
        "xim_B": dataset["xim_B"],
        "xip_amb": dataset["xip_amb"],
        "xim_amb": dataset["xim_amb"],
        "cov": dataset["cov_pure_eb"],
    }

    # Calculate PTE matrices
    print(f"Calculating PTE matrices for {params['version']}...")
    results = calculate_eb_statistics(
        results,
        cov_path_int=cov_path_int,
        n_samples=int(params["n_samples"]),
    )

    # Extract and save PTE matrices
    pte_matrices = results["pte_matrices"]

    output_data = {
        "theta": theta,
        "pte_xip_B": pte_matrices["xip_B"],
        "pte_xim_B": pte_matrices["xim_B"],
        "pte_combined": pte_matrices["combined"],
    }

    np.savez(snakemake.output[0], **output_data)
    print(f"Saved PTE matrices to {snakemake.output[0]}")


if __name__ == "__main__":
    main()
