# %%
"""Compute COSEBIS B-mode PTE for a single scale cut.

Scatter job for figure_3_cosebis_pte_matrices claim.
Reads precomputed fine-binned 2PCF and covariance, computes one PTE.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import treecorr
from scipy import stats

from sp_validation.b_modes import calculate_cosebis


def _load_snakemake():
    if hasattr(sys, "ps1"):
        from snakemake_helpers import snakemake_interactive

        return snakemake_interactive(
            "results/claims/figure_3_cosebis_pte_matrices/pte_values/SP_v1.4.6_leak_corr/pte_000_005.json",
            str(Path.cwd()),
        )
    from snakemake.script import snakemake

    return snakemake


snakemake = _load_snakemake()


def _compute_pte(values, covariance):
    """Compute PTE for B-mode null test."""
    chi2 = float(values @ np.linalg.solve(covariance, values))
    dof = len(values)
    return chi2, stats.chi2.sf(chi2, dof)


def main():
    t_start = time.time()

    # Extract wildcards
    i_min = int(snakemake.wildcards.i_min)
    i_max = int(snakemake.wildcards.i_max)
    nmodes = int(snakemake.params.nmodes)  # Should be 20 for full computation

    # Load precomputed fine-binned 2PCF
    t0 = time.time()
    gg = treecorr.GGCorrelation(min_sep=0.5, max_sep=500, nbins=1000, sep_units="arcmin")
    gg.read(snakemake.input.xi_integration)
    print(f"[TIMING] Load 2PCF: {time.time() - t0:.2f}s")

    # Compute theta grid from gg bin edges (20 bins for PTE matrix)
    theta_grid = np.geomspace(1.0, 250.0, 21)  # 21 edges -> 20 bins

    theta_min = theta_grid[i_min]
    theta_max = theta_grid[i_max]

    # Compute COSEBIS for this single scale cut with full 20 modes
    t0 = time.time()
    results = calculate_cosebis(
        gg,
        nmodes=nmodes,
        scale_cuts=[(theta_min, theta_max)],
        cov_path=snakemake.input.cov_integration,
    )
    print(f"[TIMING] calculate_cosebis (nmodes={nmodes}): {time.time() - t0:.2f}s")

    # Extract result for this scale cut
    result = results[(theta_min, theta_max)]

    # Full En, Bn arrays (20 modes)
    En_full = result["En"]
    Bn_full = result["Bn"]
    cov_full = result["cov"]

    # Compute PTEs for both 6 and 20 modes
    # 6 modes: use first 6
    cov_B_6 = cov_full[nmodes:nmodes+6, nmodes:nmodes+6]
    cov_E_6 = cov_full[:6, :6]
    chi2_B_6, pte_B_6 = _compute_pte(Bn_full[:6], cov_B_6)
    chi2_E_6, pte_E_6 = _compute_pte(En_full[:6], cov_E_6)

    # 20 modes: use all
    cov_B_20 = cov_full[nmodes:, nmodes:]
    cov_E_20 = cov_full[:nmodes, :nmodes]
    chi2_B_20, pte_B_20 = _compute_pte(Bn_full, cov_B_20)
    chi2_E_20, pte_E_20 = _compute_pte(En_full, cov_E_20)

    # Write output with both 6 and 20 mode results
    output = {
        "i_min": i_min,
        "i_max": i_max,
        "theta_min": float(theta_min),
        "theta_max": float(theta_max),
        # Full mode arrays
        "En": En_full.tolist(),
        "Bn": Bn_full.tolist(),
        # 6-mode results
        "nmodes_6": {
            "chi2_E": float(chi2_E_6),
            "chi2_B": float(chi2_B_6),
            "pte_E": float(pte_E_6),
            "pte_B": float(pte_B_6),
        },
        # 20-mode results
        "nmodes_20": {
            "chi2_E": float(chi2_E_20),
            "chi2_B": float(chi2_B_20),
            "pte_E": float(pte_E_20),
            "pte_B": float(pte_B_20),
        },
        # Legacy fields for backwards compatibility
        "pte_B": float(pte_B_6),
        "chi2_B": float(chi2_B_6),
        "chi2_E": float(chi2_E_6),
    }

    output_path = Path(snakemake.output.pte_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[TIMING] Total: {time.time() - t_start:.2f}s")
    print(f"Results: PTE_B(6)={pte_B_6:.4f}, PTE_B(20)={pte_B_20:.4f}")


if __name__ == "__main__":
    main()
