"""Harmonic-space B-mode PTE heatmap across ell-bin cuts.

Claim: Full ell range shows B-modes consistent with zero across nearly all
bin cut combinations (~96% healthy PTEs).
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
from astropy.io import fits
from scipy import stats


plt.style.use(
    "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle"
)


def _compute_pte(data, covariance):
    chi2 = float(data @ np.linalg.solve(covariance, data))
    dof = len(data)
    pte = stats.chi2.sf(chi2, dof)
    return pte, chi2, dof


def main():
    from snakemake.script import snakemake

    # Load pseudo-Cl data
    hdu = fits.open(snakemake.input["pseudo_cl"])
    data = hdu["PSEUDO_CELL"].data
    hdu.close()

    ell = data["ELL"]
    cl_ee = data["EE"]
    cl_bb = data["BB"]
    n_ell = len(ell)

    # Load covariances
    hdu_cov = fits.open(snakemake.input["pseudo_cl_cov"])
    cov_ee = hdu_cov["COVAR_EE_EE"].data
    cov_bb = hdu_cov["COVAR_BB_BB"].data
    hdu_cov.close()

    print(f"Loaded {n_ell} ell bins: ell = [{ell.min():.1f}, {ell.max():.1f}]")

    # Compute PTE matrix
    pte_bb_matrix = np.full((n_ell, n_ell), np.nan)

    for i_min in range(n_ell):
        for i_max in range(i_min + 1, n_ell):
            idx_slice = slice(i_min, i_max + 1)
            bb_slice = cl_bb[idx_slice]
            cov_bb_slice = cov_bb[idx_slice, idx_slice]

            try:
                pte_bb, chi2_bb, dof_bb = _compute_pte(bb_slice, cov_bb_slice)
                pte_bb_matrix[i_min, i_max] = pte_bb
            except np.linalg.LinAlgError:
                pass

    # Statistics
    valid_ptes = pte_bb_matrix[~np.isnan(pte_bb_matrix)]
    frac_healthy = float(
        np.sum((valid_ptes >= 0.05) & (valid_ptes <= 0.95)) / len(valid_ptes)
    )
    pte_fiducial = pte_bb_matrix[0, n_ell - 1]

    print(f"Valid scale cuts: {len(valid_ptes)}")
    print(f"Fraction healthy: {frac_healthy:.1%}")
    print(f"PTE at full ell range: {pte_fiducial:.4f}")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(3.54, 3.54))

    vlag_cmap = sns.color_palette("vlag", as_cmap=True).copy()
    vlag_cmap.set_bad(color="lightgray")

    pte_plot_data = pte_bb_matrix.T

    # Flip both axes so small angular scales (high ell) are in bottom-left
    # Y-axis: flip so large ell_max at bottom, small ell_max at top
    # X-axis: flip data so large ell_min on left, small ell_min on right
    pte_plot_data = pte_plot_data[:, ::-1]  # flip X (columns)

    im = ax.imshow(
        pte_plot_data, origin="upper", aspect="equal",
        cmap=vlag_cmap, vmin=0, vmax=1, extent=[0, n_ell, 0, n_ell],
    )

    cs = ax.contour(
        pte_plot_data, levels=[0.05, 0.95], colors="black",
        linewidths=0.8, extent=[0, n_ell, 0, n_ell],
    )
    ax.clabel(cs, inline=True, fmt="%.2f")
    ax.set_xlabel(r"Lower $\ell$ cut")
    ax.set_ylabel(r"Upper $\ell$ cut")

    # Mark full ell range (now at bottom-right after axis flips)
    # X: ell_min=0 is now at x=n_ell-1 (right side)
    # Y: ell_max=n_ell-1 is now at y=0 (bottom, due to origin="upper")
    ax.add_patch(
        Rectangle((n_ell - 1, 0), 1, 1, fill=False, edgecolor="black",
                  linewidth=1.5, hatch="///", alpha=0.8)
    )

    # Tick labels (reversed for both axes)
    tick_step = max(1, n_ell // 5)
    tick_indices = np.arange(0, n_ell, tick_step)
    ax.set_xticks(tick_indices + 0.5)
    # X-axis: large ell on left, small on right (reversed)
    ax.set_xticklabels([f"{ell[n_ell - 1 - i]:.0f}" for i in tick_indices], rotation=45, ha="right")
    ax.set_yticks(tick_indices + 0.5)
    # Y-axis: large ell at bottom (tick_index 0), small at top (tick_index n_ell-1)
    ax.set_yticklabels([f"{ell[n_ell - 1 - i]:.0f}" for i in tick_indices])

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("PTE")

    plt.tight_layout()

    # Save outputs
    output_dir = Path(snakemake.output["evidence"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = Path(snakemake.output["figure"])
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_path}")

    paper_path = Path(snakemake.output["paper_figure"])
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fig_path, paper_path)
    print(f"Copied to {paper_path}")

    plt.close(fig)

    # Build evidence
    spec_paths = snakemake.input["specs"]

    evidence_data = {
        "spec_id": "cl_pte_matrix",
        "spec_path": spec_paths[0],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "pte_at_full_range": float(pte_fiducial),
            "fraction_healthy": frac_healthy,
            "n_bin_cuts_evaluated": int(len(valid_ptes)),
            "ell_range": [float(ell.min()), float(ell.max())],
            "n_ell_bins": int(n_ell),
            "pte_statistics": {
                "mean": float(np.nanmean(valid_ptes)),
                "median": float(np.nanmedian(valid_ptes)),
                "std": float(np.nanstd(valid_ptes)),
                "min": float(np.nanmin(valid_ptes)),
                "max": float(np.nanmax(valid_ptes)),
            },
        },
        "artifacts": {
            "figure": fig_path.name,
        },
    }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"Saved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
