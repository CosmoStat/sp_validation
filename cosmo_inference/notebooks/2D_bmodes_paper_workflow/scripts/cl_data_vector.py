"""Harmonic-space fiducial B-mode power spectrum.

Claim: Fiducial catalog shows C_ell^BB consistent with zero.
Produces two figures:
- Main figure (leak-corrected, unlabeled) for paper
- Companion figure (uncorrected, labeled) for dashboard
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits

# Import shared utilities (also registers SquareRootScale)
from plotting_utils import compute_chi2_pte


plt.style.use(
    "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle"
)


def _compute_pte_with_cuts(data, covariance, ell, ell_min, ell_max):
    """Compute PTE for B-mode null test with scale cuts."""
    mask = (ell >= ell_min) & (ell <= ell_max)
    data_cut = data[mask]
    cov_cut = covariance[np.ix_(mask, mask)]
    chi2, pte, dof = compute_chi2_pte(data_cut, cov_cut)
    return pte, chi2, dof


def _load_pseudo_cl_data(pseudo_cl_path, pseudo_cl_cov_path):
    """Load pseudo-Cl data and covariance from FITS files."""
    hdu = fits.open(pseudo_cl_path)
    data = hdu["PSEUDO_CELL"].data
    hdu.close()

    ell = data["ELL"]
    cl_eb = data["EB"]
    cl_bb = data["BB"]

    hdu_cov = fits.open(pseudo_cl_cov_path)
    cov_eb = hdu_cov["COVAR_EB_EB"].data
    cov_bb = hdu_cov["COVAR_BB_BB"].data
    hdu_cov.close()

    sigma_eb = np.sqrt(np.diag(cov_eb))
    sigma_bb = np.sqrt(np.diag(cov_bb))

    return ell, cl_bb, cl_eb, cov_bb, cov_eb, sigma_bb, sigma_eb


def _create_cl_figure(ell, cl_bb, cl_eb, sigma_bb, sigma_eb, ell_min_cut, ell_max_cut, label_suffix=""):
    """Create two-panel Cl figure.

    Args:
        ell_min_cut: Lower scale cut for shading excluded region
        ell_max_cut: Upper scale cut for shading excluded region
        label_suffix: Optional suffix for labels (e.g., " (uncorrected)")
    """
    fig, (ax_bb, ax_eb) = plt.subplots(2, 1, figsize=(7.24, 5.0), sharex=True)

    sns.set_palette("husl", 2)
    colors = sns.color_palette()
    color_bb = colors[0]
    color_eb = colors[1]

    minor_ticks = [i * 10 for i in range(1, 10)] + [i * 100 for i in range(1, 21)]

    # BB panel
    bb_label = rf"$C_\ell^{{BB}}${label_suffix}"
    ax_bb.errorbar(
        ell, cl_bb / sigma_bb, yerr=np.ones_like(cl_bb),
        fmt="o", mfc=color_bb, mec=color_bb, color=color_bb,
        capsize=2, markersize=4, linewidth=1.0, label=bb_label,
    )
    ax_bb.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.8)
    ax_bb.set_xscale("squareroot")
    ax_bb.set_ylabel(r"$C_\ell / \sigma$")
    ax_bb.grid(True, which="major", axis="both", alpha=0.3)
    ax_bb.legend(loc="upper left", framealpha=0.9)

    # EB panel
    eb_label = rf"$C_\ell^{{EB}}${label_suffix}"
    ax_eb.errorbar(
        ell, cl_eb / sigma_eb, yerr=np.ones_like(cl_eb),
        fmt="s", mfc="none", mec=color_eb, color=color_eb,
        capsize=2, markersize=4, linewidth=1.0, label=eb_label,
    )
    ax_eb.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.8)
    ax_eb.set_xscale("squareroot")
    ax_eb.set_xlabel(r"$\ell$")
    ax_eb.set_ylabel(r"$C_\ell / \sigma$")
    ax_eb.grid(True, which="major", axis="both", alpha=0.3)
    ax_eb.legend(loc="upper left", framealpha=0.9)

    # Apply shading and ticks to both panels
    for ax in [ax_bb, ax_eb]:
        ell_min_data = ell.min()
        ell_max_data = ell.max()
        ax.set_xlim(ell_min_data * 0.95, ell_max_data * 1.05)

        xlim = ax.get_xlim()
        ax.axvspan(xlim[0], ell_min_cut, alpha=0.1, color="gray", zorder=0)
        ax.axvspan(ell_max_cut, xlim[1], alpha=0.1, color="gray", zorder=0)
        ax.set_xlim(xlim)

        ax.set_xticks(np.array([100, 400, 900, 1600]))
        ax.minorticks_on()
        ax.tick_params(axis="x", which="minor", length=2, width=0.8)
        ax.set_xticks(minor_ticks, minor=True)

    plt.tight_layout()
    return fig


def main():
    ell_min_cut = int(snakemake.params.ell_min_cut)
    ell_max_cut = int(snakemake.params.ell_max_cut)

    # Load leak-corrected data (main figure)
    ell, cl_bb, cl_eb, cov_bb, cov_eb, sigma_bb, sigma_eb = _load_pseudo_cl_data(
        snakemake.input["pseudo_cl"], snakemake.input["pseudo_cl_cov"]
    )

    # Compute PTEs (null tests) using full ell range
    chi2_eb_full, pte_eb_full, dof_eb_full = compute_chi2_pte(cl_eb, cov_eb)
    chi2_bb_full, pte_bb_full, dof_bb_full = compute_chi2_pte(cl_bb, cov_bb)

    # Compute PTEs with scale cuts
    pte_eb_cut, chi2_eb_cut, dof_eb_cut = _compute_pte_with_cuts(
        cl_eb, cov_eb, ell, ell_min_cut, ell_max_cut
    )
    pte_bb_cut, chi2_bb_cut, dof_bb_cut = _compute_pte_with_cuts(
        cl_bb, cov_bb, ell, ell_min_cut, ell_max_cut
    )

    # Create output directory
    output_dir = Path(snakemake.output["evidence"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Main figure (leak-corrected, unlabeled) ---
    fig = _create_cl_figure(
        ell, cl_bb, cl_eb, sigma_bb, sigma_eb, ell_min_cut, ell_max_cut, label_suffix=""
    )
    fig_path = Path(snakemake.output["figure"])
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_path}")

    # Copy to paper figures
    paper_path = Path(snakemake.output["paper_figure"])
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fig_path, paper_path)
    print(f"Copied to {paper_path}")
    plt.close(fig)

    # --- Companion figure (uncorrected, labeled) ---
    ell_uc, cl_bb_uc, cl_eb_uc, _, _, sigma_bb_uc, sigma_eb_uc = _load_pseudo_cl_data(
        snakemake.input["pseudo_cl_uncorr"], snakemake.input["pseudo_cl_cov_uncorr"]
    )
    fig_uc = _create_cl_figure(
        ell_uc, cl_bb_uc, cl_eb_uc, sigma_bb_uc, sigma_eb_uc, ell_min_cut, ell_max_cut,
        label_suffix=" (uncorrected)"
    )
    fig_uc_path = Path(snakemake.output["figure_uncorrected"])
    fig_uc.savefig(fig_uc_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_uc_path}")
    plt.close(fig_uc)

    # Build evidence (based on leak-corrected data only)
    spec_paths = snakemake.input["specs"]

    evidence_data = {
        "spec_id": "cl_data_vector",
        "spec_path": spec_paths[0],
        "generated": datetime.now().isoformat(),
        "evidence": {
            # Full range PTEs
            "pte_eb_full": float(pte_eb_full),
            "chi2_eb_full": float(chi2_eb_full),
            "dof_eb_full": int(dof_eb_full),
            "pte_bb_full": float(pte_bb_full),
            "chi2_bb_full": float(chi2_bb_full),
            "dof_bb_full": int(dof_bb_full),
            # Scale cut PTEs
            "pte_eb_cut": float(pte_eb_cut),
            "chi2_eb_cut": float(chi2_eb_cut),
            "dof_eb_cut": int(dof_eb_cut),
            "pte_bb_cut": float(pte_bb_cut),
            "chi2_bb_cut": float(chi2_bb_cut),
            "dof_bb_cut": int(dof_bb_cut),
            # Scale cut values
            "ell_min_cut": int(ell_min_cut),
            "ell_max_cut": int(ell_max_cut),
            # Data range
            "ell_min": float(ell.min()),
            "ell_max": float(ell.max()),
            "n_ell_bins": int(len(ell)),
        },
        "artifacts": {
            "figure": fig_path.name,
            "figure_uncorrected": fig_uc_path.name,
        },
    }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"Saved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
