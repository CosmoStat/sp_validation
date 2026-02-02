"""Harmonic-space fiducial B-mode power spectrum.

Claim: Fiducial catalog shows C_ell^BB consistent with zero.
Produces single-panel figure with BB and EB on same axis and evidence.json.
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from astropy.io import fits
from scipy import stats


plt.style.use(
    "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle"
)

# Scale cuts from harmonic space paper (Guerrini et al.)
ELL_MIN_CUT = 300
ELL_MAX_CUT = 1600

# X-shift for visual separation (in sqrt(ell) space, corresponds to ~2% shift)
X_SHIFT_FACTOR = 0.02


# SquareRootScale for ell axis
class SquareRootScale(mscale.ScaleBase):
    """Square root scale for x-axis (matches bandpower binning)."""

    name = "squareroot"

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self, axis, **kwargs)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(0.0, vmin), vmax

    class SquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a) ** 0.5

        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()

    class InvertedSquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a) ** 2

        def inverted(self):
            return SquareRootScale.SquareRootTransform()

    def get_transform(self):
        return self.SquareRootTransform()


mscale.register_scale(SquareRootScale)


def _compute_pte(data, covariance):
    """Compute PTE for B-mode null test."""
    chi2 = float(data @ np.linalg.solve(covariance, data))
    dof = len(data)
    pte = stats.chi2.sf(chi2, dof)
    return pte, chi2, dof


def _compute_pte_with_cuts(data, covariance, ell, ell_min, ell_max):
    """Compute PTE for B-mode null test with scale cuts."""
    mask = (ell >= ell_min) & (ell <= ell_max)
    data_cut = data[mask]
    cov_cut = covariance[np.ix_(mask, mask)]
    return _compute_pte(data_cut, cov_cut)


def main():
    from snakemake.script import snakemake

    # Load pseudo-Cl data
    hdu = fits.open(snakemake.input["pseudo_cl"])
    data = hdu["PSEUDO_CELL"].data
    hdu.close()

    ell = data["ELL"]
    cl_eb = data["EB"]
    cl_bb = data["BB"]

    # Load covariances
    hdu_cov = fits.open(snakemake.input["pseudo_cl_cov"])
    cov_eb = hdu_cov["COVAR_EB_EB"].data
    cov_bb = hdu_cov["COVAR_BB_BB"].data
    hdu_cov.close()

    # Extract errors from diagonal
    sigma_eb = np.sqrt(np.diag(cov_eb))
    sigma_bb = np.sqrt(np.diag(cov_bb))

    # Compute PTEs (null tests) using full ell range
    pte_eb_full, chi2_eb_full, dof_eb_full = _compute_pte(cl_eb, cov_eb)
    pte_bb_full, chi2_bb_full, dof_bb_full = _compute_pte(cl_bb, cov_bb)

    # Compute PTEs with scale cuts
    pte_eb_cut, chi2_eb_cut, dof_eb_cut = _compute_pte_with_cuts(
        cl_eb, cov_eb, ell, ELL_MIN_CUT, ELL_MAX_CUT
    )
    pte_bb_cut, chi2_bb_cut, dof_bb_cut = _compute_pte_with_cuts(
        cl_bb, cov_bb, ell, ELL_MIN_CUT, ELL_MAX_CUT
    )

    # Create two-panel figure (vertically stacked, two-column width)
    fig, (ax_bb, ax_eb) = plt.subplots(2, 1, figsize=(7.24, 5.0), sharex=True)

    # Use husl palette for two series
    sns.set_palette("husl", 2)
    colors = sns.color_palette()
    color_bb = colors[0]
    color_eb = colors[1]

    minor_ticks = [i * 10 for i in range(1, 10)] + [i * 100 for i in range(1, 21)]

    # BB panel (circles, filled) - top
    ax_bb.errorbar(
        ell, cl_bb / sigma_bb, yerr=np.ones_like(cl_bb),
        fmt="o", mfc=color_bb, mec=color_bb, color=color_bb,
        capsize=2, markersize=4, linewidth=1.0, label=r"$C_\ell^{BB}$",
    )
    ax_bb.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.8)
    ax_bb.set_xscale("squareroot")
    ax_bb.set_ylabel(r"$C_\ell / \sigma$")
    ax_bb.grid(True, which="major", axis="both", alpha=0.3)
    ax_bb.legend(loc="upper left", framealpha=0.9)

    # EB panel (squares, unfilled) - bottom
    ax_eb.errorbar(
        ell, cl_eb / sigma_eb, yerr=np.ones_like(cl_eb),
        fmt="s", mfc="none", mec=color_eb, color=color_eb,
        capsize=2, markersize=4, linewidth=1.0, label=r"$C_\ell^{EB}$",
    )
    ax_eb.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.8)
    ax_eb.set_xscale("squareroot")
    ax_eb.set_xlabel(r"$\ell$")
    ax_eb.set_ylabel(r"$C_\ell / \sigma$")
    ax_eb.grid(True, which="major", axis="both", alpha=0.3)
    ax_eb.legend(loc="upper left", framealpha=0.9)

    # Apply shading and ticks to both panels
    for ax in [ax_bb, ax_eb]:
        # Get data range for shading
        ell_min_data = ell.min()
        ell_max_data = ell.max()

        # Set x limits to data range before shading
        ax.set_xlim(ell_min_data * 0.95, ell_max_data * 1.05)

        # Shade excluded regions (extending to plot edges)
        xlim = ax.get_xlim()
        ax.axvspan(xlim[0], ELL_MIN_CUT, alpha=0.1, color="gray", zorder=0)
        ax.axvspan(ELL_MAX_CUT, xlim[1], alpha=0.1, color="gray", zorder=0)

        # Restore limits (shading may have expanded them)
        ax.set_xlim(xlim)

        # Ticks
        ax.set_xticks(np.array([100, 400, 900, 1600]))
        ax.minorticks_on()
        ax.tick_params(axis="x", which="minor", length=2, width=0.8)
        ax.set_xticks(minor_ticks, minor=True)

    plt.tight_layout()

    # Save figure
    output_dir = Path(snakemake.output["evidence"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = Path(snakemake.output["figure"])
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_path}")

    # Copy to paper figures
    paper_path = Path(snakemake.output["paper_figure"])
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fig_path, paper_path)
    print(f"Copied to {paper_path}")

    plt.close(fig)

    # Build evidence
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
            "ell_min_cut": int(ELL_MIN_CUT),
            "ell_max_cut": int(ELL_MAX_CUT),
            # Data range
            "ell_min": float(ell.min()),
            "ell_max": float(ell.max()),
            "n_ell_bins": int(len(ell)),
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
