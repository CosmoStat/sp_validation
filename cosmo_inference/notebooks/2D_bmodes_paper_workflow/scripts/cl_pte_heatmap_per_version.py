# %%
"""Generate Cl PTE heatmap figure for a specific catalog version.

Simplified version of claim_figure_14_cl_pte_heatmap.py for per-version output.
No epistemic trace — just the figure.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
from astropy.io import fits
from scipy import stats

from IPython import get_ipython

ipython = get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
else:
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# Apply paper style
plt.style.use(
    "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle"
)


def _load_snakemake():
    if hasattr(sys, "ps1"):
        from snakemake_helpers import snakemake_interactive

        return snakemake_interactive(
            "docs/unions_release/unions_bmodes/Figures/cl_pte_heatmap_SP_v1.4.6_leak_corr.png",
            str(Path.cwd()),
        )
    from snakemake.script import snakemake

    return snakemake


snakemake = _load_snakemake()

params = snakemake.params


def _compute_pte(data, covariance):
    """Compute PTE for B-mode null test."""
    chi2 = float(data @ np.linalg.solve(covariance, data))
    dof = len(data)
    pte = stats.chi2.sf(chi2, dof)
    return pte, chi2, dof


def _version_label(version):
    """Convert version string to label for display."""
    return version.replace("SP_", "").replace("_leak_corr", "")


def main():
    version = params["version"]

    # Load pseudo-Cl data
    hdu = fits.open(snakemake.input["pseudo_cl"])
    data = hdu["PSEUDO_CELL"].data
    hdu.close()

    ell = data["ELL"]
    cl_bb = data["BB"]
    n_ell = len(ell)

    # Load covariances
    hdu_cov = fits.open(snakemake.input["pseudo_cl_cov"])
    cov_bb = hdu_cov["COVAR_BB_BB"].data
    hdu_cov.close()

    print(f"Loaded {n_ell} ell bins for {version}: ell = [{ell.min():.1f}, {ell.max():.1f}]")

    # Compute PTE matrix over all (ell_min_idx, ell_max_idx) combinations
    pte_bb_matrix = np.full((n_ell, n_ell), np.nan)

    for i_min in range(n_ell):
        for i_max in range(i_min + 1, n_ell):
            # Extract data and covariance for this ell range
            idx_slice = slice(i_min, i_max + 1)

            bb_slice = cl_bb[idx_slice]
            cov_bb_slice = cov_bb[idx_slice, idx_slice]

            # Compute BB PTE (null test)
            try:
                pte_bb, chi2_bb, dof_bb = _compute_pte(bb_slice, cov_bb_slice)
                pte_bb_matrix[i_min, i_max] = pte_bb
            except np.linalg.LinAlgError:
                # Singular matrix - leave as NaN
                pass

    # Compute statistics
    valid_ptes = pte_bb_matrix[~np.isnan(pte_bb_matrix)]
    frac_healthy = float(
        np.sum((valid_ptes >= 0.05) & (valid_ptes <= 0.95)) / len(valid_ptes)
    )

    print(f"\nPTE matrix statistics for {version}:")
    print(f"  Valid scale cuts: {len(valid_ptes)}")
    print(f"  Fraction healthy (PTE in [0.05, 0.95]): {frac_healthy:.1%}")

    # Create single-pane figure: BB PTE heatmap
    # Size: single-column width, square (A&A standard)
    fig, ax = plt.subplots(1, 1, figsize=(3.54, 3.54))

    # Set up colormap: vlag for B-mode PTE
    vlag_cmap = sns.color_palette("vlag", as_cmap=True).copy()
    vlag_cmap.set_bad(color="lightgray")

    # Transpose for plotting (lower ell cut on x-axis, upper on y-axis)
    pte_plot_data = pte_bb_matrix.T

    # BB PTE heatmap with contours
    im = ax.imshow(
        pte_plot_data,
        origin="lower",
        aspect="equal",
        cmap=vlag_cmap,
        vmin=0,
        vmax=1,
        extent=[0, n_ell, 0, n_ell],
    )

    # Add contours at significance levels
    cs = ax.contour(
        pte_plot_data,
        levels=[0.05, 0.95],
        colors="black",
        linewidths=0.8,
        extent=[0, n_ell, 0, n_ell],
    )
    ax.clabel(cs, inline=True, fmt="%.2f")

    # Add version label to title
    version_label = _version_label(version)
    ax.set_title(rf"$C_\ell^{{BB}}$ PTE ({version_label})")
    ax.set_xlabel(r"Lower $\ell$ cut")
    ax.set_ylabel(r"Upper $\ell$ cut")

    # Mark the full ell range (fiducial) with hatched rectangle
    fid_idx_min = 0
    fid_idx_max = n_ell - 1
    ax.add_patch(
        Rectangle(
            (fid_idx_min, fid_idx_max),
            1,
            1,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
            hatch="///",
            alpha=0.8,
        )
    )

    # Set tick labels to show ell values
    tick_step = max(1, n_ell // 5)
    tick_indices = np.arange(0, n_ell, tick_step)
    x_tick_labels = [f"{ell[i]:.0f}" for i in tick_indices]
    y_tick_labels = [f"{ell[i]:.0f}" for i in tick_indices]
    x_tick_positions = tick_indices + 0.5
    y_tick_positions = tick_indices + 0.5

    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("PTE")

    plt.tight_layout()

    # Save plot
    output_path = Path(snakemake.output["figure"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
