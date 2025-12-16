# %%
"""Generate Pure E/B PTE heatmap figures for a specific catalog version.

Simplified version of claim_figure_10_pure_eb_pte_heatmap.py for per-version output.
No epistemic trace — just the figures.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns

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
            "docs/unions_release/unions_bmodes/Figures/pure_eb_pte_xip_SP_v1.4.6_leak_corr.png",
            str(Path.cwd()),
        )
    from snakemake.script import snakemake

    return snakemake


# Figure dimensions: single-column width, square (A&A standard)
FIGURE_SIZE = (3.54, 3.54)


snakemake = _load_snakemake()

params = snakemake.params


def _version_label(version):
    """Convert version string to display label."""
    return version.replace("SP_", "").replace("_leak_corr", "")


def main():
    version = params["version"]

    # Load PTE matrices
    pte_data = np.load(snakemake.input["pte_data"])
    theta = pte_data["theta"]
    pte_xip_B = pte_data["pte_xip_B"]
    pte_xim_B = pte_data["pte_xim_B"]
    nbins = len(theta)

    # Get fiducial scale cuts from config
    fiducial_config = snakemake.config["fiducial"]
    xip_cut = tuple(fiducial_config["fiducial_xip_scale_cut"])
    xim_cut = tuple(fiducial_config["fiducial_xim_scale_cut"])

    # Find bin indices for fiducial cuts
    xip_start = np.argmin(np.abs(theta - xip_cut[0]))
    xip_stop = np.argmin(np.abs(theta - xip_cut[1]))
    xim_start = np.argmin(np.abs(theta - xim_cut[0]))
    xim_stop = np.argmin(np.abs(theta - xim_cut[1]))

    # Set up colormap (vlag for B-mode PTE heatmaps)
    vlag_cmap = sns.color_palette("vlag", as_cmap=True).copy()
    vlag_cmap.set_bad(color="lightgray")

    def create_single_pte_figure(pte_matrix, title, fid_start, fid_stop, output_path):
        """Create a single-pane PTE heatmap figure."""
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        # Plot PTE heatmap
        im = ax.imshow(
            pte_matrix.T,
            origin="lower",
            aspect="equal",
            cmap=vlag_cmap,
            vmin=0,
            vmax=1,
            extent=[0, nbins, 0, nbins],
        )

        # Add contour lines at significance levels
        cs = ax.contour(
            pte_matrix.T,
            levels=[0.05, 0.95],
            colors="black",
            linewidths=0.8,
            extent=[0, nbins, 0, nbins],
        )
        ax.clabel(cs, inline=True)

        # Add fiducial scale cut as hatched rectangle
        ax.add_patch(
            Rectangle(
                (fid_start, fid_stop),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=1.5,
                hatch="///",
                alpha=0.8,
            )
        )

        # Set axis labels and title
        ax.set_xlabel("Lower scale cut")
        ax.set_ylabel("Upper scale cut")
        ax.set_title(title)

        # Set angular scale tick labels
        tick_step = max(1, nbins // 6)  # Fewer ticks for compact figure
        tick_indices = np.arange(0, nbins, tick_step)
        tick_labels = [f"{theta[i]:.1f}" for i in tick_indices]
        tick_positions = tick_indices + 0.5

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("PTE", rotation=270, labelpad=12)

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved {title} figure to {output_path}")
        plt.close(fig)

    # Create output directory
    output_xip = Path(snakemake.output["figure_xip"])
    output_xip.parent.mkdir(parents=True, exist_ok=True)

    # Create xi+ figure
    version_label = _version_label(version)
    create_single_pte_figure(
        pte_xip_B,
        rf"$\xi_+^B$ PTE ({version_label})",
        xip_start,
        xip_stop,
        output_xip,
    )

    # Create xi- figure
    output_xim = Path(snakemake.output["figure_xim"])
    create_single_pte_figure(
        pte_xim_B,
        rf"$\xi_-^B$ PTE ({version_label})",
        xim_start,
        xim_stop,
        output_xim,
    )

    print(f"Generated PTE heatmaps for {version}")


if __name__ == "__main__":
    main()
