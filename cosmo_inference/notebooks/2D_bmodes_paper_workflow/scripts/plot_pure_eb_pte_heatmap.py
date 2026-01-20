# %%
"""Plot PTE heatmaps for Pure E/B scale cut robustness (per-version)."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
            "results/paper_plots/pure_eb_pte_heatmap_v1.4.6.png",
            str(Path.cwd()),
        )
    from snakemake.script import snakemake

    return snakemake


snakemake = _load_snakemake()

params = snakemake.params


def main():
    # Load PTE matrices
    pte_data = np.load(snakemake.input["pte_data"])
    theta = pte_data["theta"]
    pte_xip_B = pte_data["pte_xip_B"]
    pte_xim_B = pte_data["pte_xim_B"]
    nbins = len(theta)

    # Get fiducial scale cuts
    fiducial_config = snakemake.config["fiducial"]
    xip_cut = tuple(fiducial_config["fiducial_xip_scale_cut"])
    xim_cut = tuple(fiducial_config["fiducial_xim_scale_cut"])

    # Find bin indices for fiducial cuts
    xip_start = np.argmin(np.abs(theta - xip_cut[0]))
    xip_stop = np.argmin(np.abs(theta - xip_cut[1]))
    xim_start = np.argmin(np.abs(theta - xim_cut[0]))
    xim_stop = np.argmin(np.abs(theta - xim_cut[1]))

    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # Set up colormap
    vlag_cmap = sns.color_palette("vlag", as_cmap=True).copy()
    vlag_cmap.set_bad(color='lightgray')
    contour_levels = [0.05, 0.95]

    # Plot configurations
    plot_configs = [
        (axs[0], pte_xip_B.T, r"$\xi_+^B$ PTE", xip_start, xip_stop),
        (axs[1], pte_xim_B.T, r"$\xi_-^B$ PTE", xim_start, xim_stop),
    ]

    for ax_idx, (ax, pte_matrix, title, fid_start, fid_stop) in enumerate(plot_configs):
        # Plot PTE heatmap
        im = ax.imshow(
            pte_matrix, origin='lower', aspect='auto', cmap=vlag_cmap,
            vmin=0, vmax=1, extent=[0, nbins, 0, nbins]
        )

        # Add contour lines
        cs = ax.contour(
            pte_matrix, levels=contour_levels, colors='black', linewidths=1,
            extent=[0, nbins, 0, nbins]
        )
        ax.clabel(cs, inline=True, fontsize=8)

        # Add fiducial scale cut as hatched rectangle
        rect_x = fid_start
        rect_y = fid_stop - 1
        rect_width = 1
        rect_height = 1

        ax.add_patch(plt.Rectangle(
            (rect_x, rect_y), rect_width, rect_height,
            fill=False, edgecolor='black', linewidth=2,
            hatch='///', alpha=0.8,
            label='Fiducial scale cut' if ax_idx == 0 else None
        ))

        # Set axis labels and title
        ax.set_xlabel('Lower scale cut')
        ax.set_ylabel('Upper scale cut' if ax_idx == 0 else '')
        ax.set_title(title)
        ax.set_aspect('equal')

    # Add colorbar
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('PTE', rotation=270, labelpad=15)

    # Add legend
    axs[0].legend(loc='lower right', fontsize=8)

    # Overall title
    version = params["version"]
    plt.suptitle(f'{version}: PTEs as a Function of Scale Cut', y=0.95)
    plt.tight_layout()

    # Save
    output_path = Path(snakemake.output[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
