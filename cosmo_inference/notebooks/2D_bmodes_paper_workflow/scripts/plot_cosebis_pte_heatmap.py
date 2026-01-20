# %%
"""Plot PTE heatmaps for COSEBIS E/B scale cut robustness (per-version)."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
            "results/paper_plots/cosebis_pte_heatmap_v1.4.6.png",
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
    pte_chi2_E = pte_data["pte_chi2_E"]
    pte_B = pte_data["pte_B"]

    # Get fiducial scale cut
    fiducial_config = snakemake.config["fiducial"]
    scale_cut = (
        int(fiducial_config["fiducial_min_scale"]),
        int(fiducial_config["fiducial_max_scale"])
    )

    # Find bin indices for fiducial cut
    fid_start = np.argmin(np.abs(theta - scale_cut[0]))
    fid_stop = np.argmin(np.abs(theta - scale_cut[1]))

    # Create figure
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot configurations
    plot_configs = [
        (axes[0], pte_chi2_E, r"E-mode $\chi^2$"),
        (axes[1], pte_B, r"B-mode PTE"),
    ]

    for ax, pte_matrix, title in plot_configs:
        # Plot PTE heatmap
        im = ax.imshow(
            pte_matrix, cmap="vlag", vmin=0, vmax=1,
            origin="lower", aspect="auto", interpolation="nearest"
        )

        # Mark fiducial scale cut
        ax.plot([fid_stop], [fid_start], marker="*", color="red",
                markersize=15, markeredgecolor="white", markeredgewidth=1.5,
                label="Fiducial cut")

        # Axis labels
        ax.set_xlabel(r"$\theta_\mathrm{max}$ bin index")
        ax.set_ylabel(r"$\theta_\mathrm{min}$ bin index")
        ax.set_title(title)
        ax.legend(loc="upper right", framealpha=0.9)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("PTE", rotation=270, labelpad=20)

    # Overall title
    version_label = params["version"].replace("SP_", "").replace("_leak_corr", "")
    fig.suptitle(f"{version_label}: COSEBIS Scale Cut Robustness", fontsize=18, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    # Save
    output_path = Path(snakemake.output[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
