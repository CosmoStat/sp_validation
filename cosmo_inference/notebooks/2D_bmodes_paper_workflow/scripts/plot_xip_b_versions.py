# %%
"""Cross-version comparison of ξ+ and ξ+^B (B-mode component only)."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
            "results/paper_plots/xip_b_versions.png",
            str(Path.cwd()),
        )
    from snakemake.script import snakemake

    return snakemake


snakemake = _load_snakemake()

params = snakemake.params


def _extract_sigma(covariance, block_index, block_size):
    """Extract standard deviations from diagonal of covariance block."""
    block_slice = slice(block_index * block_size, (block_index + 1) * block_size)
    block = covariance[block_slice, block_slice]
    return np.sqrt(np.clip(np.diag(block), 0, None))


def _version_label(version):
    """Convert version string to clean label."""
    return version.replace("SP_", "").replace("_leak_corr", "")


def main():
    versions = params["versions"]
    palette = sns.color_palette("colorblind", len(versions))

    # Get fiducial scale cuts from config
    fiducial_config = snakemake.config["fiducial"]
    fiducial_xip_scale_cut = tuple(fiducial_config["fiducial_xip_scale_cut"])
    fiducial_xim_scale_cut = tuple(fiducial_config["fiducial_xim_scale_cut"])

    # Load data for all versions
    datasets = []
    for version, color, data_path in zip(versions, palette, snakemake.input["pure_eb_data"]):
        dataset = np.load(data_path)
        theta = dataset["theta"]
        nbins = len(theta)

        # Extract ξ+/- total and B-modes
        xip_total = dataset["xip_total"]
        xim_total = dataset["xim_total"]
        xip_B = dataset["xip_B"]
        xim_B = dataset["xim_B"]

        # Extract errors
        cov_xip_xim = dataset["cov_xip_xim"]
        cov_pure_eb = dataset["cov_pure_eb"]

        sigma_xip_total = _extract_sigma(cov_xip_xim, 0, nbins)
        sigma_xim_total = _extract_sigma(cov_xip_xim, 1, nbins)
        sigma_xip_B = _extract_sigma(cov_pure_eb, 2, nbins)
        sigma_xim_B = _extract_sigma(cov_pure_eb, 3, nbins)

        datasets.append({
            "version": version,
            "label": _version_label(version),
            "color": color,
            "theta": theta,
            "xip_total": xip_total,
            "xim_total": xim_total,
            "xip_B": xip_B,
            "xim_B": xim_B,
            "sigma_xip_total": sigma_xip_total,
            "sigma_xim_total": sigma_xim_total,
            "sigma_xip_B": sigma_xip_B,
            "sigma_xim_B": sigma_xim_B,
        })

    # Create 2x2 figure (top row: xi_total, bottom row: xi_B)
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
                         'legend.fontsize': 12, 'xtick.labelsize': 14, 'ytick.labelsize': 14})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    # Scale factor for readability
    scale_factor = 1e-4

    # X-offset factors for each version
    offsets = [0.95, 1.0, 1.05]

    # Plot each version
    for data, x_offset in zip(datasets, offsets):
        theta = data["theta"]
        theta_offset = theta * x_offset
        color = data["color"]
        label = data["label"]

        # Top row: xi_total only (xi+ left, xi- right)
        axes[0, 0].errorbar(
            theta_offset, theta * data["xip_total"] / scale_factor,
            yerr=theta * data["sigma_xip_total"] / scale_factor,
            fmt=".", color=color, ms=7, alpha=0.8,
            capsize=3, label=label
        )
        axes[0, 1].errorbar(
            theta_offset, theta * data["xim_total"] / scale_factor,
            yerr=theta * data["sigma_xim_total"] / scale_factor,
            fmt=".", color=color, ms=7, alpha=0.8,
            capsize=3
        )

        # Bottom row: xi_B only (xi+ left, xi- right)
        axes[1, 0].errorbar(
            theta_offset, theta * data["xip_B"] / scale_factor,
            yerr=theta * data["sigma_xip_B"] / scale_factor,
            fmt="X", mfc="none", color=color, ms=4, alpha=0.9,
            capsize=2, capthick=1.5, label=label
        )
        axes[1, 1].errorbar(
            theta_offset, theta * data["xim_B"] / scale_factor,
            yerr=theta * data["sigma_xim_B"] / scale_factor,
            fmt="X", mfc="none", color=color, ms=4, alpha=0.9,
            capsize=2, capthick=1.5
        )

    # Styling with fiducial scale cuts
    for col_idx, (col_label, scale_cut) in enumerate(zip([r"$\xi_+$", r"$\xi_-$"],
                                                           [fiducial_xip_scale_cut, fiducial_xim_scale_cut])):
        for row_idx in range(2):
            ax = axes[row_idx, col_idx]
            # Highlight fiducial scale range
            ax.axvspan(scale_cut[0], scale_cut[1], color="0.95", zorder=0, alpha=0.5)
            ax.axhline(0, color="k", linestyle="--", alpha=0.6, linewidth=0.8)
            ax.set_xscale("log")
            ax.set_xlim(1, 250)

            # Only bottom row gets x-label
            if row_idx == 1:
                ax.set_xlabel(r"$\theta$ [arcmin]")

            # Only top row gets title
            if row_idx == 0:
                ax.set_title(col_label)

            # Y-labels on left column only
            if col_idx == 0:
                ax.set_ylabel(r"$\theta \xi \times 10^4$")

    # Legends (no PTE values, just version labels)
    axes[0, 0].legend(loc="upper left")
    axes[1, 0].legend(loc="upper left")

    # Set y-limits (auto-scale per row)
    axes[0, 0].set_ylim(-0.5, 1.5)

    fig.suptitle(r"$\xi_{\pm}$ and $\xi_{\pm}^B$ Comparison Across Catalog Versions", y=0.93, fontsize=24)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    # Save
    output_path = Path(snakemake.output[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()