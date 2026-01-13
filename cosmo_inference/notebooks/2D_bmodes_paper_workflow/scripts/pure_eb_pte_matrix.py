"""Pure E/B PTE heatmap across scale cut combinations.

Claim: Fiducial angular scale cuts justified by PTE heatmaps.
Different cuts for xi+ [6, 85] and xi- [15, 85] arcmin.
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
import yaml


plt.style.use(
    "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle"
)


def _compute_pte_stats(pte_matrix, fid_start, fid_stop):
    """Compute PTE statistics for a given matrix."""
    valid_ptes = pte_matrix[~np.isnan(pte_matrix)]

    if len(valid_ptes) > 0:
        frac_healthy = float(
            np.sum((valid_ptes >= 0.05) & (valid_ptes <= 0.95)) / len(valid_ptes)
        )
        pte_at_fid = float(pte_matrix[fid_start, fid_stop])
        return {
            "pte_at_fiducial_cut": pte_at_fid,
            "fraction_healthy": frac_healthy,
            "n_scale_cuts_evaluated": int(len(valid_ptes)),
            "pte_statistics": {
                "mean": float(np.nanmean(valid_ptes)),
                "median": float(np.nanmedian(valid_ptes)),
                "std": float(np.nanstd(valid_ptes)),
                "min": float(np.nanmin(valid_ptes)),
                "max": float(np.nanmax(valid_ptes)),
            },
        }
    return {
        "pte_at_fiducial_cut": float("nan"),
        "fraction_healthy": float("nan"),
        "n_scale_cuts_evaluated": 0,
        "pte_statistics": {},
    }


def _create_pte_figure(pte_matrix, theta, title, fid_start, fid_stop, output_path):
    """Create a single-pane PTE heatmap figure."""
    nbins = len(theta)

    vlag_cmap = sns.color_palette("vlag", as_cmap=True).copy()
    vlag_cmap.set_bad(color="lightgray")

    fig, ax = plt.subplots(figsize=(3.54, 3.54))

    im = ax.imshow(
        pte_matrix.T, origin="lower", aspect="equal",
        cmap=vlag_cmap, vmin=0, vmax=1, extent=[0, nbins, 0, nbins],
    )

    cs = ax.contour(
        pte_matrix.T, levels=[0.05, 0.95], colors="black",
        linewidths=0.8, extent=[0, nbins, 0, nbins],
    )
    ax.clabel(cs, inline=True)

    # Fiducial scale cut marker
    ax.add_patch(
        Rectangle((fid_start, fid_stop), 1, 1, fill=False, edgecolor="black",
                  linewidth=1.5, hatch="///", alpha=0.8)
    )

    ax.set_xlabel("Lower scale cut")
    ax.set_ylabel("Upper scale cut")
    ax.set_title(title)

    # Tick labels
    tick_step = max(1, nbins // 6)
    tick_indices = np.arange(0, nbins, tick_step)
    tick_labels = [f"{theta[i]:.1f}" for i in tick_indices]
    tick_positions = tick_indices + 0.5

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("PTE", rotation=270, labelpad=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {title} figure to {output_path}")
    plt.close(fig)


def main():
    from snakemake.script import snakemake

    # Read config
    with open(snakemake.input["config"]) as f:
        config = yaml.safe_load(f)

    fiducial_config = config["fiducial"]
    xip_cut = tuple(fiducial_config["fiducial_xip_scale_cut"])
    xim_cut = tuple(fiducial_config["fiducial_xim_scale_cut"])

    # Load PTE matrices
    pte_data = np.load(snakemake.input["pte_data"])
    theta = pte_data["theta"]
    pte_xip_B = pte_data["pte_xip_B"]
    pte_xim_B = pte_data["pte_xim_B"]

    # Find fiducial indices
    xip_start = np.argmin(np.abs(theta - xip_cut[0]))
    xip_stop = np.argmin(np.abs(theta - xip_cut[1]))
    xim_start = np.argmin(np.abs(theta - xim_cut[0]))
    xim_stop = np.argmin(np.abs(theta - xim_cut[1]))

    # Compute statistics
    xip_stats = _compute_pte_stats(pte_xip_B, xip_start, xip_stop)
    xim_stats = _compute_pte_stats(pte_xim_B, xim_start, xim_stop)

    print(f"xi+ PTE at fiducial: {xip_stats['pte_at_fiducial_cut']:.4f}")
    print(f"xi- PTE at fiducial: {xim_stats['pte_at_fiducial_cut']:.4f}")

    # Create output directory
    output_dir = Path(snakemake.output["evidence"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figures
    fig_xip_path = Path(snakemake.output["figure_xip"])
    fig_xim_path = Path(snakemake.output["figure_xim"])

    _create_pte_figure(pte_xip_B, theta, r"$\xi_+^B$ PTE", xip_start, xip_stop, fig_xip_path)
    _create_pte_figure(pte_xim_B, theta, r"$\xi_-^B$ PTE", xim_start, xim_stop, fig_xim_path)

    # Copy to paper figures
    paper_xip_path = Path(snakemake.output["paper_figure_xip"])
    paper_xim_path = Path(snakemake.output["paper_figure_xim"])
    paper_xip_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(fig_xip_path, paper_xip_path)
    shutil.copy2(fig_xim_path, paper_xim_path)
    print(f"Copied to paper figures")

    # Build evidence
    spec_paths = snakemake.input["specs"]

    evidence_data = {
        "spec_id": "pure_eb_pte_matrix",
        "spec_path": spec_paths[0],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "xip": xip_stats,
            "xim": xim_stats,
            "fiducial_xip_scale_cut": list(xip_cut),
            "fiducial_xim_scale_cut": list(xim_cut),
            "theta_range": [float(theta.min()), float(theta.max())],
            "n_theta_bins": int(len(theta)),
        },
        "artifacts": {
            "figure_xip": fig_xip_path.name,
            "figure_xim": fig_xim_path.name,
        },
    }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"Saved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
