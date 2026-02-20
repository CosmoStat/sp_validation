"""Pure E/B version comparison claim.

Visualizes total and B-mode correlation functions across catalog versions.
Top row: xi_total +/- (same style as data vector plot)
Bottom row: xi_B +/- normalized by error (B / sigma)
Data outside fiducial scale cuts shown greyed out.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from plotting_utils import (
    ERRORBAR_DEFAULTS,
    FIG_WIDTH_FULL,
    MARKER_STYLES,
    PAPER_MPLSTYLE,
    compute_chi2_pte,
    draw_normalized_boxes_log_scale,
    find_fiducial_index,
    get_box_style,
    get_version_alpha,
    version_label,
)


plt.style.use(PAPER_MPLSTYLE)


def _extract_sigma(covariance, block_index, block_size):
    """Extract standard deviations from diagonal of covariance block."""
    block_slice = slice(block_index * block_size, (block_index + 1) * block_size)
    block = covariance[block_slice, block_slice]
    return np.sqrt(np.clip(np.diag(block), 0, None))



def _create_version_comparison_figure(datasets, scale_cuts, fiducial_version,
                                       x_offset_factors=None, box_style=None):
    """Create figure comparing total and B-mode correlations across versions.

    Layout:
    - Top row (2/3 height): xi_total + (left), xi_total - (right)
      Plotted as theta * xi / 1e-4 (matching data vector style)
    - Bottom row (1/3 height): xi_B + (left), xi_B - (right)
      Plotted as B / sigma (normalized)

    Data outside fiducial scale cuts shown with light axvspan shading.
    For each bin, a box spans the range of values across versions,
    with a line marking the fiducial version's value.

    Parameters
    ----------
    x_offset_factors : list, optional
        Multiplicative offsets for each version (for log scale). Defaults to
        [0.88, 0.96, 1.04, 1.12] for tighter spacing.
    box_style : dict, optional
        Styling for version boxes. See _draw_version_boxes for keys.
    """
    if x_offset_factors is None:
        x_offset_factors = [0.92, 0.97, 1.03, 1.08]

    # Create figure with custom height ratios: top row 2x height of bottom
    # Bottom row shares y-axis
    fig, axes = plt.subplots(
        2, 2,
        figsize=(FIG_WIDTH_FULL, FIG_WIDTH_FULL * 0.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    # Share y-axis for bottom row
    axes[1, 1].sharey(axes[1, 0])

    scale_factor = 1e-4

    fiducial_idx = find_fiducial_index(datasets, fiducial_version)
    theta = datasets[0]["theta"]
    x_offset_range = (min(x_offset_factors), max(x_offset_factors))

    # Plotting parameters (matching data vector style)
    ms = 2.5
    capsize = 1.5
    capthick = 0.3
    elinewidth = 0.4
    mew = 0.4

    legend_handles = []
    legend_labels = []

    # Top row: xi_total +/-
    # Pre-compute theta-scaled values for box drawing
    for data in datasets:
        for key in ("xip_total", "xim_total"):
            data[f"{key}_scaled"] = (theta * data[key]) / scale_factor

    for col, (mode_key, title) in enumerate([
        ("xip_total", r"$\xi_+$"),
        ("xim_total", r"$\xi_-$"),
    ]):
        ax = axes[0, col]

        # Draw version spread boxes (before data points so they're behind)
        draw_normalized_boxes_log_scale(
            ax, theta, datasets,
            y_norm_key=f"{mode_key}_scaled",
            fiducial_idx=fiducial_idx,
            x_offset_range=x_offset_range,
            box_style=box_style,
        )

        for i, data in enumerate(datasets):
            theta_i = data["theta"]
            offset = x_offset_factors[i] if i < len(x_offset_factors) else 1.0
            marker = data.get("marker", MARKER_STYLES[i] if i < len(MARKER_STYLES) else "o")
            fillstyle = data.get("fillstyle", "full")
            mfc = data["color"] if fillstyle == "full" else "none"

            y = data[mode_key]
            sigma = data[f"sigma_{mode_key}"]

            # Plot all points (full color)
            line = ax.errorbar(
                theta_i * offset,
                (theta_i * y) / scale_factor,
                yerr=(theta_i * sigma) / scale_factor,
                fmt=marker,
                color=data["color"],
                alpha=data["alpha"],
                markerfacecolor=mfc,
                markeredgecolor=data["color"],
                markeredgewidth=mew,
                markersize=ms,
                capsize=capsize,
                capthick=capthick,
                elinewidth=elinewidth,
                zorder=2,
            )

            if col == 0:
                legend_handles.append(line)
                legend_labels.append(data["label"])

        # No y=0 line in upper row
        ax.set_xscale("log")
        ax.set_xlim(1, 250)
        ax.set_title(title)

        if col == 0:
            ax.set_ylabel(r"$\theta \xi \times 10^4$")

    # Bottom row: xi_B +/- normalized
    for col, (mode_key, title) in enumerate([
        ("xip_B", r"$\xi_+^{\mathrm{B}} / \sigma$"),
        ("xim_B", r"$\xi_-^{\mathrm{B}} / \sigma$"),
    ]):
        ax = axes[1, col]

        # Draw version spread boxes (before data points)
        draw_normalized_boxes_log_scale(
            ax, theta, datasets,
            y_norm_key=f"{mode_key}_normalized",
            fiducial_idx=fiducial_idx,
            x_offset_range=x_offset_range,
            box_style=box_style,
        )

        for i, data in enumerate(datasets):
            theta_i = data["theta"]
            offset = x_offset_factors[i] if i < len(x_offset_factors) else 1.0
            marker = data.get("marker", MARKER_STYLES[i] if i < len(MARKER_STYLES) else "o")
            fillstyle = data.get("fillstyle", "full")
            mfc = data["color"] if fillstyle == "full" else "none"

            y_norm = data[f"{mode_key}_normalized"]

            # Plot all points (full color, error bars = 1 by construction)
            ax.errorbar(
                theta_i * offset,
                y_norm,
                yerr=np.ones(len(theta_i)),
                fmt=marker,
                color=data["color"],
                alpha=data["alpha"],
                markerfacecolor=mfc,
                markeredgecolor=data["color"],
                markeredgewidth=mew,
                markersize=ms,
                capsize=capsize,
                capthick=capthick,
                elinewidth=elinewidth,
                zorder=2,
            )

        ax.axhline(0, color="k", linestyle="--", alpha=0.6, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlim(1, 250)
        ax.set_xlabel(r"$\theta$ [arcmin]")
        ax.set_title(title)

        if col == 0:
            ax.set_ylabel(r"$\xi^B / \sigma$")

    # Legend in top-left of upper-right panel
    axes[0, 1].legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        frameon=True,
        framealpha=0.9,
    )

    # Add axvspan shading for excluded regions (outside scale cuts)
    xlim = (1, 250)
    for col, scale_cut_key in enumerate(["fiducial_xip", "fiducial_xim"]):
        cut = scale_cuts[scale_cut_key]
        for row in range(2):
            ax = axes[row, col]
            # Shade below lower scale cut
            ax.axvspan(xlim[0], cut[0], alpha=0.1, color="gray", zorder=0)
            # Shade above upper scale cut
            ax.axvspan(cut[1], xlim[1], alpha=0.1, color="gray", zorder=0)

    plt.tight_layout()
    return fig


def main():
    config = snakemake.config
    version_labels = snakemake.params.version_labels
    # Version list: from params if provided (ecut comparison), else from config
    versions = getattr(snakemake.params, "versions", None)
    if versions is None:
        versions = [v for v in config["versions"] if "_leak_corr" in v]
    fiducial_version = config["fiducial"]["version"]
    plotting_config = config["plotting"]

    # Which version gets the fiducial reference line in boxes
    fiducial_for_comparison = getattr(
        snakemake.params, "fiducial_for_comparison",
        plotting_config.get("fiducial_for_comparison", fiducial_version),
    )

    # Tighter x-offsets for pure E/B (log scale, many bins)
    x_offsets = [-0.06, -0.02, 0.02, 0.06]
    x_offset_factors = [1.0 + offset for offset in x_offsets]

    # Box styling from config
    box_style = plotting_config.get("version_box", {})

    # Scale cuts
    fiducial_xip_scale_cut = tuple(config["fiducial"]["fiducial_xip_scale_cut"])
    fiducial_xim_scale_cut = tuple(config["fiducial"]["fiducial_xim_scale_cut"])

    scale_cuts = {
        "full": (1.0, 250.0),
        "fiducial_xip": fiducial_xip_scale_cut,
        "fiducial_xim": fiducial_xim_scale_cut,
    }

    # Color/marker assignment: pair by parent version if ecut versions present
    has_ecut = any("_ecut" in v for v in versions)
    if has_ecut:
        parents = []
        for v in versions:
            parent = v.split("_ecut")[0].replace("_leak_corr", "")
            if parent not in parents:
                parents.append(parent)
        parent_colors = sns.husl_palette(len(parents), l=0.4)
        parent_color_map = dict(zip(parents, parent_colors))
    else:
        colors = sns.husl_palette(len(versions), l=0.4)

    # Load data for all versions
    datasets = []

    for i, (version, data_path) in enumerate(zip(
        versions,
        snakemake.input["pure_eb_data"],
    )):
        if has_ecut:
            parent = version.split("_ecut")[0].replace("_leak_corr", "")
            color = parent_color_map[parent]
            is_cut = "_ecut" in version
            marker = "o" if not is_cut else "D"
            fillstyle = "full" if not is_cut else "none"
        else:
            color = colors[i]
            marker = MARKER_STYLES[i] if i < len(MARKER_STYLES) else "o"
            fillstyle = "full"

        data = np.load(data_path)

        theta = data["theta"]
        nbins = len(theta)

        # Total correlation functions
        xip_total = data["xip_total"]
        xim_total = data["xim_total"]

        # B-mode data
        xip_B = data["xip_B"]
        xim_B = data["xim_B"]
        cov_pure_eb = data["cov_pure_eb"]

        # Use E-mode errors for total xi (E dominates signal, good proxy for visualization)
        # Blocks: 0=xip_E, 1=xim_E, 2=xip_B, 3=xim_B, 4=xip_amb, 5=xim_amb
        sigma_xip_total = _extract_sigma(cov_pure_eb, 0, nbins)
        sigma_xim_total = _extract_sigma(cov_pure_eb, 1, nbins)

        # Extract B-mode covariance blocks
        block_xip_B = cov_pure_eb[2*nbins:3*nbins, 2*nbins:3*nbins]
        block_xim_B = cov_pure_eb[3*nbins:4*nbins, 3*nbins:4*nbins]
        sigma_xip_B = np.sqrt(np.clip(np.diag(block_xip_B), 0, None))
        sigma_xim_B = np.sqrt(np.clip(np.diag(block_xim_B), 0, None))

        # Full-range PTEs
        chi2_xip_B, pte_xip_B, dof_xip_B = compute_chi2_pte(xip_B, block_xip_B)
        chi2_xim_B, pte_xim_B, dof_xim_B = compute_chi2_pte(xim_B, block_xim_B)

        # Scale-cut PTEs (fiducial xip and xim cuts)
        xip_cut_mask = (theta >= fiducial_xip_scale_cut[0]) & (theta <= fiducial_xip_scale_cut[1])
        xim_cut_mask = (theta >= fiducial_xim_scale_cut[0]) & (theta <= fiducial_xim_scale_cut[1])
        xip_idx = np.where(xip_cut_mask)[0]
        xim_idx = np.where(xim_cut_mask)[0]
        chi2_xip_B_cut, pte_xip_B_cut, dof_xip_B_cut = compute_chi2_pte(
            xip_B[xip_idx], block_xip_B[np.ix_(xip_idx, xip_idx)])
        chi2_xim_B_cut, pte_xim_B_cut, dof_xim_B_cut = compute_chi2_pte(
            xim_B[xim_idx], block_xim_B[np.ix_(xim_idx, xim_idx)])

        datasets.append({
            "version": version,
            "label": version_label(version, version_labels),
            "color": color,
            "marker": marker,
            "fillstyle": fillstyle,
            "alpha": get_version_alpha(version, fiducial_for_comparison, plotting_config),
            "theta": theta,
            # Total correlation functions
            "xip_total": xip_total,
            "xim_total": xim_total,
            "sigma_xip_total": sigma_xip_total,
            "sigma_xim_total": sigma_xim_total,
            # B-mode (normalized)
            "xip_B_normalized": xip_B / sigma_xip_B,
            "xim_B_normalized": xim_B / sigma_xim_B,
            # PTEs
            "pte_xip_B": pte_xip_B, "chi2_xip_B": chi2_xip_B, "dof_xip_B": dof_xip_B,
            "pte_xim_B": pte_xim_B, "chi2_xim_B": chi2_xim_B, "dof_xim_B": dof_xim_B,
            "pte_xip_B_cut": pte_xip_B_cut, "chi2_xip_B_cut": chi2_xip_B_cut, "dof_xip_B_cut": dof_xip_B_cut,
            "pte_xim_B_cut": pte_xim_B_cut, "chi2_xim_B_cut": chi2_xim_B_cut, "dof_xim_B_cut": dof_xim_B_cut,
        })

    # Create output directory
    output_dir = Path(snakemake.output["evidence"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate figure
    fig = _create_version_comparison_figure(
        datasets, scale_cuts, fiducial_for_comparison,
        x_offset_factors=x_offset_factors, box_style=box_style
    )

    fig_name = "figure.png"
    fig_path = output_dir / fig_name
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_path}")

    # Copy to paper figures
    if "paper_figure" in snakemake.output.keys():
        paper_path = Path(snakemake.output["paper_figure"])
        paper_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paper_path, bbox_inches="tight")
        print(f"Saved {paper_path}")

    plt.close(fig)

    # Write evidence
    spec_paths = snakemake.input["specs"]

    # Build flat evidence dict
    evidence_versions = {}
    for data in datasets:
        v = data["version"]
        for key in ("pte_xip_B", "chi2_xip_B", "dof_xip_B",
                     "pte_xim_B", "chi2_xim_B", "dof_xim_B",
                     "pte_xip_B_cut", "chi2_xip_B_cut", "dof_xip_B_cut",
                     "pte_xim_B_cut", "chi2_xim_B_cut", "dof_xim_B_cut"):
            val = data[key]
            evidence_versions[f"{v}_{key}"] = int(val) if "dof" in key else float(val)

    evidence_data = {
        "spec_id": "pure_eb_version_comparison",
        "spec_path": spec_paths[0],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "scale_cuts": {k: list(v) for k, v in scale_cuts.items()},
            "versions": evidence_versions,
        },
        "output": {"figure": fig_name},
    }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"Saved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
