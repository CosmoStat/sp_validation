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
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns


plt.style.use(
    "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle"
)


VERSION_LABELS = {
    "SP_v1.4.5_leak_corr": "Initial",
    "SP_v1.4.6_leak_corr": "Fiducial",
    "SP_v1.4.8_leak_corr": "Masked",
    "SP_v1.4.10.1_leak_corr": "Blends",
}


def _version_label(version):
    return VERSION_LABELS.get(version, version.replace("SP_", "").replace("_leak_corr", ""))


def _extract_sigma(covariance, block_index, block_size):
    """Extract standard deviations from diagonal of covariance block."""
    block_slice = slice(block_index * block_size, (block_index + 1) * block_size)
    block = covariance[block_slice, block_slice]
    return np.sqrt(np.clip(np.diag(block), 0, None))


def _draw_version_boxes(ax, theta, datasets, y_key, sigma_key, fiducial_idx,
                        scale_factor=1.0, apply_theta_scaling=True):
    """Draw boxes spanning version spread with fiducial line through each bin.

    For each angular bin, draws:
    - A box from min(y - sigma) to max(y + sigma) across all versions
    - A horizontal fiducial line at the fiducial version's y value

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on
    theta : array
        Angular bin centers
    datasets : list of dict
        Version data dictionaries
    y_key : str
        Key for y-values in datasets
    sigma_key : str
        Key for error values in datasets
    fiducial_idx : int
        Index of fiducial version in datasets
    scale_factor : float
        Factor to divide values by (for plotting in scaled units)
    apply_theta_scaling : bool
        If True, multiply y and sigma by theta (for theta*xi plots)
    """
    nbins = len(theta)

    # Compute log-scale box widths (fraction of bin position in log space)
    log_theta = np.log10(theta)
    box_width_factor = 0.15  # Fraction of distance to next bin

    for i in range(nbins):
        # Collect all version values at this bin
        y_vals = []
        y_lower = []
        y_upper = []

        for data in datasets:
            y = data[y_key][i]
            sigma = data[sigma_key][i]

            if apply_theta_scaling:
                y = theta[i] * y
                sigma = theta[i] * sigma

            y_scaled = y / scale_factor
            sigma_scaled = sigma / scale_factor

            y_vals.append(y_scaled)
            y_lower.append(y_scaled - sigma_scaled)
            y_upper.append(y_scaled + sigma_scaled)

        # Box spans from min lower to max upper
        box_bottom = min(y_lower)
        box_top = max(y_upper)

        # Compute box width in log space
        if i < nbins - 1:
            log_width = (log_theta[i + 1] - log_theta[i]) * box_width_factor
        else:
            log_width = (log_theta[i] - log_theta[i - 1]) * box_width_factor

        x_left = 10 ** (log_theta[i] - log_width)
        x_right = 10 ** (log_theta[i] + log_width)

        # Draw box
        rect = Rectangle(
            (x_left, box_bottom),
            x_right - x_left,
            box_top - box_bottom,
            facecolor='none',
            edgecolor='0.4',
            linewidth=0.5,
            zorder=1,
        )
        ax.add_patch(rect)

        # Draw fiducial line
        fiducial_y = y_vals[fiducial_idx]
        ax.hlines(
            fiducial_y, x_left, x_right,
            colors='0.5', linewidth=0.6, zorder=1
        )


def _draw_normalized_version_boxes(ax, theta, datasets, y_norm_key, fiducial_idx):
    """Draw boxes for normalized (y/sigma) plots where error bars are unity.

    For each angular bin, draws:
    - A box from min(y_norm - 1) to max(y_norm + 1) across all versions
    - A horizontal fiducial line at the fiducial version's y_norm value
    """
    nbins = len(theta)
    log_theta = np.log10(theta)
    box_width_factor = 0.15

    for i in range(nbins):
        y_vals = [data[y_norm_key][i] for data in datasets]

        # Error is 1 by construction for normalized plots
        y_lower = [y - 1 for y in y_vals]
        y_upper = [y + 1 for y in y_vals]

        box_bottom = min(y_lower)
        box_top = max(y_upper)

        if i < nbins - 1:
            log_width = (log_theta[i + 1] - log_theta[i]) * box_width_factor
        else:
            log_width = (log_theta[i] - log_theta[i - 1]) * box_width_factor

        x_left = 10 ** (log_theta[i] - log_width)
        x_right = 10 ** (log_theta[i] + log_width)

        rect = Rectangle(
            (x_left, box_bottom),
            x_right - x_left,
            box_top - box_bottom,
            facecolor='none',
            edgecolor='0.4',
            linewidth=0.5,
            zorder=1,
        )
        ax.add_patch(rect)

        fiducial_y = y_vals[fiducial_idx]
        ax.hlines(
            fiducial_y, x_left, x_right,
            colors='0.5', linewidth=0.6, zorder=1
        )


def _create_version_comparison_figure(datasets, scale_cuts, fiducial_version):
    """Create figure comparing total and B-mode correlations across versions.

    Layout:
    - Top row (2/3 height): xi_total + (left), xi_total - (right)
      Plotted as theta * xi / 1e-4 (matching data vector style)
    - Bottom row (1/3 height): xi_B + (left), xi_B - (right)
      Plotted as B / sigma (normalized)

    Data outside fiducial scale cuts shown with light axvspan shading.
    For each bin, a box spans the range of all versions' error bars,
    with a line marking the fiducial version's value.
    """
    fig_width = 7.24

    # Create figure with custom height ratios: top row 2x height of bottom
    # Bottom row shares y-axis
    fig, axes = plt.subplots(
        2, 2,
        figsize=(fig_width, fig_width * 0.65),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    # Share y-axis for bottom row
    axes[1, 1].sharey(axes[1, 0])

    x_offset_factors = [0.91, 0.97, 1.03, 1.09]
    marker_styles = ["o", "s", "D", "^"]
    scale_factor = 1e-4

    # Find fiducial version index
    fiducial_idx = next(
        (i for i, d in enumerate(datasets) if d["version"] == fiducial_version),
        0
    )
    theta = datasets[0]["theta"]

    # Plotting parameters (matching data vector style)
    ms = 2.5
    capsize = 1.5
    capthick = 0.3
    elinewidth = 0.4
    mew = 0.4

    legend_handles = []
    legend_labels = []

    # Top row: xi_total +/-
    for col, (mode_key, title) in enumerate([
        ("xip_total", r"$\xi_+$"),
        ("xim_total", r"$\xi_-$"),
    ]):
        ax = axes[0, col]

        # Draw version spread boxes (before data points so they're behind)
        _draw_version_boxes(
            ax, theta, datasets,
            y_key=mode_key,
            sigma_key=f"sigma_{mode_key}",
            fiducial_idx=fiducial_idx,
            scale_factor=scale_factor,
            apply_theta_scaling=True,
        )

        for i, data in enumerate(datasets):
            theta_i = data["theta"]
            offset = x_offset_factors[i] if i < len(x_offset_factors) else 1.0
            marker = marker_styles[i] if i < len(marker_styles) else "o"

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
                markerfacecolor=data["color"],
                markeredgecolor="white",
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
        ax.set_title(title, fontsize=10)

        if col == 0:
            ax.set_ylabel(r"$\theta \xi \times 10^4$")

    # Bottom row: xi_B +/- normalized
    for col, (mode_key, title) in enumerate([
        ("xip_B", r"$\xi_+^B / \sigma$"),
        ("xim_B", r"$\xi_-^B / \sigma$"),
    ]):
        ax = axes[1, col]

        # Draw version spread boxes (before data points)
        _draw_normalized_version_boxes(
            ax, theta, datasets,
            y_norm_key=f"{mode_key}_normalized",
            fiducial_idx=fiducial_idx,
        )

        for i, data in enumerate(datasets):
            theta_i = data["theta"]
            offset = x_offset_factors[i] if i < len(x_offset_factors) else 1.0
            marker = marker_styles[i] if i < len(marker_styles) else "o"

            y_norm = data[f"{mode_key}_normalized"]

            # Plot all points (full color, error bars = 1 by construction)
            ax.errorbar(
                theta_i * offset,
                y_norm,
                yerr=np.ones(len(theta_i)),
                fmt=marker,
                color=data["color"],
                alpha=data["alpha"],
                markerfacecolor=data["color"],
                markeredgecolor="white",
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
        ax.set_title(title, fontsize=10)

        if col == 0:
            ax.set_ylabel(r"$\xi^B / \sigma$")

    # Legend in top-left panel
    axes[0, 0].legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        fontsize=9,
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
    versions = config["versions"]
    fiducial_version = config["fiducial"]["version"]

    # Scale cuts
    fiducial_xip_scale_cut = tuple(config["fiducial"]["fiducial_xip_scale_cut"])
    fiducial_xim_scale_cut = tuple(config["fiducial"]["fiducial_xim_scale_cut"])

    scale_cuts = {
        "full": (1.0, 250.0),
        "fiducial_xip": fiducial_xip_scale_cut,
        "fiducial_xim": fiducial_xim_scale_cut,
    }

    colors = sns.color_palette("colorblind", len(versions))
    version_alpha = {v: 1.0 if v == fiducial_version else 0.5 for v in versions}

    # Load data for all versions
    datasets = []

    for version, color, data_path in zip(
        versions,
        colors,
        snakemake.input["pure_eb_data"],
    ):
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

        # Extract B-mode errors (blocks 2 and 3 of the 6-block covariance)
        sigma_xip_B = _extract_sigma(cov_pure_eb, 2, nbins)
        sigma_xim_B = _extract_sigma(cov_pure_eb, 3, nbins)

        datasets.append({
            "version": version,
            "label": _version_label(version),
            "color": color,
            "alpha": version_alpha.get(version, 1.0),
            "theta": theta,
            # Total correlation functions
            "xip_total": xip_total,
            "xim_total": xim_total,
            "sigma_xip_total": sigma_xip_total,
            "sigma_xim_total": sigma_xim_total,
            # B-mode (normalized)
            "xip_B_normalized": xip_B / sigma_xip_B,
            "xim_B_normalized": xim_B / sigma_xim_B,
        })

    # Create output directory
    output_dir = Path(snakemake.output["evidence"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate figure
    fig = _create_version_comparison_figure(datasets, scale_cuts, fiducial_version)

    fig_name = "figure.png"
    fig_path = output_dir / fig_name
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_path}")

    # Copy to paper figures
    if "paper_figure" in snakemake.output.keys():
        paper_path = Path(snakemake.output["paper_figure"])
        paper_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fig_path, paper_path)
        print(f"Copied to {paper_path}")

    plt.close(fig)

    # Write evidence
    spec_paths = snakemake.input["specs"]

    evidence_data = {
        "spec_id": "pure_eb_version_comparison",
        "spec_path": spec_paths[0],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "scale_cuts": {k: list(v) for k, v in scale_cuts.items()},
            "versions_plotted": versions,
            "fiducial_version": fiducial_version,
            "note": "Visualization only. Statistical PTEs in pure_eb_pte_matrix and config_space_pte_matrices claims.",
        },
        "artifacts": {"figure": fig_name},
    }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"Saved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
