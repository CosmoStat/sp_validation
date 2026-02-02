"""COSEBIS version comparison claim.

Visualizes B-mode COSEBIS across catalog versions.
Produces figures at fiducial scale cut and full range.
Statistical evidence (PTEs) is in cosebis_pte_matrix.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns

import treecorr

from sp_validation.b_modes import calculate_cosebis


plt.style.use(
    "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle"
)




def _draw_normalized_version_boxes_modes(ax, modes, datasets, y_norm_key, fiducial_idx):
    """Draw boxes for normalized (y/sigma) COSEBIS modes.

    For each mode, draws:
    - A box from min(y_norm - 1) to max(y_norm + 1) across all versions
    - A horizontal fiducial line at the fiducial version's y_norm value
    """
    nmodes = len(modes)
    box_half_width = 0.35  # Half width of box in mode units

    for i in range(nmodes):
        y_vals = [data[y_norm_key][i] for data in datasets]

        # Error is 1 by construction for normalized plots
        y_lower = [y - 1 for y in y_vals]
        y_upper = [y + 1 for y in y_vals]

        box_bottom = min(y_lower)
        box_top = max(y_upper)

        x_center = modes[i]
        x_left = x_center - box_half_width
        x_right = x_center + box_half_width

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


def _version_label(version, version_labels):
    """Get human-readable label for version from config."""
    return version_labels.get(version, version.replace("SP_", "").replace("_leak_corr", ""))


def _get_cov_path(cov_base_dir, version, blind, min_sep, max_sep, nbins):
    """Construct covariance path for a specific blind.

    Versions derived from v1.4.6 footprint (v1.4.10.1, v1.4.11.2) use v1.4.6 covariance.
    """
    # Same footprint → same covariance geometry
    cov_version = version.replace("v1.4.10.1", "v1.4.6").replace("v1.4.11.2", "v1.4.6")
    base_name_masked = f"covariance_{cov_version}_{blind}_g_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_masked"
    masked_path = f"{cov_base_dir}/{base_name_masked}/{base_name_masked}_processed.txt"
    if Path(masked_path).exists():
        return masked_path

    base_name = f"covariance_{cov_version}_{blind}_g_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}"
    return f"{cov_base_dir}/{base_name}/{base_name}_processed.txt"


def _create_stacked_bmode_figure(fiducial_datasets, full_datasets, nmodes, scale_cuts, fiducial_idx, y_limits=None):
    """Create a vertically stacked B-mode COSEBIS comparison figure.

    Plots B_n / sigma_n (dimensionless, in units of standard deviation).
    Top panel: full range, bottom panel: fiducial scale cut.
    For each mode, a box spans the range of all versions' error bars,
    with a line marking the fiducial version's value.
    """
    fig_width = 7.24
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_width * 0.6), sharex=True)

    x_offsets = np.array([-0.2, -0.07, 0.07, 0.2])
    modes = np.arange(1, nmodes + 1)
    marker_styles = ["o", "s", "D", "^"]
    scale_labels = {"fiducial": "Fiducial", "full": "Full"}

    panels = [
        ("full", full_datasets, axes[0], scale_cuts["full"]),
        ("fiducial", fiducial_datasets, axes[1], scale_cuts["fiducial"]),
    ]

    legend_handles = []
    legend_labels = []

    for panel_idx, (scale_key, datasets, ax, scale_cut) in enumerate(panels):
        # Draw version spread boxes (before data points)
        _draw_normalized_version_boxes_modes(
            ax, modes, datasets,
            y_norm_key="Bn_normalized", fiducial_idx=fiducial_idx
        )

        for i, data in enumerate(datasets):
            offset = x_offsets[i] if i < len(x_offsets) else 0
            marker = marker_styles[i] if i < len(marker_styles) else "o"
            line = ax.errorbar(
                modes + offset,
                data["Bn_normalized"],
                yerr=np.ones_like(data["Bn_normalized"]),
                fmt=marker,
                color=data["color"],
                alpha=data["alpha"],
                markerfacecolor=data["color"],
                markeredgecolor="white",
                markeredgewidth=0.5,
                markersize=4,
                capsize=2,
                capthick=0.8,
                linewidth=0.8,
                elinewidth=0.8,
                zorder=2,
            )
            if panel_idx == 0:
                legend_handles.append(line)
                legend_labels.append(data["label"])

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        ax.axvspan(0.5, 6.5, color="0.95", alpha=0.5, zorder=0)
        ax.set_ylabel(r"$B_n / \sigma_n$")
        ax.set_xlim(0.5, nmodes + 0.5)
        if y_limits:
            ax.set_ylim(y_limits)
        ax.tick_params(axis="both", width=0.5, length=3)

        label = scale_labels[scale_key]
        ax.text(
            0.98, 0.95,
            rf"{label} $\theta \in [{scale_cut[0]:.0f}, {scale_cut[1]:.0f}]'$",
            transform=ax.transAxes,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    axes[1].set_xlabel("COSEBIS mode $n$")
    axes[1].set_xticks(np.arange(1, nmodes + 1))

    axes[0].legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        frameon=True,
        framealpha=0.9,
        handletextpad=0.3,
        columnspacing=1.0,
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08)
    return fig


def main():
    config = snakemake.config
    versions = config["versions"]
    nmodes = config["fiducial"]["nmodes"]

    # Use fiducial blind for plotting (B-modes are same across blinds, only covariance differs)
    blind = config["fiducial"]["blind"]
    cov_base_dir = snakemake.params.cov_base_dir
    version_labels = snakemake.params.version_labels

    fiducial_scale_cut = (
        float(config["fiducial"]["fiducial_min_scale"]),
        float(config["fiducial"]["fiducial_max_scale"]),
    )
    full_scale_cut = (
        float(config["cosebis"]["theta_min"]),
        float(config["cosebis"]["theta_max"]),
    )

    scale_cuts = {
        "fiducial": fiducial_scale_cut,
        "full": full_scale_cut,
    }

    min_sep_int = float(config["fiducial"]["min_sep_int"])
    max_sep_int = float(config["fiducial"]["max_sep_int"])
    nbins_int = int(config["fiducial"]["nbins_int"])

    fiducial_version = config["fiducial"]["version"]
    version_alpha = {v: 1.0 if v == fiducial_version else 0.4 for v in versions}

    # Find fiducial version index for box highlighting
    fiducial_idx = next(
        (i for i, v in enumerate(versions) if v == fiducial_version),
        0  # Fallback to first version
    )

    colors = sns.color_palette("colorblind", len(versions))

    # Compute COSEBIS for visualization
    all_datasets = {}

    for scale_key, scale_cut in scale_cuts.items():
        datasets = []

        for version, color, xi_path in zip(
            versions,
            colors,
            snakemake.input["xi_integration"],
        ):
            gg = treecorr.GGCorrelation(
                min_sep=min_sep_int,
                max_sep=max_sep_int,
                nbins=nbins_int,
                sep_units="arcmin",
            )
            gg.read(xi_path)

            cov_path = _get_cov_path(
                cov_base_dir, version, blind, min_sep_int, max_sep_int, nbins_int
            )

            results = calculate_cosebis(
                gg,
                nmodes=nmodes,
                scale_cuts=[scale_cut],
                cov_path=cov_path,
            )

            cosebis_result = results[scale_cut]
            Bn = cosebis_result["Bn"]
            cov = cosebis_result["cov"]
            cov_B = cov[nmodes:, nmodes:]
            sigma_B = np.sqrt(np.diag(cov_B))

            datasets.append({
                "version": version,
                "label": _version_label(version, version_labels),
                "color": color,
                "alpha": version_alpha.get(version, 1.0),
                "Bn_normalized": Bn / sigma_B,
            })

        all_datasets[scale_key] = datasets

    # Compute y-limits
    all_y = []
    for datasets in all_datasets.values():
        for d in datasets:
            all_y.extend(d["Bn_normalized"] - 1)
            all_y.extend(d["Bn_normalized"] + 1)
    y_range = max(all_y) - min(all_y)
    y_limits = (min(all_y) - 0.1 * y_range, max(all_y) + 0.1 * y_range)

    # Create output
    output_dir = Path(snakemake.output["evidence"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = _create_stacked_bmode_figure(
        all_datasets["fiducial"],
        all_datasets["full"],
        nmodes,
        scale_cuts,
        fiducial_idx,
        y_limits,
    )

    fig_name = "figure_stacked.png"
    fig_path = output_dir / fig_name
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_path}")

    if "paper_stacked" in snakemake.output.keys():
        paper_path = Path(snakemake.output["paper_stacked"])
        paper_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fig_path, paper_path)
        print(f"Copied to {paper_path}")

    plt.close(fig)

    # Write evidence (visualization only, PTEs are in cosebis_pte_matrix)
    spec_paths = snakemake.input["specs"]

    evidence_data = {
        "spec_id": "cosebis_version_comparison",
        "spec_path": spec_paths[0],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "scale_cuts": scale_cuts,
            "versions_plotted": versions,
            "nmodes": nmodes,
            "note": "Visualization only. Statistical PTEs in cosebis_pte_matrix claim.",
        },
        "artifacts": {"figure_stacked": fig_name},
    }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"Saved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
