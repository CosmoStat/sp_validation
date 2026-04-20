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
import numpy as np
import seaborn as sns
import treecorr

from sp_validation.b_modes import calculate_cosebis
from plotting_utils import (
    ERRORBAR_DEFAULTS,
    FIG_WIDTH_FULL,
    MARKER_STYLES,
    PAPER_MPLSTYLE,
    compute_chi2_pte,
    draw_normalized_boxes_linear_scale,
    find_fiducial_index,
    get_version_alpha,
    version_label,
)


plt.style.use(PAPER_MPLSTYLE)





def _create_stacked_bmode_figure(fiducial_datasets, full_datasets, nmodes, scale_cuts, fiducial_idx,
                                  y_limits=None, x_offsets=None, box_style=None):
    """Create a vertically stacked B-mode COSEBIS comparison figure.

    Plots B_n / sigma_n (dimensionless, in units of standard deviation).
    Top panel: full range, bottom panel: fiducial scale cut.
    For each mode, a box spans the range of all versions' error bars,
    with a line marking the fiducial version's value.

    Parameters
    ----------
    x_offsets : array-like, optional
        Additive offsets for each version. Default: [-0.12, -0.04, 0.04, 0.12]
    box_style : dict, optional
        Styling for version boxes.
    """
    if x_offsets is None:
        x_offsets = np.array([-0.20, -0.07, 0.07, 0.20])
    x_offsets = np.array(x_offsets)

    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH_FULL, FIG_WIDTH_FULL * 0.4), sharex=True)

    modes = np.arange(1, nmodes + 1)
    scale_labels = {"fiducial": "Fiducial", "full": "Full"}

    panels = [
        ("full", full_datasets, axes[0], scale_cuts["full"]),
        ("fiducial", fiducial_datasets, axes[1], scale_cuts["fiducial"]),
    ]

    legend_handles = []
    legend_labels = []

    for panel_idx, (scale_key, datasets, ax, scale_cut) in enumerate(panels):
        # Draw version spread boxes (before data points)
        draw_normalized_boxes_linear_scale(
            ax, modes, datasets,
            y_norm_key="Bn_normalized", fiducial_idx=fiducial_idx,
            x_offsets=x_offsets, box_style=box_style
        )

        for i, data in enumerate(datasets):
            offset = x_offsets[i] if i < len(x_offsets) else 0
            marker = data.get("marker", MARKER_STYLES[i] if i < len(MARKER_STYLES) else "o")
            fillstyle = data.get("fillstyle", "full")
            mfc = data["color"] if fillstyle == "full" else "none"
            line = ax.errorbar(
                modes + offset,
                data["Bn_normalized"],
                yerr=np.ones_like(data["Bn_normalized"]),
                fmt=marker,
                color=data["color"],
                alpha=data["alpha"],
                markerfacecolor=mfc,
                markeredgecolor=data["color"],
                **ERRORBAR_DEFAULTS,
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
            rf"{label} $\theta = {scale_cut[0]:.0f}$--${scale_cut[1]:.0f}'$",
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
    # Version list: from params if provided (ecut comparison), else from config
    versions = getattr(snakemake.params, "versions", None)
    if versions is None:
        versions = [v for v in config["versions"] if "_leak_corr" in v]
    nmodes = config["fiducial"]["nmodes"]
    plotting_config = config["plotting"]

    version_labels = snakemake.params.version_labels

    # Which version gets the fiducial reference line in boxes
    fiducial_for_comparison = getattr(
        snakemake.params, "fiducial_for_comparison",
        plotting_config.get("fiducial_for_comparison", config["fiducial"]["version"]),
    )

    # Plotting config
    x_offsets = plotting_config.get("x_offsets", [-0.12, -0.04, 0.04, 0.12])
    box_style = plotting_config.get("version_box", {})

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

    # Compute COSEBIS for visualization - need to build datasets first to get fiducial_idx
    all_datasets = {}
    # Collect PTEs across scale cuts
    all_ptes = {v: {} for v in versions}

    for scale_key, scale_cut in scale_cuts.items():
        datasets = []

        for i, (version, xi_path, cov_path) in enumerate(zip(
            versions,
            snakemake.input["xi_integration"],
            snakemake.input["cov_integration"],
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

            gg = treecorr.GGCorrelation(
                min_sep=min_sep_int,
                max_sep=max_sep_int,
                nbins=nbins_int,
                sep_units="arcmin",
            )
            gg.read(xi_path)

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

            # PTEs
            chi2, pte, dof = compute_chi2_pte(Bn, cov_B)
            all_ptes[version][f"pte_B_{scale_key}"] = float(pte)
            all_ptes[version][f"chi2_B_{scale_key}"] = float(chi2)
            all_ptes[version][f"dof_B_{scale_key}"] = int(dof)

            datasets.append({
                "version": version,
                "label": version_label(version, version_labels),
                "color": color,
                "marker": marker,
                "fillstyle": fillstyle,
                "alpha": get_version_alpha(version, fiducial_for_comparison, plotting_config),
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

    # Find fiducial version index for box highlighting (use fiducial datasets)
    fiducial_idx = find_fiducial_index(all_datasets["fiducial"], fiducial_for_comparison)

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
        x_offsets=x_offsets,
        box_style=box_style,
    )

    fig_name = "figure_stacked.png"
    fig_path = output_dir / fig_name
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_path}")

    if "paper_stacked" in snakemake.output.keys():
        paper_path = Path(snakemake.output["paper_stacked"])
        paper_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paper_path, bbox_inches="tight")
        print(f"Saved {paper_path}")

    plt.close(fig)

    # Write evidence (visualization only, PTEs are in cosebis_pte_matrix)
    spec_paths = snakemake.input["specs"]

    # Build flat evidence dict
    evidence_versions = {}
    for version, ptes in all_ptes.items():
        for key, val in ptes.items():
            evidence_versions[f"{version}_{key}"] = val

    evidence_data = {
        "spec_id": "cosebis_version_comparison",
        "spec_path": spec_paths[0],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "scale_cuts": scale_cuts,
            "nmodes": nmodes,
            "versions": evidence_versions,
        },
        "output": {"figure_stacked": fig_name},
    }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"Saved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
