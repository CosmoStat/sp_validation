"""COSEBIs data vector claim.

Single-panel figure showing B-mode COSEBIS for each catalog version.
Overplots fiducial and full angular range scale cuts.

Produces 9 figures:
- figure.png: fiducial version, leak-corrected, no title (paper)
- figure_v{X.Y.Z}.png: version X.Y.Z, leak-corrected, with title
- figure_v{X.Y.Z}_uncorrected.png: version X.Y.Z, uncorrected, with title
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
    FIG_WIDTH_SINGLE,
    PAPER_MPLSTYLE,
    iter_version_figures,
)


plt.style.use(PAPER_MPLSTYLE)


def _compute_cosebis_datasets(gg, cov_path, nmodes, scale_cuts):
    """Compute COSEBIS B-modes for given scale cuts.

    Returns dict with normalized B_n / sigma_n for each scale cut.
    """
    datasets = {}
    for scale_key, scale_cut in scale_cuts.items():
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
        datasets[scale_key] = {"Bn_normalized": Bn / sigma_B}
    return datasets


def _create_single_panel_bmode_figure(datasets, nmodes, scale_cuts, title=None):
    """Create a single-panel B-mode COSEBIS figure.

    Plots B_n / sigma_n (dimensionless, in units of standard deviation).
    Both scale cuts overplotted with horizontal offset and different colors.

    Args:
        title: Optional title for the figure (None for paper figure).
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE, FIG_WIDTH_SINGLE * 0.55))

    x_offsets = {"fiducial": -0.15, "full": 0.15}
    modes = np.arange(1, nmodes + 1)
    marker_styles = {"fiducial": "o", "full": "s"}
    colors = sns.color_palette("colorblind", 2)
    scale_colors = {"fiducial": colors[0], "full": colors[1]}

    legend_handles = []
    legend_labels = []

    for scale_key, data in datasets.items():
        offset = x_offsets[scale_key]
        marker = marker_styles[scale_key]
        color = scale_colors[scale_key]
        scale_cut = scale_cuts[scale_key]

        label = rf"$\theta = {scale_cut[0]:.0f}$--${scale_cut[1]:.0f}'$"

        line = ax.errorbar(
            modes + offset,
            data["Bn_normalized"],
            yerr=np.ones_like(data["Bn_normalized"]),
            fmt=marker,
            color=color,
            alpha=1.0,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5,
            capsize=2,
            capthick=0.8,
            linewidth=0.8,
            elinewidth=0.8,
            label=label,
        )
        legend_handles.append(line)
        legend_labels.append(label)

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.axvspan(0.5, 6.5, color="0.95", alpha=0.5, zorder=0)
    ax.set_ylabel(r"$B_n / \sigma_n$")
    ax.set_xlabel("COSEBIS mode $n$")
    ax.set_xlim(0.5, nmodes + 0.5)
    ax.set_xticks(np.arange(1, nmodes + 1))
    ax.tick_params(axis="both", width=0.5, length=3)

    if title:
        ax.set_title(f"COSEBIS B-modes ({title})")

    # Compute y-limits from data
    all_y = []
    for data in datasets.values():
        all_y.extend(data["Bn_normalized"] - 1)
        all_y.extend(data["Bn_normalized"] + 1)
    y_range = max(all_y) - min(all_y)
    ax.set_ylim(min(all_y) - 0.1 * y_range, max(all_y) + 0.1 * y_range)

    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
    )

    plt.tight_layout()
    return fig


def main():
    config = snakemake.config
    version = config["fiducial"]["version"]
    nmodes = config["fiducial"]["nmodes"]
    version_labels = config["plotting"]["version_labels"]

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

    # Build input path lookup from snakemake inputs
    xi_paths = {k: v for k, v in snakemake.input.items() if k.startswith("xi_")}
    cov_paths = {k: v for k, v in snakemake.input.items() if k.startswith("cov_")}

    # Create output directory
    output_dir = Path(snakemake.output["evidence"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track generated artifacts
    output = {}

    # Generate all 9 figures
    for fig_spec in iter_version_figures(version_labels, version):
        # Determine which input keys to use
        if fig_spec["leak_corrected"]:
            xi_key = f"xi_{fig_spec['version_leak_corr']}"
            cov_key = f"cov_{fig_spec['version_leak_corr']}"
        else:
            xi_key = f"xi_{fig_spec['version_uncorr']}"
            cov_key = f"cov_{fig_spec['version_uncorr']}"

        # Load 2PCF for this version
        gg = treecorr.GGCorrelation(
            min_sep=min_sep_int,
            max_sep=max_sep_int,
            nbins=nbins_int,
            sep_units="arcmin",
        )
        gg.read(xi_paths[xi_key])

        # Compute COSEBIS datasets
        datasets = _compute_cosebis_datasets(
            gg, cov_paths[cov_key], nmodes, scale_cuts
        )

        # Create figure with appropriate title
        fig = _create_single_panel_bmode_figure(
            datasets, nmodes, scale_cuts, title=fig_spec["title"]
        )

        # Save figure
        fig_path = output_dir / fig_spec["filename"]
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved {fig_path}")
        plt.close(fig)

        # Track artifact
        output[fig_spec["filename"].replace(".png", "")] = fig_spec["filename"]

        # Copy paper figure to paper figures directory
        if fig_spec["is_paper_figure"] and "paper_figure" in snakemake.output.keys():
            paper_path = Path(snakemake.output["paper_figure"])
            paper_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(paper_path, bbox_inches="tight")
            print(f"Saved {paper_path}")

    # Write evidence
    spec_paths = snakemake.input["specs"]

    evidence_data = {
        "spec_id": "cosebis_data_vector",
        "spec_path": spec_paths[0],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "version": version,
            "fiducial_scale_cut": list(fiducial_scale_cut),
            "full_scale_cut": list(full_scale_cut),
            "nmodes": nmodes,
            "note": "Paper data vector figure. Statistical PTEs in cosebis_pte_matrix claim.",
        },
        "output": output,
    }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"Saved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
