"""COSEBIs data vector claim.

Single-panel figure showing B-mode COSEBIS for fiducial version (from config.fiducial.version).
Overplots fiducial and full angular range scale cuts.
Paper figure for main text B-mode validation.

Produces two figures:
- Main figure (leak-corrected, unlabeled) for paper
- Companion figure (uncorrected, labeled) for dashboard
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


plt.style.use(
    "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle"
)


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


def _create_single_panel_bmode_figure(datasets, nmodes, scale_cuts, title_suffix=""):
    """Create a single-panel B-mode COSEBIS figure.

    Plots B_n / sigma_n (dimensionless, in units of standard deviation).
    Both scale cuts overplotted with horizontal offset and different colors.

    Args:
        title_suffix: Optional suffix for title (e.g., " (uncorrected)")
    """
    fig_width = 7.24
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.35))

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

        label = rf"$\theta \in [{scale_cut[0]:.0f}, {scale_cut[1]:.0f}]'$"

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

    if title_suffix:
        ax.set_title(f"COSEBIS B-modes{title_suffix}", fontsize=10)

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

    # Create output directory
    output_dir = Path(snakemake.output["evidence"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Main figure (leak-corrected) ---
    gg = treecorr.GGCorrelation(
        min_sep=min_sep_int,
        max_sep=max_sep_int,
        nbins=nbins_int,
        sep_units="arcmin",
    )
    gg.read(snakemake.input["xi_integration"])
    datasets = _compute_cosebis_datasets(
        gg, snakemake.input["cov_integration"], nmodes, scale_cuts
    )
    fig = _create_single_panel_bmode_figure(datasets, nmodes, scale_cuts, title_suffix="")

    fig_name = "figure.png"
    fig_path = output_dir / fig_name
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_path}")

    if "paper_figure" in snakemake.output.keys():
        paper_path = Path(snakemake.output["paper_figure"])
        paper_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fig_path, paper_path)
        print(f"Copied to {paper_path}")
    plt.close(fig)

    # --- Companion figure (uncorrected) ---
    gg_uncorr = treecorr.GGCorrelation(
        min_sep=min_sep_int,
        max_sep=max_sep_int,
        nbins=nbins_int,
        sep_units="arcmin",
    )
    gg_uncorr.read(snakemake.input["xi_integration_uncorr"])
    datasets_uncorr = _compute_cosebis_datasets(
        gg_uncorr, snakemake.input["cov_integration_uncorr"], nmodes, scale_cuts
    )
    fig_uncorr = _create_single_panel_bmode_figure(
        datasets_uncorr, nmodes, scale_cuts, title_suffix=" (uncorrected)"
    )

    fig_uncorr_path = Path(snakemake.output["figure_uncorrected"])
    fig_uncorr.savefig(fig_uncorr_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_uncorr_path}")
    plt.close(fig_uncorr)

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
        "artifacts": {
            "figure": fig_name,
            "figure_uncorrected": fig_uncorr_path.name,
        },
    }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"Saved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
