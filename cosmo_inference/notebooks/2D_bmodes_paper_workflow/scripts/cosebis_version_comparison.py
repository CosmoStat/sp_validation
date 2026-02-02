"""COSEBIS version comparison claim.

Main figure: leak_corr versions only (catalog evolution across v1.4.5, v1.4.6, v1.4.8)
Second figure: v1.4.6 leak_corr vs uncorrected (correction impact comparison)

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


plt.style.use(
    "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle"
)


VERSION_LABELS = {
    "SP_v1.4.5_leak_corr": "Initial",
    "SP_v1.4.6_leak_corr": "Fiducial",
    "SP_v1.4.8_leak_corr": "Masked",
    "SP_v1.4.6": "Uncorrected",
}

# Versions for main catalog evolution figure (leak_corr only)
CATALOG_VERSIONS = [
    "SP_v1.4.5_leak_corr",
    "SP_v1.4.6_leak_corr",
    "SP_v1.4.8_leak_corr",
]

# Versions for correction comparison figure
CORRECTION_VERSIONS = [
    "SP_v1.4.6_leak_corr",
    "SP_v1.4.6",
]


def _version_label(version):
    return VERSION_LABELS.get(version, version.replace("SP_", "").replace("_leak_corr", ""))


def _normalize_version_for_cov(version):
    """Normalize version to leak_corr for covariance lookups.

    Covariances depend on survey properties (area, n_eff, sigma_e), which are
    identical for corrected/uncorrected ellipticities. Use _leak_corr covariance
    for all versions.
    """
    if "leak_corr" not in version:
        return version + "_leak_corr"
    return version


def _get_cov_path(cov_base_dir, version, blind, min_sep, max_sep, nbins):
    """Construct covariance path for a specific blind."""
    # Normalize version to leak_corr for covariance
    cov_version = _normalize_version_for_cov(version)
    base_name_masked = f"covariance_{cov_version}_{blind}_g_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_masked"
    masked_path = f"{cov_base_dir}/{base_name_masked}/{base_name_masked}_processed.txt"
    if Path(masked_path).exists():
        return masked_path

    base_name = f"covariance_{cov_version}_{blind}_g_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}"
    return f"{cov_base_dir}/{base_name}/{base_name}_processed.txt"


def _create_stacked_bmode_figure(fiducial_datasets, full_datasets, nmodes, scale_cuts, y_limits=None):
    """Create a vertically stacked B-mode COSEBIS comparison figure.

    Plots B_n / sigma_n (dimensionless, in units of standard deviation).
    Top panel: full range, bottom panel: fiducial scale cut.
    """
    fig_width = 7.24
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_width * 0.6), sharex=True)

    x_offsets = np.array([-0.1, 0.0, 0.1])
    modes = np.arange(1, nmodes + 1)
    marker_styles = ["o", "s", "D"]
    scale_labels = {"fiducial": "Fiducial", "full": "Full"}

    panels = [
        ("full", full_datasets, axes[0], scale_cuts["full"]),
        ("fiducial", fiducial_datasets, axes[1], scale_cuts["fiducial"]),
    ]

    legend_handles = []
    legend_labels = []

    for panel_idx, (scale_key, datasets, ax, scale_cut) in enumerate(panels):
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
        ncol=3,
        frameon=True,
        framealpha=0.9,
        handletextpad=0.3,
        columnspacing=1.0,
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08)
    return fig


def _compute_cosebis_for_versions(versions, xi_paths, config, cov_base_dir, scale_cuts):
    """Compute COSEBIS for a set of versions.

    Returns dict mapping scale_key -> list of dataset dicts.
    """
    nmodes = config["fiducial"]["nmodes"]
    blind = "A"  # B-modes are same across blinds, only covariance differs

    min_sep_int = float(config["fiducial"]["min_sep_int"])
    max_sep_int = float(config["fiducial"]["max_sep_int"])
    nbins_int = int(config["fiducial"]["nbins_int"])

    fiducial_version = config["fiducial"]["version"]
    version_alpha = {v: 1.0 if v == fiducial_version else 0.4 for v in versions}

    colors = sns.color_palette("colorblind", len(versions))

    all_datasets = {}

    for scale_key, scale_cut in scale_cuts.items():
        datasets = []

        for version, color, xi_path in zip(versions, colors, xi_paths):
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
                "label": _version_label(version),
                "color": color,
                "alpha": version_alpha.get(version, 1.0),
                "Bn_normalized": Bn / sigma_B,
            })

        all_datasets[scale_key] = datasets

    return all_datasets


def _compute_y_limits(all_datasets):
    """Compute y-limits from all datasets."""
    all_y = []
    for datasets in all_datasets.values():
        for d in datasets:
            all_y.extend(d["Bn_normalized"] - 1)
            all_y.extend(d["Bn_normalized"] + 1)
    y_range = max(all_y) - min(all_y)
    return (min(all_y) - 0.1 * y_range, max(all_y) + 0.1 * y_range)


def main():
    config = snakemake.config
    all_versions = config["versions"]
    nmodes = config["fiducial"]["nmodes"]
    fiducial_version = config["fiducial"]["version"]
    cov_base_dir = snakemake.params.cov_base_dir

    fiducial_scale_cut = (
        float(config["fiducial"]["fiducial_min_scale"]),
        float(config["fiducial"]["fiducial_max_scale"]),
    )
    full_scale_cut = (1.0, 250.0)

    scale_cuts = {
        "fiducial": fiducial_scale_cut,
        "full": full_scale_cut,
    }

    # Build version-to-path lookup
    version_to_path = {
        ver: path
        for ver, path in zip(all_versions, snakemake.input["xi_integration"])
    }

    # Create output directory
    output_dir = Path(snakemake.output["evidence"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Main figure: catalog evolution (leak_corr only) ===
    catalog_paths = [version_to_path[v] for v in CATALOG_VERSIONS]
    catalog_datasets = _compute_cosebis_for_versions(
        CATALOG_VERSIONS, catalog_paths, config, cov_base_dir, scale_cuts
    )
    catalog_y_limits = _compute_y_limits(catalog_datasets)

    fig_catalog = _create_stacked_bmode_figure(
        catalog_datasets["fiducial"],
        catalog_datasets["full"],
        nmodes,
        scale_cuts,
        catalog_y_limits,
    )

    fig_path = output_dir / "figure_stacked.png"
    fig_catalog.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_path}")

    if "paper_stacked" in snakemake.output.keys():
        paper_path = Path(snakemake.output["paper_stacked"])
        paper_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fig_path, paper_path)
        print(f"Copied to {paper_path}")

    plt.close(fig_catalog)

    # === Second figure: correction comparison (v1.4.6 leak_corr vs uncorrected) ===
    correction_paths = [version_to_path[v] for v in CORRECTION_VERSIONS]
    # Override alpha to full opacity for direct comparison
    correction_datasets_raw = _compute_cosebis_for_versions(
        CORRECTION_VERSIONS, correction_paths, config, cov_base_dir, scale_cuts
    )
    # Set both to full alpha for direct comparison
    for scale_key in correction_datasets_raw:
        for d in correction_datasets_raw[scale_key]:
            d["alpha"] = 1.0

    correction_y_limits = _compute_y_limits(correction_datasets_raw)

    fig_correction = _create_stacked_bmode_figure(
        correction_datasets_raw["fiducial"],
        correction_datasets_raw["full"],
        nmodes,
        scale_cuts,
        correction_y_limits,
    )

    fig_correction_path = output_dir / "figure_correction.png"
    fig_correction.savefig(fig_correction_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_correction_path}")

    if "paper_correction" in snakemake.output.keys():
        paper_correction_path = Path(snakemake.output["paper_correction"])
        paper_correction_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fig_correction_path, paper_correction_path)
        print(f"Copied to {paper_correction_path}")

    plt.close(fig_correction)

    # Write evidence
    spec_paths = snakemake.input["specs"]

    evidence_data = {
        "spec_id": "cosebis_version_comparison",
        "spec_path": spec_paths[0],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "scale_cuts": scale_cuts,
            "catalog_versions": CATALOG_VERSIONS,
            "correction_versions": CORRECTION_VERSIONS,
            "fiducial_version": fiducial_version,
            "nmodes": nmodes,
            "note": "Main figure: catalog evolution (leak_corr only). Second figure: correction comparison (v1.4.6 leak_corr vs uncorrected). Statistical PTEs in cosebis_pte_matrix claim.",
        },
        "artifacts": {
            "figure_stacked": "figure_stacked.png",
            "figure_correction": "figure_correction.png",
        },
    }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"Saved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
