"""COSEBIs data vector claim.

Single-panel figure showing B-mode COSEBIS for each catalog version.
Overplots fiducial and full angular range scale cuts.

Produces 9 figures:
- figure.png: fiducial version, leak-corrected, no title (paper)
- figure_v{X.Y.Z}.png: version X.Y.Z, leak-corrected, with title
- figure_v{X.Y.Z}_uncorrected.png: version X.Y.Z, uncorrected, with title
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import treecorr
from plotting_utils import (
    FIG_WIDTH_SINGLE,
    PAPER_MPLSTYLE,
)

from sp_validation.b_modes import calculate_cosebis

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
    ax.set_xlabel("COSEBI mode $n$")
    ax.set_xlim(0.5, nmodes + 0.5)
    ax.set_xticks(np.arange(1, nmodes + 1))
    ax.set_xticklabels(
        [str(i) for i in range(1, nmodes + 1)],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="both", width=0.5, length=3)

    if title:
        ax.set_title(f"COSEBI B-modes ({title})")

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


def main(config, xi_integration, cov_integration, out_dir):
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

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the fiducial fine-binned 2PCF once (integration grid)
    gg = treecorr.GGCorrelation(
        min_sep=min_sep_int,
        max_sep=max_sep_int,
        nbins=nbins_int,
        sep_units="arcmin",
    )
    gg.read(xi_integration)

    # Compute the E_n / B_n data vectors + T^T C_xi T mode covariance at both
    # scale cuts. Same calculate_cosebis call as the original per-version loop.
    npz_payload = {"nmodes": nmodes, "version": version}
    datasets = {}
    for scale_key, scale_cut in scale_cuts.items():
        results = calculate_cosebis(
            gg,
            nmodes=nmodes,
            scale_cuts=[scale_cut],
            cov_path=cov_integration,
        )
        r = results[scale_cut]
        En = r["En"]
        Bn = r["Bn"]
        cov = r["cov"]
        sigma_E = np.sqrt(np.diag(cov[:nmodes, :nmodes]))
        sigma_B = np.sqrt(np.diag(cov[nmodes:, nmodes:]))
        datasets[scale_key] = {"Bn_normalized": Bn / sigma_B}
        npz_payload[f"{scale_key}_En"] = En
        npz_payload[f"{scale_key}_Bn"] = Bn
        npz_payload[f"{scale_key}_cov"] = cov
        npz_payload[f"{scale_key}_sigma_E"] = sigma_E
        npz_payload[f"{scale_key}_sigma_B"] = sigma_B
        npz_payload[f"{scale_key}_scale_cut"] = np.array(scale_cut)

    # Primary data artifact (cosebis_modes_data)
    npz_path = out_dir / f"cosebis_modes_{version}.npz"
    np.savez(npz_path, **npz_payload)
    print(f"Saved COSEBI modes data to {npz_path}")

    # Paper data-vector figure (fiducial version, both scale cuts overplotted)
    fig = _create_single_panel_bmode_figure(datasets, nmodes, scale_cuts, title=None)
    fig.savefig(out_dir / "figure.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "cosebis_data_vector.pdf", bbox_inches="tight")
    print(f"Saved figure to {out_dir / 'figure.png'}")
    plt.close(fig)

    evidence_data = {
        "spec_id": "cosebis_data_vector",
        "generated": datetime.now().isoformat(),
        "evidence": {
            "version": version,
            "fiducial_scale_cut": list(fiducial_scale_cut),
            "full_scale_cut": list(full_scale_cut),
            "nmodes": nmodes,
            "note": "COSEBI B_n/E_n data vector + paper figure. PTEs in cosebis_pte_per_cut.",
        },
        "output": {"data": npz_path.name, "figure": "figure.png"},
    }
    evidence_path = out_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"Saved evidence to {evidence_path}")


def _from_cli(argv=None):
    import yaml

    ap = argparse.ArgumentParser(
        description="COSEBI E_n/B_n data vector (NPZ) + paper figure for the fiducial catalog."
    )
    ap.add_argument(
        "--config", required=True, help="Absolute path to bmodes config.yaml"
    )
    ap.add_argument(
        "--xi-integration",
        required=True,
        help="Fiducial fine-binned (1000-bin) TreeCorr xi_pm .txt dump",
    )
    ap.add_argument(
        "--cov-integration",
        required=True,
        help="Fiducial 1000-bin Gaussian covariance (processed .txt) for the T^T C_xi T transform",
    )
    ap.add_argument("--out", required=True, help="Output directory (lc {output})")
    a = ap.parse_args(argv)
    with open(a.config) as f:
        config = yaml.safe_load(f)
    main(config, a.xi_integration, a.cov_integration, a.out)


if __name__ == "__main__":
    _from_cli()
