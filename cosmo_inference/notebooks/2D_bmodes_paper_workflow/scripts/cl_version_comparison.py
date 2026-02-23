"""Harmonic-space C_ell^BB version comparison.

Compares B-mode power spectra across configured catalog versions.
Visualizes C_ell^BB and C_ell^EB normalized by error bars (C_ell / sigma).
Statistical PTEs reported separately in evidence.json.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits

from plotting_utils import (
    ERRORBAR_DEFAULTS,
    FIG_WIDTH_FULL,
    MARKER_STYLES,
    PAPER_MPLSTYLE,
    compute_chi2_pte,
    draw_normalized_boxes_ell_scale,
    find_fiducial_index,
    get_version_alpha,
    version_label,
)
# Import to register SquareRootScale
import plotting_utils  # noqa: F401


plt.style.use(PAPER_MPLSTYLE)




def main():
    # Read config
    import yaml
    with open(snakemake.input["config"]) as f:
        config = yaml.safe_load(f)

    # Scale cuts from params (passed from rule, originally Guerrini et al.)
    ell_min_cut = int(snakemake.params.ell_min_cut)
    ell_max_cut = int(snakemake.params.ell_max_cut)

    version_labels = snakemake.params.version_labels
    # Version list: from params if provided (ecut comparison), else from config
    versions = getattr(snakemake.params, "versions", None)
    if versions is None:
        versions = [v for v in config["versions"] if "_leak_corr" in v]
    plotting_config = config["plotting"]

    # Which version gets the fiducial reference line in boxes
    fiducial_for_comparison = getattr(
        snakemake.params, "fiducial_for_comparison",
        plotting_config.get("fiducial_for_comparison", config["fiducial"]["version"]),
    )
    box_style = plotting_config.get("version_box", {})

    # Load data for all versions
    datasets = []

    # Color/marker assignment: pair by parent version if ecut versions present
    has_ecut = any("_ecut" in v for v in versions)
    if has_ecut:
        # Group by parent: same color, filled=uncut / open=cut
        parents = []
        for v in versions:
            parent = v.split("_ecut")[0].replace("_leak_corr", "")
            if parent not in parents:
                parents.append(parent)
        parent_colors = sns.husl_palette(len(parents), l=0.4)
        parent_color_map = dict(zip(parents, parent_colors))

    for i, version in enumerate(versions):
        if has_ecut:
            parent = version.split("_ecut")[0].replace("_leak_corr", "")
            color = parent_color_map[parent]
            is_cut = "_ecut" in version
            marker = "o" if not is_cut else "D"
            fillstyle = "full" if not is_cut else "none"
        else:
            colors = sns.husl_palette(len(versions), l=0.4)
            color = colors[i]
            marker = MARKER_STYLES[i] if i < len(MARKER_STYLES) else "o"
            fillstyle = "full"

        hdu = fits.open(snakemake.input["pseudo_cl"][i])
        data = hdu["PSEUDO_CELL"].data
        hdu.close()

        ell = data["ELL"]
        cl_bb = data["BB"]
        cl_eb = data["EB"]

        hdu_cov = fits.open(snakemake.input["pseudo_cl_cov"][i])
        cov_bb = hdu_cov["COVAR_BB_BB"].data
        cov_eb = hdu_cov["COVAR_EB_EB"].data
        hdu_cov.close()

        sigma_bb = np.sqrt(np.diag(cov_bb))
        sigma_eb = np.sqrt(np.diag(cov_eb))

        # Full-range PTEs
        chi2_bb, pte_bb, dof_bb = compute_chi2_pte(cl_bb, cov_bb)
        chi2_eb, pte_eb, dof_eb = compute_chi2_pte(cl_eb, cov_eb)

        # Scale-cut PTEs
        cut_mask = (ell >= ell_min_cut) & (ell <= ell_max_cut)
        idx = np.where(cut_mask)[0]
        cl_bb_cut = cl_bb[idx]
        cl_eb_cut = cl_eb[idx]
        cov_bb_cut = cov_bb[np.ix_(idx, idx)]
        cov_eb_cut = cov_eb[np.ix_(idx, idx)]
        chi2_bb_cut, pte_bb_cut, dof_bb_cut = compute_chi2_pte(cl_bb_cut, cov_bb_cut)
        chi2_eb_cut, pte_eb_cut, dof_eb_cut = compute_chi2_pte(cl_eb_cut, cov_eb_cut)

        datasets.append({
            "version": version,
            "label": version_label(version, version_labels),
            "color": color,
            "marker": marker,
            "fillstyle": fillstyle,
            "alpha": get_version_alpha(version, fiducial_for_comparison, plotting_config),
            "ell": ell,
            "cl_bb": cl_bb,
            "cl_eb": cl_eb,
            "sigma_bb": sigma_bb,
            "sigma_eb": sigma_eb,
            "pte_bb": pte_bb,
            "chi2_bb": chi2_bb,
            "dof_bb": dof_bb,
            "pte_eb": pte_eb,
            "chi2_eb": chi2_eb,
            "dof_eb": dof_eb,
            "pte_bb_cut": pte_bb_cut,
            "chi2_bb_cut": chi2_bb_cut,
            "dof_bb_cut": dof_bb_cut,
            "pte_eb_cut": pte_eb_cut,
            "chi2_eb_cut": chi2_eb_cut,
            "dof_eb_cut": dof_eb_cut,
        })

    # Two-panel figure: BB (top) and EB (bottom)
    fig, (ax_bb, ax_eb) = plt.subplots(2, 1, figsize=(FIG_WIDTH_FULL, FIG_WIDTH_FULL * 0.45), sharex=True)

    ell_ref = datasets[0]["ell"]
    ell_widths = np.diff(ell_ref)
    ell_widths = np.append(ell_widths, ell_widths[-1])
    jitter_fraction = 0.15

    fiducial_idx = find_fiducial_index(datasets, fiducial_for_comparison)

    # Pre-compute normalized values for box drawing
    for data in datasets:
        data["cl_bb_normalized"] = data["cl_bb"] / data["sigma_bb"]
        data["cl_eb_normalized"] = data["cl_eb"] / data["sigma_eb"]

    # Draw version spread boxes (before data points)
    draw_normalized_boxes_ell_scale(
        ax_bb, ell_ref, ell_widths, datasets,
        y_norm_key="cl_bb_normalized", fiducial_idx=fiducial_idx,
        jitter_fraction=jitter_fraction, n_versions=len(datasets), box_style=box_style
    )
    draw_normalized_boxes_ell_scale(
        ax_eb, ell_ref, ell_widths, datasets,
        y_norm_key="cl_eb_normalized", fiducial_idx=fiducial_idx,
        jitter_fraction=jitter_fraction, n_versions=len(datasets), box_style=box_style
    )

    legend_handles = []
    legend_labels = []

    for i, data in enumerate(datasets):
        ell = data["ell"]
        jitter_offset = (i - (len(datasets) - 1) / 2) * jitter_fraction
        ell_jittered = ell + jitter_offset * ell_widths
        color = data["color"]
        label = data["label"]
        alpha = data["alpha"]
        marker = data["marker"]
        fillstyle = data["fillstyle"]
        mfc = color if fillstyle == "full" else "none"

        cl_bb_normalized = data["cl_bb_normalized"]
        cl_eb_normalized = data["cl_eb_normalized"]

        line_bb = ax_bb.errorbar(
            ell_jittered, cl_bb_normalized, yerr=np.ones_like(cl_bb_normalized),
            fmt=marker, color=color, alpha=alpha,
            markerfacecolor=mfc, markeredgecolor=color,
            **ERRORBAR_DEFAULTS,
            zorder=2,
        )

        ax_eb.errorbar(
            ell_jittered, cl_eb_normalized, yerr=np.ones_like(cl_eb_normalized),
            fmt=marker, color=color, alpha=alpha,
            markerfacecolor=mfc, markeredgecolor=color,
            **ERRORBAR_DEFAULTS,
            zorder=2,
        )

        # Collect legend handles from BB panel only
        legend_handles.append(line_bb)
        legend_labels.append(label)

    all_ell = np.concatenate([d["ell"] for d in datasets])
    ell_min, ell_max = all_ell.min() * 0.9, all_ell.max() * 1.05

    major_ticks = np.array([100, 400, 900, 1600])
    minor_ticks = [i * 10 for i in range(1, 10)] + [i * 100 for i in range(1, 21)]

    for ax, ylabel in [(ax_bb, r"$C_\ell^{BB} / \sigma$"), (ax_eb, r"$C_\ell^{EB} / \sigma$")]:
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_xscale("squareroot")
        ax.set_xlim(ell_min, ell_max)

        # Shade excluded regions (matching cl_data_vector)
        xlim = ax.get_xlim()
        ax.axvspan(xlim[0], ell_min_cut, alpha=0.1, color="gray", zorder=0)
        ax.axvspan(ell_max_cut, xlim[1], alpha=0.1, color="gray", zorder=0)
        ax.set_xlim(xlim)  # Restore limits after shading

        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.minorticks_on()
        ax.tick_params(axis="x", which="minor", length=2, width=0.8)
        ax.set_ylabel(ylabel)

    # Legend only on upper panel, single row with four columns, at bottom
    ax_bb.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=True,
        framealpha=0.9,
    )

    ax_eb.set_xlabel(r"$\ell$")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08)

    # Save outputs
    output_dir = Path(snakemake.output["evidence"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = Path(snakemake.output["figure"])
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_path}")

    paper_path = Path(snakemake.output["paper_figure"])
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_path, bbox_inches="tight")
    print(f"Saved {paper_path}")

    plt.close(fig)

    # Build evidence
    evidence_versions = {}
    for data in datasets:
        v = data["version"]
        evidence_versions[f"{v}_pte_bb"] = float(data["pte_bb"])
        evidence_versions[f"{v}_chi2_bb"] = float(data["chi2_bb"])
        evidence_versions[f"{v}_dof_bb"] = int(data["dof_bb"])
        evidence_versions[f"{v}_pte_eb"] = float(data["pte_eb"])
        evidence_versions[f"{v}_chi2_eb"] = float(data["chi2_eb"])
        evidence_versions[f"{v}_dof_eb"] = int(data["dof_eb"])
        evidence_versions[f"{v}_pte_bb_cut"] = float(data["pte_bb_cut"])
        evidence_versions[f"{v}_chi2_bb_cut"] = float(data["chi2_bb_cut"])
        evidence_versions[f"{v}_dof_bb_cut"] = int(data["dof_bb_cut"])
        evidence_versions[f"{v}_pte_eb_cut"] = float(data["pte_eb_cut"])
        evidence_versions[f"{v}_chi2_eb_cut"] = float(data["chi2_eb_cut"])
        evidence_versions[f"{v}_dof_eb_cut"] = int(data["dof_eb_cut"])

    spec_paths = snakemake.input["specs"]

    evidence_data = {
        "spec_id": "cl_version_comparison",
        "spec_path": spec_paths[0],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "versions": evidence_versions,
        },
        "output": {
            "figure": fig_path.name,
        },
    }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"Saved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
