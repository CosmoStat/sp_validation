"""Pure E/B data vector claim.

Shows B-mode signals consistent with zero at fiducial scale cuts.
Writes evidence.json with PTE values including joint B-mode test.

Uses fiducial blind only (config.fiducial.blind) for PTE calculation.

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

from plotting_utils import (
    FIG_WIDTH_FULL,
    PAPER_MPLSTYLE,
    compute_chi2_pte,
    iter_version_figures,
)


plt.style.use(PAPER_MPLSTYLE)


def _extract_sigma(covariance, block_index, block_size):
    """Extract standard deviations from diagonal of covariance block."""
    block_slice = slice(block_index * block_size, (block_index + 1) * block_size)
    return np.sqrt(np.clip(np.diag(covariance[block_slice, block_slice]), 0, None))


def _compute_joint_pte(xip_B, xim_B, cov_xip_B, cov_xim_B, cov_cross, n_samples=None):
    """Compute joint PTE for combined B-mode data vector [xip_B, xim_B]."""
    data_joint = np.concatenate([xip_B, xim_B])
    n_xip, n_xim = len(xip_B), len(xim_B)

    cov_joint = np.zeros((n_xip + n_xim, n_xip + n_xim))
    cov_joint[:n_xip, :n_xip] = cov_xip_B
    cov_joint[n_xip:, n_xip:] = cov_xim_B
    cov_joint[:n_xip, n_xip:] = cov_cross
    cov_joint[n_xip:, :n_xip] = cov_cross.T

    chi2, pte, dof = compute_chi2_pte(data_joint, cov_joint, n_samples=n_samples)
    return pte, chi2, dof


def _load_pure_eb_data(pure_eb_path, cov_path):
    """Load pure E/B decomposition and covariances."""
    dataset = np.load(pure_eb_path)
    theta = dataset["theta"]
    nbins = len(theta)

    data = {
        "theta": theta,
        "nbins": nbins,
        "xip_total": dataset["xip_total"],
        "xim_total": dataset["xim_total"],
        "xip_E": dataset["xip_E"],
        "xim_E": dataset["xim_E"],
        "xip_B": dataset["xip_B"],
        "xim_B": dataset["xim_B"],
        "xip_amb": dataset["xip_amb"],
        "xim_amb": dataset["xim_amb"],
        "cov_pure_eb": dataset["cov_pure_eb"],
        "cov_xi": np.loadtxt(cov_path),
    }
    return data


def _create_pure_eb_figure(data, fiducial_xip_scale_cut, fiducial_xim_scale_cut, title_suffix=""):
    """Create pure E/B decomposition figure.

    Args:
        title_suffix: Optional suffix for panel titles (e.g., " (uncorrected)")
    """
    theta = data["theta"]
    nbins = data["nbins"]
    cov_xi = data["cov_xi"]
    cov_pure_eb = data["cov_pure_eb"]

    # Extract error bars
    sigma_xip_total = _extract_sigma(cov_xi, 0, nbins)
    sigma_xim_total = _extract_sigma(cov_xi, 1, nbins)
    sigma_xip_E = _extract_sigma(cov_pure_eb, 0, nbins)
    sigma_xim_E = _extract_sigma(cov_pure_eb, 1, nbins)
    sigma_xip_B = _extract_sigma(cov_pure_eb, 2, nbins)
    sigma_xim_B = _extract_sigma(cov_pure_eb, 3, nbins)
    sigma_xip_amb = _extract_sigma(cov_pure_eb, 4, nbins)
    sigma_xim_amb = _extract_sigma(cov_pure_eb, 5, nbins)

    # Plotting parameters
    scale_factor = 1e-4
    xlim, ylim = [1, 250], [-0.3, 1.25]
    ms, capsize, capthick, elinewidth = 2.0, 1.5, 0.3, 0.4
    color_total, color_E, color_B, color_amb = "k", "#008080", "crimson", "#7570b3"
    offsets = [0.90, 0.96, 1.04, 1.10]
    alpha_main, alpha_faint = 1.0, 0.45

    def shade_excluded_regions(ax, scale_cut):
        # Shade regions outside the fiducial scale cuts
        ax.axvspan(xlim[0], scale_cut[0], alpha=0.1, color="gray", zorder=0)
        ax.axvspan(scale_cut[1], xlim[1], alpha=0.1, color="gray", zorder=0)

    def setup_panel(ax, ylabel_text=None):
        ax.axhline(0, color="k", linestyle="--", alpha=0.6, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlim(xlim)
        ax.set_xlabel(r"$\theta$ [arcmin]")
        ylabel_text and ax.set_ylabel(ylabel_text)

    # Create 1x2 figure
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_FULL, FIG_WIDTH_FULL * 0.48), sharey=True)

    # Left panel: xi+ decomposition
    ax = axes[0]
    plot_data = [
        (data["xip_total"], sigma_xip_total, "o", color_total, alpha_main, r"$\xi_\pm$ (total)"),
        (data["xip_E"], sigma_xip_E, "s", color_E, alpha_faint, r"$\xi_\pm^E$"),
        (data["xip_B"], sigma_xip_B, "X", color_B, alpha_main, r"$\xi_\pm^B$"),
        (data["xip_amb"], sigma_xip_amb, "v", color_amb, alpha_faint, r"$\xi_\pm^\mathrm{amb}$"),
    ]
    for i, (d, sigma, marker, color, alpha, label) in enumerate(plot_data):
        ax.errorbar(
            theta * offsets[i], theta * d / scale_factor, yerr=theta * sigma / scale_factor,
            fmt=marker, color=color, markersize=ms, capsize=capsize, capthick=capthick,
            elinewidth=elinewidth, alpha=alpha, label=label
        )
    shade_excluded_regions(ax, fiducial_xip_scale_cut)
    setup_panel(ax, ylabel_text=r"$\theta \xi \times 10^4$")
    panel_label = rf"$\xi_+${title_suffix}" if title_suffix else r"$\xi_+$"
    ax.text(0.05, 0.95, panel_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"))

    # Right panel: xi- decomposition
    ax = axes[1]
    xim_plot_data = [
        (data["xim_total"], sigma_xim_total, "o", color_total, alpha_main),
        (data["xim_E"], sigma_xim_E, "s", color_E, alpha_faint),
        (data["xim_B"], sigma_xim_B, "X", color_B, alpha_main),
        (data["xim_amb"], sigma_xim_amb, "v", color_amb, alpha_faint),
    ]
    for i, (d, sigma, marker, color, alpha) in enumerate(xim_plot_data):
        ax.errorbar(
            theta * offsets[i], theta * d / scale_factor, yerr=theta * sigma / scale_factor,
            fmt=marker, color=color, markersize=ms, capsize=capsize, capthick=capthick,
            elinewidth=elinewidth, alpha=alpha
        )
    shade_excluded_regions(ax, fiducial_xim_scale_cut)
    setup_panel(ax)
    panel_label = rf"$\xi_-${title_suffix}" if title_suffix else r"$\xi_-$"
    ax.text(0.05, 0.95, panel_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"))
    handles, labels = axes[0].get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center", fontsize="small")

    axes[0].set_ylim(ylim)
    fig.tight_layout()
    return fig


def main():
    config = snakemake.config
    blind = config["fiducial"]["blind"]
    version = config["fiducial"]["version"]
    version_labels = config["plotting"]["version_labels"]
    fiducial_xip_scale_cut = tuple(config["fiducial"]["fiducial_xip_scale_cut"])
    fiducial_xim_scale_cut = tuple(config["fiducial"]["fiducial_xim_scale_cut"])

    # Build input path lookup from snakemake inputs
    pure_eb_paths = {k: v for k, v in snakemake.input.items() if k.startswith("pure_eb_")}
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
            pure_eb_key = f"pure_eb_{fig_spec['version_leak_corr']}"
            cov_key = f"cov_{fig_spec['version_leak_corr']}"
        else:
            pure_eb_key = f"pure_eb_{fig_spec['version_uncorr']}"
            cov_key = f"cov_{fig_spec['version_uncorr']}"

        # Load data
        data = _load_pure_eb_data(pure_eb_paths[pure_eb_key], cov_paths[cov_key])

        # Create figure with appropriate title
        title_suffix = f" ({fig_spec['title']})" if fig_spec["title"] else ""
        fig = _create_pure_eb_figure(
            data, fiducial_xip_scale_cut, fiducial_xim_scale_cut,
            title_suffix=title_suffix
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

    # Compute PTEs for evidence (fiducial version, leak-corrected only)
    data = _load_pure_eb_data(
        pure_eb_paths[f"pure_eb_{version}"],
        cov_paths[f"cov_{version}"]
    )
    theta = data["theta"]
    nbins = data["nbins"]
    cov_pure_eb = data["cov_pure_eb"]
    xip_B, xim_B = data["xip_B"], data["xim_B"]

    # Hartlap correction: MC-propagated covariance uses n_samples from config
    n_samples = int(config["covariance"]["n_samples"])

    # Extract B-mode covariance blocks
    cov_xip_B_full = cov_pure_eb[2 * nbins : 3 * nbins, 2 * nbins : 3 * nbins]
    cov_xim_B_full = cov_pure_eb[3 * nbins : 4 * nbins, 3 * nbins : 4 * nbins]
    cov_cross_full = cov_pure_eb[2 * nbins : 3 * nbins, 3 * nbins : 4 * nbins]

    # Scale cut masks
    mask_xip = (theta >= fiducial_xip_scale_cut[0]) & (theta <= fiducial_xip_scale_cut[1])
    mask_xim = (theta >= fiducial_xim_scale_cut[0]) & (theta <= fiducial_xim_scale_cut[1])

    # Apply scale cuts to covariances
    cov_xip_B_cut = cov_xip_B_full[np.ix_(mask_xip, mask_xip)]
    cov_xim_B_cut = cov_xim_B_full[np.ix_(mask_xim, mask_xim)]
    cov_cross_cut = cov_cross_full[np.ix_(mask_xip, mask_xim)]

    # Compute PTEs at fiducial scale cuts
    chi2_xip_fid, pte_xip_fid, dof_xip_fid = compute_chi2_pte(xip_B[mask_xip], cov_xip_B_cut, n_samples=n_samples)
    chi2_xim_fid, pte_xim_fid, dof_xim_fid = compute_chi2_pte(xim_B[mask_xim], cov_xim_B_cut, n_samples=n_samples)
    pte_joint_fid, chi2_joint_fid, dof_joint_fid = _compute_joint_pte(
        xip_B[mask_xip], xim_B[mask_xim], cov_xip_B_cut, cov_xim_B_cut, cov_cross_cut, n_samples=n_samples
    )

    # Compute PTEs at full range
    _, pte_xip_full, _ = compute_chi2_pte(xip_B, cov_xip_B_full, n_samples=n_samples)
    _, pte_xim_full, _ = compute_chi2_pte(xim_B, cov_xim_B_full, n_samples=n_samples)
    pte_joint_full, chi2_joint_full, dof_joint_full = _compute_joint_pte(
        xip_B, xim_B, cov_xip_B_full, cov_xim_B_full, cov_cross_full, n_samples=n_samples
    )

    print(f"Blind {blind} PTEs (fiducial): xi+^B={pte_xip_fid:.3f}, xi-^B={pte_xim_fid:.3f}, joint={pte_joint_fid:.3f}")

    # Write evidence.json (based on leak-corrected fiducial data only)
    evidence_data = {
        "spec_id": "pure_eb_data_vector",
        "spec_path": snakemake.input["specs"][0],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "fiducial": {
                "scale_cut_xip": list(fiducial_xip_scale_cut),
                "scale_cut_xim": list(fiducial_xim_scale_cut),
                "pte_xip_B": float(pte_xip_fid),
                "pte_xim_B": float(pte_xim_fid),
                "pte_joint": float(pte_joint_fid),
                "chi2_joint_B": float(chi2_joint_fid),
                "dof_joint_B": int(dof_joint_fid),
            },
            "full": {
                "scale_cut_arcmin": [float(theta.min()), float(theta.max())],
                "pte_xip_B": float(pte_xip_full),
                "pte_xim_B": float(pte_xim_full),
                "pte_joint": float(pte_joint_full),
                "chi2_joint_B": float(chi2_joint_full),
                "dof_joint_B": int(dof_joint_full),
            },
            "version": version,
            "blind": blind,
        },
        "output": output,
    }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"Saved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
