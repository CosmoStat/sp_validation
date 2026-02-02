"""Pure E/B data vector claim.

Shows B-mode signals consistent with zero at fiducial scale cuts.
Writes evidence.json with PTE values including joint B-mode test.

Uses fiducial blind only (config.fiducial.blind) for PTE calculation.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


plt.style.use(
    "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle"
)


def _extract_sigma(covariance, block_index, block_size):
    """Extract standard deviations from diagonal of covariance block."""
    block_slice = slice(block_index * block_size, (block_index + 1) * block_size)
    return np.sqrt(np.clip(np.diag(covariance[block_slice, block_slice]), 0, None))


def _compute_pte(data, covariance):
    """Compute PTE for B-mode null test."""
    chi2 = float(data @ np.linalg.solve(covariance, data))
    return stats.chi2.sf(chi2, len(data)), chi2, len(data)


def _compute_joint_pte(xip_B, xim_B, cov_xip_B, cov_xim_B, cov_cross):
    """Compute joint PTE for combined B-mode data vector [xip_B, xim_B]."""
    data_joint = np.concatenate([xip_B, xim_B])
    n_xip, n_xim = len(xip_B), len(xim_B)

    cov_joint = np.zeros((n_xip + n_xim, n_xip + n_xim))
    cov_joint[:n_xip, :n_xip] = cov_xip_B
    cov_joint[n_xip:, n_xip:] = cov_xim_B
    cov_joint[:n_xip, n_xip:] = cov_cross
    cov_joint[n_xip:, :n_xip] = cov_cross.T

    chi2 = float(data_joint @ np.linalg.solve(cov_joint, data_joint))
    return stats.chi2.sf(chi2, n_xip + n_xim), chi2, n_xip + n_xim


def main():
    config = snakemake.config
    blind = config["fiducial"]["blind"]
    version = config["fiducial"]["version"]
    fiducial_xip_scale_cut = tuple(config["fiducial"]["fiducial_xip_scale_cut"])
    fiducial_xim_scale_cut = tuple(config["fiducial"]["fiducial_xim_scale_cut"])

    # Load precomputed data (MC-propagated E/B covariance)
    dataset = np.load(snakemake.input["pure_eb"])
    theta = dataset["theta"]
    nbins = len(theta)

    # Data vectors
    xip_total, xim_total = dataset["xip_total"], dataset["xim_total"]
    xip_E, xim_E = dataset["xip_E"], dataset["xim_E"]
    xip_B, xim_B = dataset["xip_B"], dataset["xim_B"]
    xip_amb, xim_amb = dataset["xip_amb"], dataset["xim_amb"]

    # Covariances
    cov_pure_eb = dataset["cov_pure_eb"]
    cov_xi = np.loadtxt(snakemake.input["cov"])

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
    pte_xip_fid, chi2_xip_fid, dof_xip_fid = _compute_pte(xip_B[mask_xip], cov_xip_B_cut)
    pte_xim_fid, chi2_xim_fid, dof_xim_fid = _compute_pte(xim_B[mask_xim], cov_xim_B_cut)
    pte_joint_fid, chi2_joint_fid, dof_joint_fid = _compute_joint_pte(
        xip_B[mask_xip], xim_B[mask_xim], cov_xip_B_cut, cov_xim_B_cut, cov_cross_cut
    )

    # Compute PTEs at full range
    pte_xip_full, _, _ = _compute_pte(xip_B, cov_xip_B_full)
    pte_xim_full, _, _ = _compute_pte(xim_B, cov_xim_B_full)
    pte_joint_full, chi2_joint_full, dof_joint_full = _compute_joint_pte(
        xip_B, xim_B, cov_xip_B_full, cov_xim_B_full, cov_cross_full
    )

    print(f"Blind {blind} PTEs (fiducial): xi+^B={pte_xip_fid:.3f}, xi-^B={pte_xim_fid:.3f}, joint={pte_joint_fid:.3f}")

    # Extract error bars for plotting
    sigma_xip_total = _extract_sigma(cov_xi, 0, nbins)
    sigma_xim_total = _extract_sigma(cov_xi, 1, nbins)
    sigma_xip_E, sigma_xim_E = _extract_sigma(cov_pure_eb, 0, nbins), _extract_sigma(cov_pure_eb, 1, nbins)
    sigma_xip_B, sigma_xim_B = _extract_sigma(cov_pure_eb, 2, nbins), _extract_sigma(cov_pure_eb, 3, nbins)
    sigma_xip_amb, sigma_xim_amb = _extract_sigma(cov_pure_eb, 4, nbins), _extract_sigma(cov_pure_eb, 5, nbins)

    # Plotting parameters
    scale_factor = 1e-4
    xlim, ylim = [1, 250], [-0.3, 1.25]
    ms, capsize, capthick, elinewidth = 2.0, 1.5, 0.3, 0.4
    color_total, color_E, color_B, color_amb = "k", "#008080", "crimson", "#7570b3"
    offsets = [0.90, 0.96, 1.04, 1.10]
    alpha_main, alpha_faint = 1.0, 0.25

    def shade_excluded_regions(ax, scale_cut):
        ax.axvspan(scale_cut[0], scale_cut[1], alpha=0.1, color="gray", zorder=0)

    def setup_panel(ax, ylabel_text=None):
        ax.axhline(0, color="k", linestyle="--", alpha=0.6, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlim(xlim)
        ax.set_xlabel(r"$\theta$ [arcmin]")
        ylabel_text and ax.set_ylabel(ylabel_text)

    # Create 1x2 figure
    fig, axes = plt.subplots(1, 2, figsize=(7.24, 3.5), sharey=True)

    # Left panel: xi+ decomposition
    ax = axes[0]
    plot_data = [
        (xip_total, sigma_xip_total, "o", color_total, alpha_main, r"$\xi_\pm$ (total)"),
        (xip_E, sigma_xip_E, "s", color_E, alpha_faint, r"$\xi_\pm^E$"),
        (xip_B, sigma_xip_B, "X", color_B, alpha_main, r"$\xi_\pm^B$"),
        (xip_amb, sigma_xip_amb, "v", color_amb, alpha_faint, r"$\xi_\pm^\mathrm{amb}$"),
    ]
    for i, (data, sigma, marker, color, alpha, label) in enumerate(plot_data):
        ax.errorbar(
            theta * offsets[i], theta * data / scale_factor, yerr=theta * sigma / scale_factor,
            fmt=marker, color=color, markersize=ms, capsize=capsize, capthick=capthick,
            elinewidth=elinewidth, alpha=alpha, label=label
        )
    shade_excluded_regions(ax, fiducial_xip_scale_cut)
    setup_panel(ax, ylabel_text=r"$\theta \xi \times 10^4$")
    ax.set_title(r"$\xi_+$")

    # Right panel: xi- decomposition
    ax = axes[1]
    xim_plot_data = [
        (xim_total, sigma_xim_total, "o", color_total, alpha_main),
        (xim_E, sigma_xim_E, "s", color_E, alpha_faint),
        (xim_B, sigma_xim_B, "X", color_B, alpha_main),
        (xim_amb, sigma_xim_amb, "v", color_amb, alpha_faint),
    ]
    for i, (data, sigma, marker, color, alpha) in enumerate(xim_plot_data):
        ax.errorbar(
            theta * offsets[i], theta * data / scale_factor, yerr=theta * sigma / scale_factor,
            fmt=marker, color=color, markersize=ms, capsize=capsize, capthick=capthick,
            elinewidth=elinewidth, alpha=alpha
        )
    shade_excluded_regions(ax, fiducial_xim_scale_cut)
    setup_panel(ax)
    ax.set_title(r"$\xi_-$")
    handles, labels = axes[0].get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", fontsize="small")

    axes[0].set_ylim(ylim)
    fig.tight_layout()

    # Save outputs
    output_dir = Path(snakemake.output["evidence"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = output_dir / "figure.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_path}")

    if "paper_figure" in snakemake.output.keys():
        paper_path = Path(snakemake.output["paper_figure"])
        paper_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fig_path, paper_path)
        print(f"Copied to {paper_path}")

    plt.close(fig)

    # Write evidence.json
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
        "artifacts": {"figure": "figure.png"},
    }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"Saved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
