"""Pure E/B data vector claim.

Shows B-mode signals consistent with zero at fiducial scale cuts.
Writes evidence.json with PTE values including joint B-mode test.

Uses blinds specified in config (default: A).
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import stats


plt.style.use(
    "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle"
)


def _extract_sigma(covariance, block_index, block_size):
    """Extract standard deviations from diagonal of covariance block."""
    block_slice = slice(block_index * block_size, (block_index + 1) * block_size)
    block = covariance[block_slice, block_slice]
    return np.sqrt(np.clip(np.diag(block), 0, None))


def _compute_pte(data, covariance):
    """Compute PTE for B-mode null test."""
    chi2 = float(data @ np.linalg.solve(covariance, data))
    dof = len(data)
    return stats.chi2.sf(chi2, dof), chi2, dof


def _compute_joint_pte(xip_B, xim_B, cov_xip_B, cov_xim_B, cov_cross):
    """Compute joint PTE for combined B-mode data vector [xip_B, xim_B].

    Parameters
    ----------
    xip_B : array
        xi+^B data vector (after scale cuts)
    xim_B : array
        xi-^B data vector (after scale cuts)
    cov_xip_B : array
        Covariance of xi+^B
    cov_xim_B : array
        Covariance of xi-^B
    cov_cross : array
        Cross-covariance between xi+^B and xi-^B

    Returns
    -------
    pte : float
        Joint PTE
    chi2 : float
        Joint chi-squared
    dof : int
        Degrees of freedom
    """
    # Build joint data vector
    data_joint = np.concatenate([xip_B, xim_B])

    # Build joint covariance matrix
    n_xip = len(xip_B)
    n_xim = len(xim_B)
    n_total = n_xip + n_xim

    cov_joint = np.zeros((n_total, n_total))
    cov_joint[:n_xip, :n_xip] = cov_xip_B
    cov_joint[n_xip:, n_xip:] = cov_xim_B
    cov_joint[:n_xip, n_xip:] = cov_cross
    cov_joint[n_xip:, :n_xip] = cov_cross.T

    chi2 = float(data_joint @ np.linalg.solve(cov_joint, data_joint))
    dof = n_total
    pte = stats.chi2.sf(chi2, dof)

    return pte, chi2, dof


def _compute_per_blind_ptes(xip_B, xim_B, cov_pure_eb_blinds, nbins, mask_xip, mask_xim):
    """Compute PTEs for each blind using per-blind MC-propagated covariances.

    Parameters
    ----------
    xip_B, xim_B : array
        B-mode data vectors (same for all blinds)
    cov_pure_eb_blinds : dict
        {blind: cov_pure_eb} for each blind (MC-propagated)
    nbins : int
        Number of angular bins
    mask_xip, mask_xim : array
        Scale cut masks

    Returns
    -------
    dict
        Per-blind PTEs and statistics
    """
    results = {}

    for blind in cov_pure_eb_blinds:
        # Get the per-blind MC-propagated E/B covariance
        cov_pure_eb = cov_pure_eb_blinds[blind]

        # Extract B-mode blocks
        cov_xip_B_full = cov_pure_eb[2*nbins:3*nbins, 2*nbins:3*nbins]
        cov_xim_B_full = cov_pure_eb[3*nbins:4*nbins, 3*nbins:4*nbins]
        cov_cross_full = cov_pure_eb[2*nbins:3*nbins, 3*nbins:4*nbins]

        # Apply scale cuts
        cov_xip_B_cut = cov_xip_B_full[np.ix_(mask_xip, mask_xip)]
        cov_xim_B_cut = cov_xim_B_full[np.ix_(mask_xim, mask_xim)]
        cov_cross_cut = cov_cross_full[np.ix_(mask_xip, mask_xim)]

        xip_B_cut = xip_B[mask_xip]
        xim_B_cut = xim_B[mask_xim]

        # Compute PTEs at fiducial scale
        pte_xip, chi2_xip, dof_xip = _compute_pte(xip_B_cut, cov_xip_B_cut)
        pte_xim, chi2_xim, dof_xim = _compute_pte(xim_B_cut, cov_xim_B_cut)
        pte_joint, chi2_joint, dof_joint = _compute_joint_pte(
            xip_B_cut, xim_B_cut, cov_xip_B_cut, cov_xim_B_cut, cov_cross_cut
        )

        # Full range PTEs
        pte_xip_full, _, _ = _compute_pte(xip_B, cov_xip_B_full)
        pte_xim_full, _, _ = _compute_pte(xim_B, cov_xim_B_full)
        pte_joint_full, chi2_joint_full, dof_joint_full = _compute_joint_pte(
            xip_B, xim_B, cov_xip_B_full, cov_xim_B_full, cov_cross_full
        )

        results[blind] = {
            "fiducial": {
                "pte_xip_B": float(pte_xip),
                "pte_xim_B": float(pte_xim),
                "pte_joint_B": float(pte_joint),
                "chi2_joint_B": float(chi2_joint),
                "dof_joint_B": int(dof_joint),
            },
            "full": {
                "pte_xip_B": float(pte_xip_full),
                "pte_xim_B": float(pte_xim_full),
                "pte_joint_B": float(pte_joint_full),
                "chi2_joint_B": float(chi2_joint_full),
                "dof_joint_B": int(dof_joint_full),
            },
        }

    return results


def main():
    config = snakemake.config

    # Get blinds from config
    blinds = config["blinds"]
    default_blind = blinds[0]  # First blind is the default (typically A)

    # Get fiducial scale cuts from config
    fiducial_xip_scale_cut = tuple(config["fiducial"]["fiducial_xip_scale_cut"])
    fiducial_xim_scale_cut = tuple(config["fiducial"]["fiducial_xim_scale_cut"])
    version = config["fiducial"]["version"]

    # Load precomputed pure E/B data (data vectors are identical across blinds)
    dataset = np.load(snakemake.input.pure_eb)
    theta = dataset["theta"]
    nbins = len(theta)

    # Total correlation functions
    xip_total = dataset["xip_total"]
    xim_total = dataset["xim_total"]

    # Pure mode decompositions
    xip_E = dataset["xip_E"]
    xim_E = dataset["xim_E"]
    xip_B = dataset["xip_B"]
    xim_B = dataset["xim_B"]
    xip_amb = dataset["xip_amb"]
    xim_amb = dataset["xim_amb"]

    # Load per-blind MC-propagated E/B covariances
    cov_pure_eb_blinds = {}
    for blind in blinds:
        cov_pure_eb_blinds[blind] = dataset["cov_pure_eb"]

    # Load per-blind reporting covariances (for total xi error bars)
    cov_xi_blinds = {}
    for i, blind in enumerate(blinds):
        cov_path = snakemake.input.cov[i]
        cov_xi_blinds[blind] = np.loadtxt(cov_path)

    # Apply scale cuts
    mask_xip = (theta >= fiducial_xip_scale_cut[0]) & (theta <= fiducial_xip_scale_cut[1])
    mask_xim = (theta >= fiducial_xim_scale_cut[0]) & (theta <= fiducial_xim_scale_cut[1])

    # Compute per-blind PTEs using per-blind MC-propagated covariances
    blind_results = _compute_per_blind_ptes(
        xip_B, xim_B, cov_pure_eb_blinds, nbins, mask_xip, mask_xim
    )

    # Identify blind with minimum joint PTE at fiducial scale (most conservative)
    min_pte_blind = min(blinds, key=lambda b: blind_results[b]["fiducial"]["pte_joint_B"])
    pte_xip_B_fid = blind_results[min_pte_blind]["fiducial"]["pte_xip_B"]
    pte_xim_B_fid = blind_results[min_pte_blind]["fiducial"]["pte_xim_B"]

    # Get covariance for the min-PTE blind (used for E/B mode error bars)
    cov_pure_eb_plot = cov_pure_eb_blinds[min_pte_blind]

    # For total xi_pm, use ng reporting covariance from min-PTE blind
    # (CosmoCov non-Gaussian, consistent with official results)
    cov_xi_plotting = cov_xi_blinds[min_pte_blind]
    sigma_xip_total = _extract_sigma(cov_xi_plotting, 0, nbins)
    sigma_xim_total = _extract_sigma(cov_xi_plotting, 1, nbins)
    sigma_xip_E = _extract_sigma(cov_pure_eb_plot, 0, nbins)
    sigma_xim_E = _extract_sigma(cov_pure_eb_plot, 1, nbins)
    sigma_xip_B = _extract_sigma(cov_pure_eb_plot, 2, nbins)
    sigma_xim_B = _extract_sigma(cov_pure_eb_plot, 3, nbins)
    sigma_xip_amb = _extract_sigma(cov_pure_eb_plot, 4, nbins)
    sigma_xim_amb = _extract_sigma(cov_pure_eb_plot, 5, nbins)

    print(f"Using blind {min_pte_blind} covariance for plotting (min joint PTE = {blind_results[min_pte_blind]['fiducial']['pte_joint_B']:.4f})")

    # Plotting parameters
    scale_factor = 1e-4
    xlim = [1, 250]
    ylim = [-0.3, 1.25]

    # Sleek styling: finer markers and thinner error bars
    ms = 2.0  # marker size
    capsize = 1.5
    capthick = 0.3
    elinewidth = 0.4
    mew = 0.4

    # Colors
    color_total = "k"
    color_E = "#008080"  # teal
    color_B = "crimson"
    color_amb = "#7570b3"  # purple

    # Create 1x2 figure: decomposition layout (xi+ left, xi- right)
    # Each panel shows all four components overlaid
    fig, axes = plt.subplots(1, 2, figsize=(7.24, 3.5), sharey=True)

    def shade_excluded_regions(ax, scale_cut):
        """Shade included region (inside scale cuts)."""
        ax.axvspan(scale_cut[0], scale_cut[1], alpha=0.1, color="gray", zorder=0)

    def setup_panel(ax, ylabel_text=None):
        """Common panel setup."""
        ax.axhline(0, color="k", linestyle="--", alpha=0.6, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlim(xlim)
        ax.set_xlabel(r"$\theta$ [arcmin]")
        if ylabel_text:
            ax.set_ylabel(ylabel_text)

    # Horizontal offsets for overlaid points
    offsets = [0.90, 0.96, 1.04, 1.10]

    # Alpha values: tot and B opaque, E and amb transparent
    alpha_main = 1.0  # tot, B
    alpha_faint = 0.25  # E, amb

    # --- Left panel: xi+ decomposition ---
    ax = axes[0]
    ax.errorbar(
        theta * offsets[0], theta * xip_total / scale_factor,
        yerr=theta * sigma_xip_total / scale_factor,
        fmt="o", color=color_total, markersize=ms, capsize=capsize,
        capthick=capthick, elinewidth=elinewidth, alpha=alpha_main,
        label=r"$\xi_\pm$ (total)"
    )
    ax.errorbar(
        theta * offsets[1], theta * xip_E / scale_factor,
        yerr=theta * sigma_xip_E / scale_factor,
        fmt="s", color=color_E, markersize=ms, capsize=capsize,
        capthick=capthick, elinewidth=elinewidth, alpha=alpha_faint,
        label=r"$\xi_\pm^E$"
    )
    ax.errorbar(
        theta * offsets[2], theta * xip_B / scale_factor,
        yerr=theta * sigma_xip_B / scale_factor,
        fmt="X", color=color_B, markersize=ms, capsize=capsize,
        capthick=capthick, elinewidth=elinewidth, alpha=alpha_main,
        label=r"$\xi_\pm^B$"
    )
    ax.errorbar(
        theta * offsets[3], theta * xip_amb / scale_factor,
        yerr=theta * sigma_xip_amb / scale_factor,
        fmt="v", color=color_amb, markersize=ms, capsize=capsize,
        capthick=capthick, elinewidth=elinewidth, alpha=alpha_faint,
        label=r"$\xi_\pm^\mathrm{amb}$"
    )
    shade_excluded_regions(ax, fiducial_xip_scale_cut)
    setup_panel(ax, ylabel_text=r"$\theta \xi \times 10^4$")
    ax.set_title(r"$\xi_+$")

    # --- Right panel: xi- decomposition ---
    ax = axes[1]
    ax.errorbar(
        theta * offsets[0], theta * xim_total / scale_factor,
        yerr=theta * sigma_xim_total / scale_factor,
        fmt="o", color=color_total, markersize=ms, capsize=capsize,
        capthick=capthick, elinewidth=elinewidth, alpha=alpha_main,
    )
    ax.errorbar(
        theta * offsets[1], theta * xim_E / scale_factor,
        yerr=theta * sigma_xim_E / scale_factor,
        fmt="s", color=color_E, markersize=ms, capsize=capsize,
        capthick=capthick, elinewidth=elinewidth, alpha=alpha_faint,
    )
    ax.errorbar(
        theta * offsets[2], theta * xim_B / scale_factor,
        yerr=theta * sigma_xim_B / scale_factor,
        fmt="X", color=color_B, markersize=ms, capsize=capsize,
        capthick=capthick, elinewidth=elinewidth, alpha=alpha_main,
    )
    ax.errorbar(
        theta * offsets[3], theta * xim_amb / scale_factor,
        yerr=theta * sigma_xim_amb / scale_factor,
        fmt="v", color=color_amb, markersize=ms, capsize=capsize,
        capthick=capthick, elinewidth=elinewidth, alpha=alpha_faint,
    )
    shade_excluded_regions(ax, fiducial_xim_scale_cut)
    setup_panel(ax)
    ax.set_title(r"$\xi_-$")
    # Legend in top-left of right panel (get handles/labels from left panel)
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

    # Copy to paper figures (min-PTE blind only)
    if "paper_figure" in snakemake.output.keys():
        paper_path = Path(snakemake.output["paper_figure"])
        paper_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fig_path, paper_path)
        print(f"Copied to {paper_path}")

    plt.close(fig)

    # Generate plots for all blinds (same 1x2 layout)
    for plot_blind in blinds:
        cov_pure_eb_blind = cov_pure_eb_blinds[plot_blind]

        sigma_xip_total_b = _extract_sigma(cov_xi_blinds[plot_blind], 0, nbins)
        sigma_xim_total_b = _extract_sigma(cov_xi_blinds[plot_blind], 1, nbins)
        sigma_xip_E_b = _extract_sigma(cov_pure_eb_blind, 0, nbins)
        sigma_xim_E_b = _extract_sigma(cov_pure_eb_blind, 1, nbins)
        sigma_xip_B_b = _extract_sigma(cov_pure_eb_blind, 2, nbins)
        sigma_xim_B_b = _extract_sigma(cov_pure_eb_blind, 3, nbins)
        sigma_xip_amb_b = _extract_sigma(cov_pure_eb_blind, 4, nbins)
        sigma_xim_amb_b = _extract_sigma(cov_pure_eb_blind, 5, nbins)

        fig_b, axes_b = plt.subplots(1, 2, figsize=(7.24, 3.5), sharey=True)

        # Left panel: xi+ decomposition
        ax = axes_b[0]
        ax.errorbar(theta * offsets[0], theta * xip_total / scale_factor,
                    yerr=theta * sigma_xip_total_b / scale_factor,
                    fmt="o", color=color_total, markersize=ms, capsize=capsize,
                    capthick=capthick, elinewidth=elinewidth, alpha=alpha_main)
        ax.errorbar(theta * offsets[1], theta * xip_E / scale_factor,
                    yerr=theta * sigma_xip_E_b / scale_factor,
                    fmt="s", color=color_E, markersize=ms, capsize=capsize,
                    capthick=capthick, elinewidth=elinewidth, alpha=alpha_faint)
        ax.errorbar(theta * offsets[2], theta * xip_B / scale_factor,
                    yerr=theta * sigma_xip_B_b / scale_factor,
                    fmt="X", color=color_B, markersize=ms, capsize=capsize,
                    capthick=capthick, elinewidth=elinewidth, alpha=alpha_main)
        ax.errorbar(theta * offsets[3], theta * xip_amb / scale_factor,
                    yerr=theta * sigma_xip_amb_b / scale_factor,
                    fmt="v", color=color_amb, markersize=ms, capsize=capsize,
                    capthick=capthick, elinewidth=elinewidth, alpha=alpha_faint)
        shade_excluded_regions(ax, fiducial_xip_scale_cut)
        setup_panel(ax, ylabel_text=r"$\theta \xi \times 10^4$")
        ax.set_title(r"$\xi_+$")

        # Right panel: xi- decomposition
        ax = axes_b[1]
        ax.errorbar(theta * offsets[0], theta * xim_total / scale_factor,
                    yerr=theta * sigma_xim_total_b / scale_factor,
                    fmt="o", color=color_total, markersize=ms, capsize=capsize,
                    capthick=capthick, elinewidth=elinewidth, alpha=alpha_main)
        ax.errorbar(theta * offsets[1], theta * xim_E / scale_factor,
                    yerr=theta * sigma_xim_E_b / scale_factor,
                    fmt="s", color=color_E, markersize=ms, capsize=capsize,
                    capthick=capthick, elinewidth=elinewidth, alpha=alpha_faint)
        ax.errorbar(theta * offsets[2], theta * xim_B / scale_factor,
                    yerr=theta * sigma_xim_B_b / scale_factor,
                    fmt="X", color=color_B, markersize=ms, capsize=capsize,
                    capthick=capthick, elinewidth=elinewidth, alpha=alpha_main)
        ax.errorbar(theta * offsets[3], theta * xim_amb / scale_factor,
                    yerr=theta * sigma_xim_amb_b / scale_factor,
                    fmt="v", color=color_amb, markersize=ms, capsize=capsize,
                    capthick=capthick, elinewidth=elinewidth, alpha=alpha_faint)
        shade_excluded_regions(ax, fiducial_xim_scale_cut)
        setup_panel(ax)
        ax.set_title(r"$\xi_-$")
        # Get legend from main figure (has labels)
        handles, labels = axes[0].get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper left", fontsize="small")

        axes_b[0].set_ylim(ylim)
        fig_b.suptitle(f"Blind {plot_blind}", y=0.98)
        fig_b.tight_layout()

        fig_blind_path = output_dir / f"figure_blind_{plot_blind}.png"
        fig_b.savefig(fig_blind_path, dpi=300, bbox_inches="tight")
        print(f"Saved {fig_blind_path}")
        plt.close(fig_b)

    spec_paths = snakemake.input["specs"]

    # Compute minimum PTEs across blinds
    pte_xip_B_min_fid = min(blind_results[b]["fiducial"]["pte_xip_B"] for b in blinds)
    pte_xim_B_min_fid = min(blind_results[b]["fiducial"]["pte_xim_B"] for b in blinds)
    pte_joint_B_min_fid = min(blind_results[b]["fiducial"]["pte_joint_B"] for b in blinds)

    pte_xip_B_min_full = min(blind_results[b]["full"]["pte_xip_B"] for b in blinds)
    pte_xim_B_min_full = min(blind_results[b]["full"]["pte_xim_B"] for b in blinds)
    pte_joint_B_min_full = min(blind_results[b]["full"]["pte_joint_B"] for b in blinds)

    # Write evidence.json with per-blind PTEs and minima
    evidence_data = {
        "spec_id": "pure_eb_data_vector",
        "spec_path": spec_paths[0],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "fiducial": {
                "scale_cut_xip": list(fiducial_xip_scale_cut),
                "scale_cut_xim": list(fiducial_xim_scale_cut),
                # Per-blind PTEs
                **{f"pte_xip_B_{b}": blind_results[b]["fiducial"]["pte_xip_B"] for b in blinds},
                **{f"pte_xim_B_{b}": blind_results[b]["fiducial"]["pte_xim_B"] for b in blinds},
                **{f"pte_joint_{b}": blind_results[b]["fiducial"]["pte_joint_B"] for b in blinds},
                # Minimum across blinds (conservative)
                "pte_xip_B_min": pte_xip_B_min_fid,
                "pte_xim_B_min": pte_xim_B_min_fid,
                "pte_joint_min": pte_joint_B_min_fid,
                # Detailed stats for min-PTE blind
                "chi2_joint_B": blind_results[min_pte_blind]["fiducial"]["chi2_joint_B"],
                "dof_joint_B": blind_results[min_pte_blind]["fiducial"]["dof_joint_B"],
            },
            "full": {
                "scale_cut_arcmin": [float(theta.min()), float(theta.max())],
                # Per-blind PTEs
                **{f"pte_xip_B_{b}": blind_results[b]["full"]["pte_xip_B"] for b in blinds},
                **{f"pte_xim_B_{b}": blind_results[b]["full"]["pte_xim_B"] for b in blinds},
                **{f"pte_joint_{b}": blind_results[b]["full"]["pte_joint_B"] for b in blinds},
                # Minimum across blinds (conservative)
                "pte_xip_B_min": pte_xip_B_min_full,
                "pte_xim_B_min": pte_xim_B_min_full,
                "pte_joint_min": pte_joint_B_min_full,
                # Detailed stats for min-PTE blind
                "chi2_joint_B": blind_results[min_pte_blind]["full"]["chi2_joint_B"],
                "dof_joint_B": blind_results[min_pte_blind]["full"]["dof_joint_B"],
            },
            "version": version,
            "plotting_blind": min_pte_blind,
        },
        "artifacts": {
            "figure": "figure.png",
            "figure_blind_A": "figure_blind_A.png",
            "figure_blind_B": "figure_blind_B.png",
            "figure_blind_C": "figure_blind_C.png",
        },
    }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"Saved evidence to {evidence_path}")

    # Print summary
    print(f"\nPTE Summary (fiducial scale cuts):")
    blind_pte_strs = ", ".join(
        f"{b}={blind_results[b]['fiducial']['pte_xip_B']:.3f}" for b in blinds
    )
    print(f"  xi+^B: min={pte_xip_B_min_fid:.3f} ({blind_pte_strs})")
    blind_pte_strs = ", ".join(
        f"{b}={blind_results[b]['fiducial']['pte_xim_B']:.3f}" for b in blinds
    )
    print(f"  xi-^B: min={pte_xim_B_min_fid:.3f} ({blind_pte_strs})")
    blind_pte_strs = ", ".join(
        f"{b}={blind_results[b]['fiducial']['pte_joint_B']:.3f}" for b in blinds
    )
    print(f"  joint: min={pte_joint_B_min_fid:.3f} ({blind_pte_strs})")


if __name__ == "__main__":
    main()
