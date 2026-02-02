"""Harmonic-space PTE matrix figures for B-modes paper.

Produces:
- Results: single-panel fiducial Cl^BB PTE matrix
- Appendix: 3-panel composite (v1.4.5, v1.4.6, v1.4.8)

Uses fiducial blind covariance only (blind independence validated elsewhere).
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
from astropy.io import fits
from scipy import stats


plt.style.use(
    "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle"
)


def _load_snakemake():
    if hasattr(sys, "ps1"):
        from snakemake_helpers import snakemake_interactive

        return snakemake_interactive(
            "results/claims/harmonic_space_pte_matrices/evidence.json",
            str(Path.cwd()),
        )
    from snakemake.script import snakemake

    return snakemake


snakemake = _load_snakemake()


def _compute_pte(data, covariance):
    """Compute chi-squared PTE."""
    chi2 = float(data @ np.linalg.solve(covariance, data))
    dof = len(data)
    pte = stats.chi2.sf(chi2, dof)
    return pte, chi2, dof


def compute_pte_matrix(pseudo_cl_path, pseudo_cl_cov_path, fiducial_ell_min=None, fiducial_ell_max=None):
    """Compute PTE matrix for all ell-bin cut combinations.

    Parameters
    ----------
    pseudo_cl_path : str
        Path to pseudo-Cl FITS file.
    pseudo_cl_cov_path : str
        Path to pseudo-Cl covariance FITS file.
    fiducial_ell_min : float, optional
        Minimum ell for fiducial cut.
    fiducial_ell_max : float, optional
        Maximum ell for fiducial cut.

    Returns
    -------
    pte_matrix : ndarray
        n_ell x n_ell PTE matrix.
    ell : ndarray
        Ell bin centers.
    stats : dict
        Summary statistics.
    """
    # Load pseudo-Cl data
    hdu = fits.open(pseudo_cl_path)
    data = hdu["PSEUDO_CELL"].data
    hdu.close()

    ell = data["ELL"]
    cl_bb = data["BB"]
    n_ell = len(ell)

    # Load covariance
    hdu_cov = fits.open(pseudo_cl_cov_path)
    cov_bb = hdu_cov["COVAR_BB_BB"].data
    hdu_cov.close()

    # Compute PTE matrix
    pte_matrix = np.full((n_ell, n_ell), np.nan)

    for i_min in range(n_ell):
        for i_max in range(i_min + 1, n_ell):
            idx_slice = slice(i_min, i_max + 1)
            bb_slice = cl_bb[idx_slice]
            cov_slice = cov_bb[idx_slice, idx_slice]

            try:
                pte, chi2, dof = _compute_pte(bb_slice, cov_slice)
                pte_matrix[i_min, i_max] = pte
            except np.linalg.LinAlgError:
                pass

    # Compute statistics
    valid_ptes = pte_matrix[~np.isnan(pte_matrix)]
    pte_full_range = pte_matrix[0, n_ell - 1]

    stats_out = {
        "pte_at_full_range": float(pte_full_range),
        "n_evaluated": int(len(valid_ptes)),
        "ell_range": [float(ell.min()), float(ell.max())],
        "n_ell_bins": int(n_ell),
        "mean": float(np.nanmean(valid_ptes)),
        "median": float(np.nanmedian(valid_ptes)),
        "std": float(np.nanstd(valid_ptes)),
    }

    # Compute PTE at fiducial ell cuts if provided
    if fiducial_ell_min is not None and fiducial_ell_max is not None:
        i_fid_min = np.argmin(np.abs(ell - fiducial_ell_min))
        i_fid_max = np.argmin(np.abs(ell - fiducial_ell_max))
        pte_fiducial = pte_matrix[i_fid_min, i_fid_max]
        stats_out["pte_at_fiducial"] = float(pte_fiducial)
        stats_out["fiducial_ell_range"] = [fiducial_ell_min, fiducial_ell_max]

    return pte_matrix, ell, stats_out


def plot_cl_pte_panel(ax, pte_matrix, ell, title, show_colorbar=False,
                      show_xlabel=True, show_ylabel=True):
    """Plot a single Cl PTE heatmap panel.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes.
    pte_matrix : ndarray
        PTE matrix.
    ell : ndarray
        Ell bin centers.
    title : str
        Panel title.
    show_colorbar : bool
        Whether to add individual colorbar.
    show_xlabel, show_ylabel : bool
        Whether to show axis labels.

    Returns
    -------
    im : AxesImage
        The image object.
    """
    n_ell = len(ell)

    vlag_cmap = sns.color_palette("vlag", as_cmap=True).copy()
    vlag_cmap.set_bad(color="lightgray")

    # pte_matrix[i_min, i_max] -> transpose so rows=i_max, cols=i_min
    # With origin="lower", row 0 is at bottom (small i_max), row n-1 at top (large i_max)
    # This matches tick labels: y-axis has small ell at bottom, large ell at top
    pte_plot = pte_matrix.T

    im = ax.imshow(
        pte_plot, origin="lower", aspect="equal",
        cmap=vlag_cmap, vmin=0, vmax=1, extent=[0, n_ell, 0, n_ell],
    )

    # Contours - with origin="lower", both imshow and contour have same orientation
    cs = ax.contour(
        pte_plot, levels=[0.05, 0.95], colors="black",
        linewidths=0.8, extent=[0, n_ell, 0, n_ell],
    )
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")

    # Mark full ell range
    ax.add_patch(
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="black",
                  linewidth=1.5, hatch="///", alpha=0.8)
    )

    # Tick labels - more frequent markers per review feedback
    tick_step = max(1, n_ell // 8)
    tick_indices = np.arange(0, n_ell, tick_step)
    ax.set_xticks(tick_indices + 0.5)
    ax.set_yticks(tick_indices + 0.5)

    if show_xlabel:
        ax.set_xticklabels([f"{ell[i]:.0f}" for i in tick_indices],
                          rotation=45, ha="right", fontsize=6)
    else:
        ax.set_xticklabels([])

    if show_ylabel:
        ax.set_yticklabels([f"{ell[i]:.0f}" for i in tick_indices], fontsize=6)
    else:
        ax.set_yticklabels([])

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("PTE", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    return im


def create_single_panel(pte_matrix, ell, version_label):
    """Create single-panel figure for fiducial version."""
    fig, ax = plt.subplots(1, 1, figsize=(3.54, 3.54))

    plot_cl_pte_panel(ax, pte_matrix, ell, "", show_colorbar=True)

    ax.set_xlabel(r"$\ell_{\rm min}$")
    ax.set_ylabel(r"$\ell_{\rm max}$")

    plt.tight_layout()
    return fig


def create_3panel_composite(matrices, ells, version_labels):
    """Create 3-panel composite for appendix versions.

    Parameters
    ----------
    matrices : list of ndarray
        PTE matrices for each version.
    ells : list of ndarray
        Ell arrays for each version.
    version_labels : list of str
        Labels for each panel.

    Returns
    -------
    fig : Figure
        The composite figure.
    """
    fig_width = 10.5
    fig_height = 2.8

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[1, 1, 1, 0.05],
        wspace=0.08,
        left=0.06, right=0.95,
        bottom=0.15, top=0.88
    )

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    cax = fig.add_subplot(gs[3])

    im0 = plot_cl_pte_panel(ax0, matrices[0], ells[0], "",
                            show_ylabel=True)
    im1 = plot_cl_pte_panel(ax1, matrices[1], ells[1], "",
                            show_ylabel=False)
    im2 = plot_cl_pte_panel(ax2, matrices[2], ells[2], "",
                            show_ylabel=False)

    # Add version titles
    ax0.set_title(version_labels[0], fontsize=9)
    ax1.set_title(version_labels[1], fontsize=9)
    ax2.set_title(version_labels[2], fontsize=9)

    # Shared colorbar
    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label("PTE", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Common axis labels
    fig.text(0.5, 0.02, r"$\ell_{\rm min}$", ha="center", fontsize=9)
    fig.text(0.02, 0.53, r"$\ell_{\rm max}$",
             va="center", rotation="vertical", fontsize=9)

    return fig


def main():
    config = snakemake.config
    versions = config["versions"]
    fiducial_version = config["fiducial"]["version"]
    fiducial_blind = config["fiducial"]["blind"]

    # Fiducial ell cuts from config
    fiducial_ell_min = config["cl"]["fiducial_ell_min"]
    fiducial_ell_max = config["cl"]["fiducial_ell_max"]

    # Output paths
    output_dir = Path(snakemake.output["evidence"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    paper_dir = Path(snakemake.output["paper_figure_fiducial"]).parent
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Get input files
    pseudo_cl_files = snakemake.input["pseudo_cl"]
    pseudo_cl_cov_files = snakemake.input["pseudo_cl_cov"]

    if isinstance(pseudo_cl_files, str):
        pseudo_cl_files = [pseudo_cl_files]
    if isinstance(pseudo_cl_cov_files, str):
        pseudo_cl_cov_files = [pseudo_cl_cov_files]

    # Map versions to their files (sort by length descending to match longer version strings first)
    sorted_versions = sorted(versions, key=len, reverse=True)

    version_to_cl = {}
    for cl_file in pseudo_cl_files:
        for ver in sorted_versions:
            if ver in cl_file:
                version_to_cl[ver] = cl_file
                break

    version_to_cov = {}
    for cov_file in pseudo_cl_cov_files:
        for ver in sorted_versions:
            if ver in cov_file:
                version_to_cov[ver] = cov_file
                break

    # Compute PTE matrices for all versions using fiducial blind
    all_stats = {}
    all_matrices = {}
    all_ells = {}

    for version in versions:
        print(f"\n--- Processing {version} (blind={fiducial_blind}) ---")

        if version not in version_to_cl:
            print(f"  WARNING: No pseudo-Cl file found for {version}, skipping")
            continue

        if version not in version_to_cov:
            print(f"  WARNING: No covariance file found for {version}, skipping")
            continue

        pte_matrix, ell, stats = compute_pte_matrix(
            version_to_cl[version],
            version_to_cov[version],
            fiducial_ell_min=fiducial_ell_min,
            fiducial_ell_max=fiducial_ell_max,
        )

        all_stats[version] = stats
        all_matrices[version] = pte_matrix
        all_ells[version] = ell

        print(f"  PTE at fiducial: {stats.get('pte_at_fiducial', 'N/A'):.4f}")
        print(f"  PTE at full range: {stats.get('pte_at_full_range', 'N/A'):.4f}")

    # Create fiducial single-panel figure
    if fiducial_version in all_matrices:
        fig_fiducial = create_single_panel(
            all_matrices[fiducial_version],
            all_ells[fiducial_version],
            "v1.4.6 (fiducial)",
        )

        fig_path = Path(snakemake.output["figure_fiducial"])
        fig_fiducial.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"\nSaved fiducial figure: {fig_path}")

        paper_path = Path(snakemake.output["paper_figure_fiducial"])
        shutil.copy2(fig_path, paper_path)
        print(f"Copied to {paper_path}")

        plt.close(fig_fiducial)

    # Create appendix 3-panel composite (all versions)
    if len(versions) >= 3:
        # Use all three versions (v1.4.5, v1.4.6, v1.4.8)
        v145 = [v for v in versions if "v1.4.5" in v][0]
        v146 = [v for v in versions if "v1.4.6" in v][0]
        v148 = [v for v in versions if "v1.4.8" in v][0]

        matrices = [all_matrices[v145], all_matrices[v146], all_matrices[v148]]
        ells = [all_ells[v145], all_ells[v146], all_ells[v148]]
        labels = ["v1.4.5", "v1.4.6", "v1.4.8"]

        fig_appendix = create_3panel_composite(matrices, ells, labels)

        fig_path = Path(snakemake.output["figure_appendix"])
        fig_appendix.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"\nSaved appendix figure: {fig_path}")

        paper_path = Path(snakemake.output["paper_figure_appendix"])
        shutil.copy2(fig_path, paper_path)
        print(f"Copied to {paper_path}")

        plt.close(fig_appendix)

    # Build evidence
    spec_paths = snakemake.input["specs"]

    evidence_data = {
        "spec_id": "harmonic_space_pte_matrices",
        "spec_path": spec_paths[0],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "blind": fiducial_blind,
            "versions": {},
        },
        "artifacts": {
            "figure_fiducial": Path(snakemake.output["figure_fiducial"]).name,
            "figure_appendix": Path(snakemake.output["figure_appendix"]).name,
        },
    }

    for version, stats in all_stats.items():
        role = "fiducial" if version == fiducial_version else "appendix"
        evidence_data["evidence"]["versions"][version] = {
            "role": role,
            **stats,
        }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"\nSaved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
