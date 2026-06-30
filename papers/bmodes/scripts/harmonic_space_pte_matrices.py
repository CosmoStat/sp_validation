"""Harmonic-space PTE matrix figures for B-modes paper.

Produces:
- Results: single-panel fiducial Cl^BB PTE matrix
- Appendix: N-panel composite for all versions from config.versions

Uses fiducial blind covariance only (blind independence validated elsewhere).
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotting_utils import (
    FIG_WIDTH_SINGLE,
    PAPER_MPLSTYLE,
    compute_chi2_pte,
    ell_bin_index,
    format_pte_colorbar,
    make_pte_colormap,
    make_pte_norm,
)

plt.style.use(PAPER_MPLSTYLE)


def _load_snakemake():
    if hasattr(sys, "ps1"):
        from snakemake_helpers import snakemake_interactive

        return snakemake_interactive(
            "results/tapestry/harmonic_space_pte_matrices/evidence.json",
            str(Path.cwd()),
        )
    from snakemake.script import snakemake

    return snakemake


snakemake = _load_snakemake()


def compute_pte_matrix(
    pseudo_cl_path, pseudo_cl_cov_path, fiducial_ell_min=None, fiducial_ell_max=None
):
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
                chi2, pte, dof = compute_chi2_pte(bb_slice, cov_slice)
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
    # Use bin edges: include only bins fully within [ell_min, ell_max]
    if fiducial_ell_min is not None and fiducial_ell_max is not None:
        i_fid_min, i_fid_max = ell_bin_index(ell, fiducial_ell_min, fiducial_ell_max)
        pte_fiducial = pte_matrix[i_fid_min, i_fid_max]
        stats_out["pte_at_fiducial"] = float(pte_fiducial)
        stats_out["fiducial_ell_range"] = [fiducial_ell_min, fiducial_ell_max]

    return pte_matrix, ell, stats_out


def plot_cl_pte_panel(
    ax,
    pte_matrix,
    ell,
    title,
    show_colorbar=False,
    show_xlabel=True,
    show_ylabel=True,
    fid_i_min=None,
    fid_i_max=None,
):
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

    # Discrete colormap: solid blue below 0.05, solid red above 0.95, gradient between
    pte_cmap = make_pte_colormap()
    pte_norm = make_pte_norm()

    # pte_matrix[i_min, i_max] -> transpose so rows=i_max, cols=i_min
    # With origin="lower", row 0 is at bottom (small i_max), row n-1 at top (large i_max)
    # This matches tick labels: y-axis has small ell at bottom, large ell at top
    pte_plot = pte_matrix.T

    im = ax.imshow(
        pte_plot,
        origin="lower",
        aspect="equal",
        cmap=pte_cmap,
        norm=pte_norm,
        extent=[0, n_ell, 0, n_ell],
    )

    # Mark fiducial ell cut (black square)
    if fid_i_min is not None and fid_i_max is not None:
        ax.add_patch(
            Rectangle(
                (fid_i_min, fid_i_max),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=1.5,
            )
        )

    # Tick positioning: x-axis at left edge of bins, y-axis at top edge
    tick_step = max(1, n_ell // 5)  # Fewer ticks for cleaner labels
    tick_indices = np.arange(0, n_ell, tick_step)

    # x-axis (lower cut): ticks at left edge of bins
    ax.set_xticks(tick_indices)
    # y-axis (upper cut): ticks at top edge of bins
    ax.set_yticks(tick_indices + 1)

    if show_xlabel:
        ax.set_xticklabels(
            [f"{ell[i]:.0f}" for i in tick_indices],
            fontsize=9,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
    else:
        ax.set_xticklabels([])

    if show_ylabel:
        y_tick_labels = [f"{ell[min(i + 1, n_ell - 1)]:.0f}" for i in tick_indices]
        ax.set_yticklabels(y_tick_labels, fontsize=9)
    else:
        ax.set_yticklabels([])

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, spacing="proportional")
        format_pte_colorbar(cbar)
        # Drop "1" tick to avoid overlap with 0.95
        cbar.set_ticks([0.05, 0.5, 0.95])
        cbar.set_ticklabels(["0.05", "0.5", "0.95"])
        cbar.set_label("PTE", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    return im


def create_single_panel(pte_matrix, ell, fid_i_min=None, fid_i_max=None):
    """Create single-panel figure for fiducial version."""
    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH_SINGLE, FIG_WIDTH_SINGLE))

    plot_cl_pte_panel(
        ax,
        pte_matrix,
        ell,
        "",
        show_colorbar=True,
        fid_i_min=fid_i_min,
        fid_i_max=fid_i_max,
    )

    ax.set_xlabel(r"$\ell_{\mathrm{min}}$")
    ax.set_ylabel(r"$\ell_{\mathrm{max}}$")

    plt.tight_layout()
    return fig


def create_npanel_composite(matrices, ells, panel_labels, fid_indices=None):
    """Create N-panel composite for appendix versions.

    Parameters
    ----------
    matrices : list of ndarray
        PTE matrices for each version.
    ells : list of ndarray
        Ell arrays for each version.
    panel_labels : list of str
        Labels for each panel.

    Returns
    -------
    fig : Figure
        The composite figure.
    """
    n_panels = len(matrices)
    gs_left, gs_right = 0.08, 0.94
    gs_bottom, gs_top = 0.28, 0.92
    wspace_val = 0.08
    cbar_ratio = 0.04
    fig_width = 7.0

    # Compute fig_height so panels are exactly square — no vertical whitespace
    total_width_units = n_panels + cbar_ratio + n_panels * wspace_val
    panel_frac = (gs_right - gs_left) / total_width_units
    panel_in = panel_frac * fig_width
    fig_height = panel_in / (gs_top - gs_bottom)

    fig = plt.figure(figsize=(fig_width, fig_height))
    width_ratios = [1] * n_panels + [cbar_ratio]
    gs = fig.add_gridspec(
        1,
        n_panels + 1,
        width_ratios=width_ratios,
        wspace=wspace_val,
        left=gs_left,
        right=gs_right,
        bottom=gs_bottom,
        top=gs_top,
    )

    axes = [fig.add_subplot(gs[i]) for i in range(n_panels)]
    cax_placeholder = fig.add_subplot(gs[n_panels])  # position reference

    for i, (ax, matrix, ell, label) in enumerate(
        zip(axes, matrices, ells, panel_labels)
    ):
        fi = fid_indices[i] if fid_indices else (None, None)
        im = plot_cl_pte_panel(
            ax, matrix, ell, "", show_ylabel=(i == 0), fid_i_min=fi[0], fid_i_max=fi[1]
        )
        # Version label inside panel (bottom-right)
        ax.text(
            0.95,
            0.05,
            label,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    # Draw first to get actual panel positions after aspect="equal" constraint
    fig.canvas.draw()
    ax0_pos = axes[0].get_position()
    cax_pos = cax_placeholder.get_position()
    fig.delaxes(cax_placeholder)

    # Colorbar at exact rendered panel height
    cax = fig.add_axes([cax_pos.x0, ax0_pos.y0, cax_pos.width, ax0_pos.height])
    cbar = fig.colorbar(im, cax=cax, spacing="proportional")
    format_pte_colorbar(cbar)
    # Drop "1" tick to avoid overlap with 0.95
    cbar.set_ticks([0.05, 0.5, 0.95])
    cbar.set_ticklabels(["0.05", "0.5", "0.95"])
    cbar.set_label("PTE", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Common axis labels
    fig.text(
        0.5 * (gs_left + gs_right),
        0.4 * gs_bottom,
        r"$\ell_{\mathrm{min}}$",
        ha="center",
        va="center",
    )
    fig.text(
        0.25 * gs_left,
        0.5 * (gs_bottom + gs_top),
        r"$\ell_{\mathrm{max}}$",
        va="center",
        ha="center",
        rotation="vertical",
    )

    return fig


def main():
    config = snakemake.config
    versions = [v for v in config["versions"] if "_ecut" not in v]
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

    # Get version labels from params
    version_labels = snakemake.params.version_labels

    # Compute fiducial ell indices per version using bin edges
    version_fid_indices = {}
    for version, ell in all_ells.items():
        version_fid_indices[version] = ell_bin_index(
            ell, fiducial_ell_min, fiducial_ell_max
        )

    # Create fiducial single-panel figure
    if fiducial_version in all_matrices:
        fi_fid = version_fid_indices.get(fiducial_version, (None, None))
        fig_fiducial = create_single_panel(
            all_matrices[fiducial_version],
            all_ells[fiducial_version],
            fid_i_min=fi_fid[0],
            fid_i_max=fi_fid[1],
        )

        fig_path = Path(snakemake.output["figure_fiducial"])
        fig_fiducial.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"\nSaved fiducial figure: {fig_path}")

        paper_path = Path(snakemake.output["paper_figure_fiducial"])
        fig_fiducial.savefig(paper_path, bbox_inches="tight")
        print(f"Saved {paper_path}")

        plt.close(fig_fiducial)

    # Create appendix N-panel composite (paper versions only, no ecut variants)
    appendix_versions = [
        v for v in versions if v in all_matrices and v in version_labels
    ]
    if len(appendix_versions) >= 2:
        matrices = [all_matrices[v] for v in appendix_versions]
        ells = [all_ells[v] for v in appendix_versions]
        labels = [version_labels.get(v, v) for v in appendix_versions]

        fid_indices = [
            version_fid_indices.get(v, (None, None)) for v in appendix_versions
        ]
        fig_appendix = create_npanel_composite(
            matrices, ells, labels, fid_indices=fid_indices
        )

        fig_path = Path(snakemake.output["figure_appendix"])
        fig_appendix.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"\nSaved appendix figure: {fig_path}")

        paper_path = Path(snakemake.output["paper_figure_appendix"])
        fig_appendix.savefig(paper_path, bbox_inches="tight", facecolor="white")
        print(f"Saved {paper_path}")

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
        "output": {
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
