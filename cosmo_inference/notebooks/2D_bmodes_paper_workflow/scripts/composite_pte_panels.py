# %%
"""Composite PTE panels for paper figures.

Creates unified multi-panel PTE heatmaps with:
- Shared colorbar on right
- Common axis labels
- Tight panel spacing
- No redundant individual labels/colorbars
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns
from IPython import get_ipython


ipython = get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
    ipython.run_line_magic("matplotlib", "inline")

plt.style.use(
    "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle"
)


def _load_snakemake():
    if hasattr(sys, "ps1"):
        from snakemake_helpers import snakemake_interactive

        return snakemake_interactive(
            "results/claims/config_space_pte_composite/trace.auto.json",
            str(Path.cwd()),
        )
    from snakemake.script import snakemake

    return snakemake


snakemake = _load_snakemake()


def load_cosebis_pte_matrix(pte_files, nmodes=6):
    """Load COSEBIS PTE values from JSON files into matrix.

    Parameters
    ----------
    pte_files : list of str
        Paths to PTE JSON files.
    nmodes : int
        Number of COSEBIS modes (6 or 20).

    Returns
    -------
    pte_matrix : ndarray
        21x21 PTE matrix (theta indices 0-20).
    theta_grid : ndarray
        Angular scale grid (21 values).
    """
    theta_grid = np.geomspace(1.0, 250.0, 21)
    n_theta = len(theta_grid)
    pte_matrix = np.full((n_theta, n_theta), np.nan)

    for pte_file in pte_files:
        with open(pte_file) as f:
            data = json.load(f)

        i_min, i_max = data["i_min"], data["i_max"]

        # Skip single-bin cases
        if i_max - i_min < 2:
            continue

        key = f"nmodes_{nmodes}"
        if key in data:
            pte_matrix[i_min, i_max] = data[key]["pte_B"]
        elif nmodes == 6:
            # Legacy format fallback
            pte_matrix[i_min, i_max] = data["pte_B"]

    return pte_matrix, theta_grid


def plot_pte_panel(ax, pte_matrix, theta_grid, fid_start, fid_stop, title,
                   show_xlabel=False, show_ylabel=False, show_cbar=False):
    """Plot a single PTE heatmap panel.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    pte_matrix : ndarray
        PTE matrix to plot.
    theta_grid : ndarray
        Angular scale grid for tick labels.
    fid_start, fid_stop : int
        Fiducial scale cut indices for marker.
    title : str
        Panel title.
    show_xlabel, show_ylabel : bool
        Whether to show axis labels.
    show_cbar : bool
        Whether to add colorbar (for rightmost panel only).

    Returns
    -------
    im : AxesImage
        The image object (for shared colorbar).
    """
    n_theta = len(theta_grid)

    # Colormap
    vlag_cmap = sns.color_palette("vlag", as_cmap=True).copy()
    vlag_cmap.set_bad(color="lightgray")

    # Plot heatmap
    im = ax.imshow(
        pte_matrix.T,
        origin="lower",
        aspect="equal",
        cmap=vlag_cmap,
        vmin=0,
        vmax=1,
        extent=[0, n_theta, 0, n_theta],
    )

    # Contours at significance levels
    cs = ax.contour(
        pte_matrix.T,
        levels=[0.05, 0.95],
        colors="black",
        linewidths=0.6,
        extent=[0, n_theta, 0, n_theta],
    )
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")

    # Fiducial scale cut marker
    ax.add_patch(
        Rectangle(
            (fid_start, fid_stop),
            1, 1,
            fill=False,
            edgecolor="black",
            linewidth=1.2,
            hatch="///",
            alpha=0.8,
        )
    )

    # Title
    ax.set_title(title, fontsize=9)

    # Tick labels
    tick_step = 5
    tick_indices = np.arange(0, n_theta, tick_step)
    tick_labels = [f"{theta_grid[i]:.0f}" for i in tick_indices]
    tick_positions = tick_indices + 0.5

    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)

    if show_xlabel:
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=6)
    else:
        ax.set_xticklabels([])

    if show_ylabel:
        ax.set_yticklabels(tick_labels, fontsize=6)
    else:
        ax.set_yticklabels([])

    return im


def main():
    params = snakemake.params
    composite_type = params.composite_type

    trace_path = Path(snakemake.output.trace)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    output_path = Path(snakemake.output.composite_figure)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    # Pure E/B PTE matrices
    pte_data = np.load(snakemake.input.pure_eb_pte)
    theta_pure_eb = pte_data["theta"]
    pte_xip_B = pte_data["pte_xip_B"]
    pte_xim_B = pte_data["pte_xim_B"]
    n_theta_pure_eb = len(theta_pure_eb)

    # COSEBIS PTE matrix
    pte_cosebis, theta_cosebis = load_cosebis_pte_matrix(
        snakemake.input.cosebis_pte_files, nmodes=6
    )
    n_theta_cosebis = len(theta_cosebis)

    # Get fiducial scale cut indices
    # COSEBIS uses [6, 85] arcmin
    fid_min = params.fiducial_min_scale
    fid_max = params.fiducial_max_scale
    cosebis_fid_start = np.argmin(np.abs(theta_cosebis[:-1] - fid_min))
    cosebis_fid_stop = np.argmin(np.abs(theta_cosebis[1:] - fid_max)) + 1

    # Pure E/B uses separate scale cuts
    xip_cut = tuple(params.fiducial_xip_scale_cut)
    xim_cut = tuple(params.fiducial_xim_scale_cut)
    xip_start = np.argmin(np.abs(theta_pure_eb - xip_cut[0]))
    xip_stop = np.argmin(np.abs(theta_pure_eb - xip_cut[1]))
    xim_start = np.argmin(np.abs(theta_pure_eb - xim_cut[0]))
    xim_stop = np.argmin(np.abs(theta_pure_eb - xim_cut[1]))

    # Create figure: 3 panels with shared colorbar
    # Panel width 2.2", colorbar 0.3", total ~7.2" (full page width)
    fig_width = 7.24
    fig_height = 2.4

    # GridSpec for tight layout with colorbar
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[1, 1, 1, 0.05],
        wspace=0.08,
        left=0.08, right=0.92,
        bottom=0.18, top=0.88
    )

    ax_cosebis = fig.add_subplot(gs[0])
    ax_xip = fig.add_subplot(gs[1])
    ax_xim = fig.add_subplot(gs[2])
    cax = fig.add_subplot(gs[3])

    # Plot panels
    im0 = plot_pte_panel(
        ax_cosebis, pte_cosebis, theta_cosebis,
        cosebis_fid_start, cosebis_fid_stop,
        r"COSEBIS $B_n$",
        show_xlabel=True, show_ylabel=True
    )

    im1 = plot_pte_panel(
        ax_xip, pte_xip_B, theta_pure_eb,
        xip_start, xip_stop,
        r"$\xi_+^B$",
        show_xlabel=True, show_ylabel=False
    )

    im2 = plot_pte_panel(
        ax_xim, pte_xim_B, theta_pure_eb,
        xim_start, xim_stop,
        r"$\xi_-^B$",
        show_xlabel=True, show_ylabel=False
    )

    # Shared colorbar
    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label("PTE", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Common axis labels
    fig.text(0.5, 0.02, r"Lower scale cut $\theta_{\min}$ [arcmin]",
             ha="center", fontsize=9)
    fig.text(0.02, 0.53, r"Upper scale cut $\theta_{\max}$ [arcmin]",
             va="center", rotation="vertical", fontsize=9)

    # Save
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved composite to {output_path}")
    plt.close(fig)

    # Copy to paper figures directory
    paper_fig_path = Path(snakemake.output.paper_figure)
    paper_fig_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_path, paper_fig_path)
    print(f"Copied to paper figures: {paper_fig_path}")

    # Compute statistics for trace
    def compute_stats(pte_matrix, fid_start, fid_stop):
        valid = pte_matrix[~np.isnan(pte_matrix)]
        if len(valid) == 0:
            return {"pte_at_fiducial": float("nan"), "fraction_healthy": float("nan")}
        return {
            "pte_at_fiducial": float(pte_matrix[fid_start, fid_stop]),
            "fraction_healthy": float(np.sum((valid >= 0.05) & (valid <= 0.95)) / len(valid)),
            "n_evaluated": int(len(valid)),
        }

    cosebis_stats = compute_stats(pte_cosebis, cosebis_fid_start, cosebis_fid_stop)
    xip_stats = compute_stats(pte_xip_B, xip_start, xip_stop)
    xim_stats = compute_stats(pte_xim_B, xim_start, xim_stop)

    # Write trace
    trace = {
        "claim": params.claim,
        "implementation_plan": params.implementation_plan,
        "evidence": {
            "composite_type": composite_type,
            "panels": ["COSEBIS Bn", "xi+^B", "xi-^B"],
            "cosebis_stats": cosebis_stats,
            "xip_stats": xip_stats,
            "xim_stats": xim_stats,
            "fiducial_scale_cuts_arcmin": {
                "cosebis": [fid_min, fid_max],
                "xip": list(xip_cut),
                "xim": list(xim_cut),
            },
            "creation_timestamp": datetime.now().isoformat(),
        },
        "artifact_paths": {
            "composite_figure": str(output_path),
        },
        "parameters": {
            "kind": params.kind,
            "composite_type": composite_type,
            "version": params.version,
        },
        "depends_on": params.depends_on,
        "ai_review": None,
    }

    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"Saved trace to {trace_path}")


if __name__ == "__main__":
    main()
