"""Configuration-space PTE matrix composites for B-modes paper.

Produces 3-panel composites (xi+^B, xi-^B, COSEBIS) for:
- Results: fiducial version (from config.fiducial.version)
- Appendix: all versions (from config.versions with labels from config.plotting.version_labels)

Each composite has shared axes and a single colorbar.

Dual-mode. Under Snakemake (``script:`` directive) the injected ``snakemake``
object supplies config/inputs/outputs; as a standalone CLI (argparse) the same
compute runs from explicit flags — the version sweep lives *inside* this script,
so the CLI reads the per-version PTE intermediates (pure E/B ``*_pure_eb_ptes.npz``
and the COSEBI ``pte_*.json`` scatter tree) directly from absolute directories
and writes every product (evidence.json + PNG/PDF composites) under ``--out``.
This is the form the lightcone/ASTRA recipe calls:

    python config_space_pte_matrices.py \
        --config /path/to/config.yaml \
        --pte-intermediate-dir /abs/.../paper_plots/intermediate \
        --cosebis-pte-dir /abs/.../tapestry/cosebis_pte_matrix/pte_values \
        --blind A --out <output_dir>
"""

import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from plotting_utils import (
    PAPER_MPLSTYLE,
    format_pte_colorbar,
    make_pte_colormap,
    make_pte_norm,
)

plt.style.use(PAPER_MPLSTYLE)


def _path_matches_version(path, version):
    """Check if a file path matches a specific catalog version exactly.

    Avoids substring false positives (e.g. 'SP_v1.4.5' matching
    'SP_v1.4.5_leak_corr' paths) by rejecting cases where the path
    actually contains the leak_corr variant of an uncorrected version.
    """
    path_str = str(path)
    if version not in path_str:
        return False
    # For uncorrected versions, reject if the path contains the leak_corr variant
    if not version.endswith("_leak_corr") and (version + "_leak_corr") in path_str:
        return False
    return True


def _resolve_overrides(version, fiducial_overrides):
    """Fiducial-provenance source paths for this version, else ``(None, None)``.

    Only the fiducial version is repointed to explicit lc outputs; every other
    version keeps reading the old-tree file lists. Either element may be None.
    """
    if fiducial_overrides and version == fiducial_overrides["version"]:
        return fiducial_overrides["pure_eb"], fiducial_overrides["cosebis"]
    return None, None


def _cosebis_matrix_from_npz(npz_path, nmodes):
    """Adapt lc's gathered COSEBI PTE NPZ to the per-pair-JSON PTE matrix.

    The lc producer (compute_cosebis_pte_single.py) writes one NPZ holding
    ``pte_B_6``/``pte_B_20`` as (nbins+1, nbins+1) matrices already indexed by
    ``[i_min, i_max]`` — the same layout ``load_cosebis_pte_matrix`` builds cell
    by cell from the JSON glob. The one structural difference is that the NPZ
    also fills single-bin cuts (``i_max - i_min == 1``), which the JSON loader
    skips; re-imposing that mask makes the fiducial column numerically identical
    whichever provenance feeds it.
    """
    npz = np.load(npz_path)
    theta_grid = npz["theta_grid"]
    pte_matrix = npz[f"pte_B_{nmodes}"].copy()
    i_idx, j_idx = np.indices(pte_matrix.shape)
    pte_matrix[(j_idx - i_idx) < 2] = np.nan
    return pte_matrix, theta_grid


def load_cosebis_pte_matrix(
    pte_files, version, config, nmodes=6, cosebis_npz_override=None
):
    """Load COSEBIS PTE values from JSON files into matrix.

    Uses fiducial blind only (data vectors identical across blinds).

    Parameters
    ----------
    pte_files : list of str
        Paths to PTE JSON files for fiducial blind.
    version : str
        Version to filter for.
    config : dict
        Workflow config with fiducial.min_sep, max_sep, nbins.
    nmodes : int
        Number of COSEBIS modes (6 or 20).

    Returns
    -------
    pte_matrix : ndarray
        (nbins+1)x(nbins+1) PTE matrix (theta indices).
    theta_grid : ndarray
        Angular scale grid (nbins+1 values).
    """
    if cosebis_npz_override is not None:
        return _cosebis_matrix_from_npz(cosebis_npz_override, nmodes)

    fid = config["fiducial"]
    theta_grid = np.geomspace(fid["min_sep"], fid["max_sep"], fid["nbins"] + 1)
    n_theta = len(theta_grid)
    # Initialize with nan for cells without data
    pte_matrix = np.full((n_theta, n_theta), np.nan)

    for pte_file in pte_files:
        # Filter to this version (exact match, no substring false positives)
        if not _path_matches_version(pte_file, version):
            continue

        with open(pte_file) as f:
            data = json.load(f)

        i_min, i_max = data["i_min"], data["i_max"]

        # Skip single-bin cases
        if i_max - i_min < 2:
            continue

        key = f"nmodes_{nmodes}"
        if key in data:
            pte_val = data[key]["pte_B"]
        elif nmodes == 6:
            # Legacy format fallback
            pte_val = data.get("pte_B", np.nan)
        else:
            continue

        # Store PTE value
        if not np.isnan(pte_val):
            pte_matrix[i_min, i_max] = pte_val

    return pte_matrix, theta_grid


def load_pure_eb_pte_matrices(pte_files, version, override_path=None):
    """Load Pure E/B PTE matrices from npz files.

    Uses fiducial blind only (data vectors identical across blinds).

    Parameters
    ----------
    pte_files : list of str
        Paths to pure_eb_ptes.npz files for fiducial blind.
    version : str
        Version to filter for.

    Returns
    -------
    pte_xip_B : ndarray
        PTE matrix for xi+^B.
    pte_xim_B : ndarray
        PTE matrix for xi-^B.
    theta : ndarray
        Angular scale grid.
    pte_combined : ndarray or None
        PTE matrix for combined ξ_tot^B, or None if not available.
    """
    if override_path is not None:
        data = np.load(override_path)
        pte_combined = data["pte_combined"] if "pte_combined" in data else None
        return data["pte_xip_B"], data["pte_xim_B"], data["theta"], pte_combined

    for pte_file in pte_files:
        # Filter to this version (exact match, no substring false positives)
        if not _path_matches_version(pte_file, version):
            continue

        data = np.load(pte_file)
        pte_combined = data["pte_combined"] if "pte_combined" in data else None
        return data["pte_xip_B"], data["pte_xim_B"], data["theta"], pte_combined

    raise ValueError(f"No PTE file found for version {version}")


def _load_version_pte_data(
    pure_eb_pte_files, cosebis_pte_files, version, config, fiducial_overrides=None
):
    """Load all PTE matrices for a single version.

    Returns
    -------
    dict with keys: pte_xip_B, pte_xim_B, pte_combined (or None),
        pte_cosebis, pte_cosebis_20, theta_pure_eb, theta_cosebis.
    """
    pure_eb_override, cosebis_override = _resolve_overrides(version, fiducial_overrides)
    pte_xip_B, pte_xim_B, theta_pure_eb, pte_combined = load_pure_eb_pte_matrices(
        pure_eb_pte_files, version, override_path=pure_eb_override
    )
    pte_cosebis, theta_cosebis = load_cosebis_pte_matrix(
        cosebis_pte_files,
        version,
        config,
        nmodes=6,
        cosebis_npz_override=cosebis_override,
    )
    pte_cosebis_20, _ = load_cosebis_pte_matrix(
        cosebis_pte_files,
        version,
        config,
        nmodes=20,
        cosebis_npz_override=cosebis_override,
    )
    return {
        "pte_xip_B": pte_xip_B,
        "pte_xim_B": pte_xim_B,
        "pte_combined": pte_combined,
        "pte_cosebis": pte_cosebis,
        "pte_cosebis_20": pte_cosebis_20,
        "theta_pure_eb": theta_pure_eb,
        "theta_cosebis": theta_cosebis,
    }


def plot_pte_panel(
    ax,
    pte_matrix,
    theta_grid,
    fid_start,
    fid_stop,
    title,
    show_xticklabels=True,
    show_yticklabels=False,
):
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
    show_xticklabels, show_yticklabels : bool
        Whether to show tick labels on each axis.

    Returns
    -------
    im : AxesImage
        The image object (for shared colorbar).
    """
    n_theta = len(theta_grid)

    # Discrete colormap: solid blue below 0.05, solid red above 0.95, gradient between
    pte_cmap = make_pte_colormap()
    pte_norm = make_pte_norm()

    # Plot heatmap (no contours - colormap provides visual threshold distinction)
    im = ax.imshow(
        pte_matrix.T,
        origin="lower",
        aspect="equal",
        cmap=pte_cmap,
        norm=pte_norm,
        extent=[0, n_theta, 0, n_theta],
    )

    # Fiducial scale cut marker (black square, no hatching for cleaner look)
    ax.add_patch(
        Rectangle(
            (fid_start, fid_stop),
            1,
            1,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
        )
    )

    # Title inside plot (bottom-right of empty upper triangle)
    if title:
        ax.text(
            0.95,
            0.05,
            title,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    # Tick positioning: x-axis at left edge of bins, y-axis at top edge
    tick_step = 2
    tick_indices = np.arange(0, n_theta, tick_step)
    x_tick_labels = [f"{theta_grid[i]:.0f}" for i in tick_indices]
    y_tick_labels = [f"{theta_grid[min(i + 1, n_theta - 1)]:.0f}" for i in tick_indices]

    # x-axis (lower cut): ticks at left edge of bins
    ax.set_xticks(tick_indices)
    # y-axis (upper cut): ticks at top edge of bins
    ax.set_yticks(tick_indices + 1)

    if show_xticklabels:
        ax.set_xticklabels(
            x_tick_labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=7
        )
    else:
        ax.set_xticklabels([])

    if show_yticklabels:
        ax.set_yticklabels(y_tick_labels, fontsize=7)
    else:
        ax.set_yticklabels([])

    return im


def compute_stats(pte_matrix, fid_start, fid_stop):
    """Compute statistics for a PTE matrix."""
    valid = pte_matrix[~np.isnan(pte_matrix)]
    if len(valid) == 0:
        return {"pte_at_fiducial": float("nan")}
    return {
        "pte_at_fiducial": float(pte_matrix[fid_start, fid_stop]),
        "n_evaluated": int(len(valid)),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "std": float(np.std(valid)),
    }


def extract_full_range_ptes(
    pure_eb_pte_files, cosebis_pte_files, version, fiducial_overrides=None
):
    """Extract full-range PTEs from npz and JSON files.

    Takes minimum PTE across blinds for each statistic.

    Parameters
    ----------
    pure_eb_pte_files : list
        All Pure E/B PTE npz files (includes all blinds).
    cosebis_pte_files : list
        All COSEBIS PTE JSON files.
    version : str
        Catalog version string.

    Returns
    -------
    ptes : dict
        Full-range PTEs for xip, xim, and cosebis (fiducial blind).
    """
    pure_eb_override, cosebis_override = _resolve_overrides(version, fiducial_overrides)

    # Get full-range PTEs from pure E/B (fiducial blind)
    xip_ptes = []
    xim_ptes = []
    combined_ptes = []
    pure_eb_sources = (
        [pure_eb_override]
        if pure_eb_override
        else [p for p in pure_eb_pte_files if _path_matches_version(p, version)]
    )
    for pte_file in pure_eb_sources:
        pte_data = np.load(pte_file)
        theta = pte_data["theta"]
        full_range_idx = (0, len(theta) - 1)
        xip_pte = pte_data["pte_xip_B"][full_range_idx]
        xim_pte = pte_data["pte_xim_B"][full_range_idx]
        if not np.isnan(xip_pte):
            xip_ptes.append(xip_pte)
        if not np.isnan(xim_pte):
            xim_ptes.append(xim_pte)
        if "pte_combined" in pte_data:
            combined_pte = pte_data["pte_combined"][full_range_idx]
            if not np.isnan(combined_pte):
                combined_ptes.append(combined_pte)

    ptes = {
        "xip": float(min(xip_ptes)) if xip_ptes else float("nan"),
        "xim": float(min(xim_ptes)) if xim_ptes else float("nan"),
        "combined": float(min(combined_ptes)) if combined_ptes else float("nan"),
        "cosebis": float("nan"),  # Default if file not found
        "cosebis_20": float("nan"),  # Default for n=20
    }

    # Load COSEBIS full-range PTE (pair (0, 20) — the full theta range).
    if cosebis_override is not None:
        npz = np.load(cosebis_override)
        pte6, pte20 = float(npz["pte_B_6"][0, 20]), float(npz["pte_B_20"][0, 20])
        if not np.isnan(pte6):
            ptes["cosebis"] = pte6
        if not np.isnan(pte20):
            ptes["cosebis_20"] = pte20
        return ptes

    # Old tree: pte_000_020.json for full theta range, min across blinds.
    cosebis_ptes_6 = []
    cosebis_ptes_20 = []
    for pte_file in cosebis_pte_files:
        if _path_matches_version(pte_file, version) and "pte_000_020" in str(pte_file):
            try:
                with open(pte_file) as f:
                    data = json.load(f)
                # n=6 PTE
                pte_val = data.get("nmodes_6", {}).get("pte_B")
                if pte_val is None:
                    pte_val = data.get("pte_B")
                if pte_val is not None and not np.isnan(pte_val):
                    cosebis_ptes_6.append(pte_val)
                # n=20 PTE
                pte_20 = data.get("nmodes_20", {}).get("pte_B")
                if pte_20 is not None and not np.isnan(pte_20):
                    cosebis_ptes_20.append(pte_20)
            except FileNotFoundError:
                continue

    if cosebis_ptes_6:
        ptes["cosebis"] = float(min(cosebis_ptes_6))
    if cosebis_ptes_20:
        ptes["cosebis_20"] = float(min(cosebis_ptes_20))

    return ptes


def create_3panel_composite(
    version,
    pure_eb_pte_files,
    cosebis_pte_files,
    xip_fid,
    xim_fid,
    cosebis_fid,
    config,
    fiducial_overrides=None,
):
    """Create a 1×3 composite figure for the fiducial version (main text).

    Parameters
    ----------
    version : str
        Catalog version string (fiducial).
    pure_eb_pte_files : list
        All Pure E/B PTE npz files (includes all blinds).
    cosebis_pte_files : list
        All COSEBIS PTE JSON files.
    xip_fid, xim_fid : tuple
        Fiducial scale cuts for xi+ and xi- (arcmin).
    cosebis_fid : tuple
        Fiducial scale cuts for COSEBIS (arcmin).
    config : dict
        Workflow config with fiducial parameters.

    Returns
    -------
    fig : Figure
        The composite figure.
    stats : dict
        Statistics for this version.
    full_range_ptes : dict
        Full-range PTEs for this version.
    """
    # Create figure: 1 row × 4 columns (3 stats + colorbar)
    fig_width = 6.5
    fig_height = 2.6

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1, 1, 1],
        wspace=0.03,
        hspace=0.03,
        left=0.08,
        right=0.92,
        bottom=0.15,
        top=0.90,
    )

    # Load all PTE data for this version
    matrices = _load_version_pte_data(
        pure_eb_pte_files, cosebis_pte_files, version, config, fiducial_overrides
    )
    pte_xip_B = matrices["pte_xip_B"]
    pte_xim_B = matrices["pte_xim_B"]
    pte_combined = matrices["pte_combined"]
    pte_cosebis = matrices["pte_cosebis"]
    pte_cosebis_20 = matrices["pte_cosebis_20"]
    theta_pure_eb = matrices["theta_pure_eb"]
    theta_cosebis = matrices["theta_cosebis"]

    # Compute fiducial indices
    cosebis_fid_start = np.argmin(np.abs(theta_cosebis[:-1] - cosebis_fid[0]))
    cosebis_fid_stop = np.argmin(np.abs(theta_cosebis[1:] - cosebis_fid[1])) + 1

    xip_start = np.argmin(np.abs(theta_pure_eb - xip_fid[0]))
    xip_stop = np.argmin(np.abs(theta_pure_eb - xip_fid[1]))
    xim_start = np.argmin(np.abs(theta_pure_eb - xim_fid[0]))
    xim_stop = np.argmin(np.abs(theta_pure_eb - xim_fid[1]))

    # Create subplot axes
    ax_xip = fig.add_subplot(gs[0, 0])
    ax_xim = fig.add_subplot(gs[0, 1])
    ax_cosebis = fig.add_subplot(gs[0, 2])

    # Plot panels with titles
    plot_pte_panel(
        ax_xip,
        pte_xip_B,
        theta_pure_eb,
        xip_start,
        xip_stop,
        r"$\xi_+^{\mathrm{B}}$",
        show_xticklabels=True,
        show_yticklabels=True,
    )
    ax_xip.set_ylabel(r"$\theta_{\max}$ [arcmin]", labelpad=2)

    plot_pte_panel(
        ax_xim,
        pte_xim_B,
        theta_pure_eb,
        xim_start,
        xim_stop,
        r"$\xi_-^{\mathrm{B}}$",
        show_xticklabels=True,
        show_yticklabels=False,
    )

    im_cosebis = plot_pte_panel(
        ax_cosebis,
        pte_cosebis,
        theta_cosebis,
        cosebis_fid_start,
        cosebis_fid_stop,
        r"COSEBIS $B_n$",
        show_xticklabels=True,
        show_yticklabels=False,
    )

    # Colorbar at exact height of rendered panels (accounts for aspect="equal")
    fig.canvas.draw()
    ax_pos = ax_cosebis.get_position()
    cax = fig.add_axes([ax_pos.x1 + 0.01, ax_pos.y0, 0.02, ax_pos.height])
    cbar = fig.colorbar(im_cosebis, cax=cax, spacing="proportional")
    format_pte_colorbar(cbar)
    cbar.set_label("PTE", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    # Common x-axis label
    fig.text(0.50, 0.02, r"$\theta_{\min}$ [arcmin]", ha="center")

    # Compute statistics
    stats = {
        "xip": compute_stats(pte_xip_B, xip_start, xip_stop),
        "xim": compute_stats(pte_xim_B, xim_start, xim_stop),
        "combined": compute_stats(pte_combined, xip_start, xip_stop)
        if pte_combined is not None
        else {"pte_at_fiducial": float("nan")},
        "cosebis": compute_stats(pte_cosebis, cosebis_fid_start, cosebis_fid_stop),
        "cosebis_20": compute_stats(
            pte_cosebis_20, cosebis_fid_start, cosebis_fid_stop
        ),
    }

    # Extract full-range PTEs
    full_range_ptes = extract_full_range_ptes(
        pure_eb_pte_files, cosebis_pte_files, version, fiducial_overrides
    )

    return fig, stats, full_range_ptes


def create_9panel_composite(
    versions,
    pure_eb_pte_files,
    cosebis_pte_files,
    xip_fid,
    xim_fid,
    cosebis_fid,
    version_labels,
    config,
    fiducial_overrides=None,
):
    """Create a Nx3 composite figure for all versions (appendix).

    Parameters
    ----------
    versions : list of str
        Catalog version strings in display order.
    pure_eb_pte_files : list
        All Pure E/B PTE npz files (includes all blinds).
    cosebis_pte_files : list
        All COSEBIS PTE JSON files.
    xip_fid, xim_fid : tuple
        Fiducial scale cuts for xi+ and xi- (arcmin).
    cosebis_fid : tuple
        Fiducial scale cuts for COSEBIS (arcmin).
    version_labels : dict
        Mapping from version string to display label (from config.plotting.version_labels).
    config : dict
        Workflow config with fiducial parameters.

    Returns
    -------
    fig : Figure
        The composite figure.
    all_stats : dict
        Statistics keyed by version.
    all_full_range_ptes : dict
        Full-range PTEs keyed by version.
    """
    n_versions = len(versions)
    fig_width = 6.5
    fig_height = 2.2 * n_versions * 0.90

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        n_versions,
        3,
        width_ratios=[1, 1, 1],
        wspace=0.03,
        hspace=0.04,
        left=0.10,
        right=0.85,
        bottom=0.08,
        top=0.94,
    )

    all_stats = {}
    all_full_range_ptes = {}
    cosebis_axes = []

    for row_idx, version in enumerate(versions):
        # Load all PTE data for this version
        matrices = _load_version_pte_data(
            pure_eb_pte_files, cosebis_pte_files, version, config, fiducial_overrides
        )
        pte_xip_B = matrices["pte_xip_B"]
        pte_xim_B = matrices["pte_xim_B"]
        pte_combined = matrices["pte_combined"]
        pte_cosebis = matrices["pte_cosebis"]
        pte_cosebis_20 = matrices["pte_cosebis_20"]
        theta_pure_eb = matrices["theta_pure_eb"]
        theta_cosebis = matrices["theta_cosebis"]

        # Compute fiducial indices
        cosebis_fid_start = np.argmin(np.abs(theta_cosebis[:-1] - cosebis_fid[0]))
        cosebis_fid_stop = np.argmin(np.abs(theta_cosebis[1:] - cosebis_fid[1])) + 1

        xip_start = np.argmin(np.abs(theta_pure_eb - xip_fid[0]))
        xip_stop = np.argmin(np.abs(theta_pure_eb - xip_fid[1]))
        xim_start = np.argmin(np.abs(theta_pure_eb - xim_fid[0]))
        xim_stop = np.argmin(np.abs(theta_pure_eb - xim_fid[1]))

        # Create subplot axes for this row
        ax_xip = fig.add_subplot(gs[row_idx, 0])
        ax_xim = fig.add_subplot(gs[row_idx, 1])
        ax_cosebis = fig.add_subplot(gs[row_idx, 2])

        # Y-axis labels on each row (leftmost column only)
        # X-axis labels only on bottom row
        show_yticklabels = True
        show_xticklabels = row_idx == n_versions - 1

        # Plot titles: version label + stat name on every row's panels
        version_label = version_labels.get(version, version)
        xip_title = rf"{version_label}: $\xi_+^{{\mathrm{{B}}}}$"
        xim_title = rf"{version_label}: $\xi_-^{{\mathrm{{B}}}}$"
        cosebis_title = rf"{version_label}: $B_n$"

        # Plot panels
        plot_pte_panel(
            ax_xip,
            pte_xip_B,
            theta_pure_eb,
            xip_start,
            xip_stop,
            xip_title,
            show_xticklabels=show_xticklabels,
            show_yticklabels=show_yticklabels,
        )

        plot_pte_panel(
            ax_xim,
            pte_xim_B,
            theta_pure_eb,
            xim_start,
            xim_stop,
            xim_title,
            show_xticklabels=show_xticklabels,
            show_yticklabels=False,
        )

        im_cosebis = plot_pte_panel(
            ax_cosebis,
            pte_cosebis,
            theta_cosebis,
            cosebis_fid_start,
            cosebis_fid_stop,
            cosebis_title,
            show_xticklabels=show_xticklabels,
            show_yticklabels=False,
        )
        cosebis_axes.append(ax_cosebis)

        # Compute statistics
        stats = {
            "xip": compute_stats(pte_xip_B, xip_start, xip_stop),
            "xim": compute_stats(pte_xim_B, xim_start, xim_stop),
            "combined": compute_stats(pte_combined, xip_start, xip_stop)
            if pte_combined is not None
            else {"pte_at_fiducial": float("nan")},
            "cosebis": compute_stats(pte_cosebis, cosebis_fid_start, cosebis_fid_stop),
            "cosebis_20": compute_stats(
                pte_cosebis_20, cosebis_fid_start, cosebis_fid_stop
            ),
        }
        all_stats[version] = stats

        # Extract full-range PTEs
        full_range_ptes = extract_full_range_ptes(
            pure_eb_pte_files, cosebis_pte_files, version, fiducial_overrides
        )
        all_full_range_ptes[version] = full_range_ptes

    # Colorbar: use actual rendered positions after aspect="equal" constraint
    fig.canvas.draw()
    first_ax = fig.axes[0]
    last_ax = fig.axes[(n_versions - 1) * 3]
    y_top = first_ax.get_position().y1
    y_bot = last_ax.get_position().y0
    cbar_height = (y_top - y_bot) * 1.005
    cbar_y0 = y_bot - (y_top - y_bot) * 0.0025
    cax = fig.add_axes([0.87, cbar_y0, 0.0225, cbar_height])
    cbar = fig.colorbar(im_cosebis, cax=cax, spacing="proportional")
    format_pte_colorbar(cbar)
    cbar.set_label("PTE", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    # Common axis labels
    fig.text(0.50, 0.02, r"$\theta_{\min}$ [arcmin]", ha="center")
    fig.text(0.02, 0.51, r"$\theta_{\max}$ [arcmin]", va="center", rotation="vertical")

    return fig, all_stats, all_full_range_ptes


def main(
    config,
    pure_eb_pte_files,
    cosebis_pte_files,
    output_dir,
    spec_path=None,
    fiducial_overrides=None,
):
    # Both corrected and uncorrected versions (exclude ecut variants)
    versions = [v for v in config["versions"] if "_ecut" not in v]
    fiducial_version = config["fiducial"]["version"]

    # Scale cuts from config
    xip_fid = tuple(config["fiducial"]["fiducial_xip_scale_cut"])
    xim_fid = tuple(config["fiducial"]["fiducial_xim_scale_cut"])
    cosebis_fid = (
        config["fiducial"]["fiducial_min_scale"],
        config["fiducial"]["fiducial_max_scale"],
    )

    # Output paths — every product lands under a single output directory.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "evidence": output_dir / "evidence.json",
        "figure_fiducial": output_dir / "figure_fiducial.png",
        "figure_appendix": output_dir / "figure_appendix.png",
        "paper_figure_fiducial": output_dir / "config_space_pte_fiducial.pdf",
        "paper_figure_appendix": output_dir / "config_space_pte_composite_appendix.pdf",
    }

    all_stats = {}
    all_full_range_ptes = {}

    # Generate 1x3 composite for fiducial version (main text)
    print("\n--- Creating 1x3 composite for fiducial version ---", flush=True)

    try:
        fig_fid, fid_stats, fid_ptes = create_3panel_composite(
            version=fiducial_version,
            pure_eb_pte_files=pure_eb_pte_files,
            cosebis_pte_files=cosebis_pte_files,
            xip_fid=xip_fid,
            xim_fid=xim_fid,
            cosebis_fid=cosebis_fid,
            config=config,
            fiducial_overrides=fiducial_overrides,
        )

        # Save fiducial figure
        fig_fid_path = out["figure_fiducial"]
        fig_fid.savefig(fig_fid_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved {fig_fid_path}", flush=True)

        paper_fid_path = out["paper_figure_fiducial"]
        fig_fid.savefig(paper_fid_path, bbox_inches="tight", facecolor="white")
        print(f"  Saved {paper_fid_path}", flush=True)

        plt.close(fig_fid)

        all_stats[fiducial_version] = fid_stats
        all_full_range_ptes[fiducial_version] = fid_ptes

    except Exception:
        import traceback

        print("  ERROR creating fiducial composite:", flush=True)
        traceback.print_exc()
        raise

    # Generate Nx3 composite for all versions (appendix)
    print("\n--- Creating Nx3 composite for all versions ---", flush=True)

    try:
        # Only include paper versions (exclude ecut variants)
        paper_version_labels = config["plotting"]["version_labels"]
        paper_versions = [v for v in versions if v in paper_version_labels]
        fig, appendix_stats, appendix_ptes = create_9panel_composite(
            versions=paper_versions,
            pure_eb_pte_files=pure_eb_pte_files,
            cosebis_pte_files=cosebis_pte_files,
            xip_fid=xip_fid,
            xim_fid=xim_fid,
            cosebis_fid=cosebis_fid,
            version_labels=config["plotting"]["version_labels"],
            config=config,
            fiducial_overrides=fiducial_overrides,
        )

        # Save appendix figure
        fig_path = out["figure_appendix"]
        fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved {fig_path}", flush=True)

        paper_path = out["paper_figure_appendix"]
        fig.savefig(paper_path, bbox_inches="tight", facecolor="white")
        print(f"  Saved {paper_path}", flush=True)

        plt.close(fig)

        # Update stats with all versions from appendix (overwrites fiducial with same data)
        all_stats.update(appendix_stats)
        all_full_range_ptes.update(appendix_ptes)

    except Exception:
        import traceback

        print("  ERROR creating appendix composite:", flush=True)
        traceback.print_exc()
        raise

    # Compute PTEs for uncorrected versions (no figures, evidence only)
    paper_version_labels = config["plotting"]["version_labels"]
    uncorrected_versions = [
        v for v in versions if v not in paper_version_labels and v != fiducial_version
    ]
    if uncorrected_versions:
        print(
            f"\n--- Computing PTEs for uncorrected versions: {uncorrected_versions} ---",
            flush=True,
        )
        for version in uncorrected_versions:
            try:
                matrices = _load_version_pte_data(
                    pure_eb_pte_files,
                    cosebis_pte_files,
                    version,
                    config,
                    fiducial_overrides,
                )
                theta_pe = matrices["theta_pure_eb"]
                theta_co = matrices["theta_cosebis"]
                xip_start = np.argmin(np.abs(theta_pe - xip_fid[0]))
                xip_stop = np.argmin(np.abs(theta_pe - xip_fid[1]))
                xim_start = np.argmin(np.abs(theta_pe - xim_fid[0]))
                xim_stop = np.argmin(np.abs(theta_pe - xim_fid[1]))
                cos_start = np.argmin(np.abs(theta_co[:-1] - cosebis_fid[0]))
                cos_stop = np.argmin(np.abs(theta_co[1:] - cosebis_fid[1])) + 1

                stats = {
                    "xip": compute_stats(matrices["pte_xip_B"], xip_start, xip_stop),
                    "xim": compute_stats(matrices["pte_xim_B"], xim_start, xim_stop),
                    "combined": compute_stats(
                        matrices["pte_combined"], xip_start, xip_stop
                    )
                    if matrices["pte_combined"] is not None
                    else {"pte_at_fiducial": float("nan")},
                    "cosebis": compute_stats(
                        matrices["pte_cosebis"], cos_start, cos_stop
                    ),
                    "cosebis_20": compute_stats(
                        matrices["pte_cosebis_20"], cos_start, cos_stop
                    ),
                }
                full_range_ptes = extract_full_range_ptes(
                    pure_eb_pte_files, cosebis_pte_files, version, fiducial_overrides
                )

                all_stats[version] = stats
                all_full_range_ptes[version] = full_range_ptes
                print(
                    f"  {version}: xip_fid={stats['xip']['pte_at_fiducial']:.3f}, "
                    f"xim_fid={stats['xim']['pte_at_fiducial']:.3f}, "
                    f"cosebis_fid={stats['cosebis']['pte_at_fiducial']:.4f}",
                    flush=True,
                )
            except Exception:
                import traceback

                print(f"  WARNING: Could not compute PTEs for {version}:", flush=True)
                traceback.print_exc()

    # Build evidence
    evidence_data = {
        "spec_id": "config_space_pte_matrices",
        "spec_path": spec_path or "papers/bmodes/config/config_space_pte_matrices.md",
        "generated": datetime.now().isoformat(),
        "evidence": {
            "versions": {},
            "fiducial_scale_cuts_arcmin": {
                "xip": list(xip_fid),
                "xim": list(xim_fid),
                "cosebis": list(cosebis_fid),
            },
        },
        "output": {},
    }

    for version, stats in all_stats.items():
        role = "fiducial" if version == fiducial_version else "appendix"
        full_ptes = all_full_range_ptes[version]

        evidence_data["evidence"]["versions"][version] = {
            "role": role,
            "xip_stats": {
                **stats["xip"],
                "pte_at_full_range": full_ptes["xip"],
            },
            "xim_stats": {
                **stats["xim"],
                "pte_at_full_range": full_ptes["xim"],
            },
            "combined_stats": {
                **stats["combined"],
                "pte_at_full_range": full_ptes["combined"],
            },
            "cosebis_stats": {
                **stats["cosebis"],
                "pte_at_full_range": full_ptes["cosebis"],
            },
            "cosebis_20_stats": {
                **stats["cosebis_20"],
                "pte_at_full_range": full_ptes["cosebis_20"],
            },
        }

    evidence_data["output"]["figure_fiducial"] = out["figure_fiducial"].name
    evidence_data["output"]["figure_appendix"] = out["figure_appendix"].name

    evidence_path = out["evidence"]
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"\nSaved evidence to {evidence_path}")


def _versions_config_space(config):
    """Reproduce VERSIONS_CONFIG_SPACE_PTES from the Snakemake claims.smk:
    leak-corrected (non-ecut) versions plus their uncorrected counterparts."""
    leak_corr = [
        v for v in config["versions"] if "_leak_corr" in v and "_ecut" not in v
    ]
    uncorrected = [v.replace("_leak_corr", "") for v in leak_corr]
    return leak_corr + uncorrected


def _from_snakemake(smk):
    pure_eb_pte_files = smk.input["pure_eb_pte"]
    if isinstance(pure_eb_pte_files, str):
        pure_eb_pte_files = [pure_eb_pte_files]
    else:
        pure_eb_pte_files = list(pure_eb_pte_files)
    cosebis_pte_files = list(smk.input["cosebis_pte_files"])
    spec_paths = smk.input["specs"]
    main(
        config=smk.config,
        pure_eb_pte_files=pure_eb_pte_files,
        cosebis_pte_files=cosebis_pte_files,
        output_dir=Path(smk.output["evidence"]).parent,
        spec_path=spec_paths[0],
    )


def _from_cli(argv=None):
    ap = argparse.ArgumentParser(
        description="Configuration-space PTE matrix composites for the B-modes paper."
    )
    ap.add_argument("--config", required=True, help="Absolute path to config.yaml")
    ap.add_argument(
        "--pte-intermediate-dir",
        required=True,
        help="Directory holding {version}_{blind}_pure_eb_ptes.npz",
    )
    ap.add_argument(
        "--cosebis-pte-dir",
        required=True,
        help="COSEBI PTE scatter tree: {version}/{blind}/pte_{i:03d}_{j:03d}.json",
    )
    ap.add_argument("--blind", default="A", help="Fiducial blind (paper: A)")
    ap.add_argument("--out", required=True, help="Output directory (lc {output})")
    ap.add_argument(
        "--fiducial-version",
        default=None,
        help="Version whose inputs are repointed to explicit lc outputs "
        "(e.g. SP_v1.4.6.3_leak_corr); all other versions keep the old tree",
    )
    ap.add_argument(
        "--fiducial-pure-eb-pte-path",
        default=None,
        help="lc pure_eb PTE NPZ for the fiducial version "
        "(same format as old-tree {version}_{blind}_pure_eb_ptes.npz)",
    )
    ap.add_argument(
        "--fiducial-cosebis-pte-path",
        default=None,
        help="lc gathered COSEBI PTE NPZ for the fiducial version "
        "(adapted back to the per-pair JSON PTE-matrix structure)",
    )
    a = ap.parse_args(argv)

    if (
        a.fiducial_pure_eb_pte_path or a.fiducial_cosebis_pte_path
    ) and not a.fiducial_version:
        ap.error(
            "--fiducial-version is required when a fiducial override path is given"
        )
    fiducial_overrides = (
        {
            "version": a.fiducial_version,
            "pure_eb": a.fiducial_pure_eb_pte_path,
            "cosebis": a.fiducial_cosebis_pte_path,
        }
        if a.fiducial_version
        else None
    )

    with open(a.config) as f:
        config = yaml.safe_load(f)

    versions = _versions_config_space(config)
    pure_eb_pte_files = [
        os.path.join(a.pte_intermediate_dir, f"{v}_{a.blind}_pure_eb_ptes.npz")
        for v in versions
    ]
    pure_eb_pte_files = [p for p in pure_eb_pte_files if os.path.exists(p)]

    cosebis_pte_files = sorted(
        glob.glob(os.path.join(a.cosebis_pte_dir, "*", a.blind, "pte_*.json"))
    )

    main(
        config=config,
        pure_eb_pte_files=pure_eb_pte_files,
        cosebis_pte_files=cosebis_pte_files,
        output_dir=a.out,
        fiducial_overrides=fiducial_overrides,
    )


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821 — injected by Snakemake's script: directive
    except NameError:
        _from_cli()
    else:
        _from_snakemake(snakemake)  # noqa: F821
