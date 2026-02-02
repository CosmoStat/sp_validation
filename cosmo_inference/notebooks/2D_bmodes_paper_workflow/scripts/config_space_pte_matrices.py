"""Configuration-space PTE matrix composites for B-modes paper.

Produces 3-panel composites (xi+^B, xi-^B, COSEBIS) for:
- Results: fiducial version (v1.4.6)
- Appendix: non-fiducial versions (v1.4.5, v1.4.8)

Each composite has shared axes and a single colorbar.
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
            "results/claims/config_space_pte_matrices/evidence.json",
            str(Path.cwd()),
        )
    from snakemake.script import snakemake

    return snakemake


snakemake = _load_snakemake()


def load_cosebis_pte_matrix(pte_files, version, nmodes=6):
    """Load COSEBIS PTE values from JSON files into matrix.

    Uses blind A covariance only. BB covariances are theoretically blind-independent;
    any variation across blinds is due to MC sampling noise, not real blind dependence.
    See cosmology_for_covariance.md wiki for details.

    Parameters
    ----------
    pte_files : list of str
        Paths to PTE JSON files (blind A only).
    version : str
        Version to filter for.
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
    # Initialize with nan
    pte_matrix = np.full((n_theta, n_theta), np.nan)

    for pte_file in pte_files:
        pte_path = Path(pte_file)
        # Filter to this version (path contains version string)
        if version not in str(pte_path):
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

        # Store PTE value (blind A only)
        if not np.isnan(pte_val):
            pte_matrix[i_min, i_max] = pte_val

    return pte_matrix, theta_grid


def load_pure_eb_pte_matrices(pte_files, version):
    """Load Pure E/B PTE matrices from npz files.

    Uses blind A covariance only. BB covariances are theoretically blind-independent;
    any variation across blinds is due to MC sampling noise, not real blind dependence.
    See cosmology_for_covariance.md wiki for details.

    Parameters
    ----------
    pte_files : list of str
        Paths to pure_eb_ptes.npz files (blind A only).
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
    """
    pte_xip_B = None
    pte_xim_B = None
    theta = None

    for pte_file in pte_files:
        # Filter to this version
        if version not in pte_file:
            continue

        data = np.load(pte_file)
        theta = data["theta"]
        pte_xip_B = data["pte_xip_B"]
        pte_xim_B = data["pte_xim_B"]
        break  # Only one file per version (blind A)

    return pte_xip_B, pte_xim_B, theta


def make_pte_colormap(low=0.05, high=0.95, gradient_range=(0.15, 0.85)):
    """Create discrete colormap with sharp breaks at significance thresholds.

    Parameters
    ----------
    low, high : float
        PTE thresholds. Solid color below low and above high.
    gradient_range : tuple
        Range of vlag colormap to use for the gradient portion.
        Narrower range = shallower gradient = sharper contrast at boundaries.
    """
    vlag = sns.color_palette("vlag", as_cmap=True)

    # Solid regions use colors just outside the gradient range for sharp contrast
    solid_blue = vlag(0.0)
    solid_red = vlag(1.0)

    # Build colormap: [0, low] solid blue, [low, high] compressed gradient, [high, 1] solid red
    n_total = 256
    n_low = int(low * n_total)
    n_high = int((1 - high) * n_total)
    n_mid = n_total - n_low - n_high

    # Gradient samples from compressed range of vlag
    g_lo, g_hi = gradient_range
    gradient_colors = [vlag(g_lo + (g_hi - g_lo) * i / (n_mid - 1)) for i in range(n_mid)]

    all_colors = [solid_blue] * n_low + gradient_colors + [solid_red] * n_high
    cmap = LinearSegmentedColormap.from_list("pte_discrete", all_colors, N=256)
    cmap.set_bad(color="lightgray")
    return cmap


def plot_pte_panel(ax, pte_matrix, theta_grid, fid_start, fid_stop, title,
                   show_xticklabels=True, show_yticklabels=False):
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

    # Discrete colormap: solid blue below 0.05, solid red above 0.95, gradient in between
    pte_cmap = make_pte_colormap()

    # Plot heatmap (no contours)
    im = ax.imshow(
        pte_matrix.T,
        origin="lower",
        aspect="equal",
        cmap=pte_cmap,
        vmin=0,
        vmax=1,
        extent=[0, n_theta, 0, n_theta],
    )

    # Fiducial scale cut marker (black square, no hatching)
    ax.add_patch(
        Rectangle(
            (fid_start, fid_stop),
            1, 1,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
        )
    )

    # Title
    ax.set_title(title)

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
        ax.set_xticklabels(x_tick_labels, rotation=45, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels([])

    if show_yticklabels:
        ax.set_yticklabels(y_tick_labels, rotation=45, ha="right", rotation_mode="anchor")
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


def extract_full_range_ptes(pure_eb_pte_files, cosebis_pte_files, version):
    """Extract full-range PTEs from npz and JSON files.

    Uses blind A covariance only.

    Parameters
    ----------
    pure_eb_pte_files : list
        Pure E/B PTE npz files (blind A only).
    cosebis_pte_files : list
        COSEBIS PTE JSON files (blind A only).
    version : str
        Catalog version string.

    Returns
    -------
    ptes : dict
        Full-range PTEs for xip, xim, and cosebis.
    """
    # Get full-range PTEs from pure E/B (blind A only)
    ptes = {
        "xip": float("nan"),
        "xim": float("nan"),
        "cosebis": float("nan"),
    }

    for pte_file in pure_eb_pte_files:
        if version not in pte_file:
            continue
        pte_data = np.load(pte_file)
        theta = pte_data["theta"]
        full_range_idx = (0, len(theta) - 1)
        ptes["xip"] = float(pte_data["pte_xip_B"][full_range_idx])
        ptes["xim"] = float(pte_data["pte_xim_B"][full_range_idx])
        break  # Only one file per version (blind A)

    # Load COSEBIS full-range PTE (pte_000_020.json for full theta range)
    for pte_file in cosebis_pte_files:
        if version in str(pte_file) and "pte_000_020" in str(pte_file):
            try:
                with open(pte_file) as f:
                    data = json.load(f)
                pte_val = data.get("pte_B", data.get("nmodes_6", {}).get("pte_B"))
                if pte_val is not None and not np.isnan(pte_val):
                    ptes["cosebis"] = float(pte_val)
                    break  # Only one file per version (blind A)
            except FileNotFoundError:
                continue

    return ptes


def create_3panel_composite(version, pure_eb_pte_files, cosebis_pte_files,
                            xip_fid, xim_fid, cosebis_fid):
    """Create a 1×3 composite figure for the fiducial version (main text).

    Parameters
    ----------
    version : str
        Catalog version string (fiducial).
    pure_eb_pte_files : list
        Pure E/B PTE npz files (blind A only).
    cosebis_pte_files : list
        COSEBIS PTE JSON files (blind A only).
    xip_fid, xim_fid : tuple
        Fiducial scale cuts for xi+ and xi- (arcmin).
    cosebis_fid : tuple
        Fiducial scale cuts for COSEBIS (arcmin).

    Returns
    -------
    fig : Figure
        The composite figure.
    stats : dict
        Statistics for this version.
    full_range_ptes : dict
        Full-range PTEs for this version.
    """
    # Create figure: 1 row × 3 columns (colorbar attached to rightmost)
    fig_width = 6.5
    fig_height = 2.6

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1, 1, 1],
        wspace=0.03, hspace=0.03,
        left=0.08, right=0.90,
        bottom=0.15, top=0.90
    )

    # Load Pure E/B PTE matrices (blind A)
    pte_xip_B, pte_xim_B, theta_pure_eb = load_pure_eb_pte_matrices(
        pure_eb_pte_files, version
    )

    # Load COSEBIS PTE matrix
    pte_cosebis, theta_cosebis = load_cosebis_pte_matrix(
        cosebis_pte_files, version, nmodes=6
    )

    # Compute fiducial indices
    cosebis_fid_start = np.argmin(np.abs(theta_cosebis[:-1] - cosebis_fid[0]))
    cosebis_fid_stop = np.argmin(np.abs(theta_cosebis[1:] - cosebis_fid[1])) + 1

    xip_start = np.argmin(np.abs(theta_pure_eb - xip_fid[0]))
    xip_stop = np.argmin(np.abs(theta_pure_eb - xip_fid[1]))
    xim_start = np.argmin(np.abs(theta_pure_eb - xim_fid[0]))
    xim_stop = np.argmin(np.abs(theta_pure_eb - xim_fid[1]))

    # Create subplot axes
    ax_xip = fig.add_subplot(gs[0])
    ax_xim = fig.add_subplot(gs[1])
    ax_cosebis = fig.add_subplot(gs[2])

    # Plot panels with titles
    im_xip = plot_pte_panel(
        ax_xip, pte_xip_B, theta_pure_eb,
        xip_start, xip_stop,
        r"$\xi_+^B$",
        show_xticklabels=True, show_yticklabels=True
    )
    ax_xip.set_ylabel(r"$\theta_{\max}$ [arcmin]", labelpad=2)

    im_xim = plot_pte_panel(
        ax_xim, pte_xim_B, theta_pure_eb,
        xim_start, xim_stop,
        r"$\xi_-^B$",
        show_xticklabels=True, show_yticklabels=False
    )

    im_cosebis = plot_pte_panel(
        ax_cosebis, pte_cosebis, theta_cosebis,
        cosebis_fid_start, cosebis_fid_stop,
        r"COSEBIS $B_n$",
        show_xticklabels=True, show_yticklabels=False
    )

    # Colorbar attached to rightmost panel (matches its height)
    divider = make_axes_locatable(ax_cosebis)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cbar = fig.colorbar(im_cosebis, cax=cax)
    cbar.set_label("PTE")
    cbar.set_ticks([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    cbar.set_ticklabels(["0", "", "0.25", "0.5", "0.75", "", "1"])
    # Mark significance thresholds with horizontal lines
    for thresh in [0.05, 0.95]:
        cbar.ax.axhline(thresh, color="black", linewidth=0.8)

    # Common x-axis label
    fig.text(0.50, 0.02, r"$\theta_{\min}$ [arcmin]", ha="center")

    # Compute statistics
    stats = {
        "xip": compute_stats(pte_xip_B, xip_start, xip_stop),
        "xim": compute_stats(pte_xim_B, xim_start, xim_stop),
        "cosebis": compute_stats(pte_cosebis, cosebis_fid_start, cosebis_fid_stop),
    }

    # Extract full-range PTEs
    full_range_ptes = extract_full_range_ptes(pure_eb_pte_files, cosebis_pte_files, version)

    return fig, stats, full_range_ptes


def create_9panel_composite(versions, pure_eb_pte_files, cosebis_pte_files,
                            xip_fid, xim_fid, cosebis_fid):
    """Create a 3x3 composite figure for all versions (appendix).

    Parameters
    ----------
    versions : list of str
        Catalog version strings in display order [v1.4.5, v1.4.6, v1.4.8].
    pure_eb_pte_files : list
        Pure E/B PTE npz files (blind A only).
    cosebis_pte_files : list
        All COSEBIS PTE JSON files.
    xip_fid, xim_fid : tuple
        Fiducial scale cuts for xi+ and xi- (arcmin).
    cosebis_fid : tuple
        Fiducial scale cuts for COSEBIS (arcmin).

    Returns
    -------
    fig : Figure
        The composite figure.
    all_stats : dict
        Statistics keyed by version.
    all_full_range_ptes : dict
        Full-range PTEs keyed by version.
    """
    # Create figure: 3 rows x 4 columns (3 stats + colorbar)
    fig_width = 6.5
    fig_height = 6.5

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        3, 4,
        width_ratios=[1, 1, 1, 0.08],
        wspace=0.03, hspace=0.08,
        left=0.16, right=0.95,
        bottom=0.12, top=0.90
    )

    all_stats = {}
    all_full_range_ptes = {}
    # Use descriptive version labels per project convention
    version_labels = {
        "SP_v1.4.5_leak_corr": "Initial",
        "SP_v1.4.6_leak_corr": "Fiducial",
        "SP_v1.4.8_leak_corr": "Masked",
        "SP_v1.4.5": "Initial",
        "SP_v1.4.6": "Fiducial",
        "SP_v1.4.8": "Masked",
    }

    for row_idx, version in enumerate(versions):
        # Load Pure E/B PTE matrices (blind A)
        pte_xip_B, pte_xim_B, theta_pure_eb = load_pure_eb_pte_matrices(
            pure_eb_pte_files, version
        )

        # Load COSEBIS PTE matrix
        pte_cosebis, theta_cosebis = load_cosebis_pte_matrix(
            cosebis_pte_files, version, nmodes=6
        )

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
        show_xticklabels = (row_idx == 2)

        # Plot titles: statistic names on top row only (version labels on left side)
        version_label = version_labels.get(version, version)
        xip_title = r"$\xi_+^B$" if row_idx == 0 else ""
        xim_title = r"$\xi_-^B$" if row_idx == 0 else ""
        cosebis_title = r"COSEBIS $B_n$" if row_idx == 0 else ""

        # Plot panels
        im_xip = plot_pte_panel(
            ax_xip, pte_xip_B, theta_pure_eb,
            xip_start, xip_stop,
            xip_title,
            show_xticklabels=show_xticklabels, show_yticklabels=show_yticklabels
        )
        # Add y-axis label to leftmost panel of each row
        ax_xip.set_ylabel(r"$\theta_{\max}$ [arcmin]", labelpad=2)

        im_xim = plot_pte_panel(
            ax_xim, pte_xim_B, theta_pure_eb,
            xim_start, xim_stop,
            xim_title,
            show_xticklabels=show_xticklabels, show_yticklabels=False
        )

        im_cosebis = plot_pte_panel(
            ax_cosebis, pte_cosebis, theta_cosebis,
            cosebis_fid_start, cosebis_fid_stop,
            cosebis_title,
            show_xticklabels=show_xticklabels, show_yticklabels=False
        )

        # Add version label to the left of each row
        version_label = version_labels.get(version, version)
        ax_xip.annotate(
            version_label, xy=(-0.28, 0.5), xycoords="axes fraction",
            weight="bold", va="center", ha="right", rotation=90
        )

        # Compute statistics
        stats = {
            "xip": compute_stats(pte_xip_B, xip_start, xip_stop),
            "xim": compute_stats(pte_xim_B, xim_start, xim_stop),
            "cosebis": compute_stats(pte_cosebis, cosebis_fid_start, cosebis_fid_stop),
        }
        all_stats[version] = stats

        # Extract full-range PTEs
        full_range_ptes = extract_full_range_ptes(pure_eb_pte_files, cosebis_pte_files, version)
        all_full_range_ptes[version] = full_range_ptes

    # Shared colorbar on rightmost column
    cax = fig.add_subplot(gs[:, 3])
    cbar = fig.colorbar(im_cosebis, cax=cax)
    cbar.set_label("PTE")
    cbar.set_ticks([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    cbar.set_ticklabels(["0", "", "0.25", "0.5", "0.75", "", "1"])
    # Mark significance thresholds with horizontal lines
    for thresh in [0.05, 0.95]:
        cbar.ax.axhline(thresh, color="black", linewidth=0.8)

    # Common x-axis label
    fig.text(0.50, 0.02, r"$\theta_{\min}$ [arcmin]", ha="center")

    return fig, all_stats, all_full_range_ptes


def main():
    config = snakemake.config
    versions = config["versions"]
    fiducial_version = config["fiducial"]["version"]

    # Scale cuts from config
    xip_fid = tuple(config["fiducial"]["fiducial_xip_scale_cut"])
    xim_fid = tuple(config["fiducial"]["fiducial_xim_scale_cut"])
    cosebis_fid = (
        config["fiducial"]["fiducial_min_scale"],
        config["fiducial"]["fiducial_max_scale"],
    )

    # Output paths
    output_dir = Path(snakemake.output["evidence"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    paper_dir = Path(snakemake.output["paper_figure_appendix"]).parent
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Get input files as lists
    pure_eb_pte_files = snakemake.input["pure_eb_pte"]
    if isinstance(pure_eb_pte_files, str):
        pure_eb_pte_files = [pure_eb_pte_files]
    else:
        pure_eb_pte_files = list(pure_eb_pte_files)

    cosebis_pte_files = list(snakemake.input["cosebis_pte_files"])

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
        )

        # Save fiducial figure
        fig_fid_path = Path(snakemake.output["figure_fiducial"])
        fig_fid.savefig(fig_fid_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved {fig_fid_path}", flush=True)

        paper_fid_path = Path(snakemake.output["paper_figure_fiducial"])
        shutil.copy2(fig_fid_path, paper_fid_path)
        print(f"  Copied to {paper_fid_path}", flush=True)

        plt.close(fig_fid)

        all_stats[fiducial_version] = fid_stats
        all_full_range_ptes[fiducial_version] = fid_ptes

    except Exception as e:
        import traceback
        print(f"  ERROR creating fiducial composite:", flush=True)
        traceback.print_exc()
        raise

    # Generate 3x3 composite for leak-corrected versions (appendix)
    leak_corr_versions = [v for v in versions if "leak_corr" in v]
    print(f"\n--- Creating 3x3 composite for leak-corrected versions: {leak_corr_versions} ---", flush=True)

    try:
        fig, appendix_stats, appendix_ptes = create_9panel_composite(
            versions=leak_corr_versions,
            pure_eb_pte_files=pure_eb_pte_files,
            cosebis_pte_files=cosebis_pte_files,
            xip_fid=xip_fid,
            xim_fid=xim_fid,
            cosebis_fid=cosebis_fid,
        )

        # Save appendix figure
        fig_path = Path(snakemake.output["figure_appendix"])
        fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved {fig_path}", flush=True)

        paper_path = Path(snakemake.output["paper_figure_appendix"])
        shutil.copy2(fig_path, paper_path)
        print(f"  Copied to {paper_path}", flush=True)

        plt.close(fig)

        # Update stats with all versions from appendix
        all_stats.update(appendix_stats)
        all_full_range_ptes.update(appendix_ptes)

    except Exception as e:
        import traceback
        print(f"  ERROR creating appendix composite:", flush=True)
        traceback.print_exc()
        raise

    # Generate 3x3 composite for uncorrected versions (second appendix)
    uncorr_versions = [v for v in versions if "leak_corr" not in v]
    print(f"\n--- Creating 3x3 composite for uncorrected versions: {uncorr_versions} ---", flush=True)

    try:
        fig, uncorr_stats, uncorr_ptes = create_9panel_composite(
            versions=uncorr_versions,
            pure_eb_pte_files=pure_eb_pte_files,
            cosebis_pte_files=cosebis_pte_files,
            xip_fid=xip_fid,
            xim_fid=xim_fid,
            cosebis_fid=cosebis_fid,
        )

        # Save uncorrected appendix figure
        fig_path = Path(snakemake.output["figure_appendix_uncorrected"])
        fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved {fig_path}", flush=True)

        paper_path = Path(snakemake.output["paper_figure_appendix_uncorrected"])
        shutil.copy2(fig_path, paper_path)
        print(f"  Copied to {paper_path}", flush=True)

        plt.close(fig)

        # Update stats with uncorrected versions
        all_stats.update(uncorr_stats)
        all_full_range_ptes.update(uncorr_ptes)

    except Exception as e:
        import traceback
        print(f"  ERROR creating uncorrected appendix composite:", flush=True)
        traceback.print_exc()
        raise

    # Build evidence
    spec_paths = snakemake.input["specs"]

    evidence_data = {
        "spec_id": "config_space_pte_matrices",
        "spec_path": spec_paths[0],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "versions": {},
            "fiducial_scale_cuts_arcmin": {
                "xip": list(xip_fid),
                "xim": list(xim_fid),
                "cosebis": list(cosebis_fid),
            },
        },
        "artifacts": {},
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
            "cosebis_stats": {
                **stats["cosebis"],
                "pte_at_full_range": full_ptes["cosebis"],
            },
        }

    evidence_data["artifacts"]["figure_fiducial"] = Path(snakemake.output["figure_fiducial"]).name
    evidence_data["artifacts"]["figure_appendix"] = Path(snakemake.output["figure_appendix"]).name
    evidence_data["artifacts"]["figure_appendix_uncorrected"] = Path(snakemake.output["figure_appendix_uncorrected"]).name

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"\nSaved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
