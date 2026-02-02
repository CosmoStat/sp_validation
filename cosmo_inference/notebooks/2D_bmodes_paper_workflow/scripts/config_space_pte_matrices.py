"""Configuration-space PTE matrix composites for B-modes paper.

Produces 3-panel composites (xi+^B, xi-^B, COSEBIS) for:
- Results: fiducial version (from config.fiducial.version)
- Appendix: all versions (from config.versions with labels from config.plotting.version_labels)

Each composite has shared axes and a single colorbar.
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


def load_cosebis_pte_matrix(pte_files, version, config, nmodes=6):
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
    fid = config["fiducial"]
    theta_grid = np.geomspace(fid["min_sep"], fid["max_sep"], fid["nbins"] + 1)
    n_theta = len(theta_grid)
    # Initialize with nan for cells without data
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

        # Store PTE value
        if not np.isnan(pte_val):
            pte_matrix[i_min, i_max] = pte_val

    return pte_matrix, theta_grid


def load_pure_eb_pte_matrices(pte_files, version):
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
    """
    for pte_file in pte_files:
        # Filter to this version
        if version not in pte_file:
            continue

        data = np.load(pte_file)
        return data["pte_xip_B"], data["pte_xim_B"], data["theta"]

    raise ValueError(f"No PTE file found for version {version}")


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

    # Tick labels - more frequent markers
    tick_step = 2
    tick_indices = np.arange(0, n_theta, tick_step)
    tick_labels = [f"{theta_grid[i]:.0f}" for i in tick_indices]
    tick_positions = tick_indices + 0.5

    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)

    if show_xticklabels:
        ax.set_xticklabels(tick_labels, fontsize=6)
    else:
        ax.set_xticklabels([])

    if show_yticklabels:
        ax.set_yticklabels(tick_labels, fontsize=6)
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
    # Get full-range PTEs from pure E/B (fiducial blind)
    xip_ptes = []
    xim_ptes = []
    for pte_file in pure_eb_pte_files:
        if version not in pte_file:
            continue
        pte_data = np.load(pte_file)
        theta = pte_data["theta"]
        full_range_idx = (0, len(theta) - 1)
        xip_pte = pte_data["pte_xip_B"][full_range_idx]
        xim_pte = pte_data["pte_xim_B"][full_range_idx]
        if not np.isnan(xip_pte):
            xip_ptes.append(xip_pte)
        if not np.isnan(xim_pte):
            xim_ptes.append(xim_pte)

    ptes = {
        "xip": float(min(xip_ptes)) if xip_ptes else float("nan"),
        "xim": float(min(xim_ptes)) if xim_ptes else float("nan"),
        "cosebis": float("nan"),  # Default if file not found
    }

    # Load COSEBIS full-range PTE (pte_000_020.json for full theta range, min across blinds)
    cosebis_ptes = []
    for pte_file in cosebis_pte_files:
        if version in str(pte_file) and "pte_000_020" in str(pte_file):
            try:
                with open(pte_file) as f:
                    data = json.load(f)
                pte_val = data.get("pte_B", data.get("nmodes_6", {}).get("pte_B"))
                if pte_val is not None and not np.isnan(pte_val):
                    cosebis_ptes.append(pte_val)
            except FileNotFoundError:
                continue

    if cosebis_ptes:
        ptes["cosebis"] = float(min(cosebis_ptes))

    return ptes


def create_3panel_composite(version, pure_eb_pte_files, cosebis_pte_files,
                            xip_fid, xim_fid, cosebis_fid, config):
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
        1, 4,
        width_ratios=[1, 1, 1, 0.04],
        wspace=0.03, hspace=0.03,
        left=0.08, right=0.95,
        bottom=0.15, top=0.90
    )

    # Load Pure E/B PTE matrices (fiducial blind)
    pte_xip_B, pte_xim_B, theta_pure_eb = load_pure_eb_pte_matrices(
        pure_eb_pte_files, version
    )

    # Load COSEBIS PTE matrix
    pte_cosebis, theta_cosebis = load_cosebis_pte_matrix(
        cosebis_pte_files, version, config, nmodes=6
    )

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
        ax_xip, pte_xip_B, theta_pure_eb,
        xip_start, xip_stop,
        r"$\xi_+^B$",
        show_xticklabels=True, show_yticklabels=True
    )
    ax_xip.set_ylabel(r"$\theta_{\max}$ [arcmin]", fontsize=8, labelpad=2)

    plot_pte_panel(
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

    # Shared colorbar on rightmost column
    cax = fig.add_subplot(gs[0, 3])
    cbar = fig.colorbar(im_cosebis, cax=cax)
    cbar.set_label("PTE", fontsize=8)
    cbar.ax.tick_params(labelsize=6)

    # Common x-axis label
    fig.text(0.50, 0.02, r"$\theta_{\min}$ [arcmin]",
             ha="center", fontsize=9)

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
                            xip_fid, xim_fid, cosebis_fid, version_labels, config):
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
    fig_height = 2.2 * n_versions

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        n_versions, 4,
        width_ratios=[1, 1, 1, 0.04],
        wspace=0.03, hspace=0.08,
        left=0.10, right=0.95,
        bottom=0.08, top=0.94
    )

    all_stats = {}
    all_full_range_ptes = {}

    for row_idx, version in enumerate(versions):
        # Load Pure E/B PTE matrices (fiducial blind)
        pte_xip_B, pte_xim_B, theta_pure_eb = load_pure_eb_pte_matrices(
            pure_eb_pte_files, version
        )

        # Load COSEBIS PTE matrix
        pte_cosebis, theta_cosebis = load_cosebis_pte_matrix(
            cosebis_pte_files, version, config, nmodes=6
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
        show_xticklabels = (row_idx == n_versions - 1)

        # Plot titles: version label on top row panels
        version_label = version_labels.get(version, version)
        xip_title = rf"{version_label}: $\xi_+^B$" if row_idx == 0 else ""
        xim_title = r"$\xi_-^B$" if row_idx == 0 else ""
        cosebis_title = r"COSEBIS $B_n$" if row_idx == 0 else ""

        # Plot panels
        plot_pte_panel(
            ax_xip, pte_xip_B, theta_pure_eb,
            xip_start, xip_stop,
            xip_title,
            show_xticklabels=show_xticklabels, show_yticklabels=show_yticklabels
        )
        # Add y-axis label to leftmost panel of each row
        ax_xip.set_ylabel(r"$\theta_{\max}$ [arcmin]", fontsize=8, labelpad=2)

        plot_pte_panel(
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

        # Add version label to the right of each row
        ax_cosebis.annotate(
            version_label, xy=(1.02, 0.5), xycoords="axes fraction",
            fontsize=8, weight="bold", va="center", ha="left", rotation=-90
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
    cbar.set_label("PTE", fontsize=8)
    cbar.ax.tick_params(labelsize=6)

    # Common x-axis label
    fig.text(0.50, 0.02, r"$\theta_{\min}$ [arcmin]",
             ha="center", fontsize=9)

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
            config=config,
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

    except Exception:
        import traceback
        print("  ERROR creating fiducial composite:", flush=True)
        traceback.print_exc()
        raise

    # Generate Nx3 composite for all versions (appendix)
    print("\n--- Creating Nx3 composite for all versions ---", flush=True)

    try:
        fig, appendix_stats, appendix_ptes = create_9panel_composite(
            versions=versions,
            pure_eb_pte_files=pure_eb_pte_files,
            cosebis_pte_files=cosebis_pte_files,
            xip_fid=xip_fid,
            xim_fid=xim_fid,
            cosebis_fid=cosebis_fid,
            version_labels=config["plotting"]["version_labels"],
            config=config,
        )

        # Save appendix figure
        fig_path = Path(snakemake.output["figure_appendix"])
        fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved {fig_path}", flush=True)

        paper_path = Path(snakemake.output["paper_figure_appendix"])
        shutil.copy2(fig_path, paper_path)
        print(f"  Copied to {paper_path}", flush=True)

        plt.close(fig)

        # Update stats with all versions from appendix (overwrites fiducial with same data)
        all_stats.update(appendix_stats)
        all_full_range_ptes.update(appendix_ptes)

    except Exception:
        import traceback
        print("  ERROR creating appendix composite:", flush=True)
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

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"\nSaved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
