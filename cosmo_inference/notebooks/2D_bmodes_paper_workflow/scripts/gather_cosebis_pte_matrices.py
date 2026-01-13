# %%
"""Gather COSEBIS PTE values and generate matrices and figures.

Gather step for figure_3_cosebis_pte_matrices claim.
Reads per-scale-cut JSON files, assembles matrices, generates heatmaps.
"""

import json
import shutil
import sys
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
            "results/claims/figure_3_cosebis_pte_matrices/trace.auto.json",
            str(Path.cwd()),
        )
    from snakemake.script import snakemake

    return snakemake


snakemake = _load_snakemake()


def _version_label(version):
    return version.replace("SP_", "").replace("_leak_corr", "")


def main():
    # Load spec (if available) for quantitative parameters
    spec = snakemake.params.get("spec", {})

    # Read quantitative params from spec with defaults
    fig_dimensions = spec.get("dimensions", [3.54, 3.54])
    heatmap_spec = spec.get("heatmap", {})
    contour_spec = spec.get("contours", {})
    fiducial_marker_spec = spec.get("fiducial_marker", {})
    tick_spec = spec.get("ticks", {})
    colorbar_spec = spec.get("colorbar", {})

    # Use all versions from params (merging figures 12/13 into figure 3)
    versions = snakemake.params.versions
    fiducial_min_scale = snakemake.params.fiducial_min_scale
    fiducial_max_scale = snakemake.params.fiducial_max_scale

    # Theta grid (21 values on [1, 250] arcmin = 20 intervals)
    theta_grid = np.geomspace(1.0, 250.0, 21)
    n_theta = len(theta_grid)  # 21

    # Find fiducial scale cut indices
    fid_idx_min = np.argmin(np.abs(theta_grid[:-1] - fiducial_min_scale))
    fid_idx_max = np.argmin(np.abs(theta_grid[1:] - fiducial_max_scale)) + 1

    # Ensure output directories exist
    trace_path = Path(snakemake.output.trace)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    paper_fig_dir = Path(snakemake.output.paper_figures[0]).parent
    paper_fig_dir.mkdir(parents=True, exist_ok=True)

    # Load all PTE values and organize by version
    pte_by_version = {ver: {} for ver in versions}

    for pte_file in snakemake.input.pte_files:
        pte_path = Path(pte_file)
        # Extract version from path: .../pte_values/{version}/pte_xxx_yyy.json
        version = pte_path.parent.name

        # Skip versions we're not processing (only fiducial per human review)
        if version not in versions:
            continue

        with open(pte_path) as f:
            data = json.load(f)

        i_min, i_max = data["i_min"], data["i_max"]
        pte_by_version[version][(i_min, i_max)] = data

    # Generate figures and compute statistics for each version
    # Produce both n=6 and n=20 mode matrices
    evidence_versions = {}

    for version in versions:
        print(f"\n{'='*60}")
        print(f"Processing {version}...")
        print(f"{'='*60}")

        # Assemble PTE matrices for both 6 and 20 modes (21x21 for theta indices 0-20)
        pte_matrix_6 = np.full((n_theta, n_theta), np.nan)
        pte_matrix_20 = np.full((n_theta, n_theta), np.nan)

        for (i_min, i_max), data in pte_by_version[version].items():
            # Skip n=1 off-diagonal: only 1 data point, not meaningful for PTE
            # Require at least 2 bins between min and max scale cuts
            if i_max - i_min < 2:
                continue

            # 6-mode results
            if "nmodes_6" in data:
                pte_matrix_6[i_min, i_max] = data["nmodes_6"]["pte_B"]
            else:
                # Legacy format fallback
                pte_matrix_6[i_min, i_max] = data["pte_B"]

            # 20-mode results
            if "nmodes_20" in data:
                pte_matrix_20[i_min, i_max] = data["nmodes_20"]["pte_B"]

        # Generate figures for both n=6 and n=20 modes
        matrices_by_nmodes = {6: pte_matrix_6, 20: pte_matrix_20}

        evidence_versions[version] = {"nmodes": {}}

        for nmodes, pte_matrix in matrices_by_nmodes.items():
            # Compute statistics
            valid_ptes = pte_matrix[~np.isnan(pte_matrix)]

            if len(valid_ptes) > 0:
                frac_healthy = float(
                    np.sum((valid_ptes >= 0.05) & (valid_ptes <= 0.95)) / len(valid_ptes)
                )
                pte_at_fid = float(pte_matrix[fid_idx_min, fid_idx_max])
            else:
                frac_healthy = np.nan
                pte_at_fid = np.nan

            evidence_versions[version]["nmodes"][nmodes] = {
                "pte_at_fiducial_cut": pte_at_fid,
                "fraction_ptes_in_healthy_range": frac_healthy,
                "n_scale_cuts_evaluated": int(len(valid_ptes)),
                "pte_statistics": {
                    "mean": float(np.nanmean(valid_ptes)) if len(valid_ptes) > 0 else np.nan,
                    "median": float(np.nanmedian(valid_ptes)) if len(valid_ptes) > 0 else np.nan,
                    "std": float(np.nanstd(valid_ptes)) if len(valid_ptes) > 0 else np.nan,
                    "min": float(np.nanmin(valid_ptes)) if len(valid_ptes) > 0 else np.nan,
                    "max": float(np.nanmax(valid_ptes)) if len(valid_ptes) > 0 else np.nan,
                },
            }

            print(f"\nStatistics for {version} (n={nmodes} modes):")
            print(f"  PTE at fiducial cut: {pte_at_fid:.4f}")
            print(f"  Fraction healthy (PTE in [0.05, 0.95]): {frac_healthy:.1%}")
            if len(valid_ptes) > 0:
                print(f"  Mean PTE: {np.nanmean(valid_ptes):.4f}")
                print(f"  Median PTE: {np.nanmedian(valid_ptes):.4f}")

            # Single-pane figure from spec
            fig, ax = plt.subplots(figsize=fig_dimensions)

            # B-mode PTE colormap from spec
            colormap_name = heatmap_spec.get("colormap", "vlag")
            vlag_cmap = sns.color_palette(colormap_name, as_cmap=True).copy()
            vlag_cmap.set_bad(color=heatmap_spec.get("bad_color", "lightgray"))

            # Transpose matrix for plotting (lower scale cut on x-axis)
            pte_plot_data = pte_matrix.T

            # B-mode PTE heatmap with contours
            im = ax.imshow(
                pte_plot_data,
                origin=heatmap_spec.get("origin", "lower"),
                aspect=heatmap_spec.get("aspect", "equal"),
                cmap=vlag_cmap,
                vmin=heatmap_spec.get("vmin", 0),
                vmax=heatmap_spec.get("vmax", 1),
                extent=[0, n_theta, 0, n_theta],
            )

            # Add contours at significance levels
            contour_levels = contour_spec.get("levels", [0.05, 0.95])
            contour_linewidths = contour_spec.get("linewidths", 0.8)
            cs = ax.contour(
                pte_plot_data,
                levels=contour_levels,
                colors=contour_spec.get("colors", "black"),
                linewidths=contour_linewidths,
                extent=[0, n_theta, 0, n_theta],
            )
            ax.clabel(cs, inline=True, fontsize=tick_spec.get("fontsize", 6), fmt="%.2f")

            # Add fiducial scale cut marker
            rect_x = fid_idx_min  # lower scale cut index
            rect_y = fid_idx_max  # upper scale cut index
            ax.add_patch(
                Rectangle(
                    (rect_x, rect_y),
                    1, 1,
                    fill=fiducial_marker_spec.get("fill", False),
                    edgecolor=fiducial_marker_spec.get("edgecolor", "black"),
                    linewidth=fiducial_marker_spec.get("linewidth", 1.5),
                    hatch=fiducial_marker_spec.get("hatch", "///"),
                    alpha=fiducial_marker_spec.get("alpha", 0.8),
                )
            )

            # Axis labels
            ax.set_xlabel(r"$\theta_{\min}$ [arcmin]", fontsize=8)
            ax.set_ylabel(r"$\theta_{\max}$ [arcmin]", fontsize=8)

            # Set angular scale tick labels (sparse to avoid crowding)
            tick_step = tick_spec.get("step", 5)
            tick_fontsize = tick_spec.get("fontsize", 6)
            tick_x_rotation = tick_spec.get("x_rotation", 45)
            tick_indices = np.arange(0, n_theta, tick_step)
            tick_labels = [f"{theta_grid[i]:.0f}" for i in tick_indices]
            tick_positions = tick_indices + 0.5

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=tick_x_rotation, ha="right", fontsize=tick_fontsize)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels, fontsize=tick_fontsize)

            # Colorbar
            cbar_shrink = colorbar_spec.get("shrink", 0.8)
            cbar_pad = colorbar_spec.get("pad", 0.02)
            cbar_fontsize = colorbar_spec.get("fontsize", 7)
            cbar_label_rotation = colorbar_spec.get("label_rotation", 270)
            cbar = plt.colorbar(im, ax=ax, shrink=cbar_shrink, pad=cbar_pad)
            cbar.set_label("PTE", fontsize=cbar_fontsize, rotation=cbar_label_rotation, labelpad=10)
            cbar.ax.tick_params(labelsize=tick_fontsize)

            plt.tight_layout()

            # Save to claims directory
            claims_fig_path = trace_path.parent / f"cosebis_pte_matrices_{version}_n{nmodes}.png"
            fig.savefig(claims_fig_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {claims_fig_path}")

            # Copy to paper figures directory
            paper_fig_path = paper_fig_dir / f"cosebis_pte_matrices_{version}_n{nmodes}.png"
            shutil.copy2(claims_fig_path, paper_fig_path)
            print(f"Copied to paper figures: {paper_fig_path}")

            plt.close(fig)

    # Build realized spec (quantitative values actually used)
    realized_spec = {
        "dimensions": list(fig_dimensions),
        "heatmap": {
            "colormap": heatmap_spec.get("colormap", "vlag"),
            "vmin": heatmap_spec.get("vmin", 0),
            "vmax": heatmap_spec.get("vmax", 1),
        },
        "contours": {
            "levels": contour_spec.get("levels", [0.05, 0.95]),
            "linewidths": contour_spec.get("linewidths", 0.8),
        },
        "fiducial_marker": {
            "linewidth": fiducial_marker_spec.get("linewidth", 1.5),
            "hatch": fiducial_marker_spec.get("hatch", "///"),
        },
        "ticks": {
            "step": tick_spec.get("step", 5),
            "fontsize": tick_spec.get("fontsize", 6),
        },
        "colorbar": {
            "shrink": colorbar_spec.get("shrink", 0.8),
            "pad": colorbar_spec.get("pad", 0.02),
        },
    }

    # Write epistemic claim trace
    trace = {
        "claim": snakemake.params.claim,
        "implementation_plan": snakemake.params.implementation_plan,
        "specs": {
            "quantitative": realized_spec,
            "qualitative": {
                "emphasis": spec.get("emphasis", ""),
                "narrative": spec.get("narrative", ""),
                "style": spec.get("style", ""),
                "intent": spec.get("intent", ""),
            },
        },
        "evidence": {
            "versions": evidence_versions,
            "fiducial_scale_cuts_arcmin": [fiducial_min_scale, fiducial_max_scale],
            "nmodes_displayed": snakemake.params.nmodes,
            "healthy_pte_range": [0.05, 0.95],
            "theta_grid_arcmin": theta_grid.tolist(),
        },
        "artifact_paths": {
            f"pte_matrix_{_version_label(version)}_n{nmodes}": str(trace_path.parent / f"cosebis_pte_matrices_{version}_n{nmodes}.png")
            for version in versions
            for nmodes in [6, 20]
        },
        "parameters": {
            "kind": snakemake.params.kind,
            "versions": versions,
            "nmodes": snakemake.params.nmodes,
            "n_scale_cuts_per_version": len(pte_by_version[versions[0]]),
            "total_scatter_jobs": sum(len(v) for v in pte_by_version.values()),
        },
        "depends_on": [
            Path(p).parent.name for p in snakemake.input
            if "trace.auto.json" in str(p)
        ],
        "ai_review": None,
    }

    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"\nSaved trace to {trace_path}")


if __name__ == "__main__":
    main()
