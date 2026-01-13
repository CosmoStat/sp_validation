"""Gather COSEBIS PTE values and generate matrices and figures.

Gather step for cosebis_pte_matrix claim.
Reads per-scale-cut JSON files, assembles matrices per version/blind,
reports minimum PTE across blinds for conservative assessment.
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
import yaml


plt.style.use(
    "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle"
)


def _version_label(version):
    return version.replace("SP_", "").replace("_leak_corr", "")


def _create_pte_figure(pte_matrix, theta_grid, fid_idx_min, fid_idx_max, output_path):
    """Create a single-pane PTE heatmap figure."""
    n_theta = len(theta_grid)

    vlag_cmap = sns.color_palette("vlag", as_cmap=True).copy()
    vlag_cmap.set_bad(color="lightgray")

    fig, ax = plt.subplots(figsize=(3.54, 3.54))

    im = ax.imshow(
        pte_matrix.T, origin="lower", aspect="equal",
        cmap=vlag_cmap, vmin=0, vmax=1, extent=[0, n_theta, 0, n_theta],
    )

    cs = ax.contour(
        pte_matrix.T, levels=[0.05, 0.95], colors="black",
        linewidths=0.8, extent=[0, n_theta, 0, n_theta],
    )
    ax.clabel(cs, inline=True, fmt="%.2f")

    # Fiducial marker
    ax.add_patch(
        Rectangle((fid_idx_min, fid_idx_max), 1, 1, fill=False, edgecolor="black",
                  linewidth=1.5, hatch="///", alpha=0.8)
    )

    ax.set_xlabel(r"$\theta_{\min}$ [arcmin]")
    ax.set_ylabel(r"$\theta_{\max}$ [arcmin]")

    # Tick labels
    tick_step = 5
    tick_indices = np.arange(0, n_theta, tick_step)
    tick_labels = [f"{theta_grid[i]:.0f}" for i in tick_indices]
    tick_positions = tick_indices + 0.5

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("PTE", rotation=270, labelpad=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    from snakemake.script import snakemake

    # Read config
    with open(snakemake.input["config"]) as f:
        config = yaml.safe_load(f)

    versions = config["versions"]
    blinds = ["A", "B", "C"]
    fiducial_min_scale = config["fiducial"]["fiducial_min_scale"]
    fiducial_max_scale = config["fiducial"]["fiducial_max_scale"]

    # Theta grid
    theta_grid = np.geomspace(1.0, 250.0, 21)
    n_theta = len(theta_grid)

    # Fiducial indices
    fid_idx_min = np.argmin(np.abs(theta_grid[:-1] - fiducial_min_scale))
    fid_idx_max = np.argmin(np.abs(theta_grid[1:] - fiducial_max_scale)) + 1

    # Create output directory
    output_dir = Path(snakemake.output["evidence"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = Path(snakemake.output["paper_figures"][0]).parent
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Load PTE values organized by version/blind
    pte_by_version_blind = {ver: {blind: {} for blind in blinds} for ver in versions}

    for pte_file in snakemake.input["pte_files"]:
        pte_path = Path(pte_file)
        # Path: .../pte_values/{version}/{blind}/pte_xxx_yyy.json
        blind = pte_path.parent.name
        version = pte_path.parent.parent.name

        if version not in versions or blind not in blinds:
            continue

        with open(pte_path) as f:
            data = json.load(f)

        i_min, i_max = data["i_min"], data["i_max"]
        pte_by_version_blind[version][blind][(i_min, i_max)] = data

    # Generate figures and compute statistics
    evidence_versions = {}
    artifacts = {}

    for version in versions:
        print(f"\n{'='*60}")
        print(f"Processing {version}...")
        print(f"{'='*60}")

        evidence_versions[version] = {"blinds": {}, "nmodes": {}}

        for blind in blinds:
            pte_data = pte_by_version_blind[version][blind]

            if not pte_data:
                print(f"  No data for blind {blind}")
                continue

            # Build matrices for 6 and 20 modes
            pte_matrix_6 = np.full((n_theta, n_theta), np.nan)
            pte_matrix_20 = np.full((n_theta, n_theta), np.nan)

            for (i_min, i_max), data in pte_data.items():
                if i_max - i_min < 2:
                    continue

                if "nmodes_6" in data:
                    pte_matrix_6[i_min, i_max] = data["nmodes_6"]["pte_B"]
                else:
                    pte_matrix_6[i_min, i_max] = data["pte_B"]

                if "nmodes_20" in data:
                    pte_matrix_20[i_min, i_max] = data["nmodes_20"]["pte_B"]

            # Statistics per blind
            blind_stats = {}
            for nmodes, pte_matrix in [(6, pte_matrix_6), (20, pte_matrix_20)]:
                valid_ptes = pte_matrix[~np.isnan(pte_matrix)]

                if len(valid_ptes) > 0:
                    frac_healthy = float(
                        np.sum((valid_ptes >= 0.05) & (valid_ptes <= 0.95)) / len(valid_ptes)
                    )
                    pte_at_fid = float(pte_matrix[fid_idx_min, fid_idx_max])
                else:
                    frac_healthy = float("nan")
                    pte_at_fid = float("nan")

                blind_stats[f"n{nmodes}"] = {
                    "pte_at_fiducial_cut": pte_at_fid,
                    "fraction_healthy": frac_healthy,
                    "n_scale_cuts_evaluated": int(len(valid_ptes)),
                }

                print(f"  {blind} n={nmodes}: PTE={pte_at_fid:.4f}, healthy={frac_healthy:.1%}")

                # Generate figure
                fig_name = f"figure_{version}_{blind}_n{nmodes}.png"
                fig_path = output_dir / fig_name
                _create_pte_figure(pte_matrix, theta_grid, fid_idx_min, fid_idx_max, fig_path)
                print(f"    Saved {fig_path}")

                # Copy to paper figures
                paper_fig_name = f"cosebis_pte_matrices_{version}_{blind}_n{nmodes}.png"
                paper_path = paper_dir / paper_fig_name
                shutil.copy2(fig_path, paper_path)

                artifacts[f"{version}_{blind}_n{nmodes}"] = fig_name

            evidence_versions[version]["blinds"][blind] = blind_stats

        # Compute minimum PTE across blinds for conservative assessment
        for nmodes in [6, 20]:
            pte_values = [
                evidence_versions[version]["blinds"].get(b, {}).get(f"n{nmodes}", {}).get("pte_at_fiducial_cut", float("nan"))
                for b in blinds
            ]
            pte_values = [p for p in pte_values if not np.isnan(p)]

            if pte_values:
                evidence_versions[version]["nmodes"][f"n{nmodes}"] = {
                    "pte_min": float(min(pte_values)),
                    "pte_max": float(max(pte_values)),
                    "blinds_available": [b for b in blinds if b in evidence_versions[version]["blinds"]],
                }

    # Build evidence
    spec_paths = snakemake.input["specs"]

    evidence_data = {
        "spec_id": "cosebis_pte_matrix",
        "spec_path": spec_paths[0],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "versions": evidence_versions,
            "fiducial_scale_cuts_arcmin": [fiducial_min_scale, fiducial_max_scale],
            "theta_grid_arcmin": theta_grid.tolist(),
        },
        "artifacts": artifacts,
    }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"\nSaved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
