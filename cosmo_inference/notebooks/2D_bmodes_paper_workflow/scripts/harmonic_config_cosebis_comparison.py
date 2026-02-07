"""Harmonic vs configuration-space COSEBIS cross-validation.

Cross-validates COSEBIS E_n and B_n computed from two independent paths:
1. Harmonic: pseudo-C_ell (powspace nbins=32) -> cosebis_from_Cell()
2. Configuration: fine-binned xi_pm -> calculate_cosebis()

Visual cross-check only — no chi2/PTE on differences (cross-covariance unknown).
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import treecorr
from astropy.io import fits

from cosmo_numba.B_modes.cosebis import COSEBIS
from sp_validation.b_modes import calculate_cosebis
from plotting_utils import PAPER_MPLSTYLE, version_label

plt.style.use(PAPER_MPLSTYLE)


def _build_cell_covariance(cov_path):
    """Build joint [EE, BB] covariance from FITS pseudo-Cl covariance file."""
    with fits.open(cov_path) as hdul:
        cov_EE_EE = hdul["COVAR_EE_EE"].data
        cov_EE_BB = hdul["COVAR_EE_BB"].data
        cov_BB_EE = hdul["COVAR_BB_EE"].data
        cov_BB_BB = hdul["COVAR_BB_BB"].data

    n_ell = cov_EE_EE.shape[0]
    cov = np.zeros((2 * n_ell, 2 * n_ell))
    cov[:n_ell, :n_ell] = cov_EE_EE
    cov[:n_ell, n_ell:] = cov_EE_BB
    cov[n_ell:, :n_ell] = cov_BB_EE
    cov[n_ell:, n_ell:] = cov_BB_BB
    return cov


def _build_transforms(cosebis_obj, ell, theta_grid):
    """Build linear transform matrices T_E, T_B via basis-vector approach.

    T_E[n, i] = E_n response to unit C_ell in bin i (B=0).
    T_B[n, i] = B_n response to unit C_ell in bin i (E=0).
    """
    n_ell = len(ell)
    nmodes = cosebis_obj.Nmax
    zeros = np.zeros(n_ell)

    T_E = np.zeros((nmodes, n_ell))
    T_B = np.zeros((nmodes, n_ell))

    for idx in range(n_ell):
        basis = np.zeros(n_ell)
        basis[idx] = 1.0

        ce_basis, _ = cosebis_obj.cosebis_from_Cell(
            ell=ell, Cell_E=basis, Cell_B=zeros, theta=theta_grid, cache=True,
        )
        _, cb_basis = cosebis_obj.cosebis_from_Cell(
            ell=ell, Cell_E=zeros, Cell_B=basis, theta=theta_grid, cache=True,
        )

        T_E[:, idx] = ce_basis
        T_B[:, idx] = cb_basis

    return T_E, T_B


def _compute_harmonic_cosebis(pseudo_cl_path, pseudo_cov_path, nmodes, theta_min, theta_max):
    """Compute COSEBIS from pseudo-C_ell and propagate covariance."""
    with fits.open(pseudo_cl_path) as hdul:
        data = hdul["PSEUDO_CELL"].data
        ell = np.asarray(data["ELL"], dtype=float)
        cl_ee = np.asarray(data["EE"], dtype=float)
        cl_bb = np.asarray(data["BB"], dtype=float)

    theta_grid = np.logspace(np.log10(theta_min), np.log10(theta_max), 5000)
    cosebis_obj = COSEBIS(theta_min, theta_max, nmodes)

    ce_harm, cb_harm = cosebis_obj.cosebis_from_Cell(
        ell=ell, Cell_E=cl_ee, Cell_B=cl_bb, theta=theta_grid, cache=True,
    )

    T_E, T_B = _build_transforms(cosebis_obj, ell, theta_grid)

    cov_cell = _build_cell_covariance(pseudo_cov_path)
    n_ell = len(ell)
    transform = np.zeros((2 * nmodes, 2 * n_ell))
    transform[:nmodes, :n_ell] = T_E
    transform[nmodes:, n_ell:] = T_B
    cov_harmonic = transform @ cov_cell @ transform.T

    return ce_harm, cb_harm, cov_harmonic


def _compute_config_cosebis(xi_path, cov_path, nmodes, scale_cuts, config):
    """Compute COSEBIS from fine-binned xi_pm for multiple scale cuts.

    Returns dict keyed by scale_cut tuple -> (En, Bn, cov).
    """
    min_sep_int = float(config["fiducial"]["min_sep_int"])
    max_sep_int = float(config["fiducial"]["max_sep_int"])
    nbins_int = int(config["fiducial"]["nbins_int"])

    gg = treecorr.GGCorrelation(
        min_sep=min_sep_int, max_sep=max_sep_int, nbins=nbins_int, sep_units="arcmin",
    )
    gg.read(xi_path)

    cut_list = list(scale_cuts.values())
    results = calculate_cosebis(gg, nmodes=nmodes, scale_cuts=cut_list, cov_path=cov_path)

    out = {}
    for key, cut in scale_cuts.items():
        r = results[cut]
        out[key] = (r["En"], r["Bn"], r["cov"])
    return out


def _make_data_vector_figure(harm_data, config_data, nmodes, scale_cuts):
    """Data vector figure for fiducial catalog: E and B from both methods.

    Two rows (E-modes, B-modes), each showing both scale cuts with
    config-space and harmonic-space overlaid.
    B-modes in B_n/sigma units. E-modes in raw units.
    """
    modes = np.arange(1, nmodes + 1)
    colors = sns.color_palette("colorblind", 2)
    method_colors = {"config": colors[0], "harmonic": colors[1]}

    fig, axes = plt.subplots(2, 1, figsize=(7.24, 7.24 * 0.6), sharex=True)

    # --- E-modes (raw units) ---
    ax_e = axes[0]
    offset_map = {"full": -0.15, "fiducial": 0.15}
    marker_map = {"full": "o", "fiducial": "s"}

    for scale_key in ["full", "fiducial"]:
        cut = scale_cuts[scale_key]
        offset = offset_map[scale_key]
        marker = marker_map[scale_key]

        ce_config, _, cov_config = config_data[scale_key]
        sigma_config_E = np.sqrt(np.clip(np.diag(cov_config[:nmodes, :nmodes]), 0, None))

        ce_harm = harm_data[scale_key]["En"]
        sigma_harm_E = harm_data[scale_key]["sigma_E"]

        label_c = rf"Config $\theta \in [{cut[0]:.0f}, {cut[1]:.0f}]'$"
        label_h = rf"Harmonic $\theta \in [{cut[0]:.0f}, {cut[1]:.0f}]'$"

        ax_e.errorbar(
            modes + offset - 0.05, ce_config, yerr=sigma_config_E,
            fmt=marker, color=method_colors["config"], mfc=method_colors["config"],
            ms=4, alpha=0.8, capsize=2, capthick=0.8, elinewidth=0.8,
        )
        ax_e.errorbar(
            modes + offset + 0.05, ce_harm, yerr=sigma_harm_E,
            fmt=marker, color=method_colors["harmonic"], mfc="white",
            ms=4, alpha=0.8, capsize=2, capthick=0.8, elinewidth=0.8,
        )

    ax_e.axhline(0.0, color="black", lw=0.8, alpha=0.6)
    ax_e.set_ylabel(r"$E_n$")
    ax_e.set_xlim(0.5, nmodes + 0.5)
    ax_e.tick_params(axis="both", width=0.5, length=3)
    ax_e.set_title("E-modes", fontsize=10)

    # Manual legend for methods
    ax_e.errorbar([], [], fmt="o", color=method_colors["config"], mfc=method_colors["config"],
                  ms=4, label="Config-space")
    ax_e.errorbar([], [], fmt="o", color=method_colors["harmonic"], mfc="white",
                  ms=4, label="Harmonic-space")
    ax_e.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # --- B-modes (B_n / sigma units) ---
    ax_b = axes[1]

    for scale_key in ["full", "fiducial"]:
        cut = scale_cuts[scale_key]
        offset = offset_map[scale_key]
        marker = marker_map[scale_key]

        _, cb_config, cov_config = config_data[scale_key]
        sigma_config_B = np.sqrt(np.clip(np.diag(cov_config[nmodes:, nmodes:]), 0, None))
        sigma_safe = np.where(sigma_config_B > 0, sigma_config_B, 1.0)

        cb_harm = harm_data[scale_key]["Bn"]
        sigma_harm_B = harm_data[scale_key]["sigma_B"]
        sigma_harm_safe = np.where(sigma_harm_B > 0, sigma_harm_B, 1.0)

        ax_b.errorbar(
            modes + offset - 0.05, cb_config / sigma_safe, yerr=np.ones(nmodes),
            fmt=marker, color=method_colors["config"], mfc=method_colors["config"],
            ms=4, alpha=0.8, capsize=2, capthick=0.8, elinewidth=0.8,
        )
        ax_b.errorbar(
            modes + offset + 0.05, cb_harm / sigma_harm_safe, yerr=np.ones(nmodes),
            fmt=marker, color=method_colors["harmonic"], mfc="white",
            ms=4, alpha=0.8, capsize=2, capthick=0.8, elinewidth=0.8,
        )

    ax_b.axhline(0.0, color="black", lw=0.8, alpha=0.6)
    ax_b.set_ylabel(r"$B_n / \sigma_n$")
    ax_b.set_xlabel("COSEBIS mode $n$")
    ax_b.set_xticks(np.arange(1, nmodes + 1))
    ax_b.set_xlim(0.5, nmodes + 0.5)
    ax_b.tick_params(axis="both", width=0.5, length=3)
    ax_b.set_title("B-modes", fontsize=10)

    # Scale-cut legend
    for scale_key in ["full", "fiducial"]:
        cut = scale_cuts[scale_key]
        marker = marker_map[scale_key]
        ax_b.plot([], [], marker=marker, color="0.4", ls="none", ms=4,
                  label=rf"$\theta \in [{cut[0]:.0f}, {cut[1]:.0f}]'$")
    ax_b.legend(fontsize=8, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    return fig


def _make_version_comparison_figure(all_version_data, nmodes, scale_cuts, version_labels_map):
    """Version comparison: all versions, both methods, B-modes in B_n/sigma.

    Two rows: full range (top), fiducial (bottom).
    Each version is a color. Config=filled marker, harmonic=open marker.
    """
    modes = np.arange(1, nmodes + 1)
    n_versions = len(all_version_data)
    colors = sns.color_palette("colorblind", n_versions)

    fig, axes = plt.subplots(2, 1, figsize=(7.24, 7.24 * 0.6), sharex=True)
    scale_labels = {"full": "Full", "fiducial": "Fiducial"}

    x_offsets = np.linspace(-0.2, 0.2, n_versions)

    for panel_idx, (scale_key, ax) in enumerate(
        zip(["full", "fiducial"], axes)
    ):
        cut = scale_cuts[scale_key]

        for i, (version, vdata) in enumerate(all_version_data.items()):
            color = colors[i]
            offset = x_offsets[i]
            label = version_label(version, version_labels_map)

            _, cb_config, cov_config = vdata["config"][scale_key]
            sigma_config_B = np.sqrt(np.clip(np.diag(cov_config[nmodes:, nmodes:]), 0, None))
            sigma_safe = np.where(sigma_config_B > 0, sigma_config_B, 1.0)

            cb_harm = vdata["harm"][scale_key]["Bn"]
            sigma_harm_B = vdata["harm"][scale_key]["sigma_B"]
            sigma_harm_safe = np.where(sigma_harm_B > 0, sigma_harm_B, 1.0)

            # Config-space: filled marker
            ax.errorbar(
                modes + offset - 0.03, cb_config / sigma_safe, yerr=np.ones(nmodes),
                fmt="o", color=color, mfc=color, ms=3.5, alpha=0.85,
                capsize=1.5, capthick=0.6, elinewidth=0.6,
                label=f"{label} (config)" if panel_idx == 0 else None,
            )
            # Harmonic-space: open marker
            ax.errorbar(
                modes + offset + 0.03, cb_harm / sigma_harm_safe, yerr=np.ones(nmodes),
                fmt="s", color=color, mfc="white", ms=3.5, alpha=0.85,
                capsize=1.5, capthick=0.6, elinewidth=0.6, mew=0.8,
                label=f"{label} (harmonic)" if panel_idx == 0 else None,
            )

        ax.axhline(0.0, color="black", lw=0.8, alpha=0.6)
        ax.set_ylabel(r"$B_n / \sigma_n$")
        ax.set_xlim(0.5, nmodes + 0.5)
        ax.tick_params(axis="both", width=0.5, length=3)

        ax.text(
            0.98, 0.95,
            rf"{scale_labels[scale_key]} $\theta \in [{cut[0]:.0f}, {cut[1]:.0f}]'$",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    axes[1].set_xlabel("COSEBIS mode $n$")
    axes[1].set_xticks(np.arange(1, nmodes + 1))

    axes[0].legend(
        fontsize=7, loc="upper left", ncol=2,
        frameon=True, framealpha=0.9,
        handletextpad=0.3, columnspacing=0.8,
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08)
    return fig


def main():
    config = snakemake.config
    nmodes = int(config["fiducial"]["nmodes"])
    version_labels_map = config["plotting"].get("version_labels", {})

    fiducial_version = config["fiducial"]["version"]
    fiducial_scale_cut = (
        float(config["fiducial"]["fiducial_min_scale"]),
        float(config["fiducial"]["fiducial_max_scale"]),
    )
    full_scale_cut = (
        float(config["cosebis"]["theta_min"]),
        float(config["cosebis"]["theta_max"]),
    )
    scale_cuts = {"full": full_scale_cut, "fiducial": fiducial_scale_cut}

    versions = [v for v in config["versions"] if "_leak_corr" in v]

    pseudo_cl_inputs = snakemake.input["pseudo_cl"]
    pseudo_cov_inputs = snakemake.input["pseudo_cl_cov"]
    xi_inputs = snakemake.input["xi_integration"]
    cov_inputs = snakemake.input["cov_integration"]

    if isinstance(pseudo_cl_inputs, str):
        pseudo_cl_inputs = [pseudo_cl_inputs]
        pseudo_cov_inputs = [pseudo_cov_inputs]
        xi_inputs = [xi_inputs]
        cov_inputs = [cov_inputs]

    # Compute COSEBIS for all versions, both methods, both scale cuts
    all_version_data = {}

    for idx, version in enumerate(versions):
        print(f"Processing {version}...")

        # Config-space: multiple scale cuts in one call
        config_results = _compute_config_cosebis(
            xi_inputs[idx], cov_inputs[idx], nmodes, scale_cuts, config,
        )

        # Harmonic: compute per scale cut (different COSEBIS object per theta range)
        harm_results = {}
        for scale_key, cut in scale_cuts.items():
            ce_h, cb_h, cov_h = _compute_harmonic_cosebis(
                pseudo_cl_inputs[idx], pseudo_cov_inputs[idx], nmodes, cut[0], cut[1],
            )
            harm_results[scale_key] = {
                "En": ce_h, "Bn": cb_h,
                "sigma_E": np.sqrt(np.clip(np.diag(cov_h[:nmodes, :nmodes]), 0, None)),
                "sigma_B": np.sqrt(np.clip(np.diag(cov_h[nmodes:, nmodes:]), 0, None)),
            }

        all_version_data[version] = {"config": config_results, "harm": harm_results}

    # --- Data vector figure (fiducial version only) ---
    fid_data = all_version_data[fiducial_version]
    fig_dv = _make_data_vector_figure(
        fid_data["harm"], fid_data["config"], nmodes, scale_cuts,
    )
    fig_dv_path = Path(snakemake.output["figure"])
    fig_dv_path.parent.mkdir(parents=True, exist_ok=True)
    fig_dv.savefig(fig_dv_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_dv_path}")
    plt.close(fig_dv)

    # --- Version comparison figure ---
    fig_vc = _make_version_comparison_figure(
        all_version_data, nmodes, scale_cuts, version_labels_map,
    )
    fig_vc_path = Path(snakemake.output["figure_versions"])
    fig_vc.savefig(fig_vc_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_vc_path}")
    plt.close(fig_vc)

    # --- Evidence ---
    evidence = {
        "spec_id": "harmonic_config_cosebis_comparison",
        "spec_path": snakemake.input["specs"][0],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "nmodes": nmodes,
            "scale_cuts": {k: list(v) for k, v in scale_cuts.items()},
            "powspace_nbins": int(config["cl"]["n_ell_bins"]),
            "versions_compared": versions,
            "note": (
                "Visual cross-check only. No chi2/PTE on differences because "
                "cross-covariance between harmonic and config-space methods is unknown. "
                "E-mode FFT-log W_n(ell) has known accuracy limitations at high ell."
            ),
        },
        "artifacts": {
            "figure": "figure.png",
            "figure_versions": "figure_versions.png",
        },
    }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence, f, indent=2, default=str)
    print(f"Saved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
