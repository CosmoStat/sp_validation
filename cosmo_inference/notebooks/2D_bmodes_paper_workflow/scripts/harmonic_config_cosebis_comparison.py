"""Harmonic vs configuration-space COSEBIS cross-validation.

Cross-validates COSEBIS E_n and B_n computed from two independent paths:
1. Harmonic: pseudo-C_ell (powspace) -> cosebis_from_Cell() with 100k-point FFT-log
2. Configuration: fine-binned xi_pm -> calculate_cosebis()

Produces 9 figures per the standard data vector pattern (1 paper + 4 corrected
+ 4 uncorrected), plus a version comparison figure.

Angular range parameterized via snakemake.params.scale_cut (full or fiducial).
Modes > reliable_mode_max shown with gray band.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import treecorr
from astropy.io import fits

from cosmo_numba.B_modes.cosebis import COSEBIS
from sp_validation.b_modes import calculate_cosebis
from plotting_utils import (
    FIG_WIDTH_FULL,
    FIG_WIDTH_SINGLE,
    PAPER_MPLSTYLE,
    compute_chi2_pte,
    iter_version_figures,
    version_label,
)

plt.style.use(PAPER_MPLSTYLE)

# At 32 bins, modes > 6 are unreliable. At 96+ bins, modes 1-7 are recovered
# to <0.3% (harmonic ceiling). Mode 8 is marginal (0.8%); modes 9+ diverge
# due to W_n(ell) numerical precision at ~10^-14 amplitudes, not binning.
_RELIABLE_MODE_MAX_BY_NBINS = {32: 6, 96: 8}


def _safe_sigma(covariance_diag):
    """Standard deviation from covariance diagonal, clamped to avoid division by zero."""
    sigma = np.sqrt(np.clip(covariance_diag, 0, None))
    return np.where(sigma > 0, sigma, 1.0)


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


def _build_transforms(cosebis_obj, ell):
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
            ell=ell, Cell_E=basis, Cell_B=zeros, cache=True,
        )
        _, cb_basis = cosebis_obj.cosebis_from_Cell(
            ell=ell, Cell_E=zeros, Cell_B=basis, cache=True,
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

    cosebis_obj = COSEBIS(theta_min, theta_max, nmodes)

    ce_harm, cb_harm = cosebis_obj.cosebis_from_Cell(
        ell=ell, Cell_E=cl_ee, Cell_B=cl_bb, cache=True,
    )

    T_E, T_B = _build_transforms(cosebis_obj, ell)

    cov_cell = _build_cell_covariance(pseudo_cov_path)
    n_ell = len(ell)
    transform = np.zeros((2 * nmodes, 2 * n_ell))
    transform[:nmodes, :n_ell] = T_E
    transform[nmodes:, n_ell:] = T_B
    cov_harmonic = transform @ cov_cell @ transform.T

    return ce_harm, cb_harm, cov_harmonic


def _compute_config_cosebis(xi_path, cov_path, nmodes, scale_cut, config):
    """Compute COSEBIS from fine-binned xi_pm for a single scale cut.

    Returns (En, Bn, cov).
    """
    min_sep_int = float(config["fiducial"]["min_sep_int"])
    max_sep_int = float(config["fiducial"]["max_sep_int"])
    nbins_int = int(config["fiducial"]["nbins_int"])

    gg = treecorr.GGCorrelation(
        min_sep=min_sep_int, max_sep=max_sep_int, nbins=nbins_int, sep_units="arcmin",
    )
    gg.read(xi_path)

    results = calculate_cosebis(gg, nmodes=nmodes, scale_cuts=[scale_cut], cov_path=cov_path)
    r = results[scale_cut]
    return r["En"], r["Bn"], r["cov"]


def _shade_unreliable(ax, nmodes, reliable_mode_max=6):
    """Shade the unreliable region (modes > reliable_mode_max) with gray band."""
    ax.axvspan(reliable_mode_max + 0.5, nmodes + 0.5, color="0.90", alpha=0.5, zorder=0)


def _make_data_vector_figure(harm_data, config_data, nmodes, scale_cut, title=None, reliable_mode_max=6):
    """Data vector figure: E and B from both methods.

    Two rows (E-modes, B-modes), full angular range only.
    E-modes in raw units. B-modes in B_n/sigma units.
    Gray band over unreliable modes.
    """
    modes = np.arange(1, nmodes + 1)
    colors = sns.color_palette("colorblind", 2)
    c_cfg, c_harm = colors[0], colors[1]

    ce_config, cb_config, cov_config = config_data
    sigma_config_E = np.sqrt(np.clip(np.diag(cov_config[:nmodes, :nmodes]), 0, None))
    sigma_config_B = _safe_sigma(np.diag(cov_config[nmodes:, nmodes:]))

    ce_harm, cb_harm = harm_data["En"], harm_data["Bn"]
    sigma_harm_E = harm_data["sigma_E"]
    sigma_harm_B = np.where(harm_data["sigma_B"] > 0, harm_data["sigma_B"], 1.0)

    fig, (ax_e, ax_b) = plt.subplots(2, 1, figsize=(FIG_WIDTH_SINGLE, FIG_WIDTH_SINGLE * 0.85), sharex=True)

    # --- E-modes (factored by 10^10 for readability) ---
    e_scale = 1e10
    ax_e.errorbar(
        modes - 0.1, ce_config * e_scale, yerr=sigma_config_E * e_scale,
        fmt="o", color=c_cfg, mfc=c_cfg, ms=4, alpha=0.8,
        capsize=2, capthick=0.8, elinewidth=0.8, label="Config-space",
    )
    ax_e.errorbar(
        modes + 0.1, ce_harm * e_scale, yerr=sigma_harm_E * e_scale,
        fmt="o", color=c_harm, mfc="white", ms=4, alpha=0.8,
        capsize=2, capthick=0.8, elinewidth=0.8, label="Harmonic-space",
    )

    ax_e.axhline(0.0, color="black", lw=0.8, alpha=0.6)
    _shade_unreliable(ax_e, nmodes, reliable_mode_max)
    ax_e.set_ylabel(r"$E_n \times 10^{10}$")
    ax_e.set_xlim(0.5, nmodes + 0.5)
    ax_e.tick_params(axis="both", width=0.5, length=3)
    if title:
        ax_e.set_title(title)
    ax_e.legend(loc="upper center", framealpha=0.9)

    # --- B-modes (B_n / sigma units) ---
    ax_b.errorbar(
        modes - 0.1, cb_config / sigma_config_B, yerr=np.ones(nmodes),
        fmt="o", color=c_cfg, mfc=c_cfg, ms=4, alpha=0.8,
        capsize=2, capthick=0.8, elinewidth=0.8,
    )
    ax_b.errorbar(
        modes + 0.1, cb_harm / sigma_harm_B, yerr=np.ones(nmodes),
        fmt="o", color=c_harm, mfc="white", ms=4, alpha=0.8,
        capsize=2, capthick=0.8, elinewidth=0.8,
    )

    ax_b.axhline(0.0, color="black", lw=0.8, alpha=0.6)
    _shade_unreliable(ax_b, nmodes, reliable_mode_max)
    ax_b.set_ylabel(r"$B_n / \sigma_n$")
    ax_b.set_xlabel("COSEBIS mode $n$")
    ax_b.set_xticks(np.arange(1, nmodes + 1))
    ax_b.set_xlim(0.5, nmodes + 0.5)
    ax_b.tick_params(axis="both", width=0.5, length=3)

    ax_b.text(
        0.98, 0.95,
        rf"$\theta = {scale_cut[0]:.0f}$--${scale_cut[1]:.0f}'$",
        transform=ax_b.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    return fig


def _make_version_comparison_figure(all_version_data, nmodes, scale_cut, version_labels_map, reliable_mode_max=6):
    """Version comparison: all versions, both methods, B-modes in B_n/sigma.

    Single panel, full angular range only.
    Config=filled marker, harmonic=open marker. Gray band over unreliable modes.
    """
    modes = np.arange(1, nmodes + 1)
    n_versions = len(all_version_data)
    colors = sns.color_palette("colorblind", n_versions)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_FULL, FIG_WIDTH_FULL * 0.35))
    x_offsets = np.linspace(-0.2, 0.2, n_versions)

    for i, (version, vdata) in enumerate(all_version_data.items()):
        color = colors[i]
        offset = x_offsets[i]
        label = version_label(version, version_labels_map)

        _, cb_config, cov_config = vdata["config"]
        sigma_config_B = _safe_sigma(np.diag(cov_config[nmodes:, nmodes:]))

        cb_harm = vdata["harm"]["Bn"]
        sigma_harm_B = np.where(vdata["harm"]["sigma_B"] > 0, vdata["harm"]["sigma_B"], 1.0)

        ax.errorbar(
            modes + offset - 0.03, cb_config / sigma_config_B, yerr=np.ones(nmodes),
            fmt="o", color=color, mfc=color, ms=3.5, alpha=0.85,
            capsize=1.5, capthick=0.6, elinewidth=0.6,
            label=f"{label} (config)",
        )
        ax.errorbar(
            modes + offset + 0.03, cb_harm / sigma_harm_B, yerr=np.ones(nmodes),
            fmt="s", color=color, mfc="white", ms=3.5, alpha=0.85,
            capsize=1.5, capthick=0.6, elinewidth=0.6, mew=0.8,
            label=f"{label} (harmonic)",
        )

    ax.axhline(0.0, color="black", lw=0.8, alpha=0.6)
    _shade_unreliable(ax, nmodes, reliable_mode_max)
    ax.set_ylabel(r"$B_n / \sigma_n$")
    ax.set_xlabel("COSEBIS mode $n$")
    ax.set_xticks(np.arange(1, nmodes + 1))
    ax.set_xlim(0.5, nmodes + 0.5)
    ax.tick_params(axis="both", width=0.5, length=3)

    ax.text(
        0.98, 0.95,
        rf"$\theta = {scale_cut[0]:.0f}$--${scale_cut[1]:.0f}'$",
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.legend(
        loc="upper center", ncol=4,
        frameon=True, framealpha=0.9,
        handletextpad=0.3, columnspacing=0.8,
    )

    plt.tight_layout()
    return fig


def main():
    config = snakemake.config
    nmodes = int(config["fiducial"]["nmodes"])
    version_labels_map = config["plotting"].get("version_labels", {})

    fiducial_version = config["fiducial"]["version"]
    scale_cut = tuple(snakemake.params.scale_cut)

    cosebis_nbins = int(config["cl"].get("cosebis_nbins", 32))
    reliable_mode_max = _RELIABLE_MODE_MAX_BY_NBINS.get(
        cosebis_nbins, 8 if cosebis_nbins >= 64 else 6,
    )
    print(f"Using {cosebis_nbins}-bin pseudo-Cl, reliable modes 1-{reliable_mode_max}")

    versions_leak_corr = [v for v in config["versions"] if "_leak_corr" in v]

    # Build input path lookups from snakemake inputs
    pseudo_cl_paths = {k: v for k, v in snakemake.input.items() if k.startswith("pseudo_cl_") and not k.startswith("pseudo_cl_cov_")}
    pseudo_cov_paths = {k: v for k, v in snakemake.input.items() if k.startswith("pseudo_cl_cov_")}
    xi_paths = {k: v for k, v in snakemake.input.items() if k.startswith("xi_")}
    cov_paths = {k: v for k, v in snakemake.input.items() if k.startswith("cov_")}

    output_dir = Path(snakemake.output["evidence"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track generated artifacts
    output = {}

    # Compute B-mode PTEs for leak-corrected versions only
    harmonic_ptes = {}
    config_ptes = {}

    # Cache leak-corrected results for version comparison figure
    all_version_data = {}

    # Generate all 9 per-version data vector figures
    for fig_spec in iter_version_figures(version_labels_map, fiducial_version):
        if fig_spec["leak_corrected"]:
            ver = fig_spec["version_leak_corr"]
        else:
            ver = fig_spec["version_uncorr"]

        pseudo_cl_key = f"pseudo_cl_{ver}"
        pseudo_cov_key = f"pseudo_cl_cov_{ver}"
        xi_key = f"xi_{ver}"
        cov_key = f"cov_{ver}"

        print(f"Processing {ver} ({'corrected' if fig_spec['leak_corrected'] else 'uncorrected'})...")

        # Config-space
        ce_cfg, cb_cfg, cov_cfg = _compute_config_cosebis(
            xi_paths[xi_key], cov_paths[cov_key], nmodes, scale_cut, config,
        )

        # Harmonic-space
        ce_h, cb_h, cov_h = _compute_harmonic_cosebis(
            pseudo_cl_paths[pseudo_cl_key], pseudo_cov_paths[pseudo_cov_key],
            nmodes, scale_cut[0], scale_cut[1],
        )
        harm = {
            "En": ce_h, "Bn": cb_h,
            "sigma_E": np.sqrt(np.clip(np.diag(cov_h[:nmodes, :nmodes]), 0, None)),
            "sigma_B": np.sqrt(np.clip(np.diag(cov_h[nmodes:, nmodes:]), 0, None)),
        }

        # Create figure
        fig = _make_data_vector_figure(
            harm, (ce_cfg, cb_cfg, cov_cfg), nmodes, scale_cut,
            title=fig_spec["title"], reliable_mode_max=reliable_mode_max,
        )
        fig_path = output_dir / fig_spec["filename"]
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"  Saved {fig_path}")
        plt.close(fig)

        output[fig_spec["filename"].replace(".png", "")] = fig_spec["filename"]

        # Copy paper figure
        if fig_spec["is_paper_figure"] and "paper_figure" in snakemake.output.keys():
            paper_path = Path(snakemake.output["paper_figure"])
            paper_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(paper_path, bbox_inches="tight")
            print(f"  Saved {paper_path}")

        # Compute B-mode PTEs for leak-corrected versions
        if fig_spec["leak_corrected"] and ver not in harmonic_ptes:
            n_rel = reliable_mode_max
            # Harmonic-space PTE
            cb_h_rel = cb_h[:n_rel]
            cov_h_B_rel = cov_h[nmodes:nmodes + n_rel, nmodes:nmodes + n_rel]
            chi2, pte, dof = compute_chi2_pte(cb_h_rel, cov_h_B_rel)
            harmonic_ptes[ver] = {"chi2": chi2, "pte": pte, "dof": dof}
            print(f"  Harmonic B-mode PTE (modes 1-{n_rel}): {pte:.4f} (chi2={chi2:.2f}, dof={dof})")
            # Config-space PTE
            cb_cfg_rel = cb_cfg[:n_rel]
            cov_cfg_B_rel = cov_cfg[nmodes:nmodes + n_rel, nmodes:nmodes + n_rel]
            chi2_c, pte_c, dof_c = compute_chi2_pte(cb_cfg_rel, cov_cfg_B_rel)
            config_ptes[ver] = {"chi2": chi2_c, "pte": pte_c, "dof": dof_c}
            print(f"  Config B-mode PTE (modes 1-{n_rel}): {pte_c:.4f} (chi2={chi2_c:.2f}, dof={dof_c})")

        # Cache leak-corrected results for version comparison figure
        if fig_spec["leak_corrected"] and ver not in all_version_data:
            all_version_data[ver] = {
                "config": (ce_cfg, cb_cfg, cov_cfg),
                "harm": harm,
            }

    fig_vc = _make_version_comparison_figure(
        all_version_data, nmodes, scale_cut, version_labels_map,
        reliable_mode_max=reliable_mode_max,
    )
    fig_vc_path = Path(snakemake.output["figure_versions"])
    fig_vc.savefig(fig_vc_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_vc_path}")
    plt.close(fig_vc)
    output["figure_versions"] = "figure_versions.png"

    # --- Evidence ---
    evidence = {
        "spec_id": "harmonic_config_cosebis_comparison",
        "spec_path": snakemake.input["specs"][0],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "nmodes": nmodes,
            "reliable_modes": reliable_mode_max,
            "scale_cut": list(scale_cut),
            "powspace_nbins": cosebis_nbins,
            "versions_compared": versions_leak_corr,
            "harmonic_b_mode_ptes": harmonic_ptes,
            "config_b_mode_ptes": config_ptes,
            "note": (
                f"Harmonic-space COSEBIS computed via cosmo_numba cosebis_from_Cell() "
                f"from {cosebis_nbins}-bin powspace pseudo-C_ell. "
                f"With {cosebis_nbins} bins, modes 1-{reliable_mode_max} are reliably "
                f"recovered (validated on GLASS mocks: E_n/config within 2%). "
                f"Higher modes are limited by W_n(ell) numerical precision at "
                f"~10^-14 amplitudes, not binning. "
                f"Harmonic B-mode PTEs use modes 1-{reliable_mode_max} with "
                f"propagated covariance."
            ),
        },
        "output": output,
    }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence, f, indent=2, default=str)
    print(f"Saved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
