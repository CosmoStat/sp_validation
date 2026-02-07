"""Harmonic vs configuration-space COSEBIS cross-validation.

Cross-validates COSEBIS E_n and B_n computed from two independent paths:
1. Harmonic: pseudo-C_ell (powspace nbins=32) -> cosebis_from_Cell()
2. Configuration: fine-binned xi_pm -> calculate_cosebis()

Agreement validates that the pseudo-C_ell estimation and COSEBIS integration
give consistent results.
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import treecorr
from astropy.io import fits
from scipy import stats

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


def _chi2_pte(residual, covariance):
    """Compute chi-squared and PTE for a residual vector.

    Falls back to pseudo-inverse if covariance is singular (possible for
    propagated harmonic covariance).
    """
    try:
        chi2 = float(residual @ np.linalg.solve(covariance, residual))
    except np.linalg.LinAlgError:
        chi2 = float(residual @ np.linalg.pinv(covariance) @ residual)
    pte = float(stats.chi2.sf(chi2, len(residual)))
    return chi2, pte


def _compute_harmonic_cosebis(pseudo_cl_path, pseudo_cov_path, nmodes, theta_min, theta_max):
    """Compute COSEBIS from pseudo-C_ell and propagate covariance."""
    # Load pseudo-Cl
    with fits.open(pseudo_cl_path) as hdul:
        data = hdul["PSEUDO_CELL"].data
        ell = np.asarray(data["ELL"], dtype=float)
        cl_ee = np.asarray(data["EE"], dtype=float)
        cl_bb = np.asarray(data["BB"], dtype=float)

    # COSEBIS from C_ell (no ell cuts — use full powspace range)
    theta_grid = np.logspace(np.log10(theta_min), np.log10(theta_max), 5000)
    cosebis_obj = COSEBIS(theta_min, theta_max, nmodes)

    ce_harm, cb_harm = cosebis_obj.cosebis_from_Cell(
        ell=ell, Cell_E=cl_ee, Cell_B=cl_bb, theta=theta_grid, cache=True,
    )

    # Build transform matrices and propagate covariance
    T_E, T_B = _build_transforms(cosebis_obj, ell, theta_grid)

    cov_cell = _build_cell_covariance(pseudo_cov_path)
    n_ell = len(ell)
    transform = np.zeros((2 * nmodes, 2 * n_ell))
    transform[:nmodes, :n_ell] = T_E
    transform[nmodes:, n_ell:] = T_B
    cov_harmonic = transform @ cov_cell @ transform.T

    return ce_harm, cb_harm, cov_harmonic


def _compute_config_cosebis(xi_path, cov_path, nmodes, theta_min, theta_max, config):
    """Compute COSEBIS from fine-binned xi_pm."""
    min_sep_int = float(config["fiducial"]["min_sep_int"])
    max_sep_int = float(config["fiducial"]["max_sep_int"])
    nbins_int = int(config["fiducial"]["nbins_int"])

    gg = treecorr.GGCorrelation(
        min_sep=min_sep_int, max_sep=max_sep_int, nbins=nbins_int, sep_units="arcmin",
    )
    gg.read(xi_path)

    scale_cut = (theta_min, theta_max)
    results = calculate_cosebis(gg, nmodes=nmodes, scale_cuts=[scale_cut], cov_path=cov_path)
    result = results[scale_cut]

    return result["En"], result["Bn"], result["cov"]


def _make_figure(datasets, nmodes):
    """Create comparison figure with main panels + residuals."""
    modes = np.arange(1, nmodes + 1)

    fig = plt.figure(figsize=(14, 5 * len(datasets)))
    outer = GridSpec(len(datasets), 2, hspace=0.35, wspace=0.25)

    for row, data in enumerate(datasets):
        for col, (key, ylabel) in enumerate([("E", r"$E_n$"), ("B", r"$B_n$")]):
            sub = GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[row, col], height_ratios=[3, 1], hspace=0.05,
            )
            ax_main = fig.add_subplot(sub[0, 0])
            ax_res = fig.add_subplot(sub[1, 0], sharex=ax_main)

            config_vals = data[f"c{key.lower()}_config"]
            harm_vals = data[f"c{key.lower()}_harm"]
            sigma_config = data[f"sigma_config_{key}"]
            sigma_harm = data[f"sigma_harm_{key}"]
            residual = data[f"residual_{key}"]
            chi2 = data[f"chi2_{key}"]
            pte = data[f"pte_{key}"]

            ax_main.errorbar(
                modes - 0.1, config_vals, yerr=sigma_config,
                fmt="o", color="C0", mfc="C0", ms=4, alpha=0.9, label="Config-space",
            )
            ax_main.errorbar(
                modes + 0.1, harm_vals, yerr=sigma_harm,
                fmt="s", color="C1", mfc="white", ms=4, alpha=0.9, label="Harmonic-space",
            )

            ax_main.axhline(0.0, color="black", lw=0.8, alpha=0.6)
            ax_main.set_ylabel(ylabel)
            ax_main.grid(True, alpha=0.2)
            ax_main.set_title(f"{data['label']} {key}-modes")

            sigma_safe = np.where(sigma_config > 0, sigma_config, np.inf)
            ax_res.plot(modes, residual / sigma_safe, marker="o", color="C2", ms=3, lw=1.0)
            ax_res.axhline(0.0, color="black", lw=0.8, alpha=0.6)
            ax_res.fill_between([0.5, nmodes + 0.5], -2, 2, color="0.9", alpha=0.5, zorder=0)
            ax_res.set_ylabel(r"$\Delta/\sigma$")
            ax_res.set_xlabel("Mode $n$")
            ax_res.set_ylim(-5, 5)
            ax_res.grid(True, alpha=0.2)

            ax_main.text(
                0.02, 0.95,
                rf"$\chi^2 = {chi2:.1f}$, PTE $= {pte:.3f}$" + f"\n(dof = {nmodes})",
                transform=ax_main.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

            if row == 0 and col == 0:
                ax_main.legend(fontsize=9)

            plt.setp(ax_main.get_xticklabels(), visible=False)

    fig.suptitle("COSEBIS: harmonic vs configuration-space cross-validation", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main():
    config = snakemake.config
    nmodes = int(config["fiducial"]["nmodes"])
    theta_min = float(config["cosebis"]["theta_min"])
    theta_max = float(config["cosebis"]["theta_max"])
    version_labels = config["plotting"].get("version_labels", {})

    # Only leak-corrected versions
    versions = [v for v in config["versions"] if "_leak_corr" in v]

    # Resolve inputs by version (ensure lists for single-version case)
    pseudo_cl_inputs = snakemake.input["pseudo_cl"]
    pseudo_cov_inputs = snakemake.input["pseudo_cl_cov"]
    xi_inputs = snakemake.input["xi_integration"]
    cov_inputs = snakemake.input["cov_integration"]

    if isinstance(pseudo_cl_inputs, str):
        pseudo_cl_inputs = [pseudo_cl_inputs]
        pseudo_cov_inputs = [pseudo_cov_inputs]
        xi_inputs = [xi_inputs]
        cov_inputs = [cov_inputs]

    datasets = []
    all_stats = {}

    for idx, version in enumerate(versions):
        print(f"Processing {version}...")

        ce_harm, cb_harm, cov_harmonic = _compute_harmonic_cosebis(
            pseudo_cl_inputs[idx], pseudo_cov_inputs[idx], nmodes, theta_min, theta_max,
        )

        ce_config, cb_config, cov_config = _compute_config_cosebis(
            xi_inputs[idx], cov_inputs[idx], nmodes, theta_min, theta_max, config,
        )

        cov_config_E = cov_config[:nmodes, :nmodes]
        cov_config_B = cov_config[nmodes:, nmodes:]

        sigma_config_E = np.sqrt(np.clip(np.diag(cov_config_E), 0, None))
        sigma_config_B = np.sqrt(np.clip(np.diag(cov_config_B), 0, None))
        sigma_harm_E = np.sqrt(np.clip(np.diag(cov_harmonic[:nmodes, :nmodes]), 0, None))
        sigma_harm_B = np.sqrt(np.clip(np.diag(cov_harmonic[nmodes:, nmodes:]), 0, None))

        residual_E = ce_harm - ce_config
        residual_B = cb_harm - cb_config

        chi2_E, pte_E = _chi2_pte(residual_E, cov_config_E)
        chi2_B, pte_B = _chi2_pte(residual_B, cov_config_B)

        datasets.append({
            "version": version,
            "label": version_label(version, version_labels),
            "ce_harm": ce_harm, "cb_harm": cb_harm,
            "ce_config": ce_config, "cb_config": cb_config,
            "sigma_config_E": sigma_config_E, "sigma_config_B": sigma_config_B,
            "sigma_harm_E": sigma_harm_E, "sigma_harm_B": sigma_harm_B,
            "residual_E": residual_E, "residual_B": residual_B,
            "chi2_E": chi2_E, "pte_E": pte_E,
            "chi2_B": chi2_B, "pte_B": pte_B,
        })

        all_stats[version] = {
            "E_modes": {"chi2": chi2_E, "pte": pte_E, "dof": nmodes},
            "B_modes": {"chi2": chi2_B, "pte": pte_B, "dof": nmodes},
        }
        print(f"  E-modes: chi2={chi2_E:.2f}, PTE={pte_E:.4f}")
        print(f"  B-modes: chi2={chi2_B:.2f}, PTE={pte_B:.4f}")

    # Create figure
    fig = _make_figure(datasets, nmodes)

    fig_path = Path(snakemake.output["figure"])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_path}")
    plt.close(fig)

    # Write evidence
    spec_paths = snakemake.input["specs"]

    evidence = {
        "spec_id": "harmonic_config_cosebis_comparison",
        "spec_path": spec_paths[0],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "nmodes": nmodes,
            "theta_range": [theta_min, theta_max],
            "powspace_nbins": int(config["cl"]["n_ell_bins"]),
            "statistics": all_stats,
        },
        "artifacts": {
            "figure": "figure.png",
        },
    }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence, f, indent=2, default=str)
    print(f"Saved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
