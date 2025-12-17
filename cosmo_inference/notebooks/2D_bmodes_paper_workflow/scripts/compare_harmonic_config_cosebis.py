# %%
"""Cross-validate harmonic and configuration-space COSEBIS estimates."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import seaborn as sns
from IPython import get_ipython
from astropy.io import fits
from scipy import stats

from cosmo_numba.B_modes.cosebis import COSEBIS
from sp_validation.cosmo_val import CosmologyValidation


ipython = get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
else:
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")


plt.style.use(
    "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle"
)

sns.set_palette("husl", 3)

COSMO_VAL_DIR = "/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val"
CAT_CONFIG_PATH = f"{COSMO_VAL_DIR}/cat_config.yaml"
os.chdir(COSMO_VAL_DIR)


def _load_snakemake():
    if hasattr(sys, "ps1"):
        from snakemake_helpers import snakemake_interactive

        return snakemake_interactive(
            "output/harmonic_config_cosebis_comparison_all.png",
            str(Path.cwd()),
        )
    from snakemake.script import snakemake

    return snakemake


snakemake = _load_snakemake()
params = snakemake.params


def _version_label(version):
    return version.replace("SP_", "").replace("_leak_corr", "")


def _build_cell_covariance(cov_path):
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
    n_ell = len(ell)
    nmodes = cosebis_obj.Nmax
    zeros = np.zeros(n_ell)

    cosebis_obj.get_Wn_log(theta_grid)

    T_E = np.zeros((nmodes, n_ell))
    T_B = np.zeros((nmodes, n_ell))

    for idx in range(n_ell):
        basis = np.zeros(n_ell)
        basis[idx] = 1.0

        ce_basis, _ = cosebis_obj.cosebis_from_Cell(
            ell=ell,
            Cell_E=basis,
            Cell_B=zeros,
            theta=theta_grid,
            cache=True,
        )
        _, cb_basis = cosebis_obj.cosebis_from_Cell(
            ell=ell,
            Cell_E=zeros,
            Cell_B=basis,
            theta=theta_grid,
            cache=True,
        )

        T_E[:, idx] = ce_basis
        T_B[:, idx] = cb_basis

    return T_E, T_B


def _safe_solve(cov, vector):
    try:
        return float(vector @ np.linalg.solve(cov, vector))
    except np.linalg.LinAlgError:
        cov_pinv = np.linalg.pinv(cov)
        return float(vector @ cov_pinv @ vector)


def _compute_statistics(residual, covariance, nmodes_short):
    chi2_long = _safe_solve(covariance, residual)
    pte_long = 1.0 - stats.chi2.cdf(chi2_long, len(residual))
    chi2_short = _safe_solve(covariance[:nmodes_short, :nmodes_short], residual[:nmodes_short])
    pte_short = 1.0 - stats.chi2.cdf(chi2_short, nmodes_short)
    return {
        "chi2_long": chi2_long,
        "pte_long": pte_long,
        "chi2_short": chi2_short,
        "pte_short": pte_short,
    }


def main():
    versions = params["versions"]
    nmodes_long = int(params["nmodes_long"])
    nmodes_short = int(params["nmodes_short"])
    theta_min = float(params["theta_min"])
    theta_max = float(params["theta_max"])

    min_sep_int = float(params["min_sep_int"])
    max_sep_int = float(params["max_sep_int"])
    nbins_int = int(params["nbins_int"])
    npatch = int(params["npatch"])

    cv = CosmologyValidation(versions=versions, catalog_config=CAT_CONFIG_PATH)

    theta_grid = np.logspace(np.log10(theta_min), np.log10(theta_max), 5000)

    datasets = []
    palette = sns.color_palette("husl", len(versions))

    for idx, (version, color) in enumerate(zip(versions, palette)):
        pseudo_cl_path = snakemake.input.pseudo_cls[idx]
        pseudo_cov_path = snakemake.input.pseudo_cls_cov[idx]
        cov_path_int = snakemake.input.cov_integration[idx]

        with fits.open(pseudo_cl_path) as hdul:
            data = hdul["PSEUDO_CELL"].data
            ell = np.asarray(data["ELL"], dtype=float)
            cl_ee = np.asarray(data["EE"], dtype=float)
            cl_bb = np.asarray(data["BB"], dtype=float)

        cosebis_cell = COSEBIS(theta_min, theta_max, nmodes_long)
        ce_harm, cb_harm = cosebis_cell.cosebis_from_Cell(
            ell=ell,
            Cell_E=cl_ee,
            Cell_B=cl_bb,
            theta=theta_grid,
            cache=True,
        )

        T_E, T_B = _build_transforms(cosebis_cell, ell, theta_grid)

        cov_cell = _build_cell_covariance(pseudo_cov_path)
        transform = np.zeros((2 * nmodes_long, 2 * len(ell)))
        transform[:nmodes_long, : len(ell)] = T_E
        transform[nmodes_long:, len(ell) :] = T_B
        cov_harmonic = transform @ cov_cell @ transform.T

        results = cv.calculate_cosebis(
            version,
            min_sep_int=min_sep_int,
            max_sep_int=max_sep_int,
            nbins_int=nbins_int,
            npatch=npatch,
            nmodes=nmodes_long,
            cov_path=cov_path_int,
            evaluate_all_scale_cuts=False,
        )

        if isinstance(results, dict) and "En" in results:
            config_result = results
        else:
            config_result = next(iter(results.values()))

        ce_config = config_result["En"]
        cb_config = config_result["Bn"]
        cov_config = config_result["cov"]

        cov_config_E = cov_config[:nmodes_long, :nmodes_long]
        cov_config_B = cov_config[nmodes_long:, nmodes_long:]

        sigma_config_E = np.sqrt(np.clip(np.diag(cov_config_E), 0, None))
        sigma_config_B = np.sqrt(np.clip(np.diag(cov_config_B), 0, None))

        sigma_harm_E = np.sqrt(np.clip(np.diag(cov_harmonic[:nmodes_long, :nmodes_long]), 0, None))
        sigma_harm_B = np.sqrt(np.clip(np.diag(cov_harmonic[nmodes_long:, nmodes_long:]), 0, None))

        residual_E = ce_harm - ce_config
        residual_B = cb_harm - cb_config

        stats_E = _compute_statistics(residual_E, cov_config_E, nmodes_short)
        stats_B = _compute_statistics(residual_B, cov_config_B, nmodes_short)

        datasets.append(
            {
                "version": version,
                "label": _version_label(version),
                "color": color,
                "ell": ell,
                "ce_harm": ce_harm,
                "cb_harm": cb_harm,
                "ce_config": ce_config,
                "cb_config": cb_config,
                "sigma_config_E": sigma_config_E,
                "sigma_config_B": sigma_config_B,
                "sigma_harm_E": sigma_harm_E,
                "sigma_harm_B": sigma_harm_B,
                "residual_E": residual_E,
                "residual_B": residual_B,
                "stats_E": stats_E,
                "stats_B": stats_B,
            }
        )

    modes = np.arange(1, nmodes_long + 1)

    fig = plt.figure(figsize=(14, 12))
    outer = GridSpec(len(datasets), 2, hspace=0.35, wspace=0.25)

    for row, data in enumerate(datasets):
        for col, (series_key, ylabel) in enumerate([
            ("E", r"$E_n$"),
            ("B", r"$B_n$"),
        ]):
            sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[row, col], height_ratios=[3, 1], hspace=0.05)
            ax_main = fig.add_subplot(sub[0, 0])
            ax_res = fig.add_subplot(sub[1, 0], sharex=ax_main)

            if series_key == "E":
                config = data["ce_config"]
                harm = data["ce_harm"]
                sigma_config = data["sigma_config_E"]
                sigma_harm = data["sigma_harm_E"]
                residual = data["residual_E"]
                stats_block = data["stats_E"]
            else:
                config = data["cb_config"]
                harm = data["cb_harm"]
                sigma_config = data["sigma_config_B"]
                sigma_harm = data["sigma_harm_B"]
                residual = data["residual_B"]
                stats_block = data["stats_B"]

            ax_main.errorbar(
                modes,
                config,
                yerr=sigma_config,
                fmt="o",
                color=data["color"],
                mfc=data["color"],
                ms=4,
                alpha=0.9,
                label="Config-space",
            )
            ax_main.errorbar(
                modes,
                harm,
                yerr=sigma_harm,
                fmt="s",
                color=data["color"],
                mfc="white",
                ms=4,
                alpha=0.9,
                label="Harmonic-space",
            )

            ax_main.axhline(0.0, color="black", lw=0.8, alpha=0.6)
            ax_main.axvspan(0.5, nmodes_short + 0.5, color="0.9", zorder=0)
            ax_main.set_ylabel(ylabel)
            ax_main.grid(True, which="both", axis="both", alpha=0.2)
            ax_main.set_title(f"{data['label']} – {series_key}-modes")

            ax_res.axhline(0.0, color="black", lw=0.8, alpha=0.6)
            ax_res.axvspan(0.5, nmodes_short + 0.5, color="0.9", zorder=0)
            ax_res.plot(
                modes,
                residual / np.where(sigma_config > 0, sigma_config, np.inf),
                marker="o",
                color=data["color"],
                ms=3,
                lw=1.0,
            )
            ax_res.set_ylabel(r"$\Delta/\sigma$")
            ax_res.set_xlabel("Mode index n")
            ax_res.grid(True, which="both", axis="both", alpha=0.2)

            if row == 0 and col == 0:
                handles, labels = ax_main.get_legend_handles_labels()
                ax_main.legend(handles, labels, fontsize=9)

            if col == 1:
                ax_main.text(
                    0.02,
                    0.95,
                    (
                        f"χ²₆={stats_block['chi2_short']:.2f}, PTE₆={stats_block['pte_short']:.3f}\n"
                        f"χ²₂₀={stats_block['chi2_long']:.2f}, PTE₂₀={stats_block['pte_long']:.3f}"
                    ),
                    transform=ax_main.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                )

            if row < len(datasets) - 1:
                ax_main.set_xticklabels([])
            ax_res.set_ylim(-4, 4)

    fig.suptitle("COSEBIS harmonic vs configuration-space cross-validation", y=0.96)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_path = Path(snakemake.output["comparison"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)

    stats_lines = []
    for data in datasets:
        stats_lines.append(f"Version: {data['version']}")
        stats_lines.append(
            "  E-modes: "
            f"chi2(1-6)={data['stats_E']['chi2_short']:.3f}, "
            f"PTE(1-6)={data['stats_E']['pte_short']:.3f}, "
            f"chi2(1-20)={data['stats_E']['chi2_long']:.3f}, "
            f"PTE(1-20)={data['stats_E']['pte_long']:.3f}"
        )
        stats_lines.append(
            "  B-modes: "
            f"chi2(1-6)={data['stats_B']['chi2_short']:.3f}, "
            f"PTE(1-6)={data['stats_B']['pte_short']:.3f}, "
            f"chi2(1-20)={data['stats_B']['chi2_long']:.3f}, "
            f"PTE(1-20)={data['stats_B']['pte_long']:.3f}"
        )
        stats_lines.append("")

    stats_path = Path(snakemake.output["stats"])
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text("\n".join(stats_lines))


if __name__ == "__main__":
    main()
