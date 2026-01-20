# %%
"""Combined COSEBIS comparison across catalog versions."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
import seaborn as sns
from IPython import get_ipython
from scipy import stats

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

sns.set_palette("colorblind", 3)

COSMO_VAL_DIR = "/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val"
CAT_CONFIG_PATH = f"{COSMO_VAL_DIR}/cat_config.yaml"


def _load_snakemake():
    if hasattr(sys, "ps1"):
        from snakemake_helpers import snakemake_interactive

        return snakemake_interactive(
            "output/cosebis_version_comparison.png",
            str(Path.cwd()),
        )
    from snakemake.script import snakemake

    return snakemake


snakemake = _load_snakemake()

params = snakemake.params


def _build_cosmology(config):
    cosmo_cfg = config["covariance"]["cosmology"]
    return ccl.Cosmology(
        Omega_c=cosmo_cfg["Omega_m"] - cosmo_cfg["Omega_b"],
        Omega_b=cosmo_cfg["Omega_b"],
        h=cosmo_cfg["h"],
        sigma8=cosmo_cfg["sigma_8"],
        n_s=cosmo_cfg["n_s"],
    )


def _version_label(version):
    return version.replace("SP_", "").replace("_leak_corr", "")


def _compute_pte(values, covariance):
    chi2 = float(values @ np.linalg.solve(covariance, values))
    dof = len(values)
    return 1.0 - stats.chi2.cdf(chi2, dof)


def main():
    versions = params["versions"]

    covariance_cfg = snakemake.config["covariance"]
    use_semianalytic = covariance_cfg["use_semianalytic"]
    n_samples = covariance_cfg.get("n_samples", 1000)
    blind = snakemake.config["fiducial"]["blind"]

    numeric_params = {
        "min_sep_int": float(params["min_sep_int"]),
        "max_sep_int": float(params["max_sep_int"]),
        "nbins_int": int(params["nbins_int"]),
        "nmodes": int(params["nmodes"]),
        "npatch": int(params["npatch"]),
    }

    if numeric_params["nmodes"] != 20:
        raise ValueError("COSEBIS comparison expects nmodes=20")

    # Get fiducial scale cut from config
    fiducial_config = snakemake.config["fiducial"]
    fiducial_scale_cut = (
        int(fiducial_config["fiducial_min_scale"]),
        int(fiducial_config["fiducial_max_scale"])
    )

    cv = CosmologyValidation(
        versions=versions,
        catalog_config=CAT_CONFIG_PATH,
        output_dir=f"{COSMO_VAL_DIR}/output",
    )

    cosmo_cov = _build_cosmology(snakemake.config) if use_semianalytic else None

    datasets = []
    for version, color, cov_path in zip(
        versions,
        sns.color_palette("colorblind", len(versions)),
        snakemake.input["cov_integration"] if use_semianalytic else [None] * len(versions)
    ):
        results = cv.calculate_cosebis(
            version,
            **numeric_params,
            cov_path=cov_path if use_semianalytic else None,
            evaluate_all_scale_cuts=False,
        )

        if isinstance(results, dict) and "En" in results:
            cosebis_result = results
        else:
            cosebis_result = next(iter(results.values()))

        En = cosebis_result["En"]
        Bn = cosebis_result["Bn"]
        cov = cosebis_result["cov"]
        nmodes = numeric_params["nmodes"]

        cov_E = cov[:nmodes, :nmodes]
        cov_B = cov[nmodes:, nmodes:]

        sigma_E = np.sqrt(np.diag(cov_E))
        sigma_B = np.sqrt(np.diag(cov_B))

        pte_B_6 = _compute_pte(Bn[:6], cov_B[:6, :6])
        pte_B_20 = _compute_pte(Bn, cov_B)

        datasets.append(
            {
                "version": version,
                "label": _version_label(version),
                "color": color,
                "En": En,
                "Bn": Bn,
                "sigma_E": sigma_E,
                "sigma_B": sigma_B,
                "pte_B_6": pte_B_6,
                "pte_B_20": pte_B_20,
            }
        )

    modes = np.arange(1, numeric_params["nmodes"] + 1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for data in datasets:
        color = data["color"]
        label = data["label"]
        # E-modes: solid line, filled 'o' marker
        ax.errorbar(modes, data["En"], yerr=data["sigma_E"],
                    marker="o", ls="-", lw=1.5, color=color,
                    capsize=3, label=f"{label} E-mode")
        # B-modes: dashed line, unfilled 'o' marker
        ax.errorbar(modes, data["Bn"], yerr=data["sigma_B"],
                    marker="o", mfc="none", ls="--", lw=1.5, color=color,
                    capsize=3, label=f"{label} B-mode")

    ax.axhline(0.0, color="black", lw=0.8, alpha=0.6)
    ax.set_xlabel("Mode index n")
    ax.set_ylabel("COSEBIS modes")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(True, which="both", axis="both", alpha=0.2)
    ax.axvspan(0.5, 6.5, color="0.9", zorder=0)
    ax.set_title("COSEBIS E/B Mode Comparison")

    # Reorder legend: E-modes in column 1, B-modes in column 2
    handles, labels = ax.get_legend_handles_labels()
    reorder_idx = [0, 2, 4, 1, 3, 5]
    handles_reordered = [handles[i] for i in reorder_idx]
    labels_reordered = [labels[i] for i in reorder_idx]
    legend = ax.legend(handles_reordered, labels_reordered, title="Catalog", fontsize=9, ncol=2)
    legend._legend_box.align = "left"

    pte_lines = [
        f"{d['label']}: PTE$_B$(1-6)={d['pte_B_6']:.3f}, PTE$_B$(1-20)={d['pte_B_20']:.3f}"
        for d in datasets
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(pte_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    fig.suptitle("COSEBIS comparison across leak-corrected catalogs", y=0.95)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    output = Path(snakemake.output[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)


if __name__ == "__main__":
    main()
