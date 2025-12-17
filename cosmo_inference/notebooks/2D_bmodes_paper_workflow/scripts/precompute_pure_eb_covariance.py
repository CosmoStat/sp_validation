"""Precompute Pure E/B semi-analytic covariance for faster plotting."""

import os
import sys
from pathlib import Path

import numpy as np
import pyccl as ccl

from IPython import get_ipython
from sp_validation.cosmo_val import CosmologyValidation


ipython = get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
else:
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)


def _load_snakemake():
    if hasattr(sys, "ps1"):
        from snakemake_helpers import snakemake_interactive

        return snakemake_interactive(
            "results/paper_plots/intermediate/SP_v1.4.6_leak_corr_pure_eb_semianalytic.npz",
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


def main():

    covariance_cfg = snakemake.config["covariance"]
    cosmo_cov = _build_cosmology(snakemake.config)

    numeric_params = {
        "min_sep": float(params["min_sep"]),
        "max_sep": float(params["max_sep"]),
        "nbins": int(params["nbins"]),
        "min_sep_int": float(params["min_sep_int"]),
        "max_sep_int": float(params["max_sep_int"]),
        "nbins_int": int(params["nbins_int"]),
        "npatch": int(params["npatch"]),
    }

    cv = CosmologyValidation(
        versions=[params["version"]],
        catalog_config="/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/cat_config.yaml",
        output_dir="/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/output",
    )

    results = cv.calculate_pure_eb(
        params["version"],
        **numeric_params,
        var_method="jackknife" if numeric_params["npatch"] > 1 else "shot",
        cov_path_int=snakemake.input["cov_integration"],
        cosmo_cov=cosmo_cov,
        n_samples=int(params["n_samples"]),
    )

    gg = results["gg"]
    gg_int = results["gg_int"]

    package = {
        "theta": gg.meanr,
        "theta_int": gg_int.meanr,
        "xip_total": gg.xip,
        "xim_total": gg.xim,
        # Note: cov_xip_xim removed; use ng reporting covariance instead
        "xip_E": results["xip_E"],
        "xim_E": results["xim_E"],
        "xip_B": results["xip_B"],
        "xim_B": results["xim_B"],
        "xip_amb": results["xip_amb"],
        "xim_amb": results["xim_amb"],
        "cov_pure_eb": results["cov"],
    }

    np.savez(snakemake.output[0], **package)


if __name__ == "__main__":
    main()

