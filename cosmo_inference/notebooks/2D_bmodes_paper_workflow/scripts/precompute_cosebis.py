"""Precompute COSEBIS E/B modes with semi-analytic covariance."""

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
            "results/paper_plots/intermediate/SP_v1.4.6_leak_corr_cosebis.npz",
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
        "min_sep_int": float(params["min_sep_int"]),
        "max_sep_int": float(params["max_sep_int"]),
        "nbins_int": int(params["nbins_int"]),
        "nmodes": int(params["nmodes"]),
        "npatch": int(params["npatch"]),
    }

    cv = CosmologyValidation(
        versions=[params["version"]],
        catalog_config="/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/cat_config.yaml",
        output_dir="/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/output",
    )

    print(f"Computing COSEBIS for {params['version']}...")
    results = cv.calculate_cosebis(
        params["version"],
        **numeric_params,
        cov_path=snakemake.input["cov_integration"],
        evaluate_all_scale_cuts=False,
    )

    # Extract results (handle both dict and nested dict formats)
    if isinstance(results, dict) and "En" in results:
        cosebis_result = results
    else:
        cosebis_result = next(iter(results.values()))

    # Package for saving
    package = {
        "En": cosebis_result["En"],
        "Bn": cosebis_result["Bn"],
        "cov": cosebis_result["cov"],
        "theta": cosebis_result.get("theta"),
        "gg": cosebis_result.get("gg"),
    }

    np.savez(snakemake.output[0], **package)
    print(f"Saved COSEBIS data to {snakemake.output[0]}")


if __name__ == "__main__":
    main()
