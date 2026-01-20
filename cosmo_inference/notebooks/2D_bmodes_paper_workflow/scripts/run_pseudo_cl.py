"""Snakemake entry point for pseudo-C_ℓ measurements and covariance."""

import os
import sys
from pathlib import Path

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
            "dummy_pseudo_cl_output.txt",
            str(Path.cwd()),
        )
    from snakemake.script import snakemake

    return snakemake


snakemake = _load_snakemake()

params = snakemake.params


def main():
    cosmo_val_dir = "/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val"
    os.chdir(cosmo_val_dir)

    version = params["version"]
    n_ell_bins = int(params.get("n_ell_bins", 32))

    fiducial = snakemake.config["fiducial"]
    covariance_cfg = snakemake.config["covariance"]
    cosmo_cfg = covariance_cfg["cosmology"]

    xi_cosmo_map = {
        "h": cosmo_cfg["h"],
        "Omega_b": cosmo_cfg["Omega_b"],
        "Omega_c": cosmo_cfg["Omega_m"] - cosmo_cfg["Omega_b"],
        "n_s": cosmo_cfg["n_s"],
        "sigma8": cosmo_cfg["sigma_8"],
    }
    if "w0" in cosmo_cfg:
        xi_cosmo_map["w0"] = cosmo_cfg["w0"]
    if "wa" in cosmo_cfg:
        xi_cosmo_map["wa"] = cosmo_cfg["wa"]

    cv = CosmologyValidation(
        versions=[version],
        catalog_config=f"{cosmo_val_dir}/cat_config.yaml",
        theta_min=float(fiducial["min_sep"]),
        theta_max=float(fiducial["max_sep"]),
        nbins=int(fiducial["nbins"]),
        npatch=int(fiducial["npatch"]),
        cell_method=params.get("cell_method", "catalog"),
        nside=int(params.get("nside", 1024)),
        binning=params.get("binning", "powspace"),
        n_ell_bins=n_ell_bins,
        nrandom_cell=int(params.get("nrandom_cell", 100)),
    )

    original_cosmo = cv.cosmo
    cv.cosmo = xi_cosmo_map

    task = params["task"]

    if task == "cl":
        cv.calculate_pseudo_cl()
    elif task == "cov":
        cv.calculate_pseudo_cl_eb_cov()
    else:
        raise ValueError(f"Unknown pseudo-Cl task '{task}'")

    cv.cosmo = original_cosmo


if __name__ == "__main__":
    main()
os.environ["SLURM_CONF"] = "/etc/slurm/slurm.conf"
