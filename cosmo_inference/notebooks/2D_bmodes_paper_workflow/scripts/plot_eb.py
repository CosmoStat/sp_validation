# %%
# Pure E/B plotting script using CosmoVal plot_pure_eb function
import os
import sys

from IPython import get_ipython

ipython = get_ipython()

# enable autoreload for interactive sessions
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
else:
    # Force unbuffered stdout and stderr
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)  # line-buffered
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

import matplotlib.pyplot as plt  # noqa: E402
import pyccl as ccl  # noqa: E402
from sp_validation.cosmo_val import CosmologyValidation  # noqa: E402

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# Apply paper style
plt.style.use(
    "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle"
)

# Handle snakemake interface
if hasattr(sys, "ps1"):
    from snakemake_helpers import snakemake_interactive
    snakemake = snakemake_interactive(
        "/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/output/SP_v1.4.5_eb_minsep=1_maxsep=250_nbins=20_minsepint=0.5_maxsepint=500_nbinsint=1000_npatch=1_varmethod=semianalytic_xis.png",
        "/n17data/cdaley/unions/pure_eb",
    )
else:
    from snakemake.script import snakemake

params = snakemake.params

# %%
os.chdir("/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val")
print("Starting CosmologyValidation")

# Check if semi-analytical covariance should be used
try:
    covariance_config = snakemake.config["covariance"]
except KeyError as exc:
    raise KeyError("Missing 'covariance' section in configuration") from exc

if "use_semianalytic" not in covariance_config:
    raise KeyError("Missing 'use_semianalytic' flag in covariance configuration")
use_semianalytic = covariance_config["use_semianalytic"]

# Create CosmologyValidation instance
cv = CosmologyValidation(
    versions=[params["version"]],
)

# Convert string parameters to numeric types
numeric_params = {
    "min_sep": float(params["min_sep"]),
    "max_sep": float(params["max_sep"]),
    "nbins": int(params["nbins"]),
    "min_sep_int": float(params["min_sep_int"]),
    "max_sep_int": float(params["max_sep_int"]),
    "nbins_int": int(params["nbins_int"]),
    "npatch": int(params["npatch"]),
}

# Get fiducial scale cuts from config
try:
    fiducial_config = snakemake.config["fiducial"]
except KeyError as exc:
    raise KeyError("Missing 'fiducial' section in configuration") from exc

for key in ("fiducial_xip_scale_cut", "fiducial_xim_scale_cut"):
    if key not in fiducial_config:
        raise KeyError(f"Missing '{key}' in fiducial configuration")

fiducial_xip_scale_cut = tuple(fiducial_config["fiducial_xip_scale_cut"])
fiducial_xim_scale_cut = tuple(fiducial_config["fiducial_xim_scale_cut"])

print(f"Using fiducial scale cuts:")
print(f"  xi+ scale cut: {fiducial_xip_scale_cut}")
print(f"  xi- scale cut: {fiducial_xim_scale_cut}")

if use_semianalytic:
    # Create cosmology object from config parameters
    if "cosmology" not in covariance_config:
        raise KeyError("Missing 'cosmology' subsection in covariance configuration")
    cosmo_params = covariance_config["cosmology"]

    try:
        cosmo_cov = ccl.Cosmology(
            Omega_c=cosmo_params["Omega_m"] - cosmo_params["Omega_b"],
            Omega_b=cosmo_params["Omega_b"],
            h=cosmo_params["h"],
            sigma8=cosmo_params["sigma_8"],
            n_s=cosmo_params["n_s"]
        )
    except KeyError as e:
        raise ValueError(f"Missing cosmological parameter in config: {e}")

    # Get covariance path from snakemake inputs
    cov_path_int = snakemake.input["cov_integration"]
    if "n_samples" not in covariance_config:
        raise KeyError("Missing 'n_samples' in covariance configuration")
    n_samples = covariance_config["n_samples"]

    print(f"Using semi-analytical covariance with {n_samples} samples")
    print(f"Integration covariance: {cov_path_int}")

    # Step 1: Calculate pure E/B modes with semi-analytical covariance
    print("Calculating pure E/B modes...")
    results = cv.calculate_pure_eb(
        params["version"],
        **numeric_params,
        var_method="semi-analytic",
        cov_path_int=cov_path_int,
        cosmo_cov=cosmo_cov,
        n_samples=n_samples,
    )

    # Step 2: Generate plots using precalculated results
    print("Generating plots...")
    cv.plot_pure_eb(
        **numeric_params,
        var_method="semi-analytic",
        cov_path_int=cov_path_int,
        cosmo_cov=cosmo_cov,
        n_samples=n_samples,
        results=results,
        fiducial_xip_scale_cut=fiducial_xip_scale_cut,
        fiducial_xim_scale_cut=fiducial_xim_scale_cut,
    )
else:
    print("Using TreeCorr jackknife covariance")

    # Step 1: Calculate pure E/B modes with jackknife covariance
    print("Calculating pure E/B modes...")
    results = cv.calculate_pure_eb(
        params["version"],
        **numeric_params,
        var_method="jackknife",
    )

    # Step 2: Generate plots using precalculated results
    print("Generating plots...")
    cv.plot_pure_eb(
        **numeric_params,
        var_method="jackknife",
        results=results,
        fiducial_xip_scale_cut=fiducial_xip_scale_cut,
        fiducial_xim_scale_cut=fiducial_xim_scale_cut,
    )

print("Pure E/B plotting completed successfully")
# %%
