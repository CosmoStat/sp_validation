# %%
# COSEBIS plotting script using CosmoVal plot_cosebis function with covariances
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
        "/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/output/SP_v1.4.5_leak_corr_cosebis_minsep=0.5_maxsep=500_nbins=1000_npatch=1_varmethod=analytic_nmodes=6_scalecut=5-144_cosebis.png",
        "/home/cdaley/n17data/unions/pure_eb",
    )
else:
    from snakemake.script import snakemake

params = snakemake.params

# %%
os.chdir("/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val")
print("Starting CosmologyValidation for COSEBIS plotting")

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
    "nmodes": int(params["nmodes"]),
    "npatch": int(params["npatch"]),
}

# Get fiducial scale cut from parameters
fiducial_scale_cut = (
    int(params["fiducial_min_scale"]),
    int(params["fiducial_max_scale"])
)
print(f"Using fiducial scale cut: {fiducial_scale_cut} arcmin")


if use_semianalytic:
    # Get covariance path from snakemake inputs
    cov_path_int = snakemake.input["cov_integration"]

    print("Using CosmoCov theoretical covariance with direct CosmoNumba transformation")
    print(f"Integration covariance: {cov_path_int}")
    print(f"COSEBIS analysis with {numeric_params['nmodes']} modes")

    # Step 1: Calculate COSEBIs with all scale cuts
    print("Calculating COSEBIs...")
    results = cv.calculate_cosebis(
        params["version"],
        **numeric_params,
        cov_path=cov_path_int,
        evaluate_all_scale_cuts=True,
    )

    # Step 2: Generate plots using precalculated results
    print("Generating plots...")
    cv.plot_cosebis(
        **numeric_params,
        cov_path=cov_path_int,
        fiducial_scale_cut=fiducial_scale_cut,
        results=results
    )
else:
    print("Using TreeCorr jackknife covariance")
    print(f"COSEBIS analysis with {numeric_params['nmodes']} modes")

    # Step 1: Calculate COSEBIs with all scale cuts
    print("Calculating COSEBIs...")
    results = cv.calculate_cosebis(
        params["version"],
        **numeric_params,
        evaluate_all_scale_cuts=True,
    )

    # Step 2: Generate plots using precalculated results
    print("Generating plots...")
    cv.plot_cosebis(
        **numeric_params,
        fiducial_scale_cut=fiducial_scale_cut,
        results=results
    )

print("COSEBIS plotting completed successfully")

# Report completion with expected output files
if results:
    fiducial_scale_str = (
        f"{params['fiducial_min_scale']}-{params['fiducial_max_scale']}"
    )
    print("COSEBIS plots created successfully:")
    base_name = (
        f"{params['version']}_cosebis_minsep={float(params['min_sep_int'])}_"
        f"maxsep={params['max_sep_int']}_nbins={params['nbins_int']}_"
        f"npatch={params['npatch']}_varmethod=analytic_"
        f"nmodes={params['nmodes']}_scalecut={fiducial_scale_str}"
    )
    print(f"  - {base_name}_cosebis.png")
    print(f"  - {base_name}_covariance.png")
    print(f"  - {base_name}_ptes.png")
else:
    print("Warning: No results returned from calculate_cosebis")

print("COSEBIS analysis completed successfully")
# %%
