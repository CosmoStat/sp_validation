"""
Get the best-fit (really weighted posterior mean) parameters from the MCMC 
chain and run cosmosis to get the best-fit theory curve for plotting elsewhere.
"""

# %%
import configparser
import os
import subprocess as sp
import sys

import getdist as gd
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d

if hasattr(sys, "ps1"):
    from snakemake_helpers import get_pipe

    snakemake = get_pipe(
        "results/SP_v1.4.5_A_sc_10_60/bestfit_theory.txt",
        "/n17data/cdaley/unions/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots",
    )
else:
    from snakemake.script import snakemake

params = snakemake.params

# %% change a few values in the inference config and save it to a new file
inference_config = configparser.ConfigParser()
inference_config.optionxform = str
inference_config.read(snakemake.input.inference_ini)

inference_config["runtime"]["sampler"] = "test"
inference_config["pipeline"]["values"] = snakemake.output.bestfit_values_ini

prior_path = inference_config["pipeline"]["priors"]
if prior_path.startswith("cosmosis_config"): # sacha
    base_dir = snakemake.input.inference_ini.split("cosmosis_config")[0]
    inference_config["pipeline"]["priors"] = base_dir + prior_path
    inference_config["DEFAULT"]["FITS_FILE"] = base_dir + inference_config["DEFAULT"]["FITS_FILE"]
if "cosmosis_config/priors_" in prior_path and "lgoh" in prior_path: # lisa
    inference_config["pipeline"]["priors"] = prior_path.replace(
    "priors_", "priors/priors_")

inference_config["test"]["save_dir"] = snakemake.params["bestfit_dir"]

with open(snakemake.output.cosmosis_ini, "w", encoding="utf-8") as f:
    inference_config.write(f)


# %%
# % Get best-fit (really posterior mean) parameters from MCMC chain
chain = gd.mcsamples.loadMCSamples(
    inference_config["output"]["filename"].split(".txt")[0]
    .replace("samples_", "getdist_")
)
likestats = chain.getLikeStats()
bestfit_idx = np.argmax(chain.loglikes)
maxlike = chain.loglikes[bestfit_idx]
print(f"Maximum Likelihood: {maxlike:.5g}")
best_fit = {
    par.name: np.average(chain.samples[:, i], weights=chain.weights)
    for i, par in enumerate(likestats.names)
}
# %% create cosmosis bestfit values ini file

section_map = {
    "omch2": "cosmological_parameters",
    "ombh2": "cosmological_parameters",
    "h0": "cosmological_parameters",
    "n_s": "cosmological_parameters",
    "s_8_input": "cosmological_parameters",
    "logt_agn": "halo_model_parameters",
    "a": "intrinsic_alignment_parameters",
    "m1": "shear_calibration_parameters",
    "bias_1": "nofz_shifts",
    "alpha": "psf_leakage_parameters",
    "beta": "psf_leakage_parameters",
}

content = """
[cosmological_parameters]

tau          =  0.0544
w            = -1.0
massive_nu   =  1
massless_nu  =  2.046
omega_k      =  0.0
wa           =  0.0

[halo_model_parameters]

[intrinsic_alignment_parameters]

[shear_calibration_parameters]

[nofz_shifts]

[psf_leakage_parameters]
"""

with open(snakemake.output.bestfit_values_ini, "w", encoding="utf-8") as f:
    f.write(content)
    f.close()

best_fit_config = configparser.ConfigParser()
best_fit_config.optionxform = str
best_fit_config.read(snakemake.output.bestfit_values_ini)

for param, value in best_fit.items():
    section = section_map.get(param)
    if section is None:
        continue
    if section not in best_fit_config:
        best_fit_config.add_section(section)
    best_fit_config[section][param] = str(value)
with open(snakemake.output.bestfit_values_ini, "w", encoding="utf-8") as f:
    best_fit_config.write(f)


# %%

env = os.environ.copy()
env["LD_LIBRARY_PATH"] = (
    "/home/guerrini/.conda/envs/sp_validation/lib/python3.9/site-packages/cosmosis/datablock:"
    + env.get("LD_LIBRARY_PATH", "")
)

# Run cosmosis
result = sp.run(
    ["cosmosis", snakemake.output.cosmosis_ini], env=env, capture_output=True, text=True, check=True
)
print(f"STDOUT:\n{result.stdout}")
print(f"STDERR:\n{result.stderr}")
# %%
theta_model = (
    np.rad2deg(np.loadtxt(snakemake.params["bestfit_dir"] + "/shear_xi_plus/theta.txt"))
    * 60
)
theta_data = fits.open(inference_config["DEFAULT"]["FITS_FILE"])["XI_PLUS"].data["ANG"]
theta_out = np.geomspace(0.01, 300, 1000)

xi_plus, xi_minus = [
    interp1d(
        theta_model,
        np.loadtxt(snakemake.params["bestfit_dir"] + f"/shear_{var}/bin_1_1.txt"),
        kind="cubic",
        fill_value="extrapolate",
    )(theta_out)
    for var in ["xi_plus", "xi_minus"]
]
xi_sys_plus, xi_sys_minus = [
    interp1d(
        theta_data,
        np.loadtxt(snakemake.params["bestfit_dir"] + f"/xi_sys/shear_{var}.txt"),
        kind="cubic",
        fill_value="extrapolate",
    )(theta_out)
    for var in ["xi_plus", "xi_minus"]
]
# %%
np.savetxt(
    snakemake.output.xi_shear,
    np.column_stack([theta_out, xi_plus, xi_minus]),
    header="theta xi_plus xi_minus",
    fmt="%g",
)
np.savetxt(
    snakemake.output.xi_sys,
    np.column_stack([theta_out, xi_sys_plus, xi_sys_minus]),
    header="theta xi_sys_plus xi_sys_minus",
    fmt="%g",
)
# %%
