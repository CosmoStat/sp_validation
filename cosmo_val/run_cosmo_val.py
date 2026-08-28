# Interactive / exploratory driver for the CosmologyValidation suite.
#
# For reproducible, dependency-tracked runs the suite is now a Snakemake
# workflow: papers/cosmo_val/ composes the generic rules in
# workflow/rules/cosmo_val.smk, each cv.<method>() call is a rule, and
# `snakemake all` (run from papers/cosmo_val/) builds the whole validation.
# This script remains the cell-by-cell scratch entry point for one-off
# exploration; the workflow is the canonical batch path.

# %%
from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import sys
from astropy.cosmology import Planck18  # noqa: E402, F401

from sp_validation.cosmo_val import CosmologyValidation  # noqa: E402

# Must follow sp_validation import (which sets agg backend)
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# Fiducial COSEBIs scale cut (theta_min, theta_max) in arcmin, used for the
# B-mode summary below.
FIDUCIAL_SCALE_CUT = (10, 250)

# %%
# Specify version
versions = [
    "SP_v1.3.6.3", "SP_v1.3.6.3_leak_corr",
    "SP_v1.4.6.3", "SP_v1.4.6.3_leak_corr",
]

# Get the cosmology
planck = Planck18

h = planck.H0.value / 100.0
om = planck.Om0
ob = planck.Ob0
ns = 0.9665
sigma8 = 0.8102
mnu = 0.06
omnuh2 = 0.000644867
onu = omnuh2 / h**2
oc = om - ob - onu
halofit_version = "mead2020_feedback"
log_T_AGN = 7.8

extra_params = {
    "camb": {
        "nonlinear": True,
        "halofit_version": halofit_version,
        "HMCode_log_T_AGN": log_T_AGN,
        "kmax": 20,
        "kmax_extrapolate": 500,
    }
}

cosmo_params = {
    "Omega_m": om,
    "Omega_b": ob,
    "h": h,
    "sig8": sigma8,
    "ns": ns,
    "mnu": mnu,
    "extra_params": extra_params,
}

cv = CosmologyValidation(
    versions=versions,
    npatch=100,
    theta_min=1.0,
    theta_max=250.0,
    nbins=20,
    theta_min_plot=0.8,
    theta_max_plot=260.0,
    ylim_alpha=[-0.01, 0.05],
    nrandom_cell=100,
    cell_method="catalog",
    nside_mask=8192,
    path_onecovariance="/home/guerrini/OneCovariance/",
    cosmo_params=cosmo_params,
)
cv.treecorr_config["num_threads"] = 24


# %%
# cv.calculate_pseudo_cl_g_ng_cov()
# cv.calculate_pseudo_cl_g_ng_cov(gaussian_part="OneCovariance")

# %%
cv.plot_2pcf()
sys.exit()

# %%
cv.plot_footprints()

# %%
cv.plot_rho_stats()

# %%
cv.plot_tau_stats()

# %%
if cv.rho_tau_method != "none":
    cv.plot_rho_tau_fits()

# %% Shear diagnostics
cv.plot_objectwise_leakage()

# %%
# cv.plot_objectwise_leakage()

# %%
# cv.plot_ellipticity()

# %%
cv.plot_weights()

# %%
cv.calculate_additive_bias()

# %% Two-point correlation functions
cv.plot_2pcf()

# %%
cv.plot_ratio_xi_sys_xi(offset=0.1)

# %%
# cv.plot_aperture_mass_dispersion()

# %%
# cv.plot_pseudo_cl()

# %%
# cv.plot_pure_eb(
#    min_sep_int=0.08,
#    max_sep_int=300,
#    nbins_int=100,
#    npatch=256,
#    var_method="jackknife",
# )

# %%
"""
scv.plot_cosebis(
    min_sep=0.9,
    max_sep=250,
    nbins=2000,
    npatch=128,
    var_method="jackknife",
    nmodes=5,
    scale_cuts=[
        (1, 250),
        (2, 250),
        (5, 250),
        (10, 250),
        FIDUCIAL_SCALE_CUT,
        (15, 250),
        (20, 250),
    ],
    fiducial_scale_cut=(10, 250),
)
"""

# %% B-mode summary
cv.summarize_bmodes(fiducial_scale_cut=FIDUCIAL_SCALE_CUT)
