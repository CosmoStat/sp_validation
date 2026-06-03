# %%
from IPython import get_ipython

ipython = get_ipython()

# enable autoreload for interactive sessions
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import matplotlib.pyplot as plt  # noqa: E402, F401
import numpy as np  # noqa: E402, F401
from sp_validation.cosmo_val import CosmologyValidation  # noqa: E402
from sp_validation.cosmology import get_cosmo
from astropy.cosmology import Planck18  # noqa: E402, F401

# enable inline plotting for interactive sessions
# (must be done *after* importing package that sets agg backend)
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# %%
# Specify version
versions = [
    "SP_v1.3.6",
    "SP_v1.3.6_leak_corr"
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
        "kmax_extrapolate": 500
    }
}

cosmo_params = {
    "Omega_m": om,
    "Omega_b": ob,
    "h": h,
    "sig8":sigma8,
    "ns":ns,
    "mnu":mnu,
    "extra_params":extra_params
}

cv = CosmologyValidation(
    versions=versions,
    npatch=1,
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
    cosmo_params=cosmo_params
)


# %%
#cv.calculate_pseudo_cl_g_ng_cov()
#cv.calculate_pseudo_cl_g_ng_cov(gaussian_part="OneCovariance")

# %%
cv.plot_footprints()
# %%
cv.plot_rho_stats()

# %%
cv.plot_tau_stats()

# %%
if cv.rho_tau_method != "none":
    cv.plot_rho_tau_fits()

# %%
#cv.plot_footprints()

# %%
#cv.plot_scale_dependent_leakage()

# %%
#cv.plot_objectwise_leakage()

# %%
#cv.plot_ellipticity()

# %%
cv.plot_weights()

# %%
cv.plot_separation()

# %%
cv.npatch = 1
cv.treecorr_config["var_method"] = "shot"
cv.plot_2pcf()
cv.treecorr_config["var_method"] = "jackknife"
cv.npatch = 100

# %%
cv.plot_ratio_xi_sys_xi(offset=0.1)

# %%
#cv.plot_aperture_mass_dispersion()

# %%
#cv.plot_pseudo_cl()

# %%
#cv.plot_pure_eb(
#    min_sep_int=0.08,
#    max_sep_int=300,
#    nbins_int=100,
#    npatch=256,
#    var_method="jackknife",
#)

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
        (3, 250),
        (4, 250),
        (5, 250),
        (6, 250),
        (7, 250),
        (8, 250),
        (9, 250),
        (10, 250),
        (15, 250),
        (20, 250),
    ],
    fiducial_scale_cut=(10, 250),
) 
"""

