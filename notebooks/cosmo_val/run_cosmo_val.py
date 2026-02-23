# %%
from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import matplotlib.pyplot as plt
import numpy as np

from sp_validation.cosmo_val import CosmologyValidation  # noqa: E402

# Must follow sp_validation import (which sets agg backend)
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# %%
FIDUCIAL_SCALE_CUT = (12, 83)

cv = CosmologyValidation(
    versions=[
        "SP_v1.4.6_leak_corr",
        "SP_v1.4.11.3_leak_corr",
    ],
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
)

# %% PSF diagnostics
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
# cv.plot_scale_dependent_leakage()

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

# %% Pseudo-C_ell
cv.plot_pseudo_cl()

# %%
# cv.plot_aperture_mass_dispersion()

# %% Pure E/B modes
cv.plot_pure_eb(
    min_sep_int=0.08,
    max_sep_int=300,
    nbins_int=1000,
    fiducial_xip_scale_cut=FIDUCIAL_SCALE_CUT,
    fiducial_xim_scale_cut=FIDUCIAL_SCALE_CUT,
)

# %% COSEBIs
cv.plot_cosebis(
    min_sep_int=0.9,
    max_sep_int=300,
    nbins_int=2000,
    npatch=100,
    nmodes=20,
    scale_cuts=[
        (1, 250),
        (2, 250),
        (5, 250),
        (10, 250),
        FIDUCIAL_SCALE_CUT,
        (15, 250),
        (20, 250),
    ],
    fiducial_scale_cut=FIDUCIAL_SCALE_CUT,
)

# %% B-mode summary
cv.summarize_bmodes(fiducial_scale_cut=FIDUCIAL_SCALE_CUT)

