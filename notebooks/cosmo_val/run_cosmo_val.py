# %%
# %load_ext autoreload
# %autoreload 2

import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np
import treecorr
from sp_validation.cosmo_val import CosmologyValidation

# %%
cv = CosmologyValidation(
    versions=["SP_v1.4.2", "SP_v1.4.5"],
    data_base_dir="/n17data/mkilbing/astro/data",
    npatch=100,
)

# %%
cv.plot_rho_stats()

# %%
cv.plot_tau_stats()

# %%
if cv.rho_tau_method != "none":
    cv.plot_rho_tau_fits()

# %%
cv.plot_footprints()

# %%
cv.plot_scale_dependent_leakage()

# %%
cv.plot_objectwise_leakage()

# %%
cv.plot_objectwise_leakage_aux()

# %%
cv.plot_ellipticity()

# %%
cv.plot_separation()

# %%
cv.calculate_additive_bias()

# %%
cv.plot_2pcf()

# %%
cv.plot_aperture_mass_dispersion()

# %%
cv.plot_pseudo_cl()
