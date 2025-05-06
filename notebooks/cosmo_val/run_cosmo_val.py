# %%
from IPython import get_ipython

ipython = get_ipython()

# enable autoreload for interactive sessions
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import matplotlib.pyplot as plt
import numpy as np
from sp_validation.cosmo_val import CosmologyValidation

# enable inline plotting for interactive sessions
# (must be done *after* importing package that sets agg backend)
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# %%
cv = CosmologyValidation(
    versions=["SP_v1.4.5", "SP_v1.4.5_leak_corr", "SP_v1.4.5_glass_mock"],
    data_base_dir="/n17data/mkilbing/astro/data",
    npatch=100,
    ylim_alpha=[-0.01, 0.05],
)

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
cv.plot_scale_dependent_leakage()

# %%
cv.plot_objectwise_leakage()

# %%
cv.plot_ellipticity()

# %%
cv.plot_weights()

# %%
cv.plot_separation()

# %%
cv.plot_2pcf()

# %%
cv.plot_aperture_mass_dispersion()

# %%
cv.plot_pseudo_cl()
