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

# enable inline plotting for interactive sessions
# (must be done *after* importing package that sets agg backend)
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# %%
cv = CosmologyValidation(
    versions=["SP_v1.4.6", "SP_v1.4.5.A", "SP_v1.4.5"],
    data_base_dir="/n17data/mkilbing/astro/data/",
    npatch=100,
    nrandom_cell=100,
    cell_method="catalog"
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
#cv.plot_footprints()

# %%
#cv.plot_scale_dependent_leakage()

# %%
cv.plot_objectwise_leakage()

# %%
#cv.plot_ellipticity()

# %%
cv.plot_weights()

# %%
cv.plot_separation()

# %%
#cv.plot_2pcf()

# %%
#cv.plot_aperture_mass_dispersion()

# %%
cv.plot_pure_eb(
    min_sep_int=0.08,
    max_sep_int=300,
    nbins_int=100,
    npatch=256,
    var_method="jackknife",
)

# %%
cv.plot_cosebis(
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

# %%
cv.plot_pseudo_cl()
