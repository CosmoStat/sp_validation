# %%
from IPython import get_ipython

ipython = get_ipython()

# enable autoreload for interactive sessions
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
    ipython.run_line_magic("load_ext", "log_cell_time")

import matplotlib.pyplot as plt  # noqa: E402, F401
import numpy as np  # noqa: E402, F401
import sys  # noqa: E402
sys.path.insert(0, '/home/mkilbing/astro/repositories/gitlab.euclid-sgs/FDQA/rho_tau_stats')
from cosmo_val import CosmologyValidation  # noqa: E402

# enable inline plotting for interactive sessions
# (must be done *after* importing package that sets agg backend)
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")


# %%
cv = CosmologyValidation(
    versions=["SP_v1.4.6_GAIA_bright", "SP_v1.4.6_GAIA_medium", "SP_v1.4.6_GAIA_faint"],
    data_base_dir="/",
    catalog_config="./cat_config.yaml",
    output_dir="./output",
    theta_min=1.0,
    theta_max=250.0,
    nbins=20,
    cov_estimate_method='jk',
    theta_min_plot=0.8,
    theta_max_plot=260.0,
    rho_tau_method='emcee',
    n_cov=1,
)

# %%
cv.plot_gammat_around_stars(offset=0.05, gammax=True)
cv.plot_gammat_around_stars(offset=0.05, gammax=True, logy=True)
