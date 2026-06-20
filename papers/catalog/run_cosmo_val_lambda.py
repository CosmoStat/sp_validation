# %%
from IPython import get_ipython

ipython = get_ipython()

# enable autoreload for interactive sessions
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
    ipython.run_line_magic("load_ext", "log_cell_time")

import sys

sys.path.insert(
    0, "/home/mkilbing/astro/repositories/gitlab.euclid-sgs/FDQA/rho_tau_stats"
)
from cosmo_val import CosmologyValidation, rename_output  # noqa: E402

# enable inline plotting for interactive sessions
# (must be done *after* importing package that sets agg backend)
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# Different options
versions_146 = ["SP_v1.4.6"]  # , "SP_v1.4.6_leak_corr"]
versions_var = ["SP_v1.4.5", "SP_v1.4.6", "SP_v1.4.7", "SP_v1.4.8"]

# We use this combination of versions
versions = versions_146

output_dir = "./output"

theta_min = 1.0
theta_max = 300.0
plot_range_fac = 1.1


# %%
cv = CosmologyValidation(
    versions=versions,
    catalog_config="./cat_config.yaml",
    data_base_dir="/",
    output_dir=output_dir,
    theta_min=theta_min,
    theta_max=theta_max,
    nbins=15,
    cov_estimate_method="jk",
    theta_min_plot=theta_min / plot_range_fac,
    theta_max_plot=theta_max * plot_range_fac,
    rho_tau_method="emcee",
    n_cov=1,
    star_weight_type="uniform",
    random_multiple=5,
)

output_bases = [
    "gammat_stars_around_galaxies_lin_non_tomographic",
    "gammat_stars_around_galaxies_log_non_tomographic",
    "dgammat_stars_around_galaxies_lin_non_tomographic",
    "dsize_stars_around_galaxies_lin_non_tomographic",
]
suff = ".png"
version_string = "_".join(versions)

# lambda_1
print("Computing  λ_1...")
cv.plot_gammat_stars_around_galaxies(offset=0.025, gammax=True, wo_rand_subtr=True)
cv.plot_gammat_stars_around_galaxies(
    offset=0.025, gammax=True, logy=True, wo_rand_subtr=True
)

# lambda_2
print("Computing  λ_2...")
cv.plot_dgammat_stars_around_galaxies(offset=0.025, gammax=True, wo_rand_subtr=True)

# lambda_3
print("Computing  λ_3...")
cv.plot_dsize_stars_around_galaxies(offset=0.025, gammax=True, wo_rand_subtr=True)


rename_output(output_bases, output_dir, suff, version_string, "_wo_rand_subtr")

# %%
# lambda_1
print("Computing  λ_1...")
cv.plot_gammat_stars_around_galaxies(offset=0.025, gammax=True)
cv.plot_gammat_stars_around_galaxies(offset=0.025, gammax=True, logy=True)

# lambda_2
print("Computing  λ_2...")
cv.plot_dgammat_stars_around_galaxies(offset=0.025, gammax=True)

# lambda_3
print("Computing  λ_3...")
cv.plot_dsize_stars_around_galaxies(offset=0.025, gammax=True)

rename_output(output_bases, output_dir, suff, version_string, "")
