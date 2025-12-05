# %%
from IPython import get_ipython

ipython = get_ipython()

# enable autoreload for interactive sessions
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
    ipython.run_line_magic("load_ext", "log_cell_time")

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/home/mkilbing/astro/repositories/gitlab.euclid-sgs/FDQA/rho_tau_stats')
from cosmo_val import CosmologyValidation  # noqa: E402


def rename_output(output_bases, output_dir, suff, version_string, rand_str):

    for base in output_bases:
        old_path = f"{output_dir}/plots/{base}{suff}"
        new_path  = f"{output_dir}/plots/{base}_{version_string}{rand_str}{suff}"
        print(f"mv {old_path} -> {new_path}")
        os.rename(old_path, new_path)


# enable inline plotting for interactive sessions
# (must be done *after* importing package that sets agg backend)
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# Different options
versions_146 = ["SP_v1.4.6", "SP_v1.4.6_leak_corr"]
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
    data_base_dir="/n17data/mkilbing/astro/data/",
    catalog_config="./cat_config.yaml",
    output_dir=output_dir,
    theta_min=theta_min,
    theta_max=theta_max,
    nbins=15,
    cov_estimate_method='jk',
    theta_min_plot=theta_min / plot_range_fac,
    theta_max_plot=theta_max * plot_range_fac,
    rho_tau_method='emcee',
    n_cov=1,
    star_weight_type="uniform",
    random_multiple=5,
)

output_bases = [
	"gammat_stars_around_galaxies_lin_non_tomographic",
	"gammat_stars_around_galaxies_log_non_tomographic",
]
suff = ".png"
version_string = "_".join(versions)


cv.plot_gammat_stars_around_galaxies(offset=0.025, gammax=True, wo_rand_subtr=True)
cv.plot_gammat_stars_around_galaxies(offset=0.025, gammax=True, logy=True, wo_rand_subtr=True)
rename_output(output_bases, output_dir, suff, version_string, "_wo_rand_subtr")

# %%
cv.plot_gammat_stars_around_galaxies(offset=0.025, gammax=True)
cv.plot_gammat_stars_around_galaxies(offset=0.025, gammax=True, logy=True)
rename_output(output_bases, output_dir, suff, version_string, "")

