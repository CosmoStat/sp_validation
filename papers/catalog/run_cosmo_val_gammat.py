# %%
from IPython import get_ipython

ipython = get_ipython()

# enable autoreload for interactive sessions
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
    ipython.run_line_magic("reload_ext", "log_cell_time")

import os
import sys

sys.path.insert(
    0, "/home/mkilbing/astro/repositories/gitlab.euclid-sgs/FDQA/rho_tau_stats"
)
from cosmo_val import CosmologyValidation


def rename_output(output_bases, output_dir, suff, version_string, rand_str):

    for base in output_bases:
        old_path = f"{output_dir}/plots/{base}{suff}"
        new_path = f"{output_dir}/plots/{base}_{version_string}{rand_str}{suff}"
        print(f"mv {old_path} -> {new_path}")
        os.rename(old_path, new_path)


# enable inline plotting for interactive sessions
# (must be done *after* importing package that sets agg backend)
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

versions_146 = ["SP_v1.4.6", "SP_v1.4.6_leak_corr"]
versions_var = ["SP_v1.4.5", "SP_v1.4.6", "SP_v1.4.7", "SP_v1.4.8"]

versions = versions_146

output_dir = "./output"

# %%
cv = CosmologyValidation(
    versions=versions_146,
    data_base_dir="/n17data/mkilbing/astro/data/",
    catalog_config="./cat_config.yaml",
    output_dir=output_dir,
    theta_min=1.0,
    theta_max=250.0,
    nbins=20,
    cov_estimate_method="jk",
    theta_min_plot=0.8,
    theta_max_plot=260.0,
    rho_tau_method="emcee",
    n_cov=1,
    star_weight_type="uniform",
)


output_bases = [
    "gammat_around_stars_lin_non_tomographic",
    "gammat_around_stars_log_non_tomographic",
]
suff = ".png"
version_string = "_".join(versions)

# %%
cv.plot_gammat_around_stars(offset=0.05, gammax=True, wo_rand_subtr=True)
cv.plot_gammat_around_stars(offset=0.05, gammax=True, logy=True, wo_rand_subtr=True)
rename_output(output_bases, output_dir, suff, version_string, "_wo_rand_subtr")

# %%
cv.plot_gammat_around_stars(offset=0.05, gammax=True)
cv.plot_gammat_around_stars(offset=0.05, gammax=True, logy=True)
rename_output(output_bases, output_dir, suff, version_string, "")
