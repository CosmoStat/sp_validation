# %%
import os

# Trick to use tex in the plots
os.environ["LD_LIBRARY_PATH"] = ""
os.environ["CONDA_PREFIX"] = "/home/guerrini/.conda/envs/sp_validation_3.11"

from IPython import get_ipython

ipython = get_ipython()

# enable autoreload for interactive sessions
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from astropy.io import fits
from matplotlib import scale as mscale

from sp_validation.rho_tau import SquareRootScale

mscale.register_scale(SquareRootScale)

plt.style.use("./matplotlib_config/paper.mplstyle")

sns.set_palette("husl")

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# %%
versions = ["SP_v1.4.6_leak_corr_A", "SP_v1.4.6_leak_corr_A_KiDS"]

path_cosmo_val = "/home/guerrini/sp_validation/cosmo_val/output/"

# %%
plt.figure()

for ver in versions:
    cov = fits.open(f"{path_cosmo_val}/pseudo_cl_cov_{ver}.fits")
    plt.plot(np.sqrt(np.diag(cov["COVAR_BB_BB"].data)))
    print(np.sqrt(np.diag(cov["COVAR_BB_BB"].data)))

plt.yscale("log")

plt.show()

# %%
plt.figure()

ref = np.sqrt(
    np.diag(
        fits.open(f"{path_cosmo_val}/pseudo_cl_cov_{versions[0]}.fits")[
            "COVAR_BB_BB"
        ].data
    )
)

for ver in versions:
    cov = fits.open(f"{path_cosmo_val}/pseudo_cl_cov_{ver}.fits")
    plt.plot(np.sqrt(np.diag(cov["COVAR_BB_BB"].data)) / ref)
    print(np.sqrt(np.diag(cov["COVAR_BB_BB"].data)))

plt.show()

# %%
plt.figure()

for ver in versions:
    cov = fits.open(f"{path_cosmo_val}/pseudo_cl_cov_{ver}.fits")
    plt.plot(np.sqrt(np.diag(cov["COVAR_EE_EE"].data)))
    print(np.sqrt(np.diag(cov["COVAR_EE_EE"].data)))

plt.yscale("log")

plt.show()
# %%
plt.figure()

ref = np.sqrt(
    np.diag(
        fits.open(f"{path_cosmo_val}/pseudo_cl_cov_{versions[0]}.fits")[
            "COVAR_EE_EE"
        ].data
    )
)

for ver in versions:
    cov = fits.open(f"{path_cosmo_val}/pseudo_cl_cov_{ver}.fits")
    plt.plot(np.sqrt(np.diag(cov["COVAR_EE_EE"].data)) / ref)
    print(np.sqrt(np.diag(cov["COVAR_EE_EE"].data)))

plt.show()

# %%
# Compute PTEs for each
lower_bound_cell = 0
upper_bound_cell = 2100

for ver in versions:
    cov = fits.open(f"{path_cosmo_val}/pseudo_cl_cov_{ver}.fits")
    cl = fits.getdata(f"{path_cosmo_val}/pseudo_cl_{ver}.fits")

    ell = cl["ELL"]

    mask = (ell >= lower_bound_cell) & (ell <= upper_bound_cell)
    data_vector = cl["BB"][mask]
    cov_cut = cov["COVAR_BB_BB"].data[mask][:, mask]

    chi2 = data_vector @ np.linalg.inv(cov_cut) @ data_vector
    dof = np.sum(mask)
    pte = 1 - stats.chi2.cdf(chi2, dof)
    print(chi2, dof, pte)

# %%
