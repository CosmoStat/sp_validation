# %%
import IPython

ipython = IPython.get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import os

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import scale as mscale
import matplotlib.ticker as mticker
import seaborn as sns
from sp_validation.utils_cosmo_val import SquareRootScale

mscale.register_scale(SquareRootScale)

plt.style.use(
    "/home/guerrini/matplotlib_config/paper.mplstyle"
)

plt.rcParams["text.usetex"] = True

sns.set_palette("colorblind")

#Matplotlib inline if in jupyter
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")


# %%
base_dir = "/home/guerrini/sp_validation/notebooks/cosmo_val/output/"

versions = ["SP_v1.4.5_leak_corr", "SP_v1.4.6_leak_corr", "SP_v1.4.8_leak_corr"]
labels = ["SP v1.4.5 w/ leakage corr.", "SP v1.4.6 w/ leakage corr.", "SP v1.4.8 w/ leakage corr."]
colors = ["C0", "C1", "C2"]
markers = ["o", "h", "x"]

offset = 0.

# %%
fig, (ax0, ax1) = plt.subplots(ncols=2, nrows=1, figsize= (10, 6))

for i, ver in enumerate(versions):
    cell = fits.getdata(f"{base_dir}/pseudo_cl_{ver}.fits")
    cov_cell = fits.open(f"{base_dir}/pseudo_cl_cov_{ver}.fits")

    ell = cell['ell']
    cl_ee = cell['EE']
    cov_cl_ee = cov_cell['COVAR_EE_EE'].data
    cl_bb = cell['BB']
    cov_cl_bb = cov_cell['COVAR_BB_BB'].data

    # Better jittering: symmetric around original ell values
    jitter_factor = (i - (len(versions) - 1) / 2) * offset
    jiterred_ell = ell * (1 + jitter_factor)

    ax0.errorbar(
        jiterred_ell,
        cl_ee * jiterred_ell,
        yerr=np.sqrt(np.diag(cov_cl_ee)) * jiterred_ell,
        label=labels[i],
        color=colors[i],
        fmt=markers[i],
        capsize=2
    )
    ax1.errorbar(
        jiterred_ell,
        cl_bb * jiterred_ell,
        yerr=np.sqrt(np.diag(cov_cl_bb)) * jiterred_ell,
        label=labels[i],
        color=colors[i],
        fmt=markers[i],
        capsize=2
    )

ax0.set_xscale("squareroot")
ax0.set_xticks(np.array([100, 400, 900, 1600]))
ax0.minorticks_on()
ax0.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
ax0.set_xticks(minor_ticks, minor=True)
ax0.legend()

ax0.set_xlabel(r"$\ell$")
ax0.set_ylabel(r"$\ell \, C_\ell^{EE}$")

ax1.set_xscale("squareroot")
ax1.set_xticks(np.array([100, 400, 900, 1600]))
ax1.minorticks_on()
ax1.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
ax1.set_xticks(minor_ticks, minor=True)
ax1.legend()
ax1.axhline(0, color='gray', linestyle='--', linewidth=1)

ax1.set_xlabel(r"$\ell$")
ax1.set_ylabel(r"$\ell \, C_\ell^{BB}$")

plt.tight_layout()

plt.savefig('./plots/data_vectors_cell.png', dpi=200)
plt.show()

# %%
#Same plot with B-modes
fig, ax = plt.subplots(figsize= (8, 6))

for i, ver in enumerate(versions):
    cell = fits.getdata(f"{base_dir}/pseudo_cl_{ver}.fits")
    cov_cell = fits.open(f"{base_dir}/pseudo_cl_cov_{ver}.fits")

    ell = cell['ell']
    cl_ee = cell['BB']
    cov_cl_ee = cov_cell['COVAR_BB_BB'].data

    ax.errorbar(
        ell,
        cl_ee * ell,
        yerr=np.sqrt(np.diag(cov_cl_ee)) * ell,
        label=labels[i],
        color=colors[i],
        fmt=markers[i],
        capsize=2
    )

ax.set_xscale("squareroot")
ax.set_xticks(np.array([100, 400, 900, 1600]))
ax.minorticks_on()
ax.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
ax.set_xticks(minor_ticks, minor=True)
ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.legend()

ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$\ell \, C_\ell^{BB}$")

plt.tight_layout()

plt.savefig('./plots/data_vectors_cell_bb.png', dpi=200)
plt.show()
# %%
