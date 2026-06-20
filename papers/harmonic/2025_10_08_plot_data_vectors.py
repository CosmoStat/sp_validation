# %%
import os

# Trick to plot with tex
os.environ["LD_LIBRARY_PATH"] = ""
os.environ["CONDA_PREFIX"] = "/home/guerrini/.conda/envs/sp_validation_3.11"

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
from sp_validation.rho_tau import SquareRootScale

from utils import get_chi2_and_pte

mscale.register_scale(SquareRootScale)

plt.style.use(
    "./matplotlib_config/paper.mplstyle"
)

plt.rcParams["text.usetex"] = True

sns.set_palette("Dark2")

#Matplotlib inline if in jupyter
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")


# %%
base_dir = "/home/guerrini/sp_validation/cosmo_val/output/"

versions = ["SP_v1.4.5_leak_corr", "SP_v1.4.6_leak_corr", "SP_v1.4.8_leak_corr"]
labels = ["SP v1.4.5 w/ leakage corr.", "SP v1.4.6 w/ leakage corr.", "SP v1.4.8 w/ leakage corr."]
colors = ["C0", "C1", "C2"]
markers = ["o", "h", "x"]

offset = 0.15

# %%
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, nrows=1, figsize= (20, 12))

for i, ver in enumerate(versions):
    cell = fits.getdata(f"{base_dir}/pseudo_cl_{ver}.fits")
    cov_cell = fits.open(f"{base_dir}/pseudo_cl_cov_{ver}.fits")

    ell = cell['ell']
    cl_ee = cell['EE']
    cov_cl_ee = cov_cell['COVAR_EE_EE'].data
    cl_eb = cell['EB']
    cov_cl_eb = cov_cell['COVAR_EB_EB'].data
    chi2_eb, reduced_chi2_eb, pte_eb = get_chi2_and_pte(cl_eb, cov_cl_eb, verbose=False)
    cl_bb = cell['BB']
    cov_cl_bb = cov_cell['COVAR_BB_BB'].data
    chi2_bb, reduced_chi2_bb, pte_bb = get_chi2_and_pte(cl_bb, cov_cl_bb, verbose=False)

    ell_widths = np.diff(ell)
    ell_widths = np.append(ell_widths, ell_widths[-1])  # repeat last width

    # Better jittering: symmetric around original ell values
    jitter_fraction = (i - (len(versions) - 1) / 2) * offset
    jiterred_ell = ell + jitter_fraction * ell_widths

    ax0.errorbar(
        jiterred_ell,
        cl_ee * ell,
        yerr=np.sqrt(np.diag(cov_cl_ee)) * ell,
        label=labels[i],
        color=colors[i],
        fmt=markers[i],
        capsize=2
    )
    ax1.errorbar(
        jiterred_ell,
        cl_eb * ell,
        yerr=np.sqrt(np.diag(cov_cl_eb)) * ell,
        label=labels[i],
        color=colors[i],
        fmt=markers[i],
        capsize=2
    )
    ax2.errorbar(
        jiterred_ell,
        cl_bb * ell,
        yerr=np.sqrt(np.diag(cov_cl_bb)) * ell,
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
ax1.set_ylabel(r"$\ell \, C_\ell^{EB}$")

ax2.set_xscale("squareroot")
ax2.set_xticks(np.array([100, 400, 900, 1600]))
ax2.minorticks_on()
ax2.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
ax2.set_xticks(minor_ticks, minor=True)
ax2.legend()
ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
ax2.set_xlabel(r"$\ell$")
ax2.set_ylabel(r"$\ell \, C_\ell^{BB}$")

plt.tight_layout()

plt.savefig('./plots/data_vectors_cell.png', dpi=200)
plt.show()

# %%
#Same plot with B-modes only
fig, ax = plt.subplots(figsize= (8, 6))



for i, ver in enumerate(versions):
    cell = fits.getdata(f"{base_dir}/pseudo_cl_{ver}.fits")
    cov_cell = fits.open(f"{base_dir}/pseudo_cl_cov_{ver}.fits")

    ell = cell['ell']
    cl_bb = cell['BB']
    cov_cl_bb = cov_cell['COVAR_BB_BB'].data
    chi2, reduced_chi2, pte = get_chi2_and_pte(cl_bb, cov_cl_bb, verbose=False)

    # Better jittering: symmetric around original ell values
    jitter_fraction = (i - (len(versions) - 1) / 2) * offset
    jiterred_ell = ell + jitter_fraction * ell_widths

    ax.errorbar(
        jiterred_ell,
        cl_bb * ell,
        yerr=np.sqrt(np.diag(cov_cl_bb)) * ell,
        label=labels[i],
        color=colors[i],
        fmt=markers[i],
        capsize=2
    )
    ax.text(100, 1.25e-7 - i*0.10e-7, rf"$\chi^2_\nu={reduced_chi2:.3f}$ | PTE={pte:.3f}", color=colors[i])

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

#Same plot with EB and BB modes only of v1.4.6
fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize= (8, 6), sharex=True)

version_paper_plot = ["SP_v1.4.6.3_A", "SP_v1.4.6.3_leak_corr_A"]
labels = ["W/o leakage correction", "With leakage correction"]
colors = ["C0", "C1"]
markers = ["o", "h"]
offset = 0.2
for i, ver in enumerate(version_paper_plot):
    cell = fits.getdata(f"{base_dir}/pseudo_cl_{ver}.fits")
    cov_cell = fits.open(f"{base_dir}/pseudo_cl_cov_{ver}.fits")

    ell = cell['ell']
    cl_eb = cell['EB']
    cov_cl_eb = cov_cell['COVAR_EB_EB'].data
    cl_bb = cell['BB']
    cov_cl_bb = cov_cell['COVAR_BB_BB'].data

    ell_widths = np.diff(ell)
    ell_widths = np.append(ell_widths, ell_widths[-1])
    # Better jittering: symmetric around original ell values
    jitter_fraction = (i - (len(version_paper_plot) - 1) / 2) * offset
    jiterred_ell = ell + jitter_fraction * ell_widths

    chi2, reduced_chi2, pte = get_chi2_and_pte(cl_eb, cov_cl_eb, verbose=False)

    ax1.errorbar(
        jiterred_ell,
        cl_eb * ell,
        yerr=np.sqrt(np.diag(cov_cl_eb)) * ell,
        label=labels[i],
        color=colors[i],
        fmt=markers[i],
        capsize=2
    )
    ax1.text(100, -0.5e-7 - i*0.25e-7, rf"$\chi^2_\nu={reduced_chi2:.3f}$ | PTE={pte:.3f}", color=colors[i])

    chi2, reduced_chi2, pte = get_chi2_and_pte(cl_bb, cov_cl_bb, verbose=False)
    ax2.errorbar(
        jiterred_ell,
        cl_bb * ell,
        yerr=np.sqrt(np.diag(cov_cl_bb)) * ell,
        label=labels[i],
        color=colors[i],
        fmt=markers[i],
        capsize=2
    )
    ax2.text(100, 1.25e-7 - i*0.25e-7, rf"$\chi^2_\nu={reduced_chi2:.3f}$ | PTE={pte:.3f}", color=colors[i])

ax1.set_xscale("squareroot")
ax1.set_xticks(np.array([100, 400, 900, 1600]))
ax1.minorticks_on()
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
ax1.set_xticks(minor_ticks, minor=True)
ax1.tick_params(axis='x', which='minor', length=2, width=0.8)
ax1.axhline(0, color='gray', linestyle='--', linewidth=1)

ax2.set_xscale("squareroot")
ax2.set_xticks(np.array([100, 400, 900, 1600]))
ax2.minorticks_on()
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
ax2.set_xticks(minor_ticks, minor=True)
ax2.tick_params(axis='x', which='minor', length=2, width=0.8)
ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
ax2.legend(fontsize=14) 

ax2.set_xlabel(r"Multipole $\ell$", fontsize=16)
ax1.set_ylabel(r"$\ell \, C_\ell^{EB} \times 10^{-7}$", fontsize=16)
ax2.set_ylabel(r"$\ell \, C_\ell^{BB} \times 10^{-7}$", fontsize=16)

ax1.yaxis.get_offset_text().set_visible(False)
ax2.yaxis.get_offset_text().set_visible(False)

plt.savefig("./plots/paperplot_data_vectors_eb_bb.png", dpi=300, bbox_inches='tight')
#Save pdf
plt.savefig("./plots/paperplot_data_vectors_eb_bb.pdf", bbox_inches='tight')
plt.show()


# %%
