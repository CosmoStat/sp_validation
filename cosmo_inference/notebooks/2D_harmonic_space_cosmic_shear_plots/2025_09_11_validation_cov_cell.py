# %%
from IPython import get_ipython

ipython = get_ipython()

# enable autoreload for interactive sessions
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from matplotlib import scale as mscale
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from tqdm import tqdm
import healpy as hp

from sp_validation.utils_cosmo_val import SquareRootScale

mscale.register_scale(SquareRootScale)

plt.style.use(
    '/home/guerrini/matplotlib_config/paper.mplstyle'
)

sns.set_palette("husl")

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# %%
path_cov_namaster = "/home/guerrini/sp_validation/notebooks/cosmo_val/output/pseudo_cl_cov_SP_v1.4.6_leak_corr.fits"
path_glass_sims_output = "/n09data/guerrini/glass_mock_v1.4.6_rerun/results/"

cov_namaster = fits.open(path_cov_namaster)


# %%
#Get covariance from GLASS mock

n_sims = 350
cls_all = np.array([]).reshape((0, 32))
for i in tqdm(range(n_sims)):
    cls = np.load(f"{path_glass_sims_output}/cl_glass_mock_{str(i+1).zfill(5)}_4096.npy")
    cls_all = np.vstack((cls_all, cls[1]))

# %%
cov_sim = np.cov(cls_all.T)
# %%
def cov_to_corr(cov):
    diag = np.sqrt(np.diag(cov))
    corr = cov / np.outer(diag, diag)
    return corr

# %%
def get_cov_from_one_cov(cov_one_cov, gaussian=True):
    """
    Returns a numpy array with the covariance matrix from the OneCovariance output.
    """

    n_bins = np.sqrt(cov_one_cov.shape[0]).astype(int)
    cov = np.zeros((n_bins, n_bins))

    index_value = 10 if gaussian else 9
    for i in range(n_bins):
        for j in range(n_bins):
            cov[i, j] = cov_one_cov[i * n_bins + j, index_value]
    
    return cov

# %%
ell_eff = cls[0]

# %%
cov_one_cov = np.genfromtxt("/home/guerrini/sp_validation/notebooks/cosmo_val/output/pseudo_cl_cov_onecov_SP_v1.4.6_leak_corr/covariance_list_3x2pt_pure_Cell.dat")

gaussian_one_cov = get_cov_from_one_cov(cov_one_cov, gaussian=True)
all_one_cov = get_cov_from_one_cov(cov_one_cov, gaussian=False)

# %%
tick_labels = [str(int(ell_eff[i])) for i in [0, 10, 20, 30]]
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 3))
plt.subplots_adjust(wspace=0.3)

im0 = ax0.imshow(cov_to_corr(cov_sim), vmin=-1, vmax=1, cmap='coolwarm')
ax0.set_title("GLASS mocks")
divider = make_axes_locatable(ax0)
cax0 = divider.append_axes("right", size="5%", pad=0.1)
cbar0 = fig.colorbar(im0, cax=cax0)
ax0.set_xticks([0, 10, 20, 30])
ax0.set_yticks([0, 10, 20, 30])
ax0.set_xticklabels(tick_labels)
ax0.set_yticklabels(tick_labels)
ax0.set_xlabel(r"$\ell$", fontsize=14)
ax0.set_ylabel(r"$\ell$", fontsize=14)

im1 = ax1.imshow(cov_to_corr(cov_namaster["COVAR_EE_EE"].data), vmin=-1, vmax=1, cmap='coolwarm')
ax1.set_title("iNKA")
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.1)
cbar1 = fig.colorbar(im1, cax=cax1)
ax1.set_xticks([0, 10, 20, 30])
ax1.set_yticks([0, 10, 20, 30])
ax1.set_xticklabels(tick_labels)
ax1.set_yticklabels(tick_labels)
ax1.set_xlabel(r"$\ell$", fontsize=14)
ax1.set_ylabel(r"$\ell$", fontsize=14)

im2 = ax2.imshow(cov_to_corr(gaussian_one_cov), vmin=-1, vmax=1, cmap='coolwarm')
ax2.set_title("OneCovariance (Gaussian only)")
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.1)
cbar2 = fig.colorbar(im2, cax=cax2)
ax2.set_xticks([0, 10, 20, 30])
ax2.set_yticks([0, 10, 20, 30])
ax2.set_xticklabels(tick_labels)
ax2.set_yticklabels(tick_labels)
ax2.set_xlabel(r"$\ell$", fontsize=14)
ax2.set_ylabel(r"$\ell$", fontsize=14)

#fig.suptitle("Comparison of correlation matrices for $C_\ell^{EE}$")

plt.savefig("./plots/corr_matrix_comparison", dpi=300, bbox_inches='tight')
plt.show()

# %%
#Plot correlation matrix versus the Gaussian part.
cov_one_cov_non_gaussian = all_one_cov - gaussian_one_cov


fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 3))
plt.subplots_adjust(wspace=0.3)

im0 = ax0.imshow(cov_to_corr(cov_namaster["COVAR_EE_EE"].data + cov_one_cov_non_gaussian), vmin=-1, vmax=1, cmap='coolwarm')
ax0.set_title("iNKA + OneCovariance (NG)")
divider = make_axes_locatable(ax0)
cax0 = divider.append_axes("right", size="5%", pad=0.1)
cbar0 = fig.colorbar(im0, cax=cax0)
ax0.set_xticks([0, 10, 20, 30])
ax0.set_yticks([0, 10, 20, 30])
ax0.set_xticklabels(tick_labels)
ax0.set_yticklabels(tick_labels)
ax0.set_xlabel(r"$\ell$", fontsize=14)
ax0.set_ylabel(r"$\ell$", fontsize=14)

im1 = ax1.imshow(cov_to_corr(all_one_cov), vmin=-1, vmax=1, cmap='coolwarm')
ax1.set_title("OneCovariance (All terms)")
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.1)
cbar1 = fig.colorbar(im1, cax=cax1)
ax1.set_xticks([0, 10, 20, 30])
ax1.set_yticks([0, 10, 20, 30])
ax1.set_xticklabels(tick_labels)
ax1.set_yticklabels(tick_labels)
ax1.set_xlabel(r"$\ell$", fontsize=14)
ax1.set_ylabel(r"$\ell$", fontsize=14)

diag = np.sqrt(np.diag(all_one_cov))
non_gaussian_corr = cov_one_cov_non_gaussian / np.outer(diag, diag)
im2 = ax2.imshow(non_gaussian_corr, cmap='coolwarm')
ax2.set_title("OneCovariance (Non-Gaussian only)")
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.1)
cbar2 = fig.colorbar(im2, cax=cax2)
ax2.set_xticks([0, 10, 20, 30])
ax2.set_yticks([0, 10, 20, 30])
ax2.set_xticklabels(tick_labels)
ax2.set_yticklabels(tick_labels)
ax2.set_xlabel(r"$\ell$", fontsize=14)
ax2.set_ylabel(r"$\ell$", fontsize=14)

#fig.suptitle("Comparison of correlation matrices for $C_\ell^{EE}$")

plt.savefig("./plots/corr_matrix_comparison_non_gaussian", dpi=300, bbox_inches='tight')
plt.show()

# %%
ell = cls[0]
plt.figure()

plt.plot(ell, np.sqrt(np.diag(cov_sim)), label="GLASS mocks")
plt.plot(ell[1:], np.sqrt(np.abs(np.diag(cov_sim, k=1))), label="GLASS mocks (k=1)", linestyle='--', color='C0')
plt.plot(ell, np.sqrt(np.diag(cov_namaster["COVAR_EE_EE"].data)), label="NaMaster")
plt.plot(ell[1:], np.sqrt(np.abs(np.diag(cov_namaster["COVAR_EE_EE"].data, k=1))), label="NaMaster (k=1)", linestyle='--', color='C1')
plt.plot(ell, np.sqrt(np.diag(gaussian_one_cov)), label="OneCovariance (Gaussian only)")
#plt.plot(ell, np.diag(cov_namaster_glass), label="NaMaster (GLASS input)")
plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.yscale('log')
plt.ylabel(r"$\sigma(C_\ell^{EE})$")
plt.xlabel(r"$\ell$")
plt.legend()
plt.savefig("./plots/errorbar_comparison", dpi=300, bbox_inches='tight')

plt.show()

# %%
#Plot relative error of the errorbars on the first diagonal
ell = cls[0]
plt.figure()

plt.plot(ell, np.sqrt(np.diag(cov_sim))/np.sqrt(np.diag(cov_namaster["COVAR_EE_EE"].data)), label="GLASS mocks")
plt.plot(ell, np.sqrt(np.diag(cov_namaster["COVAR_EE_EE"].data))/np.sqrt(np.diag(cov_namaster["COVAR_EE_EE"].data)), label="NaMaster")
plt.plot(ell, np.sqrt(np.diag(gaussian_one_cov))/np.sqrt(np.diag(cov_namaster["COVAR_EE_EE"].data)), label="OneCovariance (Gaussian only)")
#plt.plot(ell, np.diag(cov_namaster_glass), label="NaMaster (GLASS input)")
plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
#plt.yscale('log')
plt.ylabel(r"$\sigma(C_\ell^{EE})$/$\sigma(C_\ell^{EE})_{NaMaster}$")
plt.xlabel(r"$\ell$")
plt.legend()
plt.savefig("./plots/relative_errorbar_comparison", dpi=300, bbox_inches='tight')

plt.show()

# %%
plt.figure()

plt.plot(ell, np.sqrt(np.diag(cov_sim)), label="GLASS mocks")
plt.plot(ell, np.sqrt(np.diag(cov_namaster["COVAR_EE_EE"].data + cov_one_cov_non_gaussian)), label="NaMaster + NG OneCov")
plt.plot(ell, np.sqrt(np.diag(all_one_cov)), label="OneCovariance (All terms)")
plt.plot(ell, np.sqrt(np.diag(cov_one_cov_non_gaussian)), label="OneCovariance (Non-Gaussian only)", linestyle='--', color='C2')
plt.plot(ell[1:], np.sqrt(np.abs(np.diag(all_one_cov, k=1))), label="OneCovariance (k=1)", linestyle='-.', color='C2')
plt.plot(ell[1:], np.sqrt(np.abs(np.diag(cov_sim, k=1))), label="GLASS mocks (k=1)", linestyle='-.', color='C0')
#plt.plot(ell, np.diag(cov_namaster_glass), label="NaMaster (GLASS input)")
plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.yscale('log')
plt.ylabel(r"$\sigma(C_\ell^{EE})$")
plt.xlabel(r"$\ell$")
plt.legend()
plt.savefig("./plots/errorbar_comparison_non_gaussian", dpi=300, bbox_inches='tight')

plt.show()

# %%
plt.figure()

plt.plot(ell, np.sqrt(np.diag(cov_sim))/np.sqrt(np.diag(cov_namaster["COVAR_EE_EE"].data + cov_one_cov_non_gaussian)), label="GLASS mocks")
plt.plot(ell, np.sqrt(np.diag(cov_namaster["COVAR_EE_EE"].data))/np.sqrt(np.diag(cov_namaster["COVAR_EE_EE"].data + cov_one_cov_non_gaussian)), label="NaMaster")
plt.plot(ell, np.sqrt(np.diag(all_one_cov))/np.sqrt(np.diag(cov_namaster["COVAR_EE_EE"].data + cov_one_cov_non_gaussian)), label="OneCovariance (All terms)")
#plt.plot(ell, np.diag(cov_namaster_glass), label="NaMaster (GLASS input)")
plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
#plt.yscale('log')
plt.ylabel(r"$\sigma(C_\ell^{EE})$/$\sigma(C_\ell^{EE})_\mathrm{NaMaster + nG OneCov}$")
plt.xlabel(r"$\ell$")
plt.legend()
plt.savefig("./plots/relative_errorbar_comparison_non_gaussian", dpi=300, bbox_inches='tight')

plt.show()

# %%
cov_one_cov = np.genfromtxt("/home/guerrini/OneCovariance/output/covariance_list_3x2pt_pure_Cell.dat")
cov_namaster_glass = fits.open("/home/guerrini/sp_validation/glass_mock/output/pseudo_cl_cov_SP_v1.4.5_glass_mock.fits")["COVAR_EE_EE"].data
ell = cls[0]
plt.figure()

#plt.plot(ell, np.diag(cov_sim), label="GLASS mocks")
#plt.plot(ell, np.diag(cov_namaster["COVAR_EE_EE"].data), label="NaMaster")
plt.plot(ell, np.diag(get_cov_from_one_cov(cov_one_cov, gaussian=True))/np.diag(cov_namaster["COVAR_EE_EE"].data), label="OneCovariance")
plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.yscale('log')
plt.legend()

plt.show()
# %%
#Get the covariance for BB
cls_all_BB = np.array([]).reshape((0, 32))
for i in tqdm(range(n_sims)):
    cls = np.load(f"{path_glass_sims_output}/cl_glass_mock_{str(i+1).zfill(5)}_4096.npy")
    cls_all_BB = np.vstack((cls_all_BB, cls[4]))

cov_sim_BB = np.cov(cls_all_BB.T)
cov_namaster_BB = cov_namaster["COVAR_BB_BB"].data
# %%
plt.figure()

plt.plot(ell, np.sqrt(np.diag(cov_sim_BB)), label="GLASS mocks")
plt.plot(ell, np.sqrt(np.diag(cov_namaster["COVAR_BB_BB"].data)), label="NaMaster")
plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.yscale('log')
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\sigma(C_\ell^{BB})$")
plt.legend()

plt.show()

# %%
# Produce paper quality plot

# Main plot: comparing error bars between NaMaster and GLASS mocks
# Bottom pannel with the relative difference between the two
cov_inka_onecov_ng = cov_namaster["COVAR_EE_EE"].data + cov_one_cov_non_gaussian
fig = plt.figure()

gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.0)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

ax1.plot(ell, np.sqrt(np.diag(cov_sim)), label="GLASS mocks")
ax1.plot(ell, np.sqrt(np.diag(cov_inka_onecov_ng)), label="iNKA + OneCovariance (NG)")
ax1.plot(ell, np.sqrt(np.diag(all_one_cov)), label="OneCovariance", color='C2')
ax1.plot([], [], label="iNKA (Gaussian only)", color='C4')  # Empty plot for legend

#Plot second diagonal
ax1.plot(ell[1:], np.sqrt(np.abs(np.diag(cov_sim, k=1))), linestyle='--', color='C0')
ax1.plot(ell[1:], np.sqrt(np.abs(np.diag(cov_inka_onecov_ng, k=1))), linestyle='--', color='C1')
ax1.plot([], [], linestyle='--', color='k', label='Second diagonal')

#Set ticks and scales for the x-axis
ax1.set_xscale('squareroot')
ax1.set_xticks(np.array([100, 400, 900, 1600]))
ax1.minorticks_on()
ax1.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
ax1.set_xticks(minor_ticks, minor=True)

# Set ticks and scale for the y-axis
ax1.set_yscale('log')
ax1.set_yticks([1e-12, 1e-11, 1e-10, 1e-9, 1e-8])
ax1.tick_params(axis='y', which='minor', length=2, width=0.8)
minor_ticks = [i * 1e-12 for i in range(1, 10)] + [i * 1e-11 for i in range(1, 10)] + [i * 1e-10 for i in range(1, 10)] + [i * 1e-9 for i in range(1, 10)]
ax1.set_yticks(minor_ticks, minor=True)

# Set labels and legend
ax1.set_ylabel(r"$\sigma(C_\ell^{EE})$")
ax1.legend()

relative_error_sim = np.sqrt(np.diag(cov_sim))/np.sqrt(np.diag(cov_inka_onecov_ng))
relative_error_onecov = np.sqrt(np.diag(all_one_cov))/np.sqrt(np.diag(cov_inka_onecov_ng))
relative_error_inka_gaussian = np.sqrt(np.diag(cov_namaster["COVAR_EE_EE"].data))/np.sqrt(np.diag(cov_inka_onecov_ng))

ax2.plot(ell, relative_error_sim - 1, label="GLASS mocks")
ax2.plot(ell, relative_error_onecov - 1, label="OneCovariance", color='C2')
ax2.plot(ell, relative_error_inka_gaussian - 1, label="iNKA (Gaussian only)", color='C4')
ax2.axhline(0, color='C1', linestyle='-')

#Set ticks and scales for the y-axis
ax2.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
ax2.set_ylim(-0.2, 0.2)
ax2.set_ylabel(r"$ \hat{\sigma}(C_\ell^{EE}) / \sigma(C_\ell^{EE})$")
ax2.set_xlabel(r"$\ell$")

plt.savefig("./plots/paper_plot_errorbar_validation.png", dpi=300, bbox_inches='tight')
plt.show()

# %%

#Validation against Gaussian simulations

path_gaussian_sims = "/n17data/sguerrini/sp_validation/notebooks/cosmo_val/harmonic_covariance_gaussian_sims/"

n_sims = 4000
cls_all_gaussian = np.array([]).reshape((0, 4, 32))
cls_noise_gaussian = np.array([]).reshape((0, 4, 32))

for i in tqdm(range(n_sims)):
    try:
        cls = np.load(f"{path_gaussian_sims}/sample_{i}.npz")
        cls_all_gaussian = np.vstack((cls_all_gaussian, cls['cl_all'][None, ...]))
        cls_noise_gaussian = np.vstack((cls_noise_gaussian, cls['cl_noise'][None, ...]))
    except Exception as e:
        print(f"Error loading {i}: {e}")
# %%
cls_all_gaussian = cls_all_gaussian - np.mean(cls_noise_gaussian, axis=0)
# %%
n_sims = cls_all_gaussian.shape[0]
cls_all_gaussian = cls_all_gaussian.reshape((n_sims, -1))

cov_sim_gaussian = np.cov(cls_all_gaussian.T)
# %%
diag_ee = np.sqrt(np.diag(cov_sim_gaussian[:32, :32]))
diag_bb = np.sqrt(np.diag(cov_sim_gaussian[96:, 96:]))
# %%
plt.figure()

plt.plot(ell, diag_ee, label="EE from Gaussian sims")
plt.plot(ell, np.sqrt(np.diag(cov_namaster["COVAR_EE_EE"].data)), label="EE from NaMaster")
plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.yscale('log')
plt.ylabel(r"$\sigma(C_\ell^{EE})$")
plt.xlabel(r"$\ell$")
plt.legend()
plt.show()

# %%
plt.figure()

plt.plot(ell, diag_bb, label="BB from Gaussian sims")
plt.plot(ell, np.sqrt(np.diag(cov_namaster["COVAR_BB_BB"].data)), label="BB from NaMaster")
plt.plot(ell, np.sqrt(np.diag(cov_sim_BB)), label="BB from GLASS mocks")
plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.yscale('log')
plt.ylabel(r"$\sigma(C_\ell^{BB})$")
plt.xlabel(r"$\ell$")
plt.legend()
plt.savefig("./plots/errorbar_namaster_vs_gaussian_BB.png", dpi=300)
plt.show()

# %%
plt.figure()

plt.plot(ell, diag_ee/np.sqrt(np.diag(cov_namaster["COVAR_EE_EE"].data)) - 1, label="EE from Gaussian sims")

plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.ylabel(r"$\Delta \sigma(C_\ell^{EE})$/$\sigma(C_\ell^{EE})_{NaMaster}$")
plt.xlabel(r"$\ell$")
plt.axhline(0, color='k', linestyle='--')
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.legend()

plt.show()
# %%
plt.figure()

plt.plot(ell, diag_bb/np.sqrt(np.diag(cov_namaster["COVAR_BB_BB"].data)) - 1, label="BB from Gaussian sims")

plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.ylabel(r"$\Delta \sigma(C_\ell^{BB})$/$\sigma(C_\ell^{BB})_{NaMaster}$")
plt.xlabel(r"$\ell$")
plt.axhline(0, color='k', linestyle='--')
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.legend()
plt.show()
# %%
# Plot comparison iNKA errorbars between versions
base_dir = "/home/guerrini/sp_validation/notebooks/cosmo_val/output/"
versions = ["SP_v1.4.5_leak_corr", "SP_v1.4.6_leak_corr", "SP_v1.4.8_leak_corr"]
labels = ["v1.4.5 w/ leakage corr.", "v1.4.6 w/ leakage corr.", "v1.4.6 w/ leakage corr."]
colors = ["C0", "C1", "C2"]

plt.figure()

for i, ver in enumerate(versions):
    cov_cell = fits.open(f"{base_dir}/pseudo_cl_cov_g_ng_iNKA_{ver}.fits")
    cov = cov_cell["COVAR_FULL"].data

    plt.plot(
        ell,
        np.sqrt(np.diag(cov)),
        label=labels[i],
        color=colors[i]
    )

plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.yscale('log')
plt.ylabel(r"$\sigma(C_\ell^{EE})$")
plt.xlabel(r"$\ell$")
plt.legend()

plt.savefig("./plots/errorbar_iNKA_versions_comparison", dpi=300, bbox_inches='tight')
plt.show()
# %%
#Plot relative errorbars w.r.t 1.4.5
plt.figure()

cov_cell_145 = fits.open(f"{base_dir}/pseudo_cl_cov_g_ng_iNKA_SP_v1.4.5_leak_corr.fits")
cov_145 = cov_cell_145["COVAR_FULL"].data

for i, ver in enumerate(versions):
    cov_cell = fits.open(f"{base_dir}/pseudo_cl_cov_g_ng_iNKA_{ver}.fits")
    cov = cov_cell["COVAR_FULL"].data

    plt.plot(
        ell,
        np.sqrt(np.diag(cov))/np.sqrt(np.diag(cov_145)),
        label=labels[i],
        color=colors[i]
    )

plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
#plt.yscale('log')
plt.ylabel(r"$\sigma(C_\ell^{EE})$/$\sigma(C_\ell^{EE})_{v1.4.5}$")
plt.xlabel(r"$\ell$")
plt.legend()

plt.savefig("./plots/relative_errorbar_iNKA_versions_comparison", dpi=300, bbox_inches='tight')
plt.show()


# %%
