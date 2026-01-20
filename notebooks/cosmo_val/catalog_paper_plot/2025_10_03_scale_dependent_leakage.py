# %%
import IPython

ipython = IPython.get_ipython()

# enable autoreload for interactive sessions
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import numpy as np
import statsmodels.api as sm
from astropy.io import fits
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from shear_psf_leakage.rho_tau_stat import RhoStat, TauStat, PSFErrorFit

plt.style.use(
    "/home/guerrini/matplotlib_config/paper.mplstyle"
)

plt.rcParams["text.usetex"] = True
sns.set_palette("colorblind")

#Matplotlib inline if in jupyter
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

path_cat = "/n17data/sguerrini/unions_shapepipe_cut_struc_2024_v1.4.5_rerun.fits"

num_bins = 20

# %%
versions = ["SP_v1.4.5", "SP_v1.4.5_leak_corr"]
labels = ["SP v1.4.5", "SP v1.4.5 w/ leakage corr."]
colors = ["C0", "C4"]

tau_stats = []
cov_taus = []
rho_stats = []
cov_rhos = []

base_dir = "/home/guerrini/sp_validation/notebooks/cosmo_val/output/rho_tau_stats/"

for ver in versions:
    tau_stats.append(fits.getdata(f"{base_dir}/tau_stats_{ver}.fits"))
    cov_taus.append(np.load(f"{base_dir}/cov_tau_{ver}_th.npy"))
    rho_stats.append(fits.getdata(f"{base_dir}/rho_stats_{ver}.fits"))
    cov_rhos.append(np.load(f"{base_dir}/cov_rho_{ver}.npy"))

# %%
alpha_leakages = []
alpha_leakages_err = []

for i, ver in enumerate(versions):
    alpha = tau_stats[i]['tau_0_p'] / rho_stats[i]['rho_0_p']
    alpha_leakages.append(alpha)
    
    var_tau = np.diag(cov_taus[i])[:num_bins]
    var_rho = np.diag(cov_rhos[i])[:num_bins]
    var_alpha = (var_tau / rho_stats[i]['rho_0_p']**2) + (tau_stats[i]['tau_0_p']**2 * var_rho / rho_stats[i]['rho_0_p']**4)
    sigma_alpha = np.sqrt(var_alpha)
    alpha_leakages_err.append(sigma_alpha)

# %%
plt.figure()

for i, (ver, color, label, alpha, alpha_err) in enumerate(zip(versions, colors, labels, alpha_leakages, alpha_leakages_err)):
    plt.errorbar(
        tau_stats[i]['theta'],
        alpha,
        yerr=alpha_err,
        fmt='o',
        color=color,
        label=label,
        capsize= 5
    )

plt.xlabel(r"$\vartheta$ [arcmin]")
plt.ylabel(r"$\alpha(\vartheta)$")

plt.xscale('log')
plt.legend()
plt.savefig('./plots/scale_dependent_alpha.png', dpi=200)
plt.show()

# %%
#Check the computation of error bars are correct.
tau_stats_sample = np.random.normal(
    loc=tau_stats[0]['tau_0_p'],
    scale=np.sqrt(np.diag(cov_taus[0])[:num_bins]),
    size=(10_000, num_bins)
)
rho_stats_sample = np.random.normal(
    loc=rho_stats[0]['rho_0_p'],
    scale=np.sqrt(np.diag(cov_rhos[0])[:num_bins]),
    size=(10_000, num_bins)
)

alpha_samples = tau_stats_sample / rho_stats_sample

sigma_alpha = np.std(alpha_samples, axis=0)
# %%
plt.figure()

plt.plot(sigma_alpha)
plt.plot(alpha_leakages_err[0], linestyle='--')

plt.show()

# %%
#Plot scale dependent xi_sys
xi_sys = []
xi_sys_err = []

for i, ver in enumerate(versions):

    xi_sys_i = alpha_leakages[i]**2 * rho_stats[i]['rho_0_p']
    xi_sys.append(xi_sys_i)

    rho_stats_sample = np.random.normal(
        loc=rho_stats[i]['rho_0_p'],
        scale=np.sqrt(np.diag(cov_rhos[i])[:num_bins]),
        size=(10_000, num_bins)
    )

    tau_stats_sample = np.random.normal(
        loc=tau_stats[i]['tau_0_p'],
        scale=np.sqrt(np.diag(cov_taus[i])[:num_bins]),
        size=(10_000, num_bins)
    )

    xi_sys_samples = tau_stats_sample**2 / rho_stats_sample
    sigma_xi_sys = np.std(xi_sys_samples, axis=0)

    xi_sys_err.append(sigma_xi_sys)

# %%
plt.figure()

for i, (ver, color, label, xi_sys_i, xi_sys_err_i) in enumerate(zip(versions, colors, labels, xi_sys, xi_sys_err)):
    plt.errorbar(
        tau_stats[i]['theta'],
        xi_sys_i,
        yerr=xi_sys_err_i,
        fmt='o',
        color=color,
        label=label,
        capsize= 5
    )

plt.xlabel(r"$\vartheta$ [arcmin]")
plt.ylabel(r"$\xi^\mathrm{sys}_+(\vartheta)$")

plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.savefig('./plots/scale_dependent_xi_sys.png', dpi=200)
plt.show()

# %%
#Comparison with rho/tau predictions

#Inference of the xi_sys parameters
sep_units = 'arcmin'
coord_units = 'degrees'
theta_min = 0.1
theta_max = 250
nbins = 20


TreeCorrConfig_xi = {
    'ra_units': coord_units,
    'dec_units': coord_units,
    'min_sep': theta_min,
    'max_sep': theta_max,
    'sep_units': sep_units,
    'nbins': nbins,
    'var_method':'jackknife',
}

rho_stats_handler = RhoStat(
    output='.',
    treecorr_config=TreeCorrConfig_xi,
    verbose=True
)

tau_stats_handler = TauStat(
    catalogs=rho_stats_handler.catalogs,
    output='.',
    treecorr_config=TreeCorrConfig_xi,
    verbose=True
    )

psf_fitter = PSFErrorFit(rho_stats_handler, tau_stats_handler, base_dir)

plt.figure()

idx = 1
ver = "SP_v1.4.5_leak_corr"
color = colors[idx]

sample = np.load(f"{base_dir}/samples_{ver}.npy")

print("Plotting verion:", ver)
psf_fitter.load_rho_stat('rho_stats_' + ver + '.fits')
quant = 0.84
quantiles = [1 - quant, quant]
xi_psf_sys_samples = np.array([]).reshape(0, num_bins)
for i in range(len(sample)):
    xi_psf_sys = psf_fitter.compute_xi_psf_sys(sample[i])
    xi_psf_sys_samples = np.vstack((xi_psf_sys_samples, xi_psf_sys))

xi_psf_sys_mean = np.mean(xi_psf_sys_samples, axis=0)
xi_psf_sys_std = np.std(xi_psf_sys_samples, axis=0)

plt.errorbar(
    psf_fitter.rho_stat_handler.rho_stats['theta'],
    xi_psf_sys_mean,
    yerr=xi_psf_sys_std,
    fmt='o',
    color=color,
    ls='-',
    label=labels[idx]+r' $\rho/\tau$ model',
    capsize=5
)

plt.errorbar(
    tau_stats[idx]['theta'],
    xi_sys[idx],
    yerr=xi_sys_err[idx],
    fmt='o',
    ls='--',
    color='C3',
    label=labels[idx]+r' direct $\xi^\mathrm{sys}_+$',
    capsize=5
)

plt.xlabel(r"$\vartheta$ [arcmin]")
plt.ylabel(r"$\xi^\mathrm{sys}_+(\vartheta)$")
plt.xscale('log')
plt.yscale('log')

plt.legend()
plt.savefig('./plots/scale_dependent_xi_sys_comparison.png', dpi=200)
plt.show()

    
# %%
