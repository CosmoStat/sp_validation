# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: sp_validation
#     language: python
#     name: sp_validation
# ---

# # Cosmological validation of UNIONS shape catalogues
# 03/2023
# 03/2024

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2


import sys
import os
import numpy as np
import matplotlib.pylab as plt
import sys
import gc
import yaml
import numpy as np
from astropy.io import fits
import treecorr
import pandas as pd
import colorama
import emcee
from uncertainties import ufloat
from astropy.io import ascii

# +
from sp_validation.plot_style import *
from cs_util import plots as cs_plots

from shear_psf_leakage import leakage
from shear_psf_leakage import run_scale
from shear_psf_leakage import run_object
from shear_psf_leakage import plots as psfleak_plots
from shear_psf_leakage.rho_tau_stat import PSFErrorFit

import utils

# +
import treecorr

n_thread = 8
treecorr.set_omp_threads(n_thread)

sep_units = 'arcmin'
coord_units = 'degrees'
# -

import pyccl
pyccl.gsl_params.LENSING_KERNEL_SPLINE_INTEGRATION = False

def color_reset():
    print(colorama.Fore.RED, end='')

def print_blue(msg):
    print(colorama.Fore.BLUE + msg)
    color_reset()

def print_start(msg):
    print()
    print_blue(msg)

def print_done(msg):
    print_blue(msg)

def print_magenta(msg):
    print(colorama.Fore.MAGENTA + msg)
    color_reset()

def print_green(msg):
    print(colorama.Fore.GREEN + msg)
    color_reset()

def print_cyan(msg):
    print(colorama.Fore.CYAN + msg)
    color_reset()


# ## Input parameters
# Catalogue versions
#versions = ['SP_v1.0', 'SP_v1.0_LFmask_8k', 'SP_v1.3', 'SP_v1.3_LFmask_8k', 'SP_axel_v0.0', 'SP_axel_v0.0_repr', 'DES']
#versions = ['SP_v1.3_LFmask_8k', 'SP_v1.3_LFmask_8k_no_alpha' , 'SP_v1.3_LFmask_8k_li_2024']
#versions = ['SP_v1.4_LFmask_8k_noalpha', 'SP_v1.4_LFmask_8k', 'SP_v1.3_LFmask_8k', 'SP_v1.3_LFmask_8k_no_alpha', 'DES']
versions = ['SP_v1.4.1', 'SP_v1.4.1_noleakage']
rho_tau_method = 'lsq' #lsq or emcee
cov_estimate_method = 'th' #or theory/jackknife
compute_cov_rho = False
n_cov = 100 #number of covariance used to marginalize on the patching in the jackknife estimate.
#Put to 1 to avoid recomputing all rho and tau statistics.
#versions = ['SP_v1.0_LFmask_4k', 'SP_v1.0_LFmask_8k', 'SP_v1.3_LFmask_4k', 'SP_v1.3_LFmask_8k']

rho_tau_method = 'lsq' #lsq, emcee, or none

all_keys = ['nz']
for ver in versions:
    all_keys.append(ver)

# Base directory for data, on candide
data_base_dir = '/n17data/mkilbing/astro/data/'

# ## Loading configuration

# +
# Read in dictionary about catalogue info from yaml file
with open('cat_config.yaml', 'r') as file:
    cat = yaml.load(file.read(), Loader=yaml.FullLoader)

# Set full paths
for ver in all_keys:
    for key in cat[ver]:
        if "path" in cat[ver][key]:
            cat[ver][key]["path"] = f"{data_base_dir}/{cat[ver]['subdir']}/{cat[ver][key]['path']}"

if not os.path.exists(cat["paths"]["output"]):
    os.mkdir(cat["paths"]["output"])

# +
# Variables
components = ['+', '-']

# Angular scales for xi_+-
theta_min = 0.1
theta_max = 250
nbins = 20

# Plotting
theta_min_plot = 0.08
theta_max_plot = 250

ylim_alpha = [-0.005, 0.05]

ylim_xi_sys_ratio = [-0.02, 0.5]
# -

# ## Set up
TreeCorrConfig_xi = {
    'ra_units': coord_units,
    'dec_units': coord_units,
    'min_sep': theta_min,
    'max_sep': theta_max,
    'sep_units': sep_units,
    'nbins': nbins,
}

# ## Processing

print_start('Start cosmology validation')


# ### Systematic tests

# +
# Save leakage coeffieicnts values
leakage_coeff = {}

for ver in versions:
    leakage_coeff[ver] = {}
# -

# #### Init dictionary for scale-dependent leakage
# MKDEBUG -> results_scale
results = {}


def set_params_leakage_scale(cat, ver):
    params_in = {}

    # Set parameters
    params_in['input_path_shear'] = cat[ver]["shear"]["path"]
    params_in['input_path_PSF'] = cat[ver]["star"]["path"]
    params_in['dndz_path'] = f"{cat['nz']['dndz']['path']}_{cat[ver]['pipeline']}_{cat['nz']['dndz']['blind']}.txt"
    params_in['output_dir'] = f'{cat["paths"]["output"]}/leakage_{ver}'

    # Note: for SP these are calibrated shear estimates
    params_in['e1_col'] = cat[ver]["shear"]["e1_col"]
    params_in['e2_col'] = cat[ver]["shear"]["e2_col"]
    params_in["R11"]=None if ver != 'DES' else cat[ver]['shear']['R11']
    params_in["R22"]=None if ver != 'DES' else cat[ver]['shear']['R22']

    params_in['ra_star_col'] = cat[ver]["star"]["ra_col"]
    params_in["dec_star_col"] = cat[ver]["star"]["dec_col"]
    params_in['e1_PSF_star_col'] = cat[ver]["star"]["e1_col"]
    params_in["e2_PSF_star_col"] = cat[ver]["star"]["e2_col"]

    params_in["theta_min_amin"] = theta_min
    params_in["theta_max_amin"] = theta_max

    params_in["verbose"] = False

    return params_in


def set_params_leakage_object(cat, ver):
    params_in = {}

    # Set parameters
    params_in['input_path_shear'] = cat[ver]["shear"]["path"]
    params_in['output_dir'] = f'{cat["paths"]["output"]}/leakage_{ver}'

    # Note: for SP these are calibrated shear estimates
    params_in['e1_col'] = cat[ver]["shear"]["e1_col"]
    params_in['e2_col'] = cat[ver]["shear"]["e2_col"]
    
    if "e1_PSF_col" in cat[ver]["shear"] and "e2_PSF_col" in cat[ver]["shear"]:
        params_in['e1_PSF_col'] = cat[ver]["shear"]["e1_PSF_col"]
        params_in["e2_PSF_col"] = cat[ver]["shear"]["e2_PSF_col"]
    else:
        raise KeyError(
            "Keys 'e1_PSF_col' and 'e2_PSF_col' not found in"
            + f" shear yaml entry for version {ver}" 
        )

    params_in["verbose"] = False

    return params_in


for ver in versions:

    # Create leakage instance
    obj = run_scale.LeakageScale()

    # Set instance parameters
    params_in = set_params_leakage_scale(cat, ver)
    for key in params_in:
        obj._params[key] = params_in[key]

    results[ver] = obj


print_start('Read catalogues')
for ver in versions:
    print_magenta(ver)
    obj = results[ver]

    obj.check_params()
    obj.prepare_output()
    obj.read_data()



# #### Rho and tau tatistics

def get_params_rho_tau(params_base, params_psf, survey="other"):

    # Set parameters
    params = params_base
    # TODO to yaml file
    if survey in ("DES", 'SP_axel_v0.0', 'SP_axel_v0.0_repr'):
        params["patch_number"] = 120
        print("DES, jackknife patch number = 120")
    elif survey == 'SP_axel_v0.0':
        params["patch_number"] = 120
        print("SP_Axel_v0.0, jackknife patch number =120")
    elif survey == 'SP_v1.4-P3' or survey == 'SP_v1.4-P3_LFmask':
        params["patch_number"] = 120
        print("SP_v1.4, jackknife patch number =120")
    else:
        params["patch_number"] = 150
    params["ra_col"] = params_psf["ra_col"]
    params["dec_col"] = params_psf["dec_col"]
    params["e1_PSF_col"] = params_psf["e1_PSF_col"]
    params["e2_PSF_col"] = params_psf["e2_PSF_col"]
    params["e1_star_col"] = params_psf["e1_star_col"]
    params["e2_star_col"] = params_psf["e2_star_col"]
    params["PSF_size"] = params_psf["PSF_size"]
    params["star_size"] = params_psf["star_size"]
    if survey != 'DES':
        params["PSF_flag"] = params_psf["PSF_flag"]
        params["star_flag"] = params_psf["star_flag"]
    params["ra_units"] = "deg"
    params["dec_units"] = "deg"

    params["w_col"] = "w"

    return params

out_dir = f"{cat['paths']['output']}/rho_tau_stats"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
print_start('Rho stats')

# Rho and Tau statistics
for ver in versions: 
    rho_stat_handler, tau_stat_handler = utils.get_rho_tau_w_cov(cat, ver, TreeCorrConfig_xi, out_dir, method=cov_estimate_method, cov_rho=compute_cov_rho)


# + Plot rho-statistics
filenames = []
colors = []
for ver in versions:
    filenames.append(f"rho_stats_{ver}.fits")
    colors.append(cat[ver]["colour"])

# Create plot
rho_stat_handler.plot_rho_stats(
    filenames,
    colors,
    versions,
    abs=False,
    savefig='rho_stats.png',
    legend="outside",
)
# -

# + Plot tau-statistics
filenames = []
colors = []
for ver in versions:
    filenames.append(f"tau_stats_{ver}.fits")
    colors.append(cat[ver]["colour"])

# Create plots
tau_stat_handler.plot_tau_stats(
    filenames,
    colors,
    versions,
    savefig='tau_stats.png',
    plot_tau_m=False,
    legend="outside",
)
# -

# ##### Parameter fits

psf_fitter = PSFErrorFit(rho_stat_handler, tau_stat_handler, out_dir)


flat_sample_list = []
result_list = []
q_list = []

for ver in versions:

    params = get_params_rho_tau(results[ver]._params, cat[ver]["psf"], survey=ver)

    if cov_estimate_method == 'sim':
        npatch = 300
    elif cov_estimate_method == 'jk':
        npatch = params['patch_number']
    else:
        npatch = None
    
    flat_samples, result, q = utils.get_samples(psf_fitter, ver, cov_type=cov_estimate_method, apply_debias=npatch, sampler=rho_tau_method)

    flat_sample_list.append(flat_samples)
    result_list.append(result)
    q_list.append(q)

if rho_tau_method != "none":
    psfleak_plots.plot_contours(
        flat_sample_list,
        names=['x0', 'x1', 'x2'],
        labels=[r'\alpha', r'\beta', r'\eta'],
        savefig=out_dir + '/contours_tau_stat.png',
        legend_labels=versions,
        legend_loc='upper right',
        contour_colors=colors,
        markers={'x0':0, 'x1':1, 'x2':1}
    )

    plt.figure(figsize=(15, 6))
    for mcmc_result, ver, color, flat_sample in zip(
        result_list,
        versions,
        colors,
        flat_sample_list
    ):
        psf_fitter.load_rho_stat('rho_stats_' + ver + '.fits')
        for i in range(100):
            psf_fitter.plot_xi_psf_sys(flat_sample[-i+1], ver, color, alpha=0.1)
        psf_fitter.plot_xi_psf_sys(mcmc_result[1], ver, color)
    plt.legend()
    cs_plots.savefig(f"{out_dir}/xi_psf_sys_samples.png")

    xi_psf_sys_mean = {}
    xi_psf_sys_var = {}

    plt.figure(figsize=(15, 6))
    quant = 0.683
    quantiles = [1 - quant, quant]
    for mcmc_result, ver, color, flat_sample in zip(
        result_list,
        versions,
        colors,
        flat_sample_list
    ):
        psf_fitter.load_rho_stat('rho_stats_' + ver + '.fits')
        nbins = psf_fitter.rho_stat_handler._treecorr_config["nbins"]
        xi_psf_sys_samples = np.array([]).reshape(0, nbins)

        for i in range(len(flat_sample)):
            xi_psf_sys = psf_fitter.compute_xi_psf_sys(flat_sample[i])
            xi_psf_sys_samples = np.vstack([xi_psf_sys_samples, xi_psf_sys])

        xi_psf_sys_mean[ver] = np.mean(xi_psf_sys_samples, axis=0)
        xi_psf_sys_var[ver] = np.var(xi_psf_sys_samples, axis=0)
        xi_psf_sys_quan = np.quantile(xi_psf_sys_samples, quantiles, axis=0)
        theta = psf_fitter.rho_stat_handler.rho_stats["theta"]
        ls = cat[ver]["ls"]
        plt.plot(theta, xi_psf_sys_mean[ver], linestyle=ls, color=color)
        plt.plot(theta, xi_psf_sys_quan[0], linestyle=ls, color=color)
        plt.plot(theta, xi_psf_sys_quan[1], linestyle=ls, color=color)
        plt.fill_between(theta, xi_psf_sys_quan[0], xi_psf_sys_quan[1], color=color, alpha=0.25, label=ver)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$\theta$ [arcmin]")
    plt.ylabel(r"$\xi^{\rm PSF}_{\rm sys}$")
    plt.title(f"{quantiles[0]:.1%}, {quantiles[1]:.1%} quantiles")
    plt.legend()
    cs_plots.savefig(f"{out_dir}/xi_psf_sys_quantiles.png")

    for mcmc_result, ver, flat_sample in zip(
        result_list,
        versions,
        flat_sample_list
    ):
        for yscale in ("linear", "log"):
            psf_fitter.load_rho_stat('rho_stats_' + ver + '.fits')
            out_path = f"{out_dir}/xi_psf_sys_terms_{yscale}_{ver}.png"
            psf_fitter.plot_xi_psf_sys_terms(ver, mcmc_result[1], out_path, yscale=yscale)


# #### Footprint

print_start('Plot footprints')
for ver in versions:
    print_magenta(ver)
    out_path = f"{cat['paths']['output']}/footprint_{ver}.png"
    if os.path.exists(out_path):
        print_green(f'Skipping footprint computation, plot {out_path} exists')
    else:
        print_cyan("Compute footprint")
        plt.clf()
        plt.plot(
            results[ver].dat_shear["RA"],
            results[ver].dat_shear["Dec"],
            ".",
            markersize=0.5
        )
        plt.xlabel("R.A. [deg]")
        plt.ylabel("Dec [deg]")
        cs_plots.savefig(out_path)
print_done('Done plotting')

# #### $\xi_\textrm{sys}$ and scale-dependent leakage

print_start('Scale-dependent leakage')
for ver in versions:
    print_magenta(ver)
    obj = results[ver]

    output_base_path = f'{cat["paths"]["output"]}/leakage_{ver}/xi_for_leak_scale'
    output_path_ab = f"{output_base_path}_a_b.txt"
    output_path_aa = f"{output_base_path}_a_a.txt"
    if os.path.exists(output_path_ab) and os.path.exists(output_path_aa):
        print_green(f'Skipping computation, reading {output_path_ab} and {output_path_aa} instead')

        # Todo: Use TreeCorrConfig_xi
        TreeCorrConfig = {
            'ra_units': coord_units,
            'dec_units': coord_units,
            'min_sep': theta_min,
            'max_sep': theta_max,
            'sep_units': sep_units,
            'nbins': obj._params["n_theta"],
            'var_method':'jackknife',
        }
        obj.r_corr_gp = treecorr.GGCorrelation(TreeCorrConfig)
        obj.r_corr_gp.read(output_path_ab)

        obj.r_corr_pp = treecorr.GGCorrelation(TreeCorrConfig)
        obj.r_corr_pp.read(output_path_aa)

    else:
        obj.compute_corr_gp_pp_alpha(output_base_path=output_base_path)

    obj.do_alpha(fast=True)
    obj.do_xi_sys()

print_done('Done scale-dependent leakage')

# +
# Plot scale-dependent leakage

theta = []
y = []
yerr = []
labels = []
colors = []
linestyles = []
markers = []

for ver in versions:

    if hasattr(results[ver], "r_corr_gp"):
        theta.append(results[ver].r_corr_gp.meanr)
        y.append(results[ver].alpha_leak)
        yerr.append(results[ver].sig_alpha_leak)
        labels.append(ver)
        colors.append(cat[ver]["colour"])
        linestyles.append(cat[ver]["ls"])
        markers.append(cat[ver]["marker"])

if len(theta) > 0:

    # Log x
    out_path = f"{cat['paths']['output']}/alpha_leak_log.pdf"

    title = r'$\alpha$ leakage'
    xlabel = r'$\theta$ [arcmin]'
    ylabel = r'$\alpha(\theta)$'
    cs_plots.plot_data_1d(
        theta,
        y,
        yerr,
        title,
        xlabel,
        ylabel,
        out_path=None,
        xlog=True,
        xlim=[theta_min_plot, theta_max_plot],
        ylim=ylim_alpha,
        labels=labels,
        colors=colors,
        linestyles=linestyles,
        shift_x=True,
    )
    cs_plots.savefig(out_path)

    # Lin x
    out_path = f"{cat['paths']['output']}/alpha_leak_lin.pdf"

    title = r'$\alpha$ leakage'
    xlabel = r'$\theta$ [arcmin]'
    ylabel = r'$\alpha(\theta)$'
    cs_plots.plot_data_1d(
        theta,
        y,
        yerr,
        title,
        xlabel,
        ylabel,
        out_path=None,
        xlog=False,
        xlim=[-10, theta_max_plot],
        ylim=ylim_alpha,
        labels=labels,
        colors=colors,
        linestyles=linestyles,
        shift_x=False,
    )
    cs_plots.savefig(out_path)


# +
# Plot xi_sys

y = []
yerr = []
colors = []
linestyles = []

for ver in versions:
    if hasattr(results[ver], "C_sys_p"):
        y.append(results[ver].C_sys_p)
        yerr.append(results[ver].C_sys_std_p)
        labels.append(ver)
        colors.append(cat[ver]["colour"])
        linestyles.append(cat[ver]["ls"])

if len(y) > 0:
    xlabel = r'$\theta$ [arcmin]'
    ylabel = r'$\xi^{\rm sys}_+(\theta)$'
    title = 'Cross-correlation leakage'
    out_path = f"{cat['paths']['output']}/xi_sys_p"
    fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
    cs_plots.plot_data_1d(
        theta,
        y,
        yerr,
        title,
        xlabel,
        ylabel,
        out_path=None,
        labels=labels,
        xlog=True,
        xlim=[theta_min_plot, theta_max_plot],
        colors=colors,
        linestyles=linestyles,
        #shift_x=True,
    )
    cs_plots.savefig(out_path)

y = []
yerr = []
for ver in versions:
    if hasattr(results[ver], "C_sys_m"):
        y.append(results[ver].C_sys_m)
        yerr.append(results[ver].C_sys_std_m)

if len(y) > 0:
    xlabel = r'$\theta$ [arcmin]'
    ylabel = r'$\xi^{\rm sys}_-(\theta)$'
    title = 'Cross-correlation leakage'
    out_path = f"{cat['paths']['output']}/xi_sys_m"
    fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
    cs_plots.plot_data_1d(
        theta,
        y,
        yerr,
        title,
        xlabel,
        ylabel,
        out_path=None,
        labels=labels,
        xlog=True,
        xlim=[theta_min_plot, theta_max_plot],
        ylim=[-1e-7, 1e-6],
        colors=colors,
        linestyles=linestyles,
        #shift_x=True,
    )
    cs_plots.savefig(out_path)


""" # #### Object-wise leakage
results_object = {}
for ver in versions:

    # Create leakage instance
    obj = run_object.LeakageObject()

    # Set instance parameters
    params_in = set_params_leakage_object(cat, ver)
    for key in params_in:
        obj._params[key] = params_in[key]

    results_object[ver] = obj


print_start("Compute object-wise leakage")
mix = True
order = "lin"
for ver in versions:
    print_magenta(ver)

    obj = results_object[ver]
    obj.check_params()
    obj.update_params()
    obj.prepare_output()

    # Skip read_data() and copy catalogue from scale instance instead
    obj._dat = results[ver].dat_shear

    out_base = obj.get_out_base(mix, order)
    out_path = f"{out_base}.pkl"
    if os.path.exists(out_path):
        print_green(f"Skipping object-wise leakage, file {out_path} exists")
        obj.par_best_fit = leakage.read_from_file(out_path)
    else:
        print_cyan("Computing object-wise leakage regression")

        # Run
        obj.PSF_leakage()


# ### Leakge coefficients

# Gather coefficients
for ver in versions:
    # Object-wise leakage
    leakage_coeff[ver]["a11"] = ufloat(results_object[ver].par_best_fit["a11"].value, results_object[ver].par_best_fit["a11"].stderr)
    leakage_coeff[ver]["a22"] = ufloat(results_object[ver].par_best_fit["a22"].value, results_object[ver].par_best_fit["a22"].stderr)
    leakage_coeff[ver]["aii_mean"] = 0.5 * (leakage_coeff[ver]["a11"] + leakage_coeff[ver]["a22"])

    # Scale-dependent leakage: mean
    leakage_coeff[ver]["alpha_mean"] = ufloat(results[ver].alpha_leak_mean, results[ver].alpha_leak_std)
    # Scale-dependent leakage: value at smallest scale
    leakage_coeff[ver]["alpha_1"] = ufloat(results[ver].alpha_leak[0], results[ver].sig_alpha_leak[0])
    # Scale-dependent leakage: value extrapolated to 0 using affine model
    leakage_coeff[ver]["alpha_0"] = ufloat(results[ver].alpha_affine_best_fit["c"].value, results[ver].alpha_affine_best_fit["c"].stderr) """


""" # +
# Plot coefficients
fig = cs_plots.figure(figsize=(15, 15))

linestyles = ["-", "--", ":"]
fillstyles = ["full", "none", "left", "right", "bottom", "top"]

for ver in versions:
    label = ver
    for key, ls, fs in zip(["alpha_mean", "alpha_1", "alpha_0"], linestyles, fillstyles):
        x = leakage_coeff[ver]["aii_mean"].nominal_value
        dx = leakage_coeff[ver]["aii_mean"].std_dev
        y = leakage_coeff[ver][key].nominal_value
        dy = leakage_coeff[ver][key].std_dev

        eb = plt.errorbar(
            x,
            y,
            xerr=dx,
            yerr=dy,
            fmt=cat[ver]["marker"],
            color=cat[ver]["colour"],
            fillstyle=fs,
            label=label
        )
        label = None
        eb[-1][0].set_linestyle(ls)


# y=x line
xlim = 0.02
x = [-xlim, xlim]
y = x
plt.plot(x, y, "k:", linewidth=0.5)

plt.legend()
plt.xlabel(r"tr $a$ (object-wise)")
plt.ylabel(r"$\alpha$ (scale-dependent)")
out_path = f"{cat['paths']['output']}/leakage_coefficients.png"
cs_plots.savefig(out_path) """

# +
# ### Cosmological analysis
# xipm correlation functions
# B-modes
# covariance
# MCMC
# -

# ### Catalogue ellipticity histograms

# +
plt.rcParams.update({'figure.figsize': [22,7]})

fig, axs = plt.subplots(1, 2)
nbins = 200

print_start("Ellipticity histograms")

out_path = f"{cat['paths']['output']}/ell_hist.png"
if os.path.exists(out_path):
    print_green(f"Skipping ellipticity histograms, {out_path} exists")
else:
    print_cyan("Compute histograms")
    for ver in versions:
        print_magenta(ver)
        R = cat[ver]["shear"]["R"]
        e1 = results[ver].dat_shear[cat[ver]['shear']['e1_col']] / R
        e2 = results[ver].dat_shear[cat[ver]['shear']['e2_col']] / R
        w = results[ver].dat_shear['w']

        n, bins, _ = axs[0].hist(e1, bins=nbins, density=False, histtype='step', weights=w,\
                                label=ver, color=cat[ver]["colour"])
        n, bins, _ = axs[1].hist(e2, bins=nbins, density=False, histtype='step', weights=w,\
                                label=ver, color=cat[ver]["colour"])

    for idx in (0, 1):
        axs[idx].set_xlabel(f"$e_{idx}$")
        axs[idx].set_ylabel('frequency')
        axs[idx].legend()
        axs[idx].set_xlim([-1.5,1.5])
    cs_plots.savefig(out_path)
print_done("Ellipticity histograms")

# Plot separation for SP/MP match
plt.rcParams.update({'figure.figsize': [10,7]})
fig, axs = plt.subplots(1, 1)
nbins = 200

if 'SP_matched_MP_v1.0' in versions:
    sep = results['SP_matched_MP_v1.0'].dat_shear['Separation']
    n, bins, _ = axs.hist(sep, bins=nbins, density=False, histtype='step',\
                            label='SP_matched_MP_v1.0', color=cat['SP_matched_MP_v1.0']["colour"])
    print('Max separation: %s arcsec' %max(sep))
    axs.set_xlabel(r'Separation $\theta$ [arcsec]')
    _ = axs.legend()
# -

# ### Cosmology calculations

# #### Compute $\xi_\pm$

# +
# Compute additive bias

print_start("Additive bias")

c1 = {}
c2 = {}

for ver in versions:
    print_magenta(ver)
    R = cat[ver]["shear"]["R"]
    c1[ver] = np.average(
        results[ver].dat_shear[cat[ver]['shear']['e1_col']] / R,
        weights=results[ver].dat_shear["w"]
    )
    c2[ver] = np.average(
        results[ver].dat_shear[cat[ver]['shear']['e2_col']] / R,
        weights=results[ver].dat_shear["w"]
    )
print_done("Done additive bias")
# -

print_start("2PCF")

# +
theta_min = 0.1
theta_max = 250
nbins = 20

npatch = 150

TreeCorrConfig_xi = {
    'ra_units': coord_units,
    'dec_units': coord_units,
    'min_sep': theta_min,
    'max_sep': theta_max,
    'sep_units': sep_units,
    'nbins': nbins,
    'var_method': 'jackknife'
}

cat_ggs = {}
for ver in versions:
    print_magenta(ver)

    gg = treecorr.GGCorrelation(TreeCorrConfig_xi)

    out_fname = f"{cat['paths']['output']}/xi_pm_{ver}.txt"
    if os.path.exists(out_fname) :
        print_green(f'Skipping 2PCF, {out_fname} exists')
        gg.read(out_fname)
    else:
        print_cyan(f'Computing 2PCF')
        if ver != 'DES':
            R = cat[ver]["shear"]["R"]
            g1 = (results[ver].dat_shear[cat[ver]['shear']['e1_col']] - c1[ver]) / R
            g2 = (results[ver].dat_shear[cat[ver]['shear']['e2_col']] - c2[ver]) / R
        else:
            R11 = cat[ver]["shear"]["R11"]
            R22 = cat[ver]["shear"]["R22"]
            g1 = (results[ver].dat_shear[cat[ver]['shear']['e1_col']] - c1[ver]) / np.average(results[ver].dat_shear[R11])
            g2 = (results[ver].dat_shear[cat[ver]['shear']['e2_col']] - c2[ver]) / np.average(results[ver].dat_shear[R22])
        cat_gal = treecorr.Catalog(
            ra=results[ver].dat_shear['RA'],
            dec=results[ver].dat_shear['Dec'],
            g1=g1,
            g2=g2,
            w=results[ver].dat_shear['w'],
            ra_units=coord_units,
            dec_units=coord_units,
            npatch=npatch,
        )
        gg.process(cat_gal)
        gg.write(out_fname)
        del(cat_gal)
        del(g1)
        del(g2)

    cat_ggs[ver] = gg
    #del(gg)

print_done("Done 2PCF")
nbins=TreeCorrConfig_xi['nbins']
for ver in versions:
    gg = cat_ggs[ver]
    lst = np.arange(1,nbins+1)

    #create fits HDU with xi_p and xi_m data
    col1 = fits.Column(name ='BIN1', format ='K', array = np.ones(len(lst)))
    col2 = fits.Column(name ='BIN2', format ='K', array = np.ones(len(lst)))
    col3 = fits.Column(name ='ANGBIN', format ='K', array = lst)
    col4 = fits.Column(name ='VALUE', format ='D', array = gg.xip)
    col5 = fits.Column(name ='ANG', format ='D', unit ='arcmin', array = gg.rnom)
    coldefs = fits.ColDefs([col1, col2, col3, col4, col5])
    xiplus_hdu = fits.BinTableHDU.from_columns(coldefs,name ='XI_PLUS')


    col4 = fits.Column(name ='VALUE', format ='D', array = gg.xim)
    coldefs = fits.ColDefs([col1, col2, col3, col4, col5])
    ximinus_hdu = fits.BinTableHDU.from_columns(coldefs,name ='XI_MINUS')

    #append xi_p/xi_m header info 
    xip_dict = {'2PTDATA':'T',
                'QUANT1':'G+R',
                'QUANT2':'G+R',
                'KERNEL_1':'NZ_SOURCE',
                'KERNEL_2':'NZ_SOURCE',
                'WINDOWS':'SAMPLE'}
    for key in xip_dict:
        xiplus_hdu.header[key] = xip_dict[key]


    xim_dict = {'2PTDATA':'T',
                'QUANT1':'G-R',
                'QUANT2':'G-R',
                'KERNEL_1':'NZ_SOURCE',
                'KERNEL_2':'NZ_SOURCE',
                'WINDOWS':'SAMPLE'}

    for key in xim_dict:
        ximinus_hdu.header[key] = xim_dict[key]

    ximinus_hdu.writeto(f"{cat['paths']['output']}/xi_minus_{ver}.fits",overwrite=True)
    xiplus_hdu.writeto(f"{cat['paths']['output']}/xi_plus_{ver}.fits",overwrite=True)


# -

# #### Plot $\xi_\pm$

# +
plt.rcParams.update({'font.size': 20,'figure.figsize':[12,10]})

#Plot of n_pairs
for ver in versions:
    plt.plot(
        cat_ggs[ver].meanr,
        cat_ggs[ver].npairs,
        label=ver,
        ls=cat[ver]['ls'],
        color=cat[ver]['colour']
    )
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.ylabel(r'$n_{\rm pair}$')
plt.legend()
out_path = f"{cat['paths']['output']}/n_pair.png"
cs_plots.savefig(out_path)
# -

# Plot of xi_+
fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
for idx, ver in enumerate(versions):
    plt.errorbar(
        cat_ggs[ver].meanr * cs_plots.dx(idx, len(ver)),
        cat_ggs[ver].xip,
        yerr=np.sqrt(cat_ggs[ver].varxip),
        label=ver,
        ls=cat[ver]["ls"],
        color=cat[ver]["colour"]
    )
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=20, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ticklabel_format(axis="y")
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.xlim([theta_min_plot, theta_max_plot])
plt.ylabel(r'$\xi_+(\theta)$')
out_path = f"{cat['paths']['output']}/xi_p.png"
cs_plots.savefig(out_path)

# Plot of xi_-
fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
for idx, ver in enumerate(versions):
    plt.errorbar(
        cat_ggs[ver].meanr * cs_plots.dx(idx, len(ver)),
        cat_ggs[ver].xim,
        yerr=np.sqrt(cat_ggs[ver].varxim),
        label=ver,
        ls=cat[ver]["ls"],
        color=cat[ver]["colour"]
    )
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=20, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ticklabel_format(axis="y")
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.xlim([theta_min_plot, theta_max_plot])
plt.ylabel(r'$\xi_-(\theta)$')
out_path = f"{cat['paths']['output']}/xi_m.png"
cs_plots.savefig(out_path)

#Plot of xi_+(theta) * theta
fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
for idx, ver in enumerate(versions):
    plt.errorbar(
        cat_ggs[ver].meanr,
        cat_ggs[ver].xip * cat_ggs[ver].meanr,
        yerr=np.sqrt(cat_ggs[ver].varxip) * cat_ggs[ver].meanr,
        label=ver,
        ls=cat[ver]["ls"],
        color=cat[ver]["colour"]
    )
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=20, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ticklabel_format(axis="y")
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.xlim([theta_min_plot, theta_max_plot])
plt.ylabel(r'$\theta \xi_+(\theta)$')
out_path = f"{cat['paths']['output']}/xi_p_theta.png"
cs_plots.savefig(out_path)

#Plot of xi_- * theta
fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
for idx, ver in enumerate(versions):
    plt.errorbar(
        cat_ggs[ver].meanr * cs_plots.dx(idx, len(ver)),
        cat_ggs[ver].xim * cat_ggs[ver].meanr,
        yerr=np.sqrt(cat_ggs[ver].varxim) * cat_ggs[ver].meanr,
        label=ver,
        ls=cat[ver]["ls"],
        color=cat[ver]["colour"]
    )
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=20, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ticklabel_format(axis="y")
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.xlim([theta_min_plot, theta_max_plot])
plt.ylabel(r'$\theta \xi_-(\theta)$')
out_path = f"{cat['paths']['output']}/xi_m_theta.png"
cs_plots.savefig(out_path)

# Plot of xi_+ with and without xi_psf_sys
for idx, ver in enumerate(versions):
    fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
    plt.errorbar(
        cat_ggs[ver].meanr * cs_plots.dx(idx, len(ver)),
        cat_ggs[ver].xip,
        yerr=np.sqrt(cat_ggs[ver].varxim),
        label=r"$\xi_+$",
        ls="solid",
        color="green",
    )
    plt.errorbar(
        cat_ggs[ver].meanr * cs_plots.dx(idx, len(ver)),
        xi_psf_sys_mean[ver],
        yerr=np.sqrt(xi_psf_sys_var[ver]),
        label=r"$\xi^{\rm psf}_{+, {\rm sys}}$",
        ls="dotted",
        color="red",
    )
    plt.errorbar(
        cat_ggs[ver].meanr * cs_plots.dx(idx, len(ver)),
        cat_ggs[ver].xip + xi_psf_sys_mean[ver],
        yerr=np.sqrt(cat_ggs[ver].varxip + xi_psf_sys_var[ver]),
        label=r"$\xi_+ + \xi^{\rm psf}_{+, {\rm sys}}$",
        ls="dashdot",
        color="magenta",
    )

    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=20, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ticklabel_format(axis="y")
    plt.xlabel(rf'$\theta$ [{sep_units}]')
    plt.xlim([theta_min_plot, theta_max_plot])
    plt.ylim(1e-8, 5e-4)
    plt.ylabel(r'$\xi_+(\theta)$')
    out_path = f"{cat['paths']['output']}/xi_p_xi_psf_sys_{ver}.png"
    cs_plots.savefig(out_path)


# +
#### Aperture-mass dispersion

# +
theta_min = 0.3
theta_max = 200
nbins = 500
npatch = 25

TreeCorrConfig_map = {
        'ra_units': coord_units,
        'dec_units': coord_units,
        'max_sep': str(theta_max),
        'min_sep': str(theta_min),
        'sep_units': sep_units,
        'nbins': nbins,
        'var_method':'jackknife',
}

# Set up angular smoothing scales for aperture-mass dispersion
n_bins_map = 15
theta_map = np.geomspace(theta_min * 5, theta_max / 2, n_bins_map)

print_start("Aperture-mass dispersion")
map2 = {}
for ver in versions:
    print_magenta(ver)

    gg = treecorr.GGCorrelation(TreeCorrConfig_map)

    out_fname = f"{cat['paths']['output']}/xi_for_map2_{ver}.txt"
    if os.path.exists(out_fname):
        print_green(f'Skipping xi for Map2, {out_fname} exists')
        gg.read(out_fname)
    else:
        print_cyan("Compute Map2")
        R = cat[ver]["shear"]["R"]
        g1 = (results[ver].dat_shear[cat[ver]['shear']['e1_col']] - c1[ver]) / R
        g2 = (results[ver].dat_shear[cat[ver]['shear']['e2_col']] - c2[ver]) / R
        cat_gal = treecorr.Catalog(
            ra=results[ver].dat_shear['RA'],
            dec=results[ver].dat_shear['Dec'],
            g1=g1,
            g2=g2,
            w=results[ver].dat_shear['w'],
            ra_units=coord_units,
            dec_units=coord_units,
            npatch=npatch,
        )

        gg.process(cat_gal)
        gg.write(out_fname)
        del(cat_gal)
        del(g1)
        del(g2)

    mapsq, mapsq_im, mxsq, mxsq_im, varmapsq = gg.calculateMapSq(
        R=theta_map,
        m2_uform='Schneider',
    )
    out_fname_map2 = f"{cat['paths']['output']}/map2_{ver}.txt"
    if os.path.exists(out_fname_map2):
        print_green(f"Skipping Map2, {out_fname_map2} exists")
    else:
        print(f"Writing Map2 to output file {out_fname_map2} ")
        gg.writeMapSq(out_fname_map2, R=theta_map, m2_uform='Schneider')
    map2[ver] = {}
    map2[ver]['mapsq'] = mapsq
    map2[ver]['mapsq_im'] = mapsq_im
    map2[ver]['mxsq'] = mxsq
    map2[ver]['mxsq_im'] = mxsq_im
    map2[ver]['varmapsq'] = varmapsq
    del(gg)

print_done("Done aperture-mass dispersion")

# +
# Plot aperture-mass dispersion

for mode in ['mapsq', 'mapsq_im', 'mxsq', 'mxsq_im']:
    x = []
    y = []
    yerr = []
    theta=[]
    labels=[]
    colors=[]
    linestyles=[]
    for idx, ver in enumerate(versions):

        x.append(theta_map)
        y.append(map2[ver][mode])
        yerr.append(np.sqrt(map2[ver]['varmapsq']))
        labels.append(ver)
        colors.append(cat[ver]["colour"])
        linestyles.append(cat[ver]["ls"])

    xlabel = r"$\theta$ [arcmin]"
    ylabel = "dispersion"
    title = f"Aperture-mass dispersion {mode}"
    out_path = f"{cat['paths']['output']}/{mode}.pdf"
    cs_plots.plot_data_1d(
        x,
        y,
        yerr,
        title,
        xlabel,
        ylabel,
        out_path=None,
        labels=labels,
        xlog=True,
        xlim=[theta_min_plot, theta_max_plot],
        ylim=[-1e-6, 2e-5],
        colors=colors,
        linestyles=linestyles,
        shift_x=True,
    )
    cs_plots.savefig(out_path)

    # Plot aperture-mass dispersion

for mode in ['mapsq', 'mapsq_im', 'mxsq', 'mxsq_im']:
    x = []
    y = []
    yerr = []
    for ver in versions:
        x.append(theta_map)
        y.append(np.abs(map2[ver][mode]))
        yerr.append(np.sqrt(map2[ver]['varmapsq']))
    xlabel = r"$\theta$ [arcmin]"
    ylabel = "dispersion"
    title = f"Aperture-mass dispersion mode {mode}"
    out_path = f"{cat['paths']['output']}/{mode}_log.pdf"
    cs_plots.plot_data_1d(
        x,
        y,
        yerr,
        title,
        xlabel,
        ylabel,
        out_path=None,
        labels=labels,
        xlog=True,
        ylog=True,
        xlim=[theta_min_plot, theta_max_plot],
        ylim=[1e-9, 3e-5],
        colors=colors,
        linestyles=linestyles,
        shift_x=True,
    )
    cs_plots.savefig(out_path)
# -

# Clean up memory
print("Clean up memory")
for ver in versions:
    del(results[ver].dat_shear)
    del(results[ver].dat_PSF)
gc.collect()
print("Done: Clean up memory")

print_done("Exiting here")
sys.exit(0)
