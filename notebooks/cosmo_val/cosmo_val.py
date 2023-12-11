# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: sp_validation
#     language: python
#     name: sp_validation
# ---

# # Cosmological validation of UNIONS shape catalogues
# 03/2023

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
from astropy.io import ascii

# +
from sp_validation.plot_style import *
from cs_util import plots

from shear_psf_leakage import run_scale

# +
import treecorr

n_thread = 1
treecorr.set_omp_threads(n_thread)

sep_units = 'arcmin'
coord_units = 'degrees'
# -

import pyccl
pyccl.gsl_params.LENSING_KERNEL_SPLINE_INTEGRATION = False

def color_reset():
    print(colorama.Fore.BLACK, end='')

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
versions = ['SP_v1.0', 'SP_v1.0_LFmask_8k', 'SP_v1.3', 'SP_v1.3_LFmask_8k', 'SP_axel_v0.0', 'SP_axel_v0.0_repr', 'DES'] 
#versions = ['SP_v1.0_LFmask_4k', 'SP_v1.0_LFmask_8k', 'SP_v1.3_LFmask_4k', 'SP_v1.3_LFmask_8k']
#versions = ["SP_v1.3_LFmask_4k"]
all_keys = ['nz']
for ver in versions:
    all_keys.append(ver)

# Base directory for data, on candide
data_base_dir = f'{os.environ["HOME"]}/astro/data'

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

# Plotting
theta_min_plot = 1
theta_max_plot = 300

ylim_alpha = [-0.005, 0.05]

ylim_xi_sys_ratio = [-0.02, 0.5]
# -

# ## Processing

print_start('Start cosmology validation')

# ### Systematic tests

# + active=""
# # TODO:
# # object-wise leakage
# -

# #### Init results dictionary

results = {}

def set_params_leakage(cat, ver):
    params_in = {}

    # Set parameters
    params_in['input_path_shear'] = cat[ver]["shear"]["path"]
    params_in['input_path_PSF'] = cat[ver]["star"]["path"]
    params_in['dndz_path'] = f"{cat['nz']['dndz']['path']}_{cat[ver]['pipeline']}_{cat['nz']['dndz']['blind']}.txt"
    params_in['output_dir'] = f'{cat["paths"]["output"]}/leakage_{ver}'

    # Note: for SP these are calibrated shear estimates
    params_in['e1_col'] = cat[ver]["shear"]["e1_col"]
    params_in['e2_col'] = cat[ver]["shear"]["e2_col"]

    params_in['ra_star_col'] = cat[ver]["star"]["ra_col"]
    params_in["dec_star_col"] = cat[ver]["star"]["dec_col"]
    params_in['e1_PSF_star_col'] = cat[ver]["star"]["e1_col"]
    params_in["e2_PSF_star_col"] = cat[ver]["star"]["e2_col"]

    params_in["verbose"] = False

    return params_in

for ver in versions:

    params_in = set_params_leakage(cat, ver)

    # Create leakage instance
    obj = run_scale.LeakageScale()

    # Set instance parameters, copy from above

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

# #### Rho statistics

import emcee
from shear_psf_leakage.rho_tau_stat import RhoStat, TauStat, PSFErrorFit

# +
out_dir = f"{cat['paths']['output']}/rho_stats"
#out_dir = "/"
if not os.path.exists(out_dir):
    os.mkdir(out_dir) 
print_start('Rho stats')

# Create class instance to compute, save, load and plot rho_stats
rho_stat_handler = RhoStat(output=out_dir, verbose=True)

for ver in versions:
    out_base = f"rho_stats_{ver}.fits"
    out_path = f"{out_dir}/{out_base}"

    # Load previous results
    if os.path.exists(out_path):
        print_green(f"Skipping rho statis computation, file {out_path} exists")
        rho_stat_handler.load_rho_stats(out_base)
    else:
        print_cyan("Computing rho stats")
        
        # Set parameters
        params = results[ver]._params
        params["patch_number"] = 120
        params["ra_col"] = cat[ver]["psf"]["ra_col"]
        params["dec_col"] = cat[ver]["psf"]["dec_col"]
        params["e1_PSF_col"] = cat[ver]["psf"]["e1_PSF_col"]
        params["e2_PSF_col"] = cat[ver]["psf"]["e2_PSF_col"]
        params["e1_star_col"] = cat[ver]["psf"]["e1_star_col"]
        params["e2_star_col"] = cat[ver]["psf"]["e2_star_col"]
        params["PSF_size"] = cat[ver]["psf"]["PSF_size"]
        params["star_size"] = cat[ver]["psf"]["star_size"]
        params["ra_units"] = "deg"
        params["dec_units"] = "deg"

        #rho_stat_handler.catalogs.params_default(out_dir)
        rho_stat_handler.catalogs.set_params(params, out_dir)

        # Build catalogues
        rho_stat_handler.build_cat_to_compute_rho(
            cat[ver]["psf"]["path"],
            catalog_id=ver, 
            square_size=True,
            mask=False
        )

        # Compute correlations
        rho_stat_handler.compute_rho_stats(ver, out_base)

# +
filenames = []
colors = []
for ver in versions:
    filenames.append(f"rho_stats_{ver}.fits")
    colors.append(cat[ver]["colour"])

# Create plots
rho_stat_handler.plot_rho_stats(
    filenames,
    colors,
    versions,
    abs=True,
    savefig='rho_stats.png'
)
# -

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
        plt.savefig(out_path)
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

        TreeCorrConfig = {
            'ra_units': coord_units,
            'dec_units': coord_units,
            'min_sep': obj._params["theta_min_amin"],
            'max_sep': obj._params["theta_max_amin"],
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
    out_path = f"{cat['paths']['output']}/alpha_leak.pdf"

    title = r'$\alpha$ leakage'
    xlabel = r'$\theta\, [arcmin]$'
    ylabel = r'$\alpha(\theta)$'
    plots.plot_data_1d(
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
        markers=markers,
        shift_x=True,
    )
    plt.savefig(out_path)

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
    plots.plot_data_1d(
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
        shift_x=True,
    )
    plt.savefig(out_path)

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
    plots.plot_data_1d(
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
        shift_x=True,
    )
    plt.savefig(out_path)
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
    plt.savefig(out_path)
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
    print(f"{ver} {c1[ver]} {c2[ver]}")
print_done("Done additive bias")
# -

print_start("2PCF")

# +
theta_min = 0.5
theta_max = 150
nbins = 20
npatch = 50

TreeCorrConfig = {
        'ra_units': coord_units,
        'dec_units': coord_units,
        'max_sep': str(theta_max),
        'min_sep': str(theta_min),
        'sep_units': sep_units,
        'nbins': nbins,
        'var_method':'jackknife',
    }

cat_ggs = {}
for ver in versions:
    print_magenta(ver)

    gg = treecorr.GGCorrelation(TreeCorrConfig)

    out_fname = f"{cat['paths']['output']}/xi_pm_{ver}.txt"
    if os.path.exists(out_fname) :
        print_green(f'Skipping 2PCF, {out_fname} exists')
        gg.read(out_fname)
    else:
        print_cyan(f'Computing 2PCF')
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

    cat_ggs[ver] = gg
    del(gg)

print_done("Done 2PCF")
lst = np.arange(1,nbins+1)

#create fits HDU with xi_p and xi_m data
col1 = fits.Column(name ='BIN1', format ='K', array = np.ones(len(lst)))
col2 = fits.Column(name ='BIN2', format ='K', array = np.ones(len(lst)))
col3 = fits.Column(name ='ANGBIN', format ='K', array = lst)
col4 = fits.Column(name ='VALUE', format ='D', array = gg.xip)
col5 = fits.Column(name ='ANG', format ='D', unit ='arcmin', array = gg.meanr)
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

ximinus_hdu.writeto(f"{cat['paths']['output']}/xi_minus.fits",overwrite=True)
xiplus_hdu.writeto(f"{cat['paths']['output']}/xi_plus.fits",overwrite=True)


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
plt.savefig(out_path)
# -

#Plot of xi_+
fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
for idx, ver in enumerate(versions):
    plt.errorbar(
        cat_ggs[ver].meanr * plots.dx(idx, len(ver)),
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
plt.savefig(out_path, bbox_inches="tight")

#Plot of xi_-
fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
for idx, ver in enumerate(versions):
    plt.errorbar(
        cat_ggs[ver].meanr * plots.dx(idx, len(ver)),
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
plt.savefig(out_path, bbox_inches="tight")

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
plt.savefig(out_path, bbox_inches="tight")

#Plot of xi_-
fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
for idx, ver in enumerate(versions):
    plt.errorbar(
        cat_ggs[ver].meanr * plots.dx(idx, len(ver)),
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
plt.savefig(out_path, bbox_inches="tight")

# +
#### Aperture-mass dispersion

# +
theta_min = 0.3
theta_max = 200
nbins = 500
npatch = 25

TreeCorrConfig = {
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

    gg = treecorr.GGCorrelation(TreeCorrConfig)

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
        print(f"Map2 output file {out_fname_map2} exists")
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
    plots.plot_data_1d(
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
    plt.savefig(out_path)

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
    plots.plot_data_1d(
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
    plt.savefig(out_path)
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

