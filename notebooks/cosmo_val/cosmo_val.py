# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Cosmological validation of UNIONS shape catalogues
# 03/2023

# +
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

print('v1')
# -

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
from astropy.io import ascii

# +
from sp_validation.plot_style import *
from cs_util import plots

from sp_validation import run

# +
import treecorr

n_thread = 8
treecorr.set_omp_threads(n_thread)
# -

import pyccl
pyccl.gsl_params.LENSING_KERNEL_SPLINE_INTEGRATION = False

# ## Input parameters

# +
# Catalogue versions
versions = ['SP_v1.0', 'LF_v1.0', 'LF_v2.0', 'SP_matched_LF_v1.0', 'LF_matched_SP_v1.0']

all_keys = ['nz']
for ver in versions:
    all_keys.append(ver)
# -

# Base directory for data, on candide
data_base_dir = f'{os.environ["HOME"]}/astro/data/CFIS'

# ## Loading data

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

# ### Systematic tests

# + active=""
# # rho stats
# # object-wise leakage
# -

# #### $\xi_\textrm{sys}$ and scale-dependent leakage

results = {}
print('Computing scale-dependent leakage')


def set_params_leakage(cat, ver):
    params_in = {}
    
    # Set parameters
    params_in['input_path_shear'] = cat[ver]["shear"]["path"]
    params_in['input_path_PSF'] = cat[ver]["star"]["path"]
    params_in['dndz_path'] = f"{cat['nz']['dndz']['path']}_{cat[ver]['pipeline']}_{cat['nz']['dndz']['blind']}.txt"
    params_in['output_dir'] = f'{cat["paths"]["output"]}/leakage_{ver}'
    params_in['sh'] = cat[ver]['shape']
    
    # Note: for SP these are calibrated shear estimates
    params_in['e1_col'] = 'e1'
    params_in['e2_col'] = 'e2'
    
    params_in['e1_PSF_star_col'] = cat[ver]["star"]["e1_col"]
    params_in["e2_PSF_star_col"] = cat[ver]["star"]["e2_col"]
    
    params_in["verbose"] = True
    
    return params_in


for ver in versions:

    params_in = set_params_leakage(cat, ver)

    # Create leakage instance
    obj = run.LeakageScale()

    # Set instance parameters, copy from above
    for key in params_in:
        obj._params[key] = params_in[key]
    
    results[ver] = obj

for ver in versions:
    print(ver)
    obj = results[ver]

    obj.check_params()
    obj.prepare_output()
    obj.read_data()

    out_fname = f'{cat["paths"]["output"]}/leakage_{ver}/alpha_leakage_{cat[ver]["shape"]}.txt'
    if os.path.exists(out_fname):
        print(f'Skipping computation, {out_fname} exists')
    else:
        obj.compute_corr_gp_pp_alpha()                                         
        obj.do_alpha()                                                         
        obj.do_xi_sys()

    print(f"done: {ver}")
print('Done scale-dependent leakage')

# +
# Plot scale-dependent leakage

theta = []
y = []
yerr = []
labels = []
colors = []
linestyles = []

for ver in versions:
    if hasattr(results[ver], "r_corr_gp"):
        theta.append(results[ver].r_corr_gp.meanr)
        y.append(results[ver].alpha_leak)
        yerr.append(results[ver].sig_alpha_leak)
        labels.append(ver)
        colors.append(cat[ver]["colour"])
        linestyles.append(cat[ver]["ls"])

if len(theta) > 0:
    out_path = f"{cat['paths']['output']}/alpha_leak.pdf"

    title = r'$\alpha$ leakage'
    xlabel = r'$\theta\, [arcmin]$'
    ylabel = r'$\alpha(\theta$'
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

for ver in versions:
    e1 = results[ver].dat_shear['e1']
    w = results[ver].dat_shear['w']
    n, bins, _ = axs[0].hist(e1, bins=nbins, density=False, histtype='step', weights=w,\
                            label=ver, color=cat[ver]["colour"])
axs[0].set_xlabel('$e_1$')
axs[0].set_ylabel('frequency')
axs[0].legend()
axs[0].set_xlim([-1.5, 1.5])

for ver in versions:
    e2 = results[ver].dat_shear['e2']
    w = results[ver].dat_shear['w']
    n, bins, _ = axs[1].hist(e2, bins=nbins, density=False, histtype='step', weights=w,\
                            label=ver, color=cat[ver]["colour"])
axs[1].set_xlabel('$e_2$')
axs[0].set_ylabel('frequency')
axs[1].legend()
_ = axs[1].set_xlim([-1.5,1.5])

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

print("Compute additive bias")

c1 = {}
c2 = {}

print("# ver c1 (from yml) c2 (from yml)")

for ver in versions:
    c1[ver] = np.average(
        results[ver].dat_shear["e1"],
        weights=results[ver].dat_shear["w"]
    )
    c2[ver] = np.average(
        results[ver].dat_shear["e2"],
        weights=results[ver].dat_shear["w"]
    )
    print(f"{ver} {c1[ver]} ({cat[ver]['shear']['e1_bias']}) {c2[ver]} ({cat[ver]['shear']['e2_bias']})")
# -

print("Compute 2PCF")
treecorr.set_omp_threads(n_thread)
sep_units = 'arcmin'
coord_units = 'degrees'

# +
theta_min = 1
theta_max = 200
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
    print(f"{ver}")
    
    gg = treecorr.GGCorrelation(TreeCorrConfig)        

    out_fname = f"{cat['paths']['output']}/xi_pm_{ver}_{cat[ver]['shape']}.txt"
    if os.path.exists(out_fname):
        print(f'Skipping 2PCF, {out_fname} exists')
        
        gg.read(out_fname)
    else:
        print(f'Computing 2PCF')

        g1 = results[ver].dat_shear["e1"] - c1[ver]
        g2 = results[ver].dat_shear["e2"] - c2[ver]
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
                 
    cat_ggs[ver] = gg

print("Done 2PCF")
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
out_path = f"{cat['paths']['output']}/n_pair"
plt.savefig(out_path)
# -

#Plot of xi_+
for ver in versions:
    plt.errorbar(
        cat_ggs[ver].meanr,
        cat_ggs[ver].xip,
        yerr=np.sqrt(cat_ggs[ver].varxip),
        label=ver,
        ls=cat[ver]["ls"],
        color=cat[ver]["colour"]
    )
plt.plot()
plt.xscale('log')
plt.legend(fontsize=20)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.xlim([theta_min_plot, theta_max_plot])
plt.ylabel(r'$\xi_+(\theta)$')
out_path = f"{cat['paths']['output']}/xi_p"
_ = plt.savefig(out_path)

#Plot of xi_-
for ver in versions:
    plt.errorbar(
        cat_ggs[ver].meanr,
        cat_ggs[ver].xim,
        yerr=np.sqrt(cat_ggs[ver].varxim),
        label=ver,
        ls=cat[ver]["ls"],
        color=cat[ver]["colour"]
    )
plt.plot()
plt.xscale('log')
plt.legend(fontsize=20)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.xlim([theta_min_plot, theta_max_plot])
plt.ylabel(r'$\xi_-(\theta)$')
out_path = f"{cat['paths']['output']}/xi_m"
_ = plt.savefig(out_path)

# #### Plot fractional difference

# +
#plt.rcParams.update({'font.size': 20,'figure.figsize':[12,10]})

#Define the two catalogues you wish to compare
#cat1 = cat_options[0]
#cat2 = cat_options[1]

#cat_key = cat1[0]+'_'+cat1[1]
#cat_key2 = cat2[0]+'_'+cat2[1]

#ratio = np.abs(cat_ggs[cat_key2].xip-cat_ggs[cat_key].xip)/cat_ggs[cat_key2].xip
#err = ratio * np.sqrt(((np.sqrt(cat_ggs[cat_key].varxip)+np.sqrt(cat_ggs[cat_key2].varxip))/(cat_ggs[cat_key2].xip-cat_ggs[cat_key].xip))**2+(cat_ggs[cat_key2].varxip/cat_ggs[cat_key2].xip)**2)
#plt.errorbar(cat_ggs[cat_key].meanr, ratio, yerr=err,
             #label=r'$\xi_+$ Fractional Diff (%s-%s)/%s' %(cat[cat2[0]][cat2[1]]['label'],cat[cat1[0]][cat1[1]]['label'],cat[cat2[0]][cat2[1]]['label']),ls='solid',color='salmon')

#ratio = np.abs(cat_ggs[cat_key2].xim-cat_ggs[cat_key].xim)/cat_ggs[cat_key2].xim
#err = ratio * np.sqrt(((np.sqrt(cat_ggs[cat_key].varxim)+np.sqrt(cat_ggs[cat_key2].varxim))/(cat_ggs[cat_key2].xim-cat_ggs[cat_key].xim))**2+(cat_ggs[cat_key2].varxim/cat_ggs[cat_key2].xim)**2)
#plt.errorbar(cat_ggs[cat_key].meanr, ratio, yerr=err,
             #label=r'$\xi_-$ Fractional Diff (%s-%s)/%s' %(cat[cat2[0]][cat2[1]]['label'],cat[cat1[0]][cat1[1]]['label'],cat[cat2[0]][cat2[1]]['label']),ls='solid',color='indigo')
#plt.hlines(0.0,0,200,colors='k')
#plt.grid()
#plt.xscale('log')
#plt.ylim([-2.5,2.5])
#plt.legend(fontsize=15)
#plt.xlabel(rf'$\theta$ [{sep_units}]')
#plt.xlim([1,200])

# +
#### Aperture-mass dispersion

# +
theta_min = 1
theta_max = 200
nbins = 200
npatch = 50

# Set up angular smoothing scales for aperture-mass dispersion
n_bins_map = 20
R = np.geomspace(theta_min * 5, theta_max / 2, n_bins_map)

TreeCorrConfig = {
    'ra_units': coord_units,
    'dec_units': coord_units,
    'max_sep': str(theta_max),
    'min_sep': str(theta_min),
    'sep_units': sep_units,
    'nbins': nbins,
    'var_method':'jackknife',
}
gg = treecorr.GGCorrelation(TreeCorrConfig)

print("Compute aperture-mass dispersion")
map2 = {}
for ver in versions:
    
    out_fname = f"{cat['paths']['output']}/xi_for_map2_{ver}.txt"
    if os.path.exists(out_fname):
        print(f'Skipping Map2, {out_fname} exists')
        gg.read(out_fname)
    else:
        print(f'Computing Map2')

        g1 = results[ver].dat_shear["e1"] - c1[ver]
        g2 = results[ver].dat_shear["e2"] - c2[ver]
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
        

        print(f"done: {ver}")

    mapsq, mapsq_im, mxsq, mxsq_im, varmapsq = gg.calculateMapSq(
        R=R,
        m2_uform='Schneider',
    )
    out_fname_map2 = f"{cat['paths']['output']}/for_map2_{ver}.txt"
    if os.path.exists(out_fname_map2):
        print("Map2 output file {out_fname_map2} exists")
    else:
        print("Writing Map2 to output file {out_fname_map2} ")
        gg.writeMapSq(out_fname_map2, R=R, m2_uform='Schneider')
    map2[ver] = {}
    map2[ver]['mapsq'] = mapsq
    map2[ver]['mapsq_im'] = mapsq_im
    map2[ver]['mxsq'] = mxsq
    map2[ver]['mxsq_im'] = mxsq_im
    map2[ver]['varmapsq'] = varmapsq

print("Done aperture-mass dispersion")

# +
# Plot aperture-mass dispersion

for mode in ['mapsq', 'mapsq_im', 'mxsq', 'mxsq_im']:
    x = []
    y = []
    yerr = []
    for ver in versions:
        x.append(R)
        y.append(map2[ver][mode])
        yerr.append(map2[ver]['varmapsq'])

    xlabel = r"$\theta$ [arcmin]"
    ylabel = "dispersion"
    title = f"Aperture-mass dispersion mode {mode}"
    out_path = f"{cat['paths']['output']}/{mode}.pdf"
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
    )
    plt.savefig(out_path)
    
    # Plot aperture-mass dispersion

for mode in ['mapsq', 'mapsq_im', 'mxsq', 'mxsq_im']:
    x = []
    y = []
    yerr = []
    for ver in versions:
        x.append(R)
        y.append(np.abs(map2[ver][mode]))
        yerr.append(map2[ver]['varmapsq'])

    xlabel = r"$\theta$ [arcmin]"
    ylabel = "dispersion"
    title = f"Aperture-mass dispersion mode {mode}"
    out_path = f"{cat['paths']['output']}/{mode}_log.pdf"
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
        ylog=True,
        xlim=[theta_min_plot, theta_max_plot],
        ylim=[1e-9, 1e-5],
        colors=colors,
        linestyles=linestyles,
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

# #### Plot covariance matrices

#Plot comparison of covariance matrices calculated by Jackknife and CosmoCov (here cosmocov files have been provided)
for ver in versions:
    try:
        cc = np.loadtxt(cat[ver]['shear']['covmat_file'])
    except KeyError:
        print('No Covmat available, skipping analysis for this catalogue: %s' %ver)
        continue
    else:
        cc_var = np.diag(cc)
        cc_varxip = cc_var[:20]
        cc_varxim = cc_var[20:]

        cc_g = np.loadtxt(cat[ver]['shear']['covmat_file'][:-4]+'_g.txt')
        cc_var = np.diag(cc_g)
        cc_varxip_g = cc_var[:20]
        cc_varxim_g = cc_var[20:]

        plt.loglog(cat_ggs[ver].meanr,cat_ggs[ver].varxip,
                   ls='-',c='k',
                   label=r'$\sigma(\xi_+)$ TreeCorr Jackknife %s' %cat[ver]['shear']['label'])
        plt.loglog(cat_ggs[ver].meanr,cc_varxip,
                   ls='--', c='%s' %cat[ver]['colour'],
                   label=r'$\sigma(\xi_+)$ CosmoCov %s' %cat[ver]['shear']['label'])
        plt.loglog(cat_ggs[ver].meanr,cc_varxip_g,
                   ls=':', c='%s' %cat[ver]['colour'],
                   label=r'$\sigma(\xi_+)$ CosmoCov Gaussian %s' %cat[ver]['shear']['label'])

        plt.grid()
        plt.xlim([cat_ggs[ver].meanr[0],cat_ggs[ver].meanr[-1]])
        plt.legend(fontsize=15)
        plt.xlabel(rf'$\theta$ [{sep_units}]')
        plt.ylabel(r'$\sigma(\xi_+)$')
        out_path = f"{cat['paths']['output']}/cov_p"
        plt.savefig(out_path)

        plt.loglog(cat_ggs[ver].meanr,cat_ggs[ver].varxim,
                   ls='-', c='k',
                    label=r'$\sigma(\xi_-)$ TreeCorr Jackknife %s' %cat[ver]['shear']['label'])
        plt.loglog(cat_ggs[ver].meanr,cc_varxim,
                   ls='--', c='%s' %cat[ver]['colour'],
                    label=r'$\sigma(\xi_-)$ CosmoCov %s' %cat[ver]['shear']['label'])
        plt.loglog(cat_ggs[ver].meanr,cc_varxim_g,
                    ls=':', c='%s' %cat[ver]['colour'],
                    label=r'$\sigma(\xi_-)$ CosmoCov Gaussian %s' %cat[ver]['shear']['label'])

        plt.grid()
        plt.xlim([cat_ggs[ver].meanr[0],cat_ggs[ver].meanr[-1]])
        plt.legend(fontsize=15)
        plt.xlabel(rf'$\theta$ [{sep_units}]')
        plt.ylabel(r'$\sigma(\xi_-)$')
        out_path = f"{cat['paths']['output']}/cov_m"
        plt.savefig(out_path)

# ## MCMC Plotting

# +
from getdist import plots, loadMCSamples
import uncertainties

g = plots.get_subplot_plotter(width_inch=30)
g.settings.axes_fontsize = 30
g.settings.axes_labelsize = 30
g.settings.alpha_filled_add = 0.6
g.settings.legend_fontsize = 30

#SPECIFY DATA DIRECTORY AND DESIRED CHAINS TO ANALYSE
scratch_dir = f'{os.environ["HOME"]}'
# -

#CREATE PARAMNAME FILE
for ver in versions:
    chain_dir = '%s/chain' %ver
    with open(scratch_dir + '%s/samples_1.txt'%(chain_dir), "r") as file:
        params = file.readline()[1:].split('\t')[:-2]
        file.close()

    with open(scratch_dir + '%s/getdist_%s_.paramnames'%(chain_dir,ver), "w") as file:
        for i in range(len(params)):
            file.write(params[i].split('--')[1] + '\n')
        file.close()
    print(params)

# +
#READ CHAIN
chains = []
colours = []
line_args = []

for ver in versions:
    chain_dir = '%s/chain' %ver
    chain = np.loadtxt(scratch_dir + '%s/samples_1.txt'%(chain_dir))

    np.savetxt(scratch_dir + '%s/getdist_%s__1.txt'%(chain_dir,ver),
               np.column_stack((np.ones_like(chain[:, -1]) ,-(chain[:, -1]-chain[:, -2]), chain[:, 0:-2])))

    chain = g.samples_for_root(scratch_dir + '%s/getdist_%s_' %(chain_dir,ver),
                                   settings={'ignore_rows':0.1,'smooth_scale_2D':0.7,'smooth_scale_1D':0.7})
    p=chain.getParams()
    chain.addDerived(p.h0*100,name='H_0',label=r'H_0')
    chain.addDerived(np.log(p.a_s*10**10), name='ln10^10A_s', label=r'ln(10^{10}A_s)')
    chain.addDerived(p.SIGMA_8*np.sqrt(p.omega_m/0.3), name='S_8', label=r'S_8')

    chains.append(chain)
    colours.append(cat[ver]['getdist_colour'])
    line_args.append({'color': cat[ver]['getdist_colour']})

# +
# %matplotlib inline
g.triangle_plot(chains,['omega_m','omega_b','ln10^10A_s','n_s','tau','h0','SIGMA_8','S_8'],
                legend_labels=versions,
                colors=colours,
                line_args=line_args)

out_path = f"{cat['paths']['output']}/corner_plot_comparison_lf.pdf"
g.export(out_path)
