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

# %matplotlib inline

import os
import numpy as np
import matplotlib.pylab as plt
import sys
import numpy as np
from astropy.io import fits
import treecorr
import pandas as pd
from astropy.io import ascii
import yaml

# +
from sp_validation.plot_style import *
from cs_util import plots

from sp_validation import run
# -

# ## Input parameters

# +
# Catalogue versions
versions = ['SP_v1.0', 'LF_v1.0', 'LF_v2.0']

all_keys = ['nz']
for ver in versions:
    all_keys.append(ver)
# -

# Base directory for data, on candide
data_base_dir = f'{os.environ["HOME"]}/astro/data/CFIS'

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
                
# ## Variables

components = ['+', '-']

# ### Plotting

# +
theta_min_plot = 1
theta_max_plot = 300

ylim_alpha = [-0.005, 0.05]

ylim_xi_sys_ratio = [-0.02, 0.5]
# -

# ## Processing

# ### Systematic tests

# + active=""
# # rho stats
# # scale-dependent leakage
# # object-wise leakage
# -

# #### $\xi_\textrm{sys}$ and scale-dependent leakage

results = {}

# +
ver = 'SP_v1.0'

params_in = {}

# Set parameters
params_in['input_path_shear'] = cat[ver]["shear"]["path"]
params_in['input_path_PSF'] = cat[ver]["star"]["path"]
params_in['dndz_path'] = f"{cat['nz']['dndz']['path']}_{cat[ver]['pipeline']}_{cat['nz']['dndz']['blind']}.txt"
params_in['output_dir'] = f'{cat["paths"]["output"]}/leakage_{ver}'
params_in['sh'] = cat[ver]['shape']
# Uncomment the following two lines to use calibrated shear estimates
# (the default is e1_uncal, uncalibrated shear estimates)
#params_in['e1_col'] = 'e1'
#params_in['e2_col'] = 'e2'
params_in['e1_PSF_star_col'] = cat[ver]["star"]["e1_col"]
params_in["e2_PSF_star_col"] = cat[ver]["star"]["e2_col"]
params_in["verbose"] = True

# Create leakage instance
obj = run.LeakageScale()

# Set instance parameters, copy from above
for key in params_in:
    obj._params[key] = params_in[key]
# -

obj.run()

results[ver] = obj

# +
ver = 'LF_v1.0'

params_in = {}

# Set parameters
params_in['input_path_shear'] = cat[ver]["shear"]["path"]
params_in['input_path_PSF'] = cat[ver]["star"]["path"]
params_in['dndz_path'] = f"{cat['nz']['dndz']['path']}_{cat[ver]['pipeline']}_{cat['nz']['dndz']['blind']}.txt"
params_in['output_dir'] = f'{cat["paths"]["output"]}/leakage_{ver}'
params_in['sh'] = cat[ver]['shape']
params_in['e1_col'] = 'e1'
params_in['e2_col'] = 'e2'
params_in['e1_PSF_star_col'] = cat[ver]["star"]["e1_col"]
params_in["e2_PSF_star_col"] = cat[ver]["star"]["e2_col"]
params_in["verbose"] = True

# Create leakage instance
obj = run.LeakageScale()

# Set instance parameters, copy from above
for key in params_in:
    obj._params[key] = params_in[key]
# -

obj.run()

results[ver] = obj

# +
ver = 'LF_v2.0'

params_in = {}

# Set parameters
params_in['input_path_shear'] = cat[ver]["shear"]["path"]
params_in['input_path_PSF'] = cat[ver]["star"]["path"]
params_in['dndz_path'] = f"{cat['nz']['dndz']['path']}_{cat[ver]['pipeline']}_{cat['nz']['dndz']['blind']}.txt"
params_in['output_dir'] = f'{cat["paths"]["output"]}/leakage_{ver}'
params_in['sh'] = cat[ver]['shape']
params_in['e1_col'] = 'e1'
params_in['e2_col'] = 'e2'
params_in['e1_PSF_star_col'] = cat[ver]["star"]["e1_col"]
params_in["e2_PSF_star_col"] = cat[ver]["star"]["e2_col"]
params_in["verbose"] = True

# Create leakage instance
obj = run.LeakageScale()

# Set instance parameters, copy from above
for key in params_in:
    obj._params[key] = params_in[key]
# -

obj.run()

results[ver] = obj

# +
# Plot scale-dependent leakage

theta = []
y = []
yerr = []
labels = []
colors = []
linestyles = []

for ver in versions:                                                           
    theta.append(results[ver].r_corr_gp.meanr)
    y.append(results[ver].alpha_leak)
    yerr.append(results[ver].sig_alpha_leak)
    labels.append(ver)
    colors.append(cat[ver]["colour"])
    linestyles.append(cat[ver]["ls"])
                      
out_path = None
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
    out_path,                                                               
    xlog=True,                                                              
    xlim=[theta_min_plot, theta_max_plot],                                                      
    ylim=ylim_alpha, 
    labels=labels,
    colors=colors,
    linestyles=linestyles,
)

# +
# Plot xi_sys

y = []
yerr = []
colors = []
linestyles = []

for ver in versions:                                                       
    y.append(results[ver].C_sys_p)
    yerr.append(results[ver].C_sys_std_p)
    labels.append(ver)
    colors.append(cat[ver]["colour"])
    linestyles.append(cat[ver]["ls"])

xlabel = r'$\theta$ [arcmin]'                                               
ylabel = r'$\xi^{\rm sys}_+(\theta)$'
title = 'Cross-correlation leakage'
out_path = None                                                                                
fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
plots.plot_data_1d(                                                         
    theta,                                                                  
    y,                                                            
    yerr,                                                                   
    title,                                                                  
    xlabel,                                                                 
    ylabel,                                                                 
    out_path,
    labels=labels,
    xlog=True,                                                              
    xlim=[theta_min_plot, theta_max_plot],
    colors=colors,
    linestyles=linestyles,
)

y = []
yerr = []
for ver in versions:                                                       
    y.append(results[ver].C_sys_m)
    yerr.append(results[ver].C_sys_std_m)

xlabel = r'$\theta$ [arcmin]'
ylabel = r'$\xi^{\rm sys}_-(\theta)$'
title = 'Cross-correlation leakage'
out_path = None
fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
plots.plot_data_1d(                                                         
    theta,                                                                  
    y,                                                            
    yerr,                                                                   
    title,                                                                  
    xlabel,                                                                 
    ylabel,                                                                 
    out_path,
    labels=labels,
    xlog=True,                                                              
    xlim=[theta_min_plot, theta_max_plot],
    ylim=[-1e-7, 1e-6],
    colors=colors,
    linestyles=linestyles,
)
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
axs[0].set_xlim([-1.5,1.5])

for ver in versions:
    e2 = results[ver].dat_shear['e2']
    w = results[ver].dat_shear['w']
    n, bins, _ = axs[1].hist(e2, bins=nbins, density=False, histtype='step', weights=w,\
                            label=ver, color=cat[ver]["colour"])
axs[1].set_xlabel('$e_2$')
axs[0].set_ylabel('frequency')
axs[1].legend()
_ = axs[1].set_xlim([-1.5,1.5])
# -

# ### Cosmology calculations

# #### Compute $\xi_\pm$

# +
treecorr.set_omp_threads(8)

sep_units = 'arcmin'
coord_units = 'degrees'
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
    # TODO: Compute bias here
    g1 = results[ver].dat_shear["e1"] - cat[ver]["shear"]["e1_bias"]
    g2 = results[ver].dat_shear["e2"] - cat[ver]["shear"]["e2_bias"]
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
    print(f"{ver}...", end="")
    gg = treecorr.GGCorrelation(TreeCorrConfig)
    gg.process(cat_gal)
    cat_ggs[ver] = gg
    print("done")
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
        ls=cat[ver]["shear"]['ls'],
        color=cat[ver]["shear"]['colour']
    )
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.ylabel(r'$n_{\rm pair}$')
plt.legend()
plt.show()
# -

#Plot of xi_+
for ver in versions:
    plt.errorbar(
        cat_ggs[ver].meanr, 
        cat_ggs[ver].xip,
        yerr=np.sqrt(cat_ggs[ver].varxip),
        label=ver,
        ls=cat[ver]["shear"]["ls"],
        color=cat[ver]["shear"]["colour"]
    )
plt.plot()
plt.xscale('log')
plt.legend(fontsize=20)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.xlim([0,200])
_ = plt.ylabel(r'$\xi_+(\theta)$')

#Plot of xi_-
for cat_option in cat_options:
    cat_key = cat_option[0]+'_'+cat_option[1]
    plt.errorbar(cat_ggs[cat_key].meanr, cat_ggs[cat_key].xim, yerr=np.sqrt(cat_ggs[cat_key].varxim), \
                 label=r'$\xi_-$ %s' %(cat[cat_option[0]][cat_option[1]]['label']),ls=cat[cat_option[0]][cat_option[1]]['ls'],color=cat[cat_option[0]][cat_option[1]]['colour'])
plt.plot()
plt.xscale('log')
plt.legend(fontsize=20)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.xlim([0,200])
_ = plt.ylabel(r'$\xi_-(\theta)$')
plt.show()

# #### Plot fractional difference

# +
plt.rcParams.update({'font.size': 20,'figure.figsize':[12,10]})

#Define the two catalogues you wish to compare
cat1 = cat_options[0]
cat2 = cat_options[1]

cat_key = cat1[0]+'_'+cat1[1]
cat_key2 = cat2[0]+'_'+cat2[1]

ratio = np.abs(cat_ggs[cat_key2].xip-cat_ggs[cat_key].xip)/cat_ggs[cat_key2].xip
err = ratio * np.sqrt(((np.sqrt(cat_ggs[cat_key].varxip)+np.sqrt(cat_ggs[cat_key2].varxip))/(cat_ggs[cat_key2].xip-cat_ggs[cat_key].xip))**2+(cat_ggs[cat_key2].varxip/cat_ggs[cat_key2].xip)**2)
plt.errorbar(cat_ggs[cat_key].meanr, ratio, yerr=err, 
             label=r'$\xi_+$ Fractional Diff (%s-%s)/%s' %(cat[cat2[0]][cat2[1]]['label'],cat[cat1[0]][cat1[1]]['label'],cat[cat2[0]][cat2[1]]['label']),ls='solid',color='salmon')

ratio = np.abs(cat_ggs[cat_key2].xim-cat_ggs[cat_key].xim)/cat_ggs[cat_key2].xim
err = ratio * np.sqrt(((np.sqrt(cat_ggs[cat_key].varxim)+np.sqrt(cat_ggs[cat_key2].varxim))/(cat_ggs[cat_key2].xim-cat_ggs[cat_key].xim))**2+(cat_ggs[cat_key2].varxim/cat_ggs[cat_key2].xim)**2)
plt.errorbar(cat_ggs[cat_key].meanr, ratio, yerr=err, 
             label=r'$\xi_-$ Fractional Diff (%s-%s)/%s' %(cat[cat2[0]][cat2[1]]['label'],cat[cat1[0]][cat1[1]]['label'],cat[cat2[0]][cat2[1]]['label']),ls='solid',color='indigo')
plt.hlines(0.0,0,200,colors='k')
plt.grid()
plt.xscale('log')
plt.ylim([-2.5,2.5])
plt.legend(fontsize=15)
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.xlim([1,200])
# -

# #### Plot covariance matrices

#Plot comparison of covariance matrices calculated by Jackknife and CosmoCov (here cosmocov files have been provided)
for cat_option in cat_options:
    try:
        cc=np.loadtxt(cat[cat_option[0]][cat_option[1]]['covmat_file'])
    except KeyError:
        print('No Covmat available, skipping analysis for this catalogue')
        continue
    else:
        cat_key = cat_option[0]+'_'+cat_option[1]
        cc_var = np.diag(cc)
        cc_varxip = cc_var[:20]
        cc_varxim = cc_var[20:]

        cc_g = np.loadtxt(cat[cat_option[0]][cat_option[1]]['covmat_file'][:-4]+'_g.txt')
        cc_var = np.diag(cc_g)
        cc_varxip_g = cc_var[:20]
        cc_varxim_g = cc_var[20:]

        plt.loglog(cat_ggs[cat_key].meanr,cat_ggs[cat_key].varxip,'-k', label=r'$\sigma(\xi_+)$ TreeCorr Jackknife %s' %cat[cat_option[0]][cat_option[1]]['label'])
        plt.loglog(cat_ggs[cat_key].meanr,cc_varxip, ls='--', c='%s' %cat[cat_option[0]][cat_option[1]]['colour'], label=r'$\sigma(\xi_+)$ CosmoCov %s' %cat[cat_option[0]][cat_option[1]]['label'])
        plt.loglog(cat_ggs[cat_key].meanr,cc_varxip_g, '.', c='%s' %cat[cat_option[0]][cat_option[1]]['colour'], label=r'$\sigma(\xi_+)$ CosmoCov Gaussian %s' %cat[cat_option[0]][cat_option[1]]['label'])
        plt.grid()
        plt.xlim([cat_ggs[cat_key].meanr[0],cat_ggs[cat_key].meanr[-1]])
        plt.legend(fontsize=15)
        plt.xlabel(rf'$\theta$ [{sep_units}]')
        plt.ylabel(r'$\sigma(\xi_+)$')
        plt.show()

        plt.loglog(cat_ggs[cat_key].meanr,cat_ggs[cat_key].varxim,'-k', label=r'$\sigma(\xi_-)$ TreeCorr Jackknife %s' %cat[cat_option[0]][cat_option[1]]['label'])
        plt.loglog(cat_ggs[cat_key].meanr,cc_varxim, ls='--', c='%s' %cat[cat_option[0]][cat_option[1]]['colour'], label=r'$\sigma(\xi_-)$ CosmoCov %s' %cat[cat_option[0]][cat_option[1]]['label'])
        plt.loglog(cat_ggs[cat_key].meanr,cc_varxim_g, '.', c='%s' %cat[cat_option[0]][cat_option[1]]['colour'], label=r'$\sigma(\xi_-)$ CosmoCov Gaussian %s' %cat[cat_option[0]][cat_option[1]]['label'])
        plt.grid()
        plt.xlim([cat_ggs[cat_key].meanr[0],cat_ggs[cat_key].meanr[-1]])
        plt.legend(fontsize=15)
        plt.xlabel(rf'$\theta$ [{sep_units}]')
        plt.ylabel(r'$\sigma(\xi_-)$')
        plt.show()





