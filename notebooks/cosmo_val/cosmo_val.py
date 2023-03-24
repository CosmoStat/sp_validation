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

# ## Input data

# +
# Catalogue versions
versions = ['SP_v1.0', 'LF_v1.0', 'SP_matched_LF_v1.0','SP_matched_MP_v1.0']

all_keys = ['nz']
for ver in versions:
    all_keys.append(ver)
# -

# Base directory for data, on candide
data_base_dir = f'{os.environ["WORK"]}'

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
                
# ## Variables

components = ['+', '-']

# ### Plotting

# +
theta_min_plot = 1
theta_max_plot = 300

ylim_alpha = [-0.03, 0.1]

ylim_xi_sys_ratio = [-0.05, 0.5]
# -

# ## Processing

# ### Systematic tests

# + active=""
# # rho stats
# # scale-dependent leakage
# # object-wise leakage
# -

# #### $\xi_\textrm{sys}$ and scale-dependent leakage

leak_scale = {}

# +
ver = 'SP_v1.0'

params_in = {}

# Set parameters
params_in['input_path_shear'] = cat[ver]["ext"]["path"]
params_in['input_path_PSF'] = cat[ver]["star"]["path"]
params_in['dndz_path'] = f"{cat['nz']['dndz']['path']}_{cat[ver]['pipeline']}_{cat['nz']['dndz']['blind']}.txt"
params_in['output_dir'] = f'{cat["paths"]["output"]}/leakage_{ver}'
params_in['sh'] = cat[ver]['shape']
params_in['e1_PSF_star_col'] = "e1"
params_in["e2_PSF_star_col"] = "e2"
params_in["verbose"] = True

# Create leakage instance
obj = run.LeakageScale()

# Set instance parameters, copy from above
for key in params_in:
    obj._params[key] = params_in[key]
# -

obj.run()

leak_scale[ver] = obj

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
params_in['e1_PSF_star_col'] = "e1"
params_in["e2_PSF_star_col"] = "e2"
params_in["verbose"] = True

# Create leakage instance
obj = run.LeakageScale()

# Set instance parameters, copy from above
for key in params_in:
    obj._params[key] = params_in[key]
# -

obj.run()

leak_scale[ver] = obj

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
params_in['e1_PSF_star_col'] = "e1_psf"
params_in["e2_PSF_star_col"] = "e2_psf"
params_in["verbose"] = True

# Create leakage instance
obj = run.LeakageScale()

# Set instance parameters, copy from above
for key in params_in:
    obj._params[key] = params_in[key]
# -

obj.run()

# +
# Plot scale-dependent leakage
for ver in versions:
    shape_method = cat[ver]['shape']
    output_dir = f'leakage_{ver}'

    alpha_name = f'{output_dir}/alpha_leakage_{shape_method}.txt'
    alpha = ascii.read(f'{alpha_name}')
                                                           
    theta.append(alpha['theta[arcmin]'])
    y.append(alpha['alpha'])
    yerr.append(alpha['sig_alpha'])
    labels.append(ver)
                                                                                
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
    linestyles=linestyles,
    labels=labels,
)

# +
# Plot xi_sys relative to xi (theory)

theta = []
y = []
yerr = []

for ver in versions:

    shape_method = cat[ver]['shape']
    output_dir = f'{cat["paths"]["output"]/leakage_{ver}'

    xi_sys_name = f'{output_dir}/xi_sys_{shape_method}.txt'
    xi_sys = ascii.read(f'{xi_sys_name}')
                                                           
    theta.append(xi_sys['theta[arcmin]'])

    y.append(
        xi_sys[f'xi_{comp}_sys'] / xi_sys[f'xi_{comp}_theo'],
    )
    yerr.append(
        xi_sys[f'sigma(xi_{comp}_sys)'] / xi_sys[f'xi_{comp}_theo'],
    )
    labels.append(rf'$\xi^{{sys}}_{comp} ver')

labels = [rf'$\xi^{{sys}}_{comp}' for comp in components]
xlabel = r'$\theta$ [arcmin]'                                               
ylabel = r'correlation function ratio'
linestyles = ['-', '-']
title = xi_sys_name
out_path = None #f'{output_dir}/xi_sys_{shape_method}_ratio_nb.png'                                                                     
                                                                                
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
    ylim=ylim_xi_sys_ratio, 
)
plt.show()
# +
# ### Cosmological analysis
# xipm correlation functions
# B-modes
# covariance
# MCMC
# -

# ### Load catalogues

# +
ra = {}
dec = {}
e1 = {}
e2 = {}
w = {}

for ver in versions:
    print(ver)
    in_path = cat[ver]["shear"]["path"]
    # TODO: add to io.py: open_fits_or_npy
    print(f'Loading {in_path}')
    _, file_extension = os.path.splitext(in_path)
    if file_extension == '.fits':
        hdu_list = fits.open(in_path)
        data = hdu_list[1].data
        ra[ver] = data['ra']
        dec[ver] = data['dec']
        e1[ver] = data['e1']
        e2[ver] = data['e2']
        w[ver] = data['w']
        hdu_list.close()
    elif file_extension == '.parquet':
        df = pd.read_parquet(in_path, engine='pyarrow')
        sep_array = df['Separation'].to_numpy()
        idx = np.argwhere(np.isfinite(sep_array))
        ra[ver] = df['ra'].to_numpy()[idx].flatten()
        dec[ver] = df['dec'].to_numpy()[idx].flatten()
        e1[ver] = df['e1'].to_numpy()[idx].flatten()
        e2[ver] = df['e2'].to_numpy()[idx].flatten()
        w[ver] = df['w'].to_numpy()[idx].flatten()
# -

# ### Catalogue histograms

# +
# Plot separation for SP/MP match
plt.rcParams.update({'figure.figsize': [10,7]})
fig, axs = plt.subplots(1, 1)
nbins = 200

print('Max separation: %s arcsec' %max(sep_array))
n, bins, _ = axs.hist(sep_array, bins=nbins, density=False, histtype='step',label='ShapePipe MegaPipe Match')
axs.set_xlabel(r'Separation $\theta$ [arcsec]')
axs.legend()

# +
# Plot ellipticities
plt.rcParams.update({'figure.figsize': [22,7]})
fig, axs = plt.subplots(1, 2)
nbins = 200

for ver in versions:
    n, bins, _ = axs[0].hist(e1[ver], bins=nbins, density=False, histtype='step', weights=w[ver],\
                            label=f'$e_1$ {ver}')
axs[0].set_xlabel('$e_1$')
axs[0].set_ylabel('frequency')
axs[0].legend()
axs[0].set_xlim([-1.5,1.5])
# axs[0].set_ylim([0,2e4])

for ver in versions:
    n, bins, _ = axs[1].hist(e2[ver], bins=nbins, density=False, histtype='step', weights=w[ver],\
                            label=f'$e_2$ {ver}')
axs[1].set_xlabel('$e_2$')
axs[0].set_ylabel('frequency')
axs[1].legend()
_ = axs[1].set_xlim([-1.5,1.5])
# -

# ### Compute xi_pm

# +
treecorr.set_omp_threads(8)

sep_units = 'arcmin'
theta_min = 1
theta_max = 200

TreeCorrConfig = {
        'ra_units': 'degrees',
        'dec_units': 'degrees',
        'max_sep': str(theta_max),
        'min_sep': str(theta_min),
        'sep_units': sep_units,
        'nbins': 20,
        'var_method':'jackknife',
    }

cat_ggs = {}
for ver in versions:
    cat_gal = treecorr.Catalog(
        ra=ra[ver],
        dec=dec[ver],
        g1=e1[ver]-cat[ver]['shear']['e1_bias'],
        g2=e2[ver]-cat[ver]['shear']['e2_bias'],
        w=w[ver],
        ra_units='degrees',
        dec_units='degrees',
        npatch=50
    )
    gg = treecorr.GGCorrelation(TreeCorrConfig)
    gg.process(cat_gal)
    cat_ggs.update({ver:gg})
    print("done for cat %s" %(ver))
# -

# ### Plot the xi_pm

# +
plt.rcParams.update({'font.size': 20,'figure.figsize':[12,10]})

#Plot of n_pairs
for ver in versions:
    plt.plot(cat_ggs[ver].meanr, cat_ggs[ver].npairs, \
             label=r'$n_{pairs}$ %s' %(cat[ver]['shear']['label']), \
             ls=cat[ver]['shear']['ls'],color=cat[ver]['shear']['colour'])
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.ylabel(r'$n_{pairs}$')
plt.legend()
plt.show()

#Plot of xi_+
for ver in versions:
    plt.errorbar(cat_ggs[ver].meanr, cat_ggs[ver].xip, yerr=np.sqrt(cat_ggs[ver].varxip), \
                 label=r'$\xi_+$ %s' %(cat[ver]['shear']['label']),ls=cat[ver]['shear']['ls'],color=cat[ver]['shear']['colour'])
plt.plot()
plt.xscale('log')
plt.legend(fontsize=20)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.xlim([0,200])
_ = plt.ylabel(r'$\xi_+(\theta)$')
plt.show()

#Plot of xi_-
for ver in versions:
    plt.errorbar(cat_ggs[ver].meanr, cat_ggs[ver].xim, yerr=np.sqrt(cat_ggs[ver].varxim), \
                 label=r'$\xi_-$ %s' %(cat[ver]['shear']['label']),ls=cat[ver]['shear']['ls'],color=cat[ver]['shear']['colour'])
plt.plot()
plt.xscale('log')
plt.legend(fontsize=20)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.xlim([0,200])
_ = plt.ylabel(r'$\xi_-(\theta)$')
plt.show()
# -

# ### Plot Fractional Difference

# +
plt.rcParams.update({'font.size': 20,'figure.figsize':[12,10]})

#Define the two catalogues you wish to compare
ver = versions[0]
ver2 = versions[1]

ratio = np.abs(cat_ggs[ver2].xip-cat_ggs[ver].xip)/cat_ggs[ver2].xip
err = ratio * np.sqrt(((np.sqrt(cat_ggs[ver].varxip)+np.sqrt(cat_ggs[ver2].varxip))/(cat_ggs[ver2].xip-cat_ggs[ver].xip))**2+(cat_ggs[ver2].varxip/cat_ggs[ver2].xip)**2)
plt.errorbar(cat_ggs[ver].meanr, ratio, yerr=err, 
             label=r'$\xi_+$ Fractional Diff (%s-%s)/%s' %(cat[ver2]['shear']['label'],cat[ver]['shear']['label'],cat[ver2]['shear']['label']), \
                ls='solid',color='salmon')

ratio = np.abs(cat_ggs[ver2].xim-cat_ggs[ver].xim)/cat_ggs[ver2].xim
err = ratio * np.sqrt(((np.sqrt(cat_ggs[ver].varxim)+np.sqrt(cat_ggs[ver2].varxim))/(cat_ggs[ver2].xim-cat_ggs[ver].xim))**2+(cat_ggs[ver2].varxim/cat_ggs[ver2].xim)**2)
plt.errorbar(cat_ggs[ver].meanr, ratio, yerr=err, 
             label=r'$\xi_-$ Fractional Diff (%s-%s)/%s' %(cat[ver2]['shear']['label'],cat[ver]['shear']['label'],cat[ver2]['shear']['label']), \
                ls='solid',color='indigo')
plt.hlines(0.0,0,200,colors='k')
plt.grid()
plt.xscale('log')
# plt.ylim([-2.5,2.5])
plt.legend(fontsize=15)
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.xlim([1,200])
# -

# ### Plot CovMats

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

        plt.loglog(cat_ggs[ver].meanr,cat_ggs[ver].varxip,'-k', label=r'$\sigma(\xi_+)$ TreeCorr Jackknife %s' %cat[ver]['shear']['label'])
        plt.loglog(cat_ggs[ver].meanr,cc_varxip, ls='--', c='%s' %cat[ver]['shear']['colour'], \
                   label=r'$\sigma(\xi_+)$ CosmoCov %s' %cat[ver]['shear']['label'])
        plt.loglog(cat_ggs[ver].meanr,cc_varxip_g, '.', c='%s' %cat[ver]['shear']['colour'], \
                   label=r'$\sigma(\xi_+)$ CosmoCov Gaussian %s' %cat[ver]['shear']['label'])
        plt.grid()
        plt.xlim([cat_ggs[ver].meanr[0],cat_ggs[ver].meanr[-1]])
        plt.legend(fontsize=15)
        plt.xlabel(rf'$\theta$ [{sep_units}]')
        plt.ylabel(r'$\sigma(\xi_+)$')
        plt.show()

        plt.loglog(cat_ggs[ver].meanr,cat_ggs[ver].varxim,'-k', label=r'$\sigma(\xi_-)$ TreeCorr Jackknife %s' %cat[ver]['shear']['label'])
        plt.loglog(cat_ggs[ver].meanr,cc_varxim, ls='--', c='%s' %cat[ver]['shear']['colour'],\
                    label=r'$\sigma(\xi_-)$ CosmoCov %s' %cat[ver]['shear']['label'])
        plt.loglog(cat_ggs[ver].meanr,cc_varxim_g, '.', c='%s' %cat[ver]['shear']['colour'], \
                    label=r'$\sigma(\xi_-)$ CosmoCov Gaussian %s' %cat[ver]['shear']['label'])
        plt.grid()
        plt.xlim([cat_ggs[ver].meanr[0],cat_ggs[ver].meanr[-1]])
        plt.legend(fontsize=15)
        plt.xlabel(rf'$\theta$ [{sep_units}]')
        plt.ylabel(r'$\sigma(\xi_-)$')
        plt.show()

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
scratch_dir = f'{os.environ["WORK"]}'
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
    colours.append(cat[ver]['shear']['getdist_colour'])
    line_args.append({'color': cat[ver]['shear']['getdist_col']})

# +
# %matplotlib inline
g.triangle_plot(chains,['omega_m','omega_b','ln10^10A_s','n_s','tau','h0','SIGMA_8','S_8'],
                legend_labels=versions,
                colors=colours,
                line_args=line_args)

# g.export('plots/corner_plot_comparison_lf.pdf')
