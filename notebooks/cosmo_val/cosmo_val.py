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

from sp_validation.plot_style import *
from cs_util import plots

# ## Input data

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

ylim_alpha = [-0.03, 0.1]

ylim_xi_sys_ratio = [-0.05, 0.5]
# -

# ## Loading data

# ## Processing

# ### Systematic tests

# + active=""
# # rho stats
# # scale-dependent leakage
# # object-wise leakage
# -

# #### $\xi_\textrm{sys}$ and scale-dependent leakage

# +
ver = 'SP_v1.0'

output_dir = f'{cat["paths"]["output"]}/leakage_{ver}'
ext_path = cat[ver]["ext"]["path"]
star_path = cat[ver]["star"]["path"]
shape_ver = cat[ver]['shape']

nz = f"{cat['nz']['dndz']['path']}_SP_{cat['nz']['dndz']['blind']}.txt"

cmd = (
    f'leakage_scale.py -i {ext_path} -I {star_path} -o {output_dir}'
    + f' --sh {shape_ver} --e1_PSF_star_col e1 --e2_PSF_star_col e2'
    + f' --dndz_path {nz} -v'
)
print(f'Running shell command {cmd}...')
os.system(cmd)

# +
ver = 'LF_v1.0'

output_dir = f'{cat["paths"]["output"]}/leakage_{ver}'

ext_path = cat[ver]["shear"]["path"]
star_path = cat[ver]["star"]["path"]
shape_ver = cat[ver]['shape']

nz = f"{cat['nz']['dndz']['path']}_LF_{cat['nz']['dndz']['blind']}.txt"

cmd = (
    f'leakage_scale.py -i {ext_path} -I {star_path} -o {output_dir}'
    + f' --sh {shape_ver} --e1_col e1 --e2_col e2'
    + f' --e1_PSF_star_col e1 --e2_PSF_star_col e2'
    + f' --dndz_path {nz} -v'
)
print(f'Running shell command {cmd}...')
os.system(cmd)

# +
ver = 'LF_v2.0'

output_dir = f'{cat["paths"]["output"]}/leakage_{ver}'

cmd = (
    f'leakage_scale.py -i {cat_shear[ver]} -I {cat_star[ver]} -o {output_dir}'
    + f' --sh {shapes[ver]} --ra_star_col=ra --dec_star_col=dec'
    + f' --e1_col=e1 --e2_col=e2'
    + f' --e1_PSF_star_col=e1_psf --e2_PSF_star_col=e2_psf'
    + f' --dndz_path {nz} -v'
)
print(f'Running shell command {cmd}...')
os.system(cmd)

# +
# Plot scale-dependent leakage

theta = []
y = []
yerr = []

xlabel = r'$\theta$ [arcmin]'                                               
ylabel = r'$\alpha(\theta)$'
labels = []
linestyles = ['-', ':', '-.']
title = ''
out_path = None

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
        array = df['Separation'].to_numpy()
        idx = np.argwhere(np.isfinite(array))
        # TODO: Check separation distribution

        ra[ver] = df['ra'].to_numpy()[idx].flatten()
        dec[ver] = df['dec'].to_numpy()[idx].flatten()
        e1[ver] = df['e1'].to_numpy()[idx].flatten()
        e2[ver] = df['e2'].to_numpy()[idx].flatten()
        w[ver] = df['w'].to_numpy()[idx].flatten()
# -

# ### Catalogue ellipticity histograms

# +
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
for cat_option in cat_options:
    cat_key = cat_option[0]+'_'+cat_option[1]
    
    cat_gal = treecorr.Catalog(
        ra=ra[cat_key],
        dec=dec[cat_key],
        g1=e1[cat_key]-cat[cat_option[0]][cat_option[1]]['e1_bias'],
        g2=e2[cat_key]-cat[cat_option[0]][cat_option[1]]['e2_bias'],
        w=w[cat_key],
        ra_units='degrees',
        dec_units='degrees',
        npatch=50
    )
    gg = treecorr.GGCorrelation(TreeCorrConfig)
    gg.process(cat_gal)
    cat_ggs.update({cat_key:gg})
    print("done for cat %s" %(cat_key))
# -

# ### Plot the xi_pm

# +
plt.rcParams.update({'font.size': 20,'figure.figsize':[12,10]})

#Plot of n_pairs
for cat_option in cat_options:
    cat_key = cat_option[0]+'_'+cat_option[1]
    plt.plot(cat_ggs[cat_key].meanr, cat_ggs[cat_key].npairs, \
             label=r'$n_{pairs}$ %s' %(cat[cat_option[0]][cat_option[1]]['label']), \
             ls=cat[cat_option[0]][cat_option[1]]['ls'],color=cat[cat_option[0]][cat_option[1]]['colour'])
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.ylabel(r'$n_{pairs}$')
plt.legend()
plt.show()

#Plot of xi_+
for cat_option in cat_options:
    cat_key = cat_option[0]+'_'+cat_option[1]
    plt.errorbar(cat_ggs[cat_key].meanr, cat_ggs[cat_key].xip, yerr=np.sqrt(cat_ggs[cat_key].varxip), \
                 label=r'$\xi_+$ %s' %(cat[cat_option[0]][cat_option[1]]['label']),ls=cat[cat_option[0]][cat_option[1]]['ls'],color=cat[cat_option[0]][cat_option[1]]['colour'])
plt.plot()
plt.xscale('log')
plt.legend(fontsize=20)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.xlim([0,200])
_ = plt.ylabel(r'$\xi_+(\theta)$')
plt.show()

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
# -

# ### Plot Fractional Difference

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

# ### Plot CovMats

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





