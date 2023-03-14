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

import os
import numpy as np
import matplotlib.pylab as plt
import sys
import numpy as np
from astropy.io import fits
import treecorr
import pandas as pd
from cs_util import plots

# ## Input data

# +
# Base directory for data, on candide
data_base_dir = '/feynman/work/dap/lcs/lg268561/UNIONS/Catalogues'

# Base directory for v1.0 data
v1_base_dir = f'{data_base_dir}/v1.0'
# -

# catalogue versions
versions = {
    'SP_v1.0',
    'LF_v1.0',
}

# +
# Catalogues

# Basic shear catalogue
cat_shear = {}
cat_shear['SP_v1.0'] = f'{v1_base_dir}/ShapePipe/unions_shapepipe_2022_v1.0.2.fits'

# Extended shear catalogue
cat_shear_ext = {}
cat_shear_ext['SP_v1.0'] = f'{v1_base_dir}/ShapePipe/unions_shapepipe_extended_2022_v1.0.fits'


# PSF catalogue
cat_psf = {}
# Updated to 1.0.2
cat_psf['SP_v1.0'] = f'{v1_base_dir}/ShapePipe/unions_psf_2022_v1.0.2.fits'
<<<<<<< HEAD
=======

# Star catalogue (with ngmix shapes)
cat_star = {}
# Updated to 1.0.3
cat_star['SP_v1.0'] = f'{v1_base_dir}/ShapePipe/unions_star_2022_v1.0.3.fits'


yyyy = 1
>>>>>>> 6b0c7c9e26bf863732168d518285daed030a54bb
# -

# ## Loading data

# ## Processing

# ### Systematic tests

# + active=""
# # xi_sys
# # rho stats
# # scale-dependent leakage
# # object-wise leakage

# +
# xi_sys

ver = 'SP_v1.0'
cmd = f'leakage_scale.py -i {cat_shear_ext[ver]} -I {cat_shear_ext[ver]} -o . --e1_PSF_star_col e1_PSF --e2_PSF_star_col e2_PSF -v'
print(f'Running shell command {cmd}...')
os.system(cmd)
# -

# ## Cosmological analysis

# ### Load in Catalogues

# +
# CATALOGUE OPTIONS: [HARDCODED, DO NOT CHANGE!]
# 1: LensFit Full
# 2: ShapePipe Full
# 3: LF Match SP
# 4: SP Match LF
# 5: SP Matched in LF footprint
# 6: SP Match MegaPipe

#Dictionary of the different catalogues and their options [HARDCODED, DO NOT CHANGE]
cat_dict = {1:{'dir': data_base_dir + '/lensfit_goldshape_2022v1.fits',
                  'label': 'LF_Full',
                  'e1_bias': 0,
                  'e2_bias': 0,
                  'ls': 'solid',
                  'colour': 'g',
                  'covmat_file': './covs/lensfit_A/cov_lensfit_A.txt'},
           2:{'dir': data_base_dir + '/unions_shapepipe_2022_v1.0.fits',
                 'label': 'SP_Full',
                 'e1_bias': 0,
                 'e2_bias': 0,
                  'ls': 'solid',
                  'colour': 'b',
                  'covmat_file': './covs/shapepipe_A/cov_shapepipe_A.txt'},
           3:{'dir': data_base_dir + '/masked_matched_lensfit_goldshape_2022v1.fits',
                  'label': 'LF_Match_SP',
                  'e1_bias': 3.939e-4,
                  'e2_bias': 6.482e-5,
                  'ls': 'dotted',
                  'colour': 'g',
                  'covmat_file': './covs/lensfit_matched/cov_lensfit_matched.txt'},
           4:{'dir': data_base_dir + '/masked_matched_unions_shapepipe_extended_2022_v1.0.fits',
                  'label': 'SP_Match_LF',
                  'e1_bias': -5.6726e-5,
                  'e2_bias': 8.218e-4,
                  'ls': 'dotted',
                  'colour': 'b',
                  'covmat_file': './covs/shapepipe_matched/cov_shapepipe_matched.txt'},
           5:{'dir': data_base_dir + '/matched_footprint_shapepipe.fits',
                  'label': 'SP_Match LF_Footprint',
                  'e1_bias': 0,
                  'e2_bias': 0,
                  'ls': 'dashed',
                  'colour': 'b'},
           6:{'dir': data_base_dir + '/cfis-shapepipe.parquet',
                  'label': 'SP_Match_MegaPipe',
                  'e1_bias': 0,
                  'e2_bias': 0,
                  'ls': 'dashdot',
                  'colour': 'b'}}

# +
# User input catalogue option 
cat_options=[3,4]

# Read in catalogue data and store as arrays
ra = {}
dec = {}
e1 = {}
e2 = {}
w = {}
for cat_option in cat_options:
    if cat_option==6:
        df = pd.read_parquet(cat_dict[cat_option]['dir'], engine='pyarrow')
        array = df['Separation'].to_numpy()
        idx = np.argwhere(np.isfinite(array))
        ra.update({cat_option: df['ra'].to_numpy()[idx].flatten()})
        dec.update({cat_option:df['dec'].to_numpy()[idx].flatten()})
        e1.update({cat_option:df['e1'].to_numpy()[idx].flatten()})
        e2.update({cat_option:df['e2'].to_numpy()[idx].flatten()})
        w.update({cat_option:df['w'].to_numpy()[idx].flatten()})
    else:
        hdu = fits.open(cat_dict[cat_option]['dir'])
        data = hdu[1].data
        ra.update({cat_option:data['ra']})
        dec.update({cat_option:data['dec']})
        e1.update({cat_option:data['e1']})
        e2.update({cat_option:data['e2']})
        w.update({cat_option:data['w']})
# -

# ### Catalogue histograms

# +
plt.rcParams.update({'font.size': 20,'figure.figsize':[22,7]})

fig, axs = plt.subplots(1, 2)
nbins = 200

for cat in cat_options:
    (n,bins,_)= axs[0].hist(e1[cat], bins=nbins, density=False, histtype='step', weights=w[cat],label='e1 %s' %cat_dict[cat]['label'])
axs[0].set_xlabel('e1')
axs[0].legend()
axs[0].set_xlim([-1.5,1.5])
# axs[0].set_ylim([0,2e4])

for cat in cat_options:
    (n,bins,_)= axs[1].hist(e2[cat], bins=nbins, density=False, histtype='step', weights=w[cat],label='e2 %s' %cat_dict[cat]['label'])
axs[1].set_xlabel('e2')
axs[1].legend()
axs[1].set_xlim([-1.5,1.5])
# axs[1].set_ylim([0,2e4])
# -

# ### Compute xi_pm's

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
for cat in cat_options:
    cat_gal = treecorr.Catalog(
        ra=ra[cat],
        dec=dec[cat],
        g1=e1[cat]-cat_dict[cat]['e1_bias'],
        g2=e2[cat]-cat_dict[cat]['e2_bias'],
        w=w[cat],
        ra_units='degrees',
        dec_units='degrees',
        npatch=50
    )
    gg = treecorr.GGCorrelation(TreeCorrConfig)
    gg.process(cat_gal)
    cat_ggs.update({cat:gg})
    print("done for cat %s" %cat)
# -

# ### Plot the xi_pm's

# +
plt.rcParams.update({'font.size': 20,'figure.figsize':[12,10]})

#Plot of n_pairs
for cat in cat_options:
    plt.plot(cat_ggs[cat].meanr, cat_ggs[cat].npairs, label=r'$n_{pairs}$ %s' %(cat_dict[cat]['label']),ls=cat_dict[cat]['ls'],color=cat_dict[cat]['colour'])
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.ylabel(r'$n_{pairs}$')
plt.legend()
plt.show()

#Plot of xi_+
for cat in cat_options:
    plt.errorbar(cat_ggs[cat].meanr, cat_ggs[cat].xip, yerr=np.sqrt(cat_ggs[cat].varxip), label=r'$\xi_+$ %s' %(cat_dict[cat]['label']),ls=cat_dict[cat]['ls'],color=cat_dict[cat]['colour'])
plt.plot()
plt.xscale('log')
plt.legend(fontsize=20)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.xlim([0,200])
_ = plt.ylabel(r'$\xi_+(\theta)$')
# plt.savefig('plots/xi_pm_3500.pdf')
plt.show()

#Plot of xi_-
for cat in cat_options:
    plt.errorbar(cat_ggs[cat].meanr, cat_ggs[cat].xim, yerr=np.sqrt(cat_ggs[cat].varxim), label=r'$\xi_-$ %s' %(cat_dict[cat]['label']),ls=cat_dict[cat]['ls'],color=cat_dict[cat]['colour'])
plt.plot()
plt.xscale('log')
plt.legend(fontsize=20)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.xlim([0,200])
_ = plt.ylabel(r'$\xi_-(\theta)$')
# plt.savefig('plots/xi_pm_3500.pdf')
plt.show()
# -

# ### Plot Fractional Difference

# +
plt.rcParams.update({'font.size': 20,'figure.figsize':[12,10]})

#Define the two catalogues you wish to compare
cat1 = 3
cat2 = 4

ratio = np.abs(cat_ggs[cat2].xip-cat_ggs[cat1].xip)/cat_ggs[cat2].xip
err = ratio * np.sqrt(((np.sqrt(cat_ggs[cat1].varxip)+np.sqrt(cat_ggs[cat2].varxip))/(cat_ggs[cat2].xip-cat_ggs[cat1].xip))**2+(cat_ggs[cat2].varxip/cat_ggs[cat2].xip)**2)
plt.errorbar(cat_ggs[cat1].meanr, ratio, yerr=err, 
             label=r'$\xi_+$ Fractional Diff (%s-%s)/%s' %(cat_dict[cat2]['label'],cat_dict[cat1]['label'],cat_dict[cat2]['label']),ls='solid',color='salmon')

ratio = np.abs(cat_ggs[cat2].xim-cat_ggs[cat1].xim)/cat_ggs[cat2].xim
err = ratio * np.sqrt(((np.sqrt(cat_ggs[cat1].varxim)+np.sqrt(cat_ggs[cat2].varxim))/(cat_ggs[cat2].xim-cat_ggs[cat1].xim))**2+(cat_ggs[cat2].varxim/cat_ggs[cat2].xim)**2)
plt.errorbar(cat_ggs[cat1].meanr, ratio, yerr=err, 
             label=r'$\xi_-$ Fractional Diff (%s-%s)/%s' %(cat_dict[cat2]['label'],cat_dict[cat1]['label'],cat_dict[cat2]['label']),ls='solid',color='indigo')
plt.hlines(0.0,0,200,colors='k')
plt.grid()
plt.xscale('log')
plt.ylim([-2.5,2.5])
plt.legend(fontsize=15)
plt.xlabel(rf'$\theta$ [{sep_units}]')
plt.xlim([1,200])
# -

# ### Plot CovMats

# +
#Plot comparison of covariance matrices calculated by Jackknife and CosmoCov (here cosmocov files have been provided)
for cat in cat_options:
    try:
        cc=np.loadtxt(cat_dict[cat]['covmat_file'])
    except KeyError:
        print('No Covmat available, skipping analysis for this catalogue')
        continue
    else:
        cc_var = np.diag(cc)
        cc_varxip = cc_var[:20]
        cc_varxim = cc_var[20:]

        cc_g = np.loadtxt(cat_dict[cat]['covmat_file'][:-4]+'_g.txt')
        cc_var = np.diag(cc_g)
        cc_varxip_g = cc_var[:20]
        cc_varxim_g = cc_var[20:]

        plt.loglog(cat_ggs[cat].meanr,cat_ggs[cat].varxip,'-k', label=r'$\sigma(\xi_+)$ TreeCorr Jackknife %s' %cat_dict[cat]['label'])
        plt.loglog(cat_ggs[cat].meanr,cc_varxip, ls='--', c='%s' %cat_dict[cat]['colour'], label=r'$\sigma(\xi_+)$ CosmoCov %s' %cat_dict[cat]['label'])
        plt.loglog(cat_ggs[cat].meanr,cc_varxip_g, '.', c='%s' %cat_dict[cat]['colour'], label=r'$\sigma(\xi_+)$ CosmoCov Gaussian %s' %cat_dict[cat]['label'])
        plt.grid()
        plt.xlim([cat_ggs[cat].meanr[0],cat_ggs[cat].meanr[-1]])
        plt.legend(fontsize=15)
        plt.xlabel(rf'$\theta$ [{sep_units}]')
        plt.ylabel(r'$\sigma(\xi_+)$')
        plt.show()

        plt.loglog(cat_ggs[cat].meanr,cat_ggs[cat].varxim,'-k', label=r'$\sigma(\xi_-)$ TreeCorr Jackknife %s' %cat_dict[cat]['label'])
        plt.loglog(cat_ggs[cat].meanr,cc_varxim, ls='--', c='%s' %cat_dict[cat]['colour'], label=r'$\sigma(\xi_-)$ CosmoCov %s' %cat_dict[cat]['label'])
        plt.loglog(cat_ggs[cat].meanr,cc_varxim_g, '.', c='%s' %cat_dict[cat]['colour'], label=r'$\sigma(\xi_-)$ CosmoCov Gaussian %s' %cat_dict[cat]['label'])
        plt.grid()
        plt.xlim([cat_ggs[cat].meanr[0],cat_ggs[cat].meanr[-1]])
        plt.legend(fontsize=15)
        plt.xlabel(rf'$\theta$ [{sep_units}]')
        plt.ylabel(r'$\sigma(\xi_-)$')
        plt.show()


# -


