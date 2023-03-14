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

# +
import os
import numpy as np
import matplotlib.pylab as plt

from astropy.io import ascii

# +
from sp_validation.plot_style import *                                          

from cs_util import plots
# -

# ## Input data

# +
# Base directory for data, on candide
data_base_dir = f'{os.environ["HOME"]}/astro/data/CFIS'

# Base directory for v1.0 data
v1_base_dir = f'{data_base_dir}/v1.0'
# -

# catalogue versions
versions = {
    'SP_v1.0',
    'LF_v1.0',
}

# Shape measurement methods (for file names, plot titles, etc.)
shapes = {
    'SP_v1.0' : 'ngmix',
    'LF_v1.0' : 'lensfit'
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
cat_psf['SP_v1.0'] = f'{v1_base_dir}/ShapePipe/unions_shapepipe_psf_2022_v1.0.2.fits'

# Star catalogue (with ngmix shapes)
cat_star = {}
# Updated to 1.0.3
cat_star['SP_v1.0'] = f'{v1_base_dir}/ShapePipe/unions_shapepipe_star_2022_v1.0.3.fits'
# -

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
# # xi_sys
# # rho stats
# # scale-dependent leakage
# # object-wise leakage

# +
# xi_sys

ver = 'SP_v1.0'

output_dir = f'leakage_{ver}'

cmd = f'leakage_scale.py -i {cat_shear_ext[ver]} -I {cat_star[ver]} -o {output_dir} --e1_PSF_star_col e1 --e2_PSF_star_col e2 -v'
print(f'Running shell command {cmd}...')
os.system(cmd)

# +
# Plot scale-dependent leakage

shape_method = shapes[ver]

alpha_name = f'{output_dir}/alpha_leakage_{shape_method}.txt'
alpha = ascii.read(f'{alpha_name}')
                                                           
theta = [alpha['theta[arcmin]']]   
y = [alpha['alpha']]
yerr = [alpha['sig_alpha']]                                                     
xlabel = r'$\theta$ [arcmin]'                                               
ylabel = r'$\alpha(\theta)$'
linestyles = ['-']
title = alpha_name                                                        
out_path = None # f'{output_dir}/alpha_leakage_{shape_method}_nb.png'                                                                     
                                                                                
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
)

# +
# Plot xi_sys relative to xi (theory)

shape_method = shapes[ver]

xi_sys_name = f'{output_dir}/xi_sys_{shape_method}.txt'
xi_sys = ascii.read(f'{xi_sys_name}')
                                                           
theta = [xi_sys['theta[arcmin]']]   
y = [
    xi_sys['xi_+_sys'] / xi_sys['xi_+_theo'],
    xi_sys['xi_-_sys'] / xi_sys['xi_-_theo'],
]
yerr = [
    xi_sys['sigma(xi_+_sys)'] / xi_sys['xi_+_theo'],
    xi_sys['sigma(xi_-_sys)'] / xi_sys['xi_-_theo'],           
]
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
    xlog=True,                                                              
    xlim=[theta_min_plot, theta_max_plot],                                                      
    ylim=ylim_xi_sys_ratio, 
    linestyles=linestyles,
)
# -

# ### Cosmological analysis

# +
# xipm correlation functions
# B-modes
# covariance
# MCMC
# -

keys = [f'xi_{comp}_sys' for comp in components]
xi_sys[keys]

# +
xi_sys_name = f'{output_dir}/xi_sys_{shape_method}.txt'
xi_sys = ascii.read(f'{xi_sys_name}')
                                                           
theta = [xi_sys['theta[arcmin]']]   
y = [xi_sys['xi_+_sys']]
y
# -


