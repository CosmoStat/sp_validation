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

from sp_validation import plot_style 
from cs_util import plots

# ## Input data

# +
# Base directory for data, on candide
data_base_dir = '/n17data/mkilbing/astro/data/CFIS'

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

# ### Cosmological analysis

# +
# xipm correlation functions
# B-modes
# covariance
# MCMC
