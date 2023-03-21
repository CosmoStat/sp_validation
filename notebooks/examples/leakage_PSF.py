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

# # SP validation

# ## PSF leakage

import os
from sp_validation import run

# ### Set input catalogue paths

# +
# Data directory
data_dir = f"{os.environ['HOME']}/astro/data/CFIS/v1.0/ShapePipe"

# Input galaxy shear catalogue
input_path_shear = f"{data_dir}/unions_shapepipe_extended_2022_v1.0.fits"

# Input star/PSF catalogue
input_path_PSF = f"{data_dir}/unions_shapepipe_star_2022_v1.0.3.fits"
# -

# ### Compute leakage

# Create leakage instance
obj = run.LeakageScale()

# +
# Set instance parameters, copy from above
obj._params['input_path_shear'] = input_path_shear
obj._params['input_path_PSF'] = input_path_PSF
obj._params['e1_PSF_star_col'] =  'e1'
obj._params['e2_PSF_star_col'] =  'e2'

obj._params['verbose'] = True
# -

obj.run()


