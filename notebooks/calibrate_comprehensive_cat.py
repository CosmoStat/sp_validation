# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: sp_validation
#     language: python
#     name: python3
# ---

# # Calibrate comprehensive catalogue

#

# %reload_ext autoreload
# %autoreload 2

# General library imports
import sys
import os
import numpy as np
from astropy.io import fits

from sp_validation import run_calibrate_cat as calibrate

obj = calibrate.CalibrateCat()

obj._params["input_path"] = "unions_shapepipe_comprehensive_2024_v1.4.2.fits"

dat = obj.read_cat()

obj.read_cat()


