# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Test to apply hsp masks

# %reload_ext autoreload
# %autoreload 2

# +
import os
import numpy as np

import healsparse as hsp
# -

from sp_validation import run_calibrate_cat as calibrate

obj = calibrate.ApplyHspMasks()


# Bits set for non-tomographic catalogues
bit_list = [1, 2, 4, 8, 64, 1024]

# +
# combine bit list with & operator
# equivalent to bits = sum(bit_list)
# if elements in bit_list are of type 2^n

bits = 0
for b in bit_list:
    bits = bits | b
# -

obj._params["input_path"] = "unions_shapepipe_comprehensive_2024_v1.4.2.hdf5"
obj._params["output_path"] = "unions_shapepipe_comprehensive_struc_2024_v1.4.2.hdf5"
obj._params["mask_dir"] = f"{os.environ['HOME']}/v1.4.x/masks"
obj._params["nside"] = 131072
obj._params["file_base"] = "mask_r_"
obj._params["bits"] = bits
obj._params["verbose"] = True

#dat = obj.read_cat(load_into_memory=True, mode="a")
import h5py
f = h5py.File(obj._params["input_path"], mode="a")
dat = f["data"]

masks = obj.get_masks(dat=dat)

dat_new = obj.append_masks(dat, masks)

# MKDEBUG TODO:
# Use JointCat function; write header
with h5py.File(obj._params["output_path"], "w") as f:
    dset = f.create_dataset("data", data=dat_new)
