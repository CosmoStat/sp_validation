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
import tracemalloc
import healsparse as hsp

from sp_validation import run_joint_cat as sp_joint

# +
# Trace and print used memory if True
trace_mem = True

if trace_mem:
    tracemalloc.start()
# -

# Create instance of ApplyHspMasks object
obj = sp_joint.ApplyHspMasks()


# Bits set for non-tomographic catalogues
bit_list = [1, 2, 4, 8, 64, 1024]

# +
# combine bit list with & operator
# equivalent to bits = sum(bit_list)
# if elements in bit_list are of type 2^n

bits = 0
for b in bit_list:
    bits = bits | b
print(bits)

# +
# Set parameters
obj._params["input_path"] = "unions_shapepipe_comprehensive_2024_v1.5.c.hdf5"
obj._params["output_path"] = "unions_shapepipe_comprehensive_struc_2024_v1.5.c.hdf5"
obj._params["mask_dir"] = f"{os.environ['HOME']}/v1.5.x/masks"
obj._params["nside"] = 131072
obj._params["file_base"] = "mask_r_"
obj._params["bits"] = bits

obj._params["aux_mask_files"] = f"{obj._params['mask_dir']}/coverage.hsp"
obj._params["aux_mask_labels"] = "npoint3"
obj._params["verbose"] = True
# -

# ## Run

# +
# Check parameter validity
obj.check_params()

# Update parameters (here: strings to list)
obj.update_params()
# -

if trace_mem:
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current (peak) memory usage: {current / 1024**2:.2f} ({peak / 1024**2:.2f}) MB")

# Read catalogue
dat = obj.read_cat(load_into_memory=False, mode="r")

if trace_mem:
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current (peak) memory usage: {current / 1024**2:.2f} ({peak / 1024**2:.2f}) MB")

# Get bit-coded masks
masks = obj.get_masks(dat=dat)

if trace_mem:
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current (peak) memory usage: {current / 1024**2:.2f} ({peak / 1024**2:.2f}) MB")

# Add mask bits as new columns
dat_new = obj.append_masks(dat, masks)

if trace_mem:
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current (peak) memory usage: {current / 1024**2:.2f} ({peak / 1024**2:.2f}) MB")

# Write extended data to new HDF5 file
obj.write_hdf5_file(dat, dat_new)

# Close input HDF5 catalogue file
obj.close_hd5()

if trace_mem:
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current (peak) memory usage: {current / 1024**2:.2f} ({peak / 1024**2:.2f}) MB")
    tracemalloc.stop()


