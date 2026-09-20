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

# # Demo notebook to apply hsp masks
#
# Read healsparse mask files, compute mask values for input catalogue, and add those
# values as new columns.

# %reload_ext autoreload
# %autoreload 2

# +
import os
import tracemalloc

from sp_validation import catalog_builders as sp_joint

# +
# Trace and print used memory if True
trace_mem = False

if trace_mem:
    tracemalloc.start()
# -

# Create instance of ApplyHspMasks object
obj = sp_joint.ApplyHspMasks()



# +
# Set parameters
base = "unions_shapepipe_comprehensive"

# Major version
ver_maj = "v1.6"

# Mask release, subdirectory of the mask directory
mask_release = "dr6-2026-08"

year = 2022 if ver_maj == "v1.3" else 2024

# $HOME/masks is a link to the shared UNIONS mask directory
mask_base = f"{os.environ['HOME']}/masks"

if ver_maj in ("v1.5", "v1.6"):
    # Multi-band (tomographic)
    bit_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048]
    obj._params["mask_dir"] = f"{mask_base}/{mask_release}"
    obj._params["file_base"] = "mask_ugriz_"
else:
    # r-band (non-tomographic)
    bit_list = [1, 2, 4, 8, 64, 1024]
    obj._params["mask_dir"] = mask_base
    obj._params["file_base"] = "mask_r_"

# combine bit list with & operator
# equivalent to bits = sum(bit_list)
# if elements in bit_list are of type 2^n
bits = 0
for b in bit_list:
    bits = bits | b
print(bits)

obj._params["input_path"] = f"{base}_{year}_{ver_maj}.c.hdf5"
obj._params["output_path"] = f"{base}_struc_{year}_{ver_maj}.c.hdf5"

obj._params["nside"] = 131072
obj._params["bits"] = bits

# Coverage map is in the mask base directory, not in the mask release one
obj._params["aux_mask_files"] = f"{mask_base}/coverage_{ver_maj}.x.hsp"
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
    print(
        f"Current (peak) memory usage: {current / 1024**2:.2f} ({peak / 1024**2:.2f}) MB"
    )

# Read catalogue
dat = obj.read_cat(load_into_memory=False, mode="r")

if trace_mem:
    current, peak = tracemalloc.get_traced_memory()
    print(
        f"Current (peak) memory usage: {current / 1024**2:.2f} ({peak / 1024**2:.2f}) MB"
    )

# Get bit-coded masks
masks = obj.get_masks(dat=dat)

# Add mask bits as new columns
dat_new = obj.append_masks(dat, masks)

if trace_mem:
    current, peak = tracemalloc.get_traced_memory()
    print(
        f"Current (peak) memory usage: {current / 1024**2:.2f} ({peak / 1024**2:.2f}) MB"
    )

# Write extended data to new HDF5 file
obj.write_hdf5_file(dat, dat_new)

# Close input HDF5 catalogue file
obj.close_hd5()
