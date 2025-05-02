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

# # Demo notebook to add (u,g,i,z,z2) bands to an r-band catalogue

# %reload_ext autoreload
# %autoreload 2

# +
import os
import numpy as np
import healsparse as hsp
from astropy.io import fits

from sp_validation import run_joint_cat as sp_joint
# -

# Create instance of object
obj = sp_joint.BaseCat()


# +
# Set parameters
base = "unions_shapepipe_comprehensive"
year = 2024
ver = "v1.5.c.P37"

obj._params = {}

obj._params["input_path"] = f"{base}_{year}_{ver}.hdf5"
obj._params["output_path"] = f"{base}_ugriz_{year}_{ver}.hdf5"
obj._params["verbose"] = True

# +
path_bands = "../UNIONS5000"

subdir_base = "UNIONS."
path_base = subdir_base
path_suff = "_SP_ugriz_photoz_ext.cat"

# NUMBER key in photo-z catalogue
key_num = "SeqNr"

keys_mag = [f"MAG_GAAP_0p7_{band}" for band in ("u", "g", "r", "i", "z", "z2")]

keys = ["Z_B", "Z_B_MIN", "Z_B_MAX", "T_B"] + keys_mag

hdu_no = 1
# -

# ## Run

# +
# Check parameter validity
#obj.check_params()

# Update parameters (here: strings to list)
#obj.update_params()
# -

# Read catalogue
dat = obj.read_cat(load_into_memory=False, mode="r")

number = dat["NUMBER"]

tile_ID_raw = dat["TILE_ID"]

# +
tile_IDs_raw_list = list(set(tile_IDs_raw))

# Transform (back) to 2x3 digits by zero-padding
tile_IDs = [f"{float(tile_ID):07.3f}" for tile_ID in tile_IDs_raw_list]
# -

len(tile_IDs_raw_list)

# +

#for tile_ID in tile_IDs:
tile_ID = tile_IDs[0]

if True:
    path = os.path.join(path_bands, f"{path_base}{tile_ID}", f"{path_base}{tile_ID}{path_suff}")
    print(tile_ID, path)
    
    hdu_list = fits.open(path)
    data = hdu_list[hdu_no].data
    
    numbers = data[key_num]
    
    hdu_list.close()
    

# -

print(tile_ID, tile_IDs_raw_list[0])

w = dat["TILE_ID"] == tile_IDs_raw_list[0]

dat[w]["NUMBER"][32900]

dat = fits.getdata("P7/sp_output/shape_catalog_comprehensive_ngmix.fits", 1)

np.min(dat["NUMBER"])

# Write extended data to new HDF5 file
obj.write_hdf5_file(dat, dat_new)

# Close input HDF5 catalogue file
obj.close_hd5()

if trace_mem:
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current (peak) memory usage: {current / 1024**2:.2f} ({peak / 1024**2:.2f}) MB")
    tracemalloc.stop()


