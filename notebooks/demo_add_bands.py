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
import numpy.lib.recfunctions as rfn

from timeit import default_timer as timer
import tqdm
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

# +
# Get Tile IDs
tile_IDs_raw = dat["TILE_ID"]
tile_IDs_raw_list = list(set(tile_IDs_raw))

# Transform (back) to 2x3 digits by zero-padding
tile_IDs = [f"{float(tile_ID):07.3f}" for tile_ID in tile_IDs_raw_list]

# +
dist_sqr = {}
n_rows = len(dat)

# Loop over tile IDs
for idx, tile_ID in tqdm.tqdm(enumerate(tile_IDs), total=len(tile_IDs), disable=True):

    print(idx/len(tile_ID), tile_ID)
    
    path = os.path.join(path_bands, f"{path_base}{tile_ID}", f"{path_base}{tile_ID}{path_suff}")
    
    print("  Read data from file:", path, end=" ")
    start = timer()
    hdu_list = fits.open(path)
    dat_mb = hdu_list[hdu_no].data
    end = timer()                                                           
    print(f" {end - start:.1f}s") 
    
    print("  Get numbers", end=" ")
    start = timer()
    numbers = dat_mb[key_num]
    end = timer()                                                           
    print(f" {end - start:.1f}s") 
    
    print("  Identify matches", end= " ")
    start = timer()
    # Select indices in dat with current tile ID
    w = dat["TILE_ID"] == tile_IDs_raw_list[idx]
    indices = np.where(w)[0]
    end = timer()                                                           
    print(f" {end - start:.1f}s") 
    
    print("  Compute distance check", end=" ")
    start = timer()
    # Compute coordinate distances as matching check    
    dist_sqr["TILE_ID"] = sum(
        (dat[indices]["RA"] - dat_mb["ALPHA_J2000"]) ** 2
        + (dat[indices]["Dec"] - dat_mb["DELTA_J2000"]) ** 2
    ) / len(dat_mb)
    end = timer()                                                           
    print(f" {end - start:.1f}s") 
    
    if idx == 0:
        print("  Create new combined array", end=" ")
        start = timer()    
        # Get dtype from multiband keys
        dtype_keys = np.dtype([dt for dt in dat_mb.dtype.descr if dt[0] in keys])

        # Create structured array from multi-band columns
        
        # Create new empty array with rows as original data and new multi-band columns
        new_empty = np.zeros(n_rows, dtype=dtype_keys)
        end = timer()                                                           
        print(f" {end - start:.1f}s") 
    
        print("    Merge empty to original", end=" ")
        start = timer()
        # Combine with original data (slow)
        combined = rfn.merge_arrays([dat, new_empty], flatten=True)
        end = timer()                                                           
        print(f" {end - start:.1f}s") 

    print(  "  Copy mb data to combined array", end=" ")
    start = timer()
    # Copy multi-band values to combined array
    for key in keys:
        combined[indices][key] = dat_mb[key]
    end = timer()                                                           
    print(f" {end - start:.1f}s") 

    hdu_list.close()

# -

obj.write_hdf5_file(combined)

# Close input HDF5 catalogue file
obj.close_hd5()
