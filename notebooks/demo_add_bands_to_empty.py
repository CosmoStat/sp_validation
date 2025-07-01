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

# # Demo notebook to add multi-band (u,g,i,z,z2) and photo-z data to r-band + empty multi-band catalogue.
#
# (Fill empty columns.)

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
base = "unions_shapepipe_comprehensive_empty_ugriz"
year = 2024
ver = "v1.5.c"

obj._params = {}

obj._params["input_path"] = f"{base}_{year}_{ver}.hdf5"
obj._params["output_path"] = obj._params["input_path"].replace("_empty", "")
obj._params["verbose"] = True

# +
path_bands = "./UNIONS5000"

path_base = "UNIONS."
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
dat = obj.read_cat(load_into_memory=False, mode="r", name="dat_comb")

# +
# Get tile IDs
tile_IDs_raw = dat["TILE_ID"]
tile_IDs_raw_list = list(set(tile_IDs_raw))

# Transform (back) to 2x3 digits by zero-padding
tile_IDs = [f"{float(tile_ID):07.3f}" for tile_ID in tile_IDs_raw_list]
# -

missing_IDs = []
print("Check for missing mb catalogue files...")
for tile_ID in tqdm.tqdm(tile_IDs, total=len(tile_IDs)):
    path = os.path.join(path_bands, f"{path_base}{tile_ID}{path_suff}")
    if not os.path.exists(path):
        missing_IDs.append(tile_ID)

n_missing = len(missing_IDs)
print(f"{n_missing} missing multi-band catalogue files, saving to file")
if n_missing > 0:
    with open("missing_IDs.txt", "w") as f:
        for tile_ID in missing_IDs:
            path = os.path.join(path_bands, f"{path_base}{tile_ID}{path_suff}")
            print(f"{path_base}{tile_ID}{path_suff}", file=f)

# +
dist_sqr = {}
do_dist_check = False

# Loop over tile IDs
for idx, tile_ID in tqdm.tqdm(enumerate(tile_IDs), total=len(tile_IDs), disable=False):

    path = os.path.join(path_bands, f"{path_base}{tile_ID}{path_suff}")

    if not os.path.exists(path):
        continue
 
    hdu_list = fits.open(path)
    dat_mb = hdu_list[hdu_no].data
    
    numbers = dat_mb[key_num]
    
    # Select indices in dat with current tile ID
    w = dat["TILE_ID"] == tile_IDs_raw_list[idx]
    indices = np.where(w)[0]
    
    # Compute coordinate distances as matching check
    if do_dist_check:
        dist_sqr[TILE_ID] = sum(
            (dat[indices]["RA"] - dat_mb["ALPHA_J2000"]) ** 2
            + (dat[indices]["Dec"] - dat_mb["DELTA_J2000"]) ** 2
        ) / len(dat_mb)

    # Copy multi-band values to combined array
    for key in keys:
        dat[indices][key] = dat_mb[key]

    hdu_list.close()
# -


obj.write_hdf5_file(dat)

# Close input HDF5 catalogue file
obj.close_hd5()
