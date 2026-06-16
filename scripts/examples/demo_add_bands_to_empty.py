# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tqdm
from astropy.io import fits

from sp_validation import run_joint_cat as sp_joint

# -

# Create instance of object
obj = sp_joint.BaseCat()


# +
# Set parameters
base = "unions_shapepipe_comprehensive_struc_empty_ugriz"
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


bands = ("u", "g", "r", "i", "z", "z2")
base_keys = ["MAGERR_GAAP", "FLUX_GAAP", "FLUXERR_GAAP", "FLAG_GAAP", "MAG_LIM"]
keys_mag = [f"MAG_GAAP_0p7_{band}" for band in bands]
for base_key in base_keys:
    keys_mag.extend([f"_{base_key}_{band}" for band in bands])

keys = [
    "EXTINCTION",
    "MP_NAME",
    "Z_B",
    "Z_B_MIN",
    "Z_B_MAX"
    "T_B",
    "ODDS",
] + keys_mag

hdu_no = 1

do_missing_check = False

parallel = False
n_cpu = 48
# -

# ## Run

# +
# Check parameter validity
# obj.check_params()

# Update parameters (here: strings to list)
# obj.update_params()
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

if do_missing_check:
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
                path = os.path.join(
                    path_bands, f"{path_base}{tile_ID}{path_suff}"
                )
                print(f"{path_base}{tile_ID}{path_suff}", file=f)


# Build TILE_ID -> list of indices mapping once
tile_to_indices = defaultdict(list)
for i, tile_id in enumerate(dat["TILE_ID"]):
    tile_to_indices[tile_id].append(i)

# Optionally convert lists to arrays
for k in tile_to_indices:
    tile_to_indices[k] = np.array(tile_to_indices[k])


def fill_one_tile(path, keys):

    try:
        with fits.open(path, memmap=True) as hdu_list:
            dat_mb = hdu_list[hdu_no].data
            result = {key: dat_mb[key] for key in keys}

            return result

    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


if parallel:

    # Parallel processing
    with ProcessPoolExecutor(max_workers=n_cpu) as executor:
        futures = {}
        for idx, tile_ID in enumerate(tile_IDs):

            path = os.path.join(path_bands, f"{path_base}{tile_ID}{path_suff}")
            if not os.path.exists(path):
                continue

            tile_key = tile_IDs_raw_list[idx]
            indices = tile_to_indices.get(tile_key, None)
            if indices is None:
                continue

            futures[
                executor.submit(
                    fill_one_tile,
                    path,
                    keys,
                )
            ] = (tile_key, indices)

        # As results complete, update dat
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            tile_key, indices = futures[future]
            result = future.result()
            if result is None:
                continue

            for key in keys:
                dat[indices][key] = result[key]

else:

    # Loop over tile IDs
    for idx, tile_ID in tqdm.tqdm(
        enumerate(tile_IDs), total=len(tile_IDs), disable=False
    ):

        path = os.path.join(path_bands, f"{path_base}{tile_ID}{path_suff}")
        if not os.path.exists(path):
            continue

        indices = tile_to_indices.get(tile_IDs_raw_list[idx], None)
        if indices is None:
            continue

        result = fill_one_tile(path, keys)
        for key in keys:
            dat[indices][key] = result[key]

obj.write_hdf5_file(dat)

# Close input HDF5 catalogue file
obj.close_hd5()
