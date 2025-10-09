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
#     name: sp_validation
# ---

# # Demo notebook to add (u,g,i,z,z2) bands to an r-band catalogue

# %reload_ext autoreload
# %autoreload 2

# +
import os
import numpy as np
import numpy.lib.recfunctions as rfn
import h5py

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
base = "unions_shapepipe_comprehensive_struc"
year = 2024
ver = "v1.5.c"

obj._params = {}

obj._params["input_path"] = f"{base}_{year}_{ver}.hdf5"
obj._params["verbose"] = True

# +
path_bands = "./UNIONS5000"
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
n_rows = len(dat)


def get_dtype_keys(keys,path=None, hdu_no=1):
    
    if path is None:

        dtype = np.dtype([(key, np.float32) for key in keys])        
    
    else:
        
        print("  Read data from file:", path, end=" ")
        start = timer()
        hdu_list = fits.open(path)
        dat_mb = hdu_list[hdu_no].data 
        dtype = np.dtype([dt for dt in dat_mb.dtype.descr if dt[0] in keys])
        end = timer()                                                           
        print(f" {end - start:.1f}s") 

    return dtype


# +
# Get dtype of new keys

path = None
#path = os.path.join(path_bands, f"{path_base}{tile_ID}", f"{path_base}{tile_IDs[0]}{path_suff}")

dtype_keys = get_dtype_keys(keys, path=path, hdu_no=hdu_no)


# -

def strip_h5py_metadata_dtype(dat_dtype, dat_ext_dtype):                         
    cleaned_fields = []                                                          
    for name, dt in dat_dtype.descr + dat_ext_dtype.descr:                       
        # If dt is a tuple (e.g., ('S7', {'h5py_encoding': 'ascii'}))            
        if isinstance(dt, tuple):                                                
            cleaned_fields.append((name, dt[0]))  # keep only the base dtype string
        else:                                                                    
            cleaned_fields.append((name, dt))     # use as-is                    
    return cleaned_fields


# +
# Create empty array with new keys
# Initialise with -199 to later be able to check for unfilled values

total_bytes = n_rows * np.dtype(dtype_keys).itemsize
print("  Create new combined array.", end=" ")
print(f"Expected size = {total_bytes / 1_048_576:.2f} MB", end=" ")
start = timer()

obj._params["output_path"] = f"{base}_empty_ugriz_{year}_{ver}.hdf5"
dtype_sp = dat.dtype
dtype_comb = strip_h5py_metadata_dtype(dtype_sp, dtype_keys)
with h5py.File(obj._params["output_path"], "w") as f:

    # Create new dataset
    dset_comb = f.create_dataset(
        "dat_comb",
        shape=(n_rows,),
        dtype=dtype_comb,
    )

    # Copy old data field-by-field
    for name in dtype_sp.names:
        dset_comb[name] = dat[name]

    # Fill new fields with default value (-199)
    for name in dtype_keys.names:
        dset_comb[name] = -199

#new_empty = np.full(n_rows, -199, dtype=dtype_keys)

end = timer()                                                           
print(f" {end - start:.1f}s") 



# +
# Merge with original data

#print("    Merge empty to original", end=" ")
#start = timer()
#combined = rfn.merge_arrays([dat, new_empty], flatten=True)
#end = timer()                                                           
#print(f" {end - start:.1f}s") 
# -

# obj.write_hdf5_file(combined)


# Close input HDF5 catalogue file
# obj.close_hd5()
