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

# +
labels_struct = {
    "1" : "Faint_star_halos",
    "2" : "Bright_star_halos",
    "4" : "Stars",
    "8" : "Manual",
    "16" : "u",
    "32" : "g",
    "64" : "r",
    "128" : "i",
    "256" : "z",
    "512"	: "Tile_RA_DEC_cut",
    "1024" : "Maximask"
}


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
# -

obj._params["input_path"] = "unions_shapepipe_comprehensive_2024_v1.4.2.hdf5"
obj._params["output_path"] = "unions_shapepipe_comprehensive_struc_2024_v1.4.2.hdf5"
obj._params["mask_dir"] = f"{os.environ['HOME']}/v1.4.x/masks"
obj._params["nside"] = 131072
obj._params["file_base"] = "mask_r_"
obj._params["bits"] = bits
obj._params["verbose"] = True


# +
def read_hsp_mask(self, path):
    """Read Hsp Mask.
    
    Parameters
    ----------
    path : str
        Path to the mask file.
        
    Returns
    -------
    np.ndarray
        Mask array.
    """
    
    if self._params["verbose"]:
        print(f"Reading healsparse mask file {path}...")
    return hsp.HealSparseMap.read(path)

def reverse_bit_list(bit):
    bit_list = []
    while bit:
        lowest_bit = bit & -bit  # Extract lowest set bit
        bit_list.append(lowest_bit)
        bit -= lowest_bit  # Remove this bit from bit
    return bit_list

def get_paths(self):
    
    paths = {}
    bit_list = reverse_bit_list(self._params["bits"])
    for bit in bit_list:
        paths[bit] = (
            f"{self._params['mask_dir']}/{self._params['file_base']}"
            + f"nside{self._params['nside']}_n{bit}.hsp"
        )
    return paths

def get_masks(self, dat=None):
    
    masks = {}
    
    # Get mask file paths
    #paths = self.get_paths()
    paths = get_paths(self)
    
    # Get coordinates from data
    if dat is None:
        dat = self.read_cat()
    if self._params["verbose"]:
        print("Reading coordinates from data...")
    ra = dat["RA"]
    dec = dat["Dec"]
    
    # Read healsparse files and apply masks to coordinate
    for bit in paths:
        if self._params["verbose"]:
            print(f"Reading mask for bit {bit}...")
        hsp_mask = hsp.HealSparseMap.read(paths[bit])
    
        if obj._params["verbose"]:
            print(f"Computing mask bit={bit}...")
        masks[bit] = ~hsp_mask.get_values_pos(ra, dec, lonlat=True)
        
    return masks


# -

#dat = obj.read_cat(load_into_memory=True, mode="a")
import h5py
f = h5py.File(obj._params["input_path"], mode="a")
dat = f["data"]

masks = get_masks(obj, dat=dat)


def append_masks(self, dat, masks):
    
    n_masks = len(masks)
    new_dtype = np.dtype(dat.dtype.descr + [(str(bit), np.bool_) for bit in masks])
    new_data = np.zeros(dat.shape, dtype=new_dtype)

    # Copy previous columns
    for name in dat.dtype.names:
        new_data[name] = dat[name]
        
    # Copy masks as new columns
    for bit in masks:
        new_data[str(bit)] = masks[bit]

    return new_data


dat_new = append_masks(obj, dat, masks)

# MKDEBUG TODO:
# Use JointCat function; write header
with h5py.File(obj._params["output_path"], "w") as f:
    dset = f.create_dataset("data", data=dat_new)
