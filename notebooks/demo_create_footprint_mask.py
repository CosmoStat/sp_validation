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

# # Demo notebook to create footprint mask

# %reload_ext autoreload
# %autoreload 2

# +
import os
import numpy as np
import healpy as hp


import matplotlib
matplotlib.use("agg")

import matplotlib.pylab as plt
import healsparse as hsp

from sp_validation import run_joint_cat as sp_joint
from sp_validation import cosmo_val
# -


# Create instance of ApplyHspMasks object
obj = sp_joint.ApplyHspMasks()

# Set parameters
obj._params["mask_dir"] = f"{os.environ['HOME']}/masks"
obj._params["aux_mask_files"] = f"{obj._params['mask_dir']}/coverage.hsp"
obj._params["aux_mask_labels"] = "npoint3"
obj._params["verbose"] = True


# Read configuration file and set parameters
config = obj.read_config_set_params("config_mask.yaml")

# Get data sections
config_data = {key: config[key] for key in ["dat", "dat_ext"] if key in config}

# Get names of possible mask names
all_masks_bits = {}
for bit in obj._labels_struct:
    all_masks_bits[obj.get_mask_col_name(bit)] = bit

# Identify masks specified in the config file
bits = 0
auxiliary_masks = []
auxiliary_labels = []
for section, mask_list in config_data.items():
    for mask_params in mask_list:
        
        # Check bit-coded masks
        if mask_params["col_name"] in all_masks_bits:
            bits = bits | all_masks_bits[mask_params["col_name"]]
            
        # Check auxillary masks
        if "npoint3" == mask_params["col_name"]:
            auxiliary_masks.append(f"{obj._params['mask_dir']}/coverage.hsp")
            auxiliary_labels.append("npoint3")

# Update bits for following function call
obj._params["bits"] = bits

# Get bit-coded mask file paths
paths = obj.get_paths_bit_masks()

# Assemble all mask paths
for label, path in zip(auxiliary_labels, auxiliary_masks):
    paths[label] = path
if obj._params["verbose"]:
    print("Using masks", paths)

hsp_maps = []
for label in paths:
    if obj._params["verbose"]:
        print(f"Reading mask {paths[label]} for label {label}...")
    hsp_maps.append(hsp.HealSparseMap.read(paths[label]))

map_comb = cosmo_val.hsp_map_logical_or(hsp_maps, verbose=obj._params["verbose"])

# +
fp = cosmo_val.FootprintPlotter()
nside = 512

for idx, label in enumerate(paths):
    fp.plot_region(hsp_maps[idx], fp._regions["fullsky"], outpath=f"mask_{label}.png", title=label)
    fp.plot_footprint_as_hp(hsp_maps[idx], nside, outpath=f"footprint_{label}.png", title=label)

label = "combined"
fp.plot_region(map_comb, fp._regions["fullsky"], outpath=f"mask_{label}.png", title=label)
fp.plot_footprint_as_hp(map_comb, nside, outpath=f"footprint_{label}.png", title=label)
# -

if obj._params["verbose"]:
    print("Deleting no-longer used maps...")
for map in hsp_maps:
    del map

if obj._params["verbose"]:
    print("Writing combined mask file...")    
map_comb.write("mask_combined.hsp", clobber=True)


# +
# Check parameter validity
obj.check_params()

# Update parameters (here: strings to list)
obj.update_params()
