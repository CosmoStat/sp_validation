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

# # Demo notebook to create footprint mask

# %reload_ext autoreload
# %autoreload 2

# +
import os
import numpy as np
import healsparse as hsp

from sp_validation import run_joint_cat as sp_joint
# -

# Create instance of ApplyHspMasks object
obj = sp_joint.ApplyHspMasks()

# +
# Set parameters
obj._params["output_path"] = "unions_shapepipe_comprehensive_struc_2024_v1.4.c.hdf5"
obj._params["mask_dir"] = f"{os.environ['HOME']}/masks"
obj._params["nside"] = 131072
obj._params["file_base"] = "mask_r_"

obj._params["aux_mask_files"] = f"{obj._params['mask_dir']}/coverage.hsp"
obj._params["aux_mask_labels"] = "npoint3"
obj._params["verbose"] = True
# -


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

# Asseumble all mask paths
for label, path in zip(auxiliary_labels, auxiliary_masks):
    paths[label] = path
if obj._params["verbose"]:
    print("Using masks", paths)


def hsp_map_logicaL_and(maps):
    """
    Logical AND of HealSparseMaps.
    """
    # Ensure consistency in coverage and data type
    nside_coverage = maps[0].nside_coverage
    nside_sparse = maps[0].nside_sparse
    dtype = maps[0].dtype

    for m in maps:
        if m.nside_coverage != nside_coverage:
            raise ValueError(
                f"Coverage nside={m.nside_coverage} does not match {nside_coverage}"
            )
        if m.dtype != dtype:
            raise ValueError(
                f"Data type {m.dtype} does not match {dtype}"
            )

    # Find intersection of valid pixels across all maps
    common_pixels = set(maps[0].valid_pixels)
    for m in maps[1:]:
        common_pixels.intersection_update(m.valid_pixels)

    # Create an empty HealSparse map
    map_and = hsp.HealSparseMap.make_empty(nside_coverage, nside_sparse, dtype=dtype)

    # Apply logical AND operation over common pixels
    for pix in common_pixels:
        map_and[pix] = np.bitwise_and.reduce([m[pix] for m in maps])

    return map_and

hsp_maps = []
for label in paths:
    if obj._params["verbose"]:
        print(f"Reading mask {paths[label]} for label {label}...")
        hsp_maps.append(hsp.HealSparseMap.read(paths[label]))
if obj._params["verbose"]:
    print("Combine all maps with logical and...")
map_and = hsp_map_logicaL_and(hsp_maps)

if obj._params["verbose"]:
    print("Writing combined mask file...")    
map_and.write("mask_combined.hsp")



# +
# Check parameter validity
obj.check_params()

# Update parameters (here: strings to list)
obj.update_params()
# -
