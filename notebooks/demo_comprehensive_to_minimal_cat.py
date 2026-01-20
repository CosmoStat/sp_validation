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

# # Comprehensive to minimal catalogue
#
# Transform an input comprehensive catalogue to an output minimal catalogue.

# %reload_ext autoreload
# %autoreload 2

# +
import sys
import os
import re

import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt
# -

from sp_validation import run_joint_cat as sp_joint
from sp_validation import util
from sp_validation.basic import metacal
from sp_validation import calibration
import sp_validation.cat as cat

# Initialize calibration class instance
obj = sp_joint.CalibrateCat()

config = obj.read_config_set_params("config_mask.P37.yaml")

# !pwd

# Get data. Set load_into_memory to False for very large files
dat, dat_ext = obj.read_cat(load_into_memory=False)

if False:
    n_max = 1_000_000
    print(f"MKDEBUG testing only first {n_max} objects")
    dat = dat[:n_max]
    dat_ext = dat_ext[:n_max]

# ## Masking

# +
# List of masks to apply
masks_to_apply = [
    "overlap",
    "IMAFLAGS_ISO",
    "NGMIX_MOM_FAIL",
    "NGMIX_ELL_PSFo_NOSHEAR_0",
    "NGMIX_ELL_PSFo_NOSHEAR_1",
    "8_Manual",
]

# List of masks not to apply and not to copy to minimal catalogue
masks_not_to_include = [
]
# -

# ### Pre-processing ShapePipe flags

masks, labels = sp_joint.get_masks_from_config(
    config,
    dat,
    dat_ext,
    masks_to_apply=masks_to_apply,
    verbose=obj._params["verbose"],
)

mask_combined = sp_joint.Mask.from_list(
    masks,
    label="combined",
    verbose=obj._params["verbose"],
)

if obj._params["sky_regions"]:

    # MKDBEUG TODO: zooms as list in config
    zoom_ra = [200, 205]
    zoom_dec = [55, 60]

    sp_joint.sky_plots(dat, masks, labels, zoom_ra, zoom_dec)

# +
# Output some mask statistics

num_obj = dat.shape[0]

with open("masks.txt", "w") as f_out:

    for my_mask in masks:
        my_mask.print_summary(f_out)
        
    sp_joint.Mask.print_strings(
        "flag", "label", f"{'num_ok':>10}", f"{'num_ok[%]':>10}",
        f_out=f_out,
    )
    print(file=f_out)

    for my_mask in masks:
        my_mask.print_stats(num_obj, f_out=f_out)

    mask_combined.print_stats(num_obj, f_out=f_out)


# -

# Get some memory back
for mask in masks:
    del mask


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
# Remove mask columns that were applied earlier

# Columns to keep
names_to_keep = [name for name in dat.dtype.names + dat_ext.dtype.names if name not in masks_not_to_include]

# Remove metadata from the dtype (in particular, encoding for TILE_ID)
clean_dtype_descr = strip_h5py_metadata_dtype(dat.dtype, dat_ext.dtype)

new_dtype = [
    (name, dt) for name, dt in
    clean_dtype_descr
    if name in names_to_keep
]

# Create a new structured array
new_dat = np.zeros(len(dat[mask_combined._mask]), dtype=new_dtype
)

# Copy relevant columns and lines from each source array
for name in names_to_keep:
    if name in dat.dtype.names:
        new_dat[name] = dat[name][mask_combined._mask]
    else:
        new_dat[name] = dat_ext[name][mask_combined._mask]

# +
# Add information to FITS header

# Generate new header
header = fits.Header()

# Add general config information to FITS header
obj.add_params_to_FITS_header(header)

# Create mask description metainfo and add to FITS header
for my_mask in masks:
    my_mask.add_summary_to_FITS_header(header)

# +
# Write extended data to new HDF5 file

# Create ApplyHspMask instance to write the new file
obj_appl = sp_joint.ApplyHspMasks()

# Replace "c" (comprehensive) with "m" (minimal)
output_path = re.sub(r'1\.[A-Za-z0-9]\.c', '1.X.m', obj._params["input_path"])
obj_appl._params["output_path"] = output_path
obj_appl._params["aux_mask_file_liast"] = []

print("Saving file to", output_path)

# Adding masks for metainfo description
obj_appl.write_hdf5_file(new_dat, masks=masks)
print("Done")
# -

from scipy import stats


def correlation_matrix(masks, confidence_level=0.9):

    n_key = len(masks)
    print(n_key)

    cm = np.empty((n_key, n_key))
    r_val = np.zeros_like(cm)
    r_cl = np.empty((n_key, n_key, 2))

    for idx, mask_idx in enumerate(masks):
        for jdx, mask_jdx in enumerate(masks):
            res = stats.pearsonr(mask_idx._mask, mask_jdx._mask)
            r_val[idx][jdx] = res.statistic
            r_cl[idx][jdx] = res.confidence_interval(
                confidence_level=confidence_level
            )

    return r_val, r_cl


#

all_masks = masks[:-3]

# +
if not obj._params["cmatrices"]:
    print("Skipping cmatric calculations")
    sys.exit(0)

r_val, r_cl = correlation_matrix(all_masks)

# +

n = len(all_masks)
keys = [my_mask._label for my_mask in all_masks]

plt.imshow(r_val, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(np.arange(n), keys)
plt.yticks(np.arange(n), keys)
plt.xticks(rotation=90)
plt.colorbar()
plt.savefig("correlation_matrix.png")

# -


def confusion_matrix(prediction, observation):

    result = {}

    pred_pos = sum(prediction)
    result["true_pos"] = sum(prediction & observation)
    result["true_neg"] = sum(
        np.logical_not(prediction) & np.logical_not(observation)
    )
    result["false_neg"] = sum(prediction & np.logical_not(observation))
    result["false_pos"] = sum(np.logical_not(prediction) & observation)
    result["false_pos_rate"] = result["false_pos"] / (
        result["false_pos"] + result["true_neg"]
    )
    result["false_neg_rate"] = result["false_neg"] / (
        result["false_neg"] + result["true_pos"]
    )
    result["sensitivity"] = result["true_pos"] / (
        result["true_pos"] + result["false_neg"]
    )
    result["specificity"] = result["true_neg"] / (
        result["true_neg"] + result["false_pos"]
    )

    cm = np.zeros((2, 2))
    cm[0][0] = result["true_pos"]
    cm[1][1] = result["true_neg"]
    cm[0][1] = result["false_neg"]
    cm[1][0] = result["false_pos"]
    cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    result["cmn"] = cmn

    return result


n_key = len(all_masks)
cms = np.zeros((n_key, n_key, 2, 2))
for idx in range(n_key):
    for jdx in range(n_key):

        if idx == jdx:
            continue

        print(idx, jdx)
        res = confusion_matrix(masks[idx]._mask, masks[jdx]._mask)
        cms[idx][jdx] = res["cmn"]

# +
import seaborn as sns

fig = plt.figure(figsize=(30, 30))
axes = np.empty((n_key, n_key), dtype=object)
for idx in range(n_key):
    for jdx in range(n_key):
        if idx == jdx:
            continue
        axes[idx][jdx] = plt.subplot2grid((n_key, n_key), (idx, jdx), fig=fig)

matrix_elements = ["True", "False"]

for idx in range(n_key):
    for jdx in range(n_key):

        if idx == jdx:
            continue

        ax = axes[idx, jdx]
        sns.heatmap(
            cms[idx][jdx],
            annot=True,
            fmt=".2f",
            xticklabels=matrix_elements,
            yticklabels=matrix_elements,
            ax=ax,
        )
        ax.set_ylabel(masks[idx]._label)
        ax.set_xlabel(masks[jdx]._label)

plt.show(block=False)
plt.savefig("confusion_matrix.png")
# -

obj.close_hd5()
