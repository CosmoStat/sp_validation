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

# # Calibrate comprehensive catalogue

# %reload_ext autoreload
# %autoreload 2

import sys
import os
import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt

from sp_validation import run_joint_cat as sp_joint
from sp_validation import util
from sp_validation.basic import metacal
from sp_validation import calibration
import sp_validation.cat as cat

obj = sp_joint.CalibrateCat()

obj._params["input_path"] = "unions_shapepipe_comprehensive_2024_v1.4.2.hdf5"
obj._params["cmatrices"] = True
obj._params["verbose"] = True

# !pwd

dat = obj.read_cat(load_into_memory=False)

print(f"Found {len(dat)} (~{util.millify(len(dat))}) objects in catalogue")

# ## Masking

# ## Pre-processing ShapePipe flags

cut_pre = {}
labels = {}

# +
# SExtractor flags (see galaxy.py:classification_galaxy_base)

name = "FLAGS"
good_mask_value = 0

# MKDBEUG TODO: implement values other than 0 as "good"

cut_pre[name] = (dat[name] == good_mask_value)
labels[name] = "SE FLAGS"

# +
# Duplicate objects

name = "overlap"
good_mask_value = True
cut_pre[name] = (dat[name] == good_mask_value)
labels[name] = "tile overlap"
# -

# ShapePipe mask
name = "IMAFLAGS_ISO"
good_mask_value = 0
cut_pre[name] = (dat[name] == good_mask_value)
labels[name] = "SP mask"

# +
# Number of epochs
name = "N_EPOCH"
val_min = 2
cut_pre[name] = (dat[name] >= val_min)

# MKDEBUG check NGMIX_N_EPOCH
labels[name] = r"$n_{\rm epoch}$"
# -

# Magnitude range
name = "mag"
min_max = [15, 30]
cut_pre[name] = (
    (dat[name] >= min_max[0])
    & (dat[name] <= min_max[1])
)
labels[name] = "mag range"

# +
# ngmix flags
names = ["NGMIX_MCAL_FLAGS", "NGMIX_MOM_FAIL"]
good_mask_values = [0, 0]
for name, good_mask_value in zip(names, good_mask_values):
    cut_pre[name] = (dat[name] == good_mask_value) 
labels[names[0]] = "ngmix flag"
labels[names[1]] = "ngmix moments fail"
    
name = "NGMIX_ELL_PSFo_NOSHEAR_0"
bad_mask_value = -10
cut_pre[name] = (
    dat[name] != bad_mask_value
)
# MKDEBUG TODO: check should be two components, see galaxy.py.
labels[name] = "bad PSF ell"

# +
# Initialise combined mask
cut_pre_combined = np.ones_like(cut_pre["FLAGS"], dtype=bool)

# Combine all masks with &
cut_pre_combined = np.logical_and.reduce(list(cut_pre.values()))

# +
# Output some mask statistics

num_obj = dat.shape[0]

print(f"{'flag':30s} {'label':30s} {'n_ok':>10} {'n_ok[%]':>10}")
for name in cut_pre:
    num_ok = sum(cut_pre[name])
    print(f"{name:30s} {labels[name]:30s} {num_ok:10d} {num_ok/num_obj:10.2%}")
name = "combined"
num_ok = sum(cut_pre_combined)
print(f"{name:30s} {num_ok:10d} {num_ok/num_obj:10.2%}")

# +
# Number of "galaxies" (cut_common in main_set_up)

cut_common = cut_pre["overlap"] & cut_pre["FLAGS"] & cut_pre["mag"] & cut_pre["IMAFLAGS_ISO"] & cut_pre["N_EPOCH"]
name = "common"
num_ok = sum(cut_common)
print(f"{name:30s} {num_ok:10d} {num_ok/num_obj:10.2%}")

cut_galaxy = cut_common & cut_pre["NGMIX_MCAL_FLAGS"] & cut_pre["NGMIX_ELL_PSFo_NOSHEAR_0"] & cut_pre["NGMIX_MOM_FAIL"]
name = "galaxy"
num_ok = sum(cut_galaxy)
print(f"{name:30s} {num_ok:10d} {num_ok/num_obj:10.2%}")
# -

help(hsp.get_values)

# +
# Initialise combined mask
cut_struct_combined = np.ones_like(cut_pre["FLAGS"], dtype=bool)

# Combine all masks with &
cut_struct_combined = np.logical_and.reduce(list(cut_struct.values()))

# +
print(f"{'flag':30s} {'label':30s} {'n_ok':>10} {'n_ok[%]':>10}")
for name in cut_struct:
    num_ok = sum(cut_struct[name])
    print(f"{name:30s} {name:30s} {num_ok:10d} {num_ok/num_obj:10.2%}")

name = "combined"
num_ok = sum(cut_struct_combined)
print(f"{name:30s} {num_ok:10d} {num_ok/num_obj:10.2%}")

# -

#

#

cut_combined = cut_pre_combined & cutc_struct_combined

# ### Calibration

# +
# Define cuts and metacal input parameters

# Ellipticity dispersion
sigma_eps_prior = 0.34

# Signal-to-noise range
gal_snr_min = 10
gal_snr_max = 500

# Relative-size (hlr / hlr_psf) range
gal_rel_size_min = 0.5
gal_rel_size_max = 3

# Correct relative size for ellipticity?
gal_size_corr_ell = False

# +
# Call metacal

gal_metacal = metacal(
    dat,
    cut_combined,
    snr_min=gal_snr_min,
    snr_max=gal_snr_max, 
    rel_size_min=gal_rel_size_min,
    rel_size_max=gal_rel_size_max,
    size_corr_ell=gal_size_corr_ell,
    sigma_eps=sigma_eps_prior,
    col_2d=False,
    verbose=True,
)

# +
# Get calibrated quantities

g_corr, g_uncorr, w, mask = calibration.get_calibrated_quantities(gal_metacal)

# Additive bias
c = np.zeros(2)
c_err = np.zeros(2)


for comp in (0, 1):
    c[comp] = np.mean(g_uncorr[comp])
    
    # MKDEBUG TODO: Use std of mean instead, which is consistent with jackknife
    c_err[comp] = np.std(g_uncorr[comp])

# Shear estimate corrected for additive bias
g_corr_mc = np.zeros_like(g_corr)
c_corr = np.linalg.inv(gal_metacal.R).dot(c)
for comp in (0, 1):
    g_corr_mc[comp] = g_corr[comp] - c_corr[comp]
# -

name = "after cuts"
num_ok = len(mask)
print(f"{name:30s} {num_ok:10d} {num_ok/num_obj:10.2%}")

# +
# Additional quantities
R_shear = np.mean(gal_metacal.R_shear, 2)

ra = cat.get_col(dat, "RA", cut_combined,  mask)
dec = cat.get_col(dat, "Dec", cut_combined,  mask)
mag = cat.get_col(dat, "mag", cut_combined,  mask)
#snr = cat.get_snr(dat, cut_combined,  mask)

add_cols = ["FLUX_RADIUS", "FWHM_IMAGE", "FWHM_WORLD", "MAGERR_AUTO", "MAG_WIN", "MAGERR_WIN", "FLUX_AUTO", "FLUXERR_AUTO", "FLUX_APER", "FLUXERR_APER"]
add_cols_data = {}    
for key in add_cols:
    add_cols_data[key] = dat[key][cut_combined][mask]

# +
output_shape_cat_path = obj._params["input_path"].replace("comprehensive", "cut")
output_shape_cat_path = output_shape_cat_path.replace("hdf5", "fits")

cat.write_shape_catalog(
    output_shape_cat_path,
    ra,
    dec,
    w,
    mag=mag,
    g=g_corr_mc,
    g1_uncal=g_uncorr[0],
    g2_uncal=g_uncorr[1],
    R=gal_metacal.R,
    R_shear=R_shear,
    R_select=gal_metacal.R_selection,
    c=c,
    c_err=c_err,
    add_cols=add_cols_data
)

# +


# Correct for PSF leakage

# Compute DES weights
# -

from scipy import stats
def correlation_matrix(mask, confidence_level=0.9):
    
    n_key = len(mask)
    print(n_key)

    cm = np.empty((n_key, n_key))
    r_val = np.zeros_like(cm)
    r_cl = np.empty((n_key, n_key, 2))

    for idx, key1 in enumerate(mask):
        for jdx, key2 in enumerate(mask):
            res = stats.pearsonr(mask[key1], mask[key2])
            r_val[idx][jdx] = res.statistic
            r_cl[idx][jdx] = res.confidence_interval(confidence_level=confidence_level)
            
    return r_val, r_cl


# +
if not obj._params["cmatrices"]:
    print("Skipping cmatric calculations")
    sys.exit(0)

r_val, r_cl = correlation_matrix(cut_pre)

# +

n = len(cut_pre)
keys = [key for key in cut_pre]
labs = [labels[key] for key in cut_pre]

plt.imshow(r_val, cmap='coolwarm', vmin=-1, vmax=1)
plt.xticks(np.arange(n), labs)
plt.yticks(np.arange(n), labs)
plt.xticks(rotation=90)
plt.colorbar()

# -

def confusion_matrix(prediction, observation):                                  
                                                                                
    result = {}                                                                 
                                                                                
    pred_pos = sum(prediction)                                                  
    result["true_pos"] = sum(prediction & observation)                  
    result["true_neg"] = sum(np.logical_not(prediction) & np.logical_not(observation))               
    result["false_neg"] = sum(prediction & np.logical_not(observation))        
    result["false_pos"] = sum(np.logical_not(prediction) & observation)            
    result["false_pos_rate"] = result["false_pos"] / (result["false_pos"] + result["true_neg"])
    result["false_neg_rate"] = result["false_neg"] / (result["false_neg"] + result["true_pos"])
    result["sensitivity"] = result["true_pos"] / (result["true_pos"] + result["false_neg"])
    result["specificity"] = result["true_neg"] / (result["true_neg"] + result["false_pos"])

    cm = np.zeros((2, 2))
    cm[0][0] = result["true_pos"]
    cm[1][1] = result["true_neg"]
    cm[0][1] = result["false_neg"]
    cm[1][0] = result["false_pos"]
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    result["cmn"] = cmn
     
    return result


cms[0][0]

n_key = len(keys)
cms = np.zeros((n_key, n_key, 2, 2))
for idx, key_1 in enumerate(keys):
    for jdx, key_2 in enumerate(keys):

        if idx == jdx: continue

        print(idx, jdx, key_1, key_2)
        res = confusion_matrix(cut_pre[key_1], cut_pre[key_2])
        cms[idx][jdx] = res["cmn"]

help(plt.subplot2grid)

# +
import seaborn as sns

fig = plt.figure(figsize=(30, 30))
axes = np.empty((n_key, n_key), dtype=object)
for idx, key_1 in enumerate(keys):
    for jdx, key_2 in enumerate(keys):
        if idx == jdx: continue
        axes[idx][jdx] = plt.subplot2grid((n_key, n_key), (idx, jdx), fig=fig)


#fig, axes = plt.subplots(nrows=n_key, ncols=n_key, figsize=(20,20))
matrix_elements = ["True", "False"]

for idx, key_1 in enumerate(keys):
    for jdx, key_2 in enumerate(keys):

        if idx == jdx: continue
        
        ax = axes[idx, jdx]
        sns.heatmap(cms[idx][jdx], annot=True, fmt='.2f', xticklabels=matrix_elements, yticklabels=matrix_elements, ax=ax)
        ax.set_ylabel(labels[key_1])
        ax.set_xlabel(labels[key_2])

plt.show(block=False)
# -

obj.close_hd5()
