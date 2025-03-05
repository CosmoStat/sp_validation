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
import matplotlib.pylab as plt

from sp_validation import run_joint_cat as sp_joint
from sp_validation import util
from sp_validation.basic import metacal
from sp_validation import calibration
import sp_validation.cat as cat

obj = sp_joint.CalibrateCat()

config = obj.read_config_set_params("config_mask.yaml")

# !pwd

dat, dat_ext = obj.read_cat(load_into_memory=False)

print(f"Found {len(dat)} (~{util.millify(len(dat))}) objects in catalogue")

# ## Masking

# ### Pre-processing ShapePipe flags

masks = []

for section, mask_list in config.items():

    dat_source = dat if section == "dat" else dat_ext
    for mask_params in mask_list:
        value = mask_params["value"]
        
        # Ensure 'range' kind has exactly two values
        if mask_params["kind"] == "range" and (not isinstance(value, list) or len(value) != 2):
            raise ValueError(f"Range kind requires a list of two values, got {value}")

        #print(mask_params)
        my_mask = sp_joint.Mask(**mask_params, dat=dat_source, verbose=obj._params["verbose"])
        masks.append(my_mask)

#

print(f"Combining {len(masks)} masks")
mask_combined = sp_joint.Mask.from_list(masks, label="combined")

# +
# Output some mask statistics

num_obj = dat.shape[0]

Mask.print_strings("flag", "label", f"{'num_ok':>10}", f"{'num_ok[%]':>10}")
for my_mask in masks:
    my_mask.print_stats(num_obj)

mask_combined.print_stats(num_obj)
# -

#

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

# For output
mask_metacal = []

col_name = "T_gal/T_PSF"
label = "metacal relaive galaxy_size"
my_mask = sp_joint.Mask(col_name=col_name, label=label, kind="range", value=[gal_rel_size_min, gal_rel_size_max])
masks.append(my_mask)

col_name = "snr"
label = "metacal signal-to-noise ratio"
my_mask = sp_joint.Mask(col_name=col_name, label=label, kind="range", value=[gal_snr_min, gal_snr_max])
masks.append(my_mask)

# +
# Call metacal

gal_metacal = metacal(
    dat,
    mask_combined._mask,
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

g_corr, g_uncorr, w, mask_metacal = calibration.get_calibrated_quantities(gal_metacal)

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
num_ok = len(masks)
print(f"{name:30s} {num_ok:10d} {num_ok/num_obj:10.2%}")

# +
# Additional quantities
R_shear = np.mean(gal_metacal.R_shear, 2)

ra = cat.get_col(dat, "RA", mask_combined._mask,  mask_metacal)
dec = cat.get_col(dat, "Dec", mask_combined._mask,  mask_metacal)
mag = cat.get_col(dat, "mag", mask_combined._mask,  mask_metacal)
snr = cat.get_snr("ngmix", dat, mask_combined._mask,  mask_metacal)

add_cols = ["FLUX_RADIUS", "FWHM_IMAGE", "FWHM_WORLD", "MAGERR_AUTO", "MAG_WIN", "MAGERR_WIN", "FLUX_AUTO", "FLUXERR_AUTO", "FLUX_APER", "FLUXERR_APER"]
add_cols_data = {}    
for key in add_cols:
    add_cols_data[key] = dat[key][mask_combined._mask][mask_metacal]

# +
# Compute DES weights

cat_gal = {}
cat_gal["e1_uncal"] = g_uncorr[0]
cat_gal["e2_uncal"] = g_uncorr[1]
cat_gal["R_g11"] = gal_metacal.R11
cat_gal["R_g12"] = gal_metacal.R12
cat_gal["R_g21"] = gal_metacal.R21
cat_gal["R_g22"] = gal_metacal.R22
cat_gal["NGMIX_T_NOSHEAR"] = dat["NGMIX_T_NOSHEAR"][mask_combined._mask][mask_metacal]
cat_gal["NGMIX_Tpsf_NOSHEAR"] = dat["NGMIX_Tpsf_NOSHEAR"][mask_combined._mask][mask_metacal]
cat_gal["snr"] = snr

name = 'w_des'
num_bins = 20
w = calibration.get_w_des(cat_gal, num_bins)

# +
# Correct for PSF leakage

cat_gal["e1"] = g_corr_mc[0]
cat_gal["e2"] = g_corr_mc[1]
#cat_gal["e1_PSF"] = e_psf

num_bins = 20
weight_type = 'des'
#alpha_1, alpha_2 = calibration.get_alpha_leakage_per_object(cat_gal, num_bins, weight_type)

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
# -

with open("masks.txt", "w") as f_out:
    for my_mask in masks:
        my_mask.print_summary(f_out)
    for my_mask in masks:
        my_mask.print_summary(f_out)

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
            r_cl[idx][jdx] = res.confidence_interval(confidence_level=confidence_level)
            
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

plt.imshow(r_val, cmap='coolwarm', vmin=-1, vmax=1)
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


n_key = len(all_masks)
cms = np.zeros((n_key, n_key, 2, 2))
for idx in range(n_key):
    for jdx in range(n_key):

        if idx == jdx: continue

        print(idx, jdx)
        res = confusion_matrix(masks[idx]._mask, masks[jdx]._mask)
        cms[idx][jdx] = res["cmn"]

# +
import seaborn as sns

fig = plt.figure(figsize=(30, 30))
axes = np.empty((n_key, n_key), dtype=object)
for idx in range(n_key):
    for jdx in range(n_key):
        if idx == jdx: continue
        axes[idx][jdx] = plt.subplot2grid((n_key, n_key), (idx, jdx), fig=fig)

matrix_elements = ["True", "False"]

for idx in range(n_key):
    for jdx in range(n_key):

        if idx == jdx: continue
        
        ax = axes[idx, jdx]
        sns.heatmap(cms[idx][jdx], annot=True, fmt='.2f', xticklabels=matrix_elements, yticklabels=matrix_elements, ax=ax)
        ax.set_ylabel(masks[idx]._label)
        ax.set_xlabel(masks[jdx]._label)

plt.show(block=False)
plt.savefig("confusion_matrix.png")
# -

obj.close_hd5()


