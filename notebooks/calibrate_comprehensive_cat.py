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

obj._params["input_path"] = "unions_shapepipe_comprehensive_struc_2024_v1.4.2.hdf5"
obj._params["cmatrices"] = True
obj._params["verbose"] = True

# !pwd

dat, dat_ext = obj.read_cat(load_into_memory=False)

print(f"Found {len(dat)} (~{util.millify(len(dat))}) objects in catalogue")


# ## Masking

class mask():
    def __init__(self, col_name, label, kind="equal", value=0, dat=None):
        self._col_name = col_name
        self._label = label
        self._value = value
        self._kind = kind
        self._num_ok = None

        if dat is not None:
            self.apply(dat)
        
    def from_list(cls, masks, label="combined"):
        my_mask = cls(label, label, kind="combined", value=None)

        my_mask._mask = np.logical_and.reduce([m._mask for m in masks])

        return my_mask

    def apply(self, dat):
        if self._kind == "equal":
            self._mask = dat[self._col_name] == self._value
        elif self._kind == "not_equal":
            self._mask = dat[self._col_name] != self._value
        elif self._kind == "greater_equal":
            self._mask = dat[self._col_name] >= self._value
        elif self._kind == "range":
            self._mask = (dat[self._col_name] >= self._value[0]) & (dat[self._col_name] <= self._value[1])
        else:
            raise ValueError(f"Invalid kind {kind}")
        
    @classmethod
    def print_strings(cls, coln, lab, num, fnum):
        print(f"{coln:30s} {lab:30s} {num:10s} {fnum:10s}")
        
    def print_stats(self, num_obj):
        if self._num_ok is None:
            self._num_ok = sum(self._mask)

        si = f"{self._num_ok:10d}"
        sf = f"{self._num_ok/num_obj:10.2%}"
        self.print_strings(self._col_name, self._label, si, sf)
        
    def print_summary(self, f_out):
        print(f"[{self._label}]\t\t\t", file=f_out, end="")
        sign = None
        if self._kind =="equal":
            sign = "="
        elif self._kind =="not_equal":
            sign = "!="
        elif self._kind =="greater_equal":
            sign = ">="
        if sign is not None:
            print(f"{self._col_name} {sign} {self._value}", file=f_out)
            
        if self._kind == "range":
            print(f"{self._value[0]} <= {self._col_name} <= {self._value[1]}", file=f_out)            


# ### Pre-processing ShapePipe flags

masks = []

# +
# SExtractor flags (see galaxy.py:classification_galaxy_base)

my_mask = mask(col_name="FLAGS", label="SE FLAGS", kind="equal", value=0, dat=dat)
masks.append(my_mask)

# +
# Duplicate objects

my_mask = mask(col_name="overlap", label="tile overlap", kind="equal", value=True, dat=dat)
masks.append(my_mask)
# -

my_mask = mask(col_name="IMAFLAGS_ISO", label="SP mask", kind="equal", value=0, dat=dat)
masks.append(my_mask)

# +
# Number of epochs

my_mask = mask(col_name="N_EPOCH", label=r"$n_{\rm epoch}$", kind="greater_equal", value=2, dat=dat)
masks.append(my_mask)

# +
# Magnitude range

my_mask = mask(col_name="mag", label="mag range", kind="range", value=[15, 30], dat=dat)
masks.append(my_mask)

# +
# ngmix flags

col_names = ["NGMIX_MCAL_FLAGS", "NGMIX_MOM_FAIL"]
tmp_labels = ["ngmix flag", "ngmix moments fail"]
good_mask_value = 0

for col_name, label in zip(col_names, tmp_labels):
    my_mask = mask(col_name=col_name, label=label, kind="equal", value=good_mask_value, dat=dat)
    masks.append(my_mask)

# MKDEBUG TODO: check should be two components, see galaxy.py.
col_name = "NGMIX_ELL_PSFo_NOSHEAR_0"
label = "bad PSF ell"
kind = "not_equal"
value = -10
my_mask = mask(col_name=col_name, label=label, kind=kind, value=value, dat=dat)
masks.append(my_mask)

# -

print(f"Combining {len(masks)} pre-processing masks")
mask_combined_pre = mask.from_list(masks, label="combined_pre")

# +
# Output some mask statistics

num_obj = dat.shape[0]

mask.print_strings("flag", "label", f"{'num_ok':>10}", f"{'num_ok[%]':>10}")
for my_mask in masks:
    my_mask.print_stats(num_obj)

mask_combined_pre.print_stats(num_obj)
# -

# ### Structural masks

# Get all columns of type ``bool'' from the extended catalogue
column_ext_info = {
    name: "bool" if dat_ext.dtype.fields[name][0] == "int8" else dat_ext.dtype.fields[name][0]
    for name in dat_ext.dtype.names
}

# +
# Get all columns of type ``bool'' from the extended catalogue.
# 0 = good mask value

# Store the index of the first structural mask = current length of masks list
idx_first_struct = len(masks)

value = False
for name in column_ext_info:
    if column_ext_info[name] == "bool":
        my_mask = mask(col_name=name, label=name, kind="equal", value=value, dat=dat_ext)
        masks.append(my_mask)
    else:
        print(f"Type for column {name} not bool, skipping")
# -

# Number of pointings
name = "npoint3"
label = r"$n_{\rm pointing}$"
value = 3
kind = "greater_equal"
my_mask = mask(col_name=name, label=label, kind=kind, value=value, dat=dat_ext)
masks.append(my_mask)

n_struct = len(masks) - idx_first_struct
print(f"Combining {n_struct} structural masks")
mask_combined_struct = mask.from_list(masks[idx_first_struct:], label="combined_struct")

for my_mask in masks[idx_first_struct:]:
    my_mask.print_stats(num_obj)
mask_combined_struct.print_stats(num_obj)

print(f"Combining pre-processing and structural masks, {len(masks)} in total")
mask_combined = mask.from_list([mask_combined_pre, mask_combined_struct], label="combined")

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
my_mask = mask(col_name=col_name, label=label, kind="range", value=[gal_rel_size_min, gal_rel_size_max])
masks.append(my_mask)

col_name = "snr"
label = "metacal signal-to-noise ratio"
my_mask = mask(col_name=col_name, label=label, kind="range", value=[gal_snr_min, gal_snr_max])
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


