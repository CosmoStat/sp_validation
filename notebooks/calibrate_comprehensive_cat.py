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

print("MKDEBUG: only use first 10k objects")
dat = dat[:10000]
dat_ext = dat_ext[:10000]

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
        
    @classmethod    
    def from_list(cls, masks, label="combined"):
        mask = cls(label, label, kind="combined", value=None)

        mask._mask = np.logical_and.reduce([m._mask for m in masks])

        return mask

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
            print(f"{self._col_name} {sign} {value}", file=f_out)
            
        if self._kind == "range":
            print(f"{self._value[0]} <= {self._col_name} <= {self._value[1]}", file=f_out)            


# ### Pre-processing ShapePipe flags

# +
cut_pre = {}
labels = {}

masks = []

# +
# SExtractor flags (see galaxy.py:classification_galaxy_base)

my_mask = mask(col_name="FLAGS", label="SE FLAGS", kind="equal", value=0, dat=dat)
masks.append(my_mask)

# +
# Duplicate objects

my_mask = mask(col_name="overlap", label="tile overlap", kind="equal", value=True, dat=dat)
masks.append(my_mask)

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

# +
# ShapePipe mask

name = "IMAFLAGS_ISO"
good_mask_value = 0
cut_pre[name] = (dat[name] == good_mask_value)
labels[name] = "SP mask"
# -

my_mask = mask(col_name="IMAFLAGS_ISO", label="SP mask", kind="equal", value=0, dat=dat)
masks.append(my_mask)

# +
# Number of epochs

my_mask = mask(col_name="N_EPOCH", label=r"$n_{\rm epoch}$", kind="greater_equal", value=2, dat=dat)
masks.append(my_mask)

# +
# Number of epochs

name = "N_EPOCH"
val_min = 2
cut_pre[name] = (dat[name] >= val_min)

# MKDEBUG check NGMIX_N_EPOCH
labels[name] = r"$n_{\rm epoch}$"

# +
# Magnitude range

my_mask = mask(col_name="mag", label="mag range", kind="range", value=[15, 30], dat=dat)
masks.append(my_mask)

# +
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
# -

# Combine all masks with &
cut_pre_combined = np.logical_and.reduce(list(cut_pre.values()))

print(f"Combining {len(masks)} pre-processing masks")
mask_combined_pre = mask.from_list(masks, label="combined_pre")

# +

# Output some mask statistics

num_obj = dat.shape[0]

mask.print_strings("flag", "label", f"{'num_ok':>10}", f"{'num_ok[%]':>10}")
for my_mask in masks:
    my_mask.print_stats(num_obj)

mask_combined_pre.print_stats(num_obj)

# +
# Output some mask statistics

num_obj = dat.shape[0]

print(f"{'flag':30s} {'label':30s} {'num_ok':>10} {'num_ok[%]':>10}")
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

# ### Structural masks

cut_struct = {}

# Get all columns of type ``bool'' from the extended catalogue
column_ext_info = {
    name: "bool" if dat_ext.dtype.fields[name][0] == "int8" else dat_ext.dtype.fields[name][0]
    for name in dat_ext.dtype.names
}

# +
good_mask_values = False

for name in column_ext_info:
    if column_ext_info[name] == "bool":
        cut_struct[name] = (dat_ext[name] == good_mask_values)
        labels[name] = name
    else:
        print(f"Type for column {name} not bool, skipping")

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
val_min = 3
cut_struct[name] = (dat_ext[name] >= val_min)
labels[name] = r"$n_{\rm pointing}$"

# Number of pointings
name = "npoint3"
label = r"$n_{\rm pointing}$"
value = 3
kind = "greater_equal"
my_mask = mask(col_name=name, label=label, kind=kind, value=value, dat=dat_ext)
masks.append(my_mask)

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

n_struct = len(masks) - idx_first_struct
print(f"Combining {n_struct} structural masks")
mask_combined_struct = mask.from_list(masks[idx_first_struct:], label="combined_struct")

for my_mask in masks[idx_first_struct:]:
    my_mask.print_stats(num_obj)
mask_combined_struct.print_stats(num_obj)

# # Plots:
#
# # mag histogram, footprint, when applying different cuts

print(f"Combining pre-processing and structural masks, {len(masks)} in total")
mask_combined = mask.from_list([mask_combined_pre, mask_combined_struct], label="combined")

#

cut_combined = cut_pre_combined & cut_struct_combined

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

ra = cat.get_col(dat, "RA", mask_combined._mask,  mask)
dec = cat.get_col(dat, "Dec", mask_combined._mask,  mask)
mag = cat.get_col(dat, "mag", mask_combined._mask,  mask)
snr = cat.get_snr("ngmix", dat, mask_combined._mask,  mask)

add_cols = ["FLUX_RADIUS", "FWHM_IMAGE", "FWHM_WORLD", "MAGERR_AUTO", "MAG_WIN", "MAGERR_WIN", "FLUX_AUTO", "FLUXERR_AUTO", "FLUX_APER", "FLUXERR_APER"]
add_cols_data = {}    
for key in add_cols:
    add_cols_data[key] = dat[key][mask_combined._mask][mask]

# +
# Compute DES weights

cat_gal = {}
cat_gal["e1_uncal"] = g_uncorr[0]
cat_gal["e2_uncal"] = g_uncorr[1]
cat_gal["R_g11"] = gal_metacal.R11
cat_gal["R_g12"] = gal_metacal.R12
cat_gal["R_g21"] = gal_metacal.R21
cat_gal["R_g22"] = gal_metacal.R22
cat_gal["NGMIX_T_NOSHEAR"] = dat["NGMIX_T_NOSHEAR"][mask_combined._mask][mask]
cat_gal["NGMIX_Tpsf_NOSHEAR"] = dat["NGMIX_Tpsf_NOSHEAR"][mask_combined._mask][mask]
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


n_key = len(keys)
cms = np.zeros((n_key, n_key, 2, 2))
for idx, key_1 in enumerate(keys):
    for jdx, key_2 in enumerate(keys):

        if idx == jdx: continue

        print(idx, jdx, key_1, key_2)
        res = confusion_matrix(cut_pre[key_1], cut_pre[key_2])
        cms[idx][jdx] = res["cmn"]

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




























