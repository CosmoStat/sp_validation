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

# # Calibrate comprehensive catalogue

# %reload_ext autoreload
# %autoreload 2

# General library imports
import sys
import os
import numpy as np
from astropy.io import fits

from sp_validation import run_calibrate_cat as calibrate
from sp_validation import util
from sp_validation.basic import metacal
from sp_validation import calibration
import sp_validation.cat as cat

obj = calibrate.CalibrateCat()

obj._params["input_path"] = "unions_shapepipe_comprehensive_2024_v1.4.2.fits"

dat = obj.read_cat()

print(f"Found {len(dat)} (~{util.millify(len(dat))}) objects in catalogue")

# ## Masking

# ## Pre-processing ShapePipe flags

cut_pre = {}

sum(dat["FLAGS"] == 0)

# +
# SExtractor flags (see galaxy.py:classification_galaxy_base)

name = "FLAGS"
good_mask_value = 0

# MKDBEUG TODO: implement values other than 0 as "good"

cut_pre[name] = (dat[name] == good_mask_value)

# +
# Duplicate objects

name = "overlap"
good_mask_value = True
cut_pre[name] = (dat[name] == good_mask_value)
# -

# ShapePipe mask
name = "IMAFLAGS_ISO"
good_mask_value = 0
cut_pre[name] = (dat[name] == good_mask_value)

# +
# Number of epochs
name = "N_EPOCH"
val_min = 2
cut_pre[name] = (dat[name] >= val_min)

# MKDEBUG check NGMIX_N_EPOCH
# -

# Magnitude range
name = "mag"
min_max = [15, 30]
cut_pre[name] = (
    (dat[name] >= min_max[0])
    & (dat[name] <= min_max[1])
)

# +
# ngmix flags
names = ["NGMIX_MCAL_FLAGS", "NGMIX_MOM_FAIL"]
good_mask_values = [0, 0]
for name, good_mask_value in zip(names, good_mask_values):
    cut_pre[name] = (dat[name] == good_mask_value) 
    
name = "NGMIX_ELL_PSFo_NOSHEAR_0"
bad_mask_value = -10
cut_pre[name] = (
    dat[name] != bad_mask_value
)
# MKDEBUG TODO: check should be two components, see galaxy.py.
# -

cut_pre_combined = np.ones_like(cut_pre["FLAGS"], dtype=bool)

cut_pre_combined = np.logical_and.reduce(list(cut_pre.values()))

# +
# Output some mask statistics

n_obj = dat.shape[0]

print(f"{'flag':30s} {'n_ok':>10} {'n_ok[%]':>10}")
for name in cut_pre:
    n_ok = sum(cut_pre[name])
    print(f"{name:30s} {n_ok:10d} {n_ok/n_obj:10.2%}")
name = "combined"
n_ok = sum(cut_pre_combined)
print(f"{name:30s} {n_ok:10d} {n_ok/n_obj:10.2%}")

# +
# Number of "galaxies" (cut_common in main_set_up)

cut_common = cut_pre["overlap"] & cut_pre["FLAGS"] & cut_pre["mag"] & cut_pre["IMAFLAGS_ISO"] & cut_pre["N_EPOCH"]
name = "common"
n_ok = sum(cut_common)
print(f"{name:30s} {n_ok:10d} {n_ok/n_obj:10.2%}")

cut_galaxy = cut_common & cut_pre["NGMIX_MCAL_FLAGS"] & cut_pre["NGMIX_ELL_PSFo_NOSHEAR_0"] & cut_pre["NGMIX_MOM_FAIL"]
name = "galaxy"
n_ok = sum(cut_galaxy)
print(f"{name:30s} {n_ok:10d} {n_ok/n_obj:10.2%}")

# +
# Apply post-proc structural masks

cut_combined = cut_pre_combined
# -

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
n_ok = len(mask)
print(f"{name:30s} {n_ok:10d} {n_ok/n_obj:10.2%}")

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


