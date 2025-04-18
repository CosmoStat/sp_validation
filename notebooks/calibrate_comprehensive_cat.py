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

# Initialize calibration class instance
obj = sp_joint.CalibrateCat()

# Read configuration file and set parameters
config = obj.read_config_set_params("config_mask.yaml")

# !pwd

# Get data. Set load_into_memory to False for very large files
dat, dat_ext = obj.read_cat(load_into_memory=False)

# +
# print("MKDEBUG testing only first 100k objects")
# dat = dat[:100000]
# dat_ext = dat_ext[:100000]
# -

# ## Masking

# ### Pre-processing ShapePipe flags

masks, labels = sp_joint.get_masks_from_config(config)

mask_combined = sp_joint.Mask.from_list(
    masks,
    label="combined",
    verbose=obj,_params["verbose"],
)

# +
# Output some mask statistics

num_obj = dat.shape[0]

sp_joint.Mask.print_strings(
    "flag", "label", f"{'num_ok':>10}", f"{'num_ok[%]':>10}"
)
for my_mask in masks:
    my_mask.print_stats(num_obj)

mask_combined.print_stats(num_obj)
# -

if obj._params["sky_regions"]:

    # MKDBEUG TODO: zooms as list in config
    zoom_ra = [200, 205]
    zoom_dec = [55, 60]

    sp_joint.sky_plots(dat, masks, labels, zoom_ra, zoom_dec)

# ### Calibration

# +
# Call metacal

cm = config["metacal"]

gal_metacal = metacal(
    dat,
    mask_combined._mask,
    snr_min=cm["gal_snr_min"],
    snr_max=cm["gal_snr_max"],
    rel_size_min=cm["gal_rel_size_min"],
    rel_size_max=cm["gal_rel_size_max"],
    size_corr_ell=cm["gal_size_corr_ell"],
    sigma_eps=cm["sigma_eps_prior"],
    col_2d=False,
    verbose=True,
)
# -

g_corr_mc, g_uncorr, w, nask)metacal, c, c_err = get_calibrated_m_c(gal_metacal

num_ok = len(g_corr_mc[0])
sp_joint.Mask.print_strings(
    "metacal", "gal selection", str(num_ok), f"{num_ok / num_obj:10.2%}"
)

# +
# Compute DES weights

cat_gal = {}

w_des = sp_joint.compute_weights_gatti(
    cat_gal,
    g_uncorr,
    gal_metacal,
    dat,
    mask_combined,
    mask_metacal,
    num_bins=20,
)

# +
# Correct for PSF leakage

alpha_1, alpha_2 = sp_joint.compute_PSF_leakage(
    cat_gal,
    g_corr_mc,
    dat,
    mask_combined,
    mask_metacal,
    num_bins=20,    
)
# -

# Compute leakage-corrected ellipticities
e1_leak_corrected = g_corr_mc[0] - alpha_1 * cat_gal["e1_PSF"]
e2_leak_corrected = g_corr_mc[1] - alpha_2 * cat_gal["e2_PSF"]

# Get some memory back
for mask in masks:
    del mask

# +
# Additional quantities
R_shear = np.mean(gal_metacal.R_shear, 2)

ra = cat.get_col(dat, "RA", mask_combined._mask, mask_metacal)
dec = cat.get_col(dat, "Dec", mask_combined._mask, mask_metacal)
mag = cat.get_col(dat, "mag", mask_combined._mask, mask_metacal)


# +

add_cols = [
    "w_iv",
    "FLUX_RADIUS",
    "FWHM_IMAGE",
    "FWHM_WORLD",
    "MAGERR_AUTO",
    "MAG_WIN",
    "MAGERR_WIN",
    "FLUX_AUTO",
    "FLUXERR_AUTO",
    "FLUX_APER",
    "FLUXERR_APER",
]
add_cols_data = {}
for key in add_cols:
    add_cols_data[key] = dat[key][mask_combined._mask][mask_metacal]

# +

add_cols_data["e1_leak_corrected"] = e1_leak_corrected
add_cols_data["e2_leak_corrected"] = e2_leak_corrected

add_cols_data["e1_PSF"] = cat_gal["e1_PSF"]
add_cols_data["e2_PSF"] = cat_gal["e2_PSF"]
add_cols_data["fwhm_PSF"] = cat.get_col(
    dat, "fwhm_PSF", mask_combined._mask, mask_metacal
)

# +
# Add information to FITS header

# Generate new header
header = fits.Header()

# Add general config information to FITS header
obj.add_params_to_FITS_header(header)

# Add mask information to FITS header
for my_mask in masks:
    my_mask.add_summary_to_FITS_header(header)

# +
output_shape_cat_path = obj._params["input_path"].replace(
    "comprehensive", "cut"
)
output_shape_cat_path = output_shape_cat_path.replace("hdf5", "fits")

cat.write_shape_catalog(
    output_shape_cat_path,
    ra,
    dec,
    w_des,
    mag=mag,
    snr=cat_gal["snr"],
    g=g_corr_mc,
    g1_uncal=g_uncorr[0],
    g2_uncal=g_uncorr[1],
    R=gal_metacal.R,
    R_shear=R_shear,
    R_select=gal_metacal.R_selection,
    c=c,
    c_err=c_err,
    w_type="des",
    add_cols=add_cols_data,
    add_header=header,
)
# -

with open("masks.txt", "w") as f_out:
    for my_mask in masks:
        my_mask.print_summary(f_out)

from scipy import stats

#

all_masks = masks[:-3]

# +
if not obj._params["cmatrices"]:
    print("Skipping cmatric calculations")
    sys.exit(0)

r_val, r_cl = sp_joint.correlation_matrix(all_masks)

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


n_key = len(all_masks)
cms = np.zeros((n_key, n_key, 2, 2))
for idx in range(n_key):
    for jdx in range(n_key):

        if idx == jdx:
            continue

        print(idx, jdx)
        res = sp_joint.confusion_matrix(masks[idx]._mask, masks[jdx]._mask)
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
