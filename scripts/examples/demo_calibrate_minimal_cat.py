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

# # Calibrate minimal catalogue

# %reload_ext autoreload
# %autoreload 2

import json

import numpy as np
from astropy.io import fits

import sp_validation.catalog as cat
from sp_validation import catalog_builders as sp_joint
from sp_validation import response

# Initialize calibration class instance
obj = sp_joint.CalibrateCat()

# Read configuration file and set parameters
config = obj.read_config_set_params("config_minimal.yaml")

print(obj._params)

# !pwd

# Get data. Set load_into_memory to False for very large files
dat = obj.read_cat(load_into_memory=False)

if True:
    n_max = 1_000_000
    print(f"MKDEBUG testing only first {n_max} objects")
    dat = dat[:n_max]


# ## Masking

masks_to_apply = [
    "FLAGS",
    "MASK_4_Stars",
    "MASK_64_r",
    "MASK_1024_Maximask",
    "N_EPOCH",
    "mag",
    "NGMIX_MCAL_TYPES_FAIL",
    "NGMIX_G1_PSF_ORIG_NOSHEAR",
    "NGMIX_G2_PSF_ORIG_NOSHEAR",
]

masks, labels = sp_joint.get_masks_from_config(
    config, dat, masks_to_apply=masks_to_apply, verbose=obj._params["verbose"]
)

mask_combined = sp_joint.Mask.from_list(
    masks,
    label="combined",
    verbose=obj._params["verbose"],
)

# +
# Output some mask statistics

num_obj = dat.shape[0]

sp_joint.Mask.print_strings("flag", "label", f"{'num_ok':>10}", f"{'num_ok[%]':>10}")
for my_mask in masks:
    my_mask.print_stats(num_obj)

mask_combined.print_stats(num_obj)
# -

if obj._params["sky_regions"]:
    # MKDBEUG TODO: zooms as list in config
    zoom_ra = [200, 205]
    zoom_dec = [55, 60]

    sp_joint.sky_plots(dat, masks, labels, zoom_ra, zoom_dec)

# ### PSF leakage

# ### Calibration

# +
# The response core: weights, response R, additive bias c_e and the
# calibrated shapes of the selected galaxies.

cm = config["metacal"]
cal = response.calibrate_catalogue(dat, cm, preselection=mask_combined._mask)
scheme = cm["weight_scheme"]
index = cal["index"]
# -

num_ok = len(index)
sp_joint.Mask.print_strings(
    "metacal", "gal selection", str(num_ok), f"{num_ok / num_obj:10.2%}"
)

# +
# Correct for PSF leakage

snr = (
    np.asarray(dat["NGMIX_FLUX_NOSHEAR"])[index]
    / np.asarray(dat["NGMIX_FLUX_ERR_NOSHEAR"])[index]
)
cat_gal = {
    f"w_{scheme}": cal["w"],
    "snr": snr,
    "NGMIX_T_NOSHEAR": np.asarray(dat["NGMIX_T_NOSHEAR"])[index],
    "NGMIX_T_PSF_RECONV_NOSHEAR": np.asarray(dat["NGMIX_T_PSF_RECONV_NOSHEAR"])[index],
}
alpha_1, alpha_2 = sp_joint.compute_PSF_leakage(
    cat_gal, cal["e"], dat, index, num_bins=20, weight_type=scheme
)
# -

# Compute leakage-corrected ellipticities
e1_leak_corrected = cal["e"][0] - alpha_1 * cat_gal["e1_PSF"]
e2_leak_corrected = cal["e"][1] - alpha_2 * cat_gal["e2_PSF"]

# +
add_cols = [
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
    "fwhm_PSF",
]
add_cols_data = {key: np.asarray(dat[key])[index] for key in add_cols}
add_cols_data["e1_leak_corrected"] = e1_leak_corrected
add_cols_data["e2_leak_corrected"] = e2_leak_corrected
for i in range(2):
    for j in range(2):
        add_cols_data[f"R_g{i + 1}{j + 1}"] = cal["R_g"][i, j]
add_cols_data["e1_PSF"] = cat_gal["e1_PSF"]
add_cols_data["e2_PSF"] = cat_gal["e2_PSF"]

# +
# FITS header: configuration, masks and the calibration contract
header = fits.Header()
obj.add_params_to_FITS_header(header)
for my_mask in masks:
    my_mask.add_summary_to_FITS_header(header)
header["WGHTSCH"] = scheme
header["RAPPLIED"] = True
header["MEANSUB"] = cm.get("additive_correction", True)

# +
output_shape_cat_path = obj._params["input_path"].replace("comprehensive", "cut")
output_shape_cat_path = output_shape_cat_path.replace("hdf5", "fits")

cat.write_shape_catalog(
    output_shape_cat_path,
    np.asarray(dat["RA"])[index],
    np.asarray(dat["Dec"])[index],
    cal["w"],
    mag=np.asarray(dat["mag"])[index],
    snr=snr,
    g=cal["e"],
    g1_uncal=cal["g"][0],
    g2_uncal=cal["g"][1],
    w_type=scheme,
    add_cols=add_cols_data,
    add_header=header,
)
stem = output_shape_cat_path.removesuffix(".fits")
with open(f"{stem}_response.json", "w") as f:
    json.dump(response.record(cal["result"]), f, indent=2)
# -

with open("masks.txt", "w") as f_out:
    for my_mask in masks:
        my_mask.print_summary(f_out)

#

obj.close_cat()
