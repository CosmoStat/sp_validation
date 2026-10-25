# %%
# Calibrate comprehensive catalogue

# %%
from IPython import get_ipython

# %%
# enable autoreload for interactive sessions
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("reload_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
    ipython.run_line_magic("reload_ext", "log_cell_time")


# %%
import sys

import healpy as hp
import matplotlib.pylab as plt
import numpy as np
from astropy.io import fits
from cs_util import cat as cs_cat

import sp_validation.catalog as cat
from sp_validation import calibration
from sp_validation import catalog_builders as sp_joint

# %%
# Initialize calibration class instance
obj = sp_joint.CalibrateCat()

# %%
# Read configuration file and set parameters
config = obj.read_config_set_params("config_mask.yaml")


# %%
obj._params

# %%
# Get data. Set load_into_memory to False for very large files
dat = obj.read_cat(load_into_memory=False)

# %%
n_test = -1
# n_test = 100000
if n_test > 0:
    print(f"MKDEBUG testing only first {n_test} objects")
    dat = dat[:n_test]


# ## Masking

# %%
# ### Pre-processing ShapePipe flags
masks, labels = sp_joint.get_masks_from_config(config, dat, verbose=True)

mask_combined = sp_joint.Mask.from_list(
    masks,
    label="combined",
    verbose=obj._params["verbose"],
)

# %%
# Output some mask statistics
sp_joint.print_mask_stats(dat.shape[0], masks, mask_combined)

if obj._params["sky_regions"]:
    # MKDBEUG TODO: zooms as list in config
    zoom_ra = [200, 205]
    zoom_dec = [55, 60]

    sp_joint.sky_plots(dat, masks, labels, zoom_ra, zoom_dec)

# %%
# ### Calibration

# Call metacal
cm = config["metacal"]

# %%
gal_metacal = calibration.metacal_from_config(
    dat, mask_combined._mask, cm, verbose=True
)

# %%
additive_correction = cm.get("additive_correction", True)
if not additive_correction:
    print("Additive bias correction disabled (additive_correction: False)")

g_corr_mc, g_uncorr, w, mask_metacal, c, c_err = calibration.get_calibrated_m_c(
    gal_metacal,
    additive_correction=additive_correction,
)

num_ok = len(g_corr_mc[0])
num_obj = len(dat)
sp_joint.Mask.print_strings(
    "metacal", "gal selection", str(num_ok), f"{num_ok / num_obj:10.2%}"
)

# %%
# Compute DES weights

cat_gal = {}

sp_joint.compute_weights_gatti(
    cat_gal,
    g_uncorr,
    gal_metacal,
    dat,
    mask_combined,
    mask_metacal,
    num_bins=20,
    snr_min=cm["gal_snr_min"],
    snr_max=cm["gal_snr_max"],
    size_ratio_min=cm["gal_rel_size_min"],
    size_ratio_max=cm["gal_rel_size_max"],
)

# %%
# Correct for PSF leakage

alpha_1, alpha_2 = sp_joint.compute_PSF_leakage(
    cat_gal,
    g_corr_mc,
    dat,
    mask_combined,
    mask_metacal,
    num_bins=20,
)

# Compute leakage-corrected ellipticities
e1_leak_corrected = g_corr_mc[0] - alpha_1 * cat_gal["e1_PSF"]
e2_leak_corrected = g_corr_mc[1] - alpha_2 * cat_gal["e2_PSF"]

# %%
# Get some memory back
for mask in masks:
    del mask

# %%
# Additional quantities
ra = cat.get_col(dat, "RA", mask_combined._mask, mask_metacal)
dec = cat.get_col(dat, "Dec", mask_combined._mask, mask_metacal)
mag = cat.get_col(dat, "mag", mask_combined._mask, mask_metacal)


# %%
# Create hp mask file
if "binned_mask" in config:
    cbm = config["binned_mask"]
    print("Binning data to obtain mask...")
    area_deg2, pix = cs_cat.get_binned_area(
        ra, dec, nside=cbm["nside"], return_pix=True
    )
    healpix_map = np.full(hp.nside2npix(cbm["nside"]), not cbm["good"])
    healpix_map[pix] = cbm["good"]
    print(f"Writing binned mask to file {cbm['output_path']}...")
    hp.write_map(cbm["output_path"], healpix_map, overwrite=True)
else:
    print("No binned mask file on output")


# %%
# Additional data columns to write to output cat
add_cols = [
    "FLUX_RADIUS",
    "MAGERR_AUTO",
    "NGMIX_T_NOSHEAR",
    "NGMIX_T_PSF_RECONV_NOSHEAR",
    "fwhm_PSF",
]
add_cols_data = {}
for key in add_cols:
    add_cols_data[key] = cat.get_col(dat, key, mask_combined._mask, mask_metacal)

# %%
# Never copy stale unsquared IV weights from a comprehensive input.
if gal_metacal.weight_scheme != "iv":
    add_cols_data["w_iv"] = gal_metacal.ns["w_iv"][mask_metacal]
add_cols_format = {}
if "TILE_ID" in dat.dtype.names:
    add_cols_data["TILE_ID"] = np.asarray(
        cat.get_col(dat, "TILE_ID", mask_combined._mask, mask_metacal)
    ).astype(str)
    add_cols_format["TILE_ID"] = f"{max(map(len, add_cols_data['TILE_ID']))}A"

# Additional post-processing columns to write to output cat
add_cols_post = [
    "R_g11",
    "R_g12",
    "R_g21",
    "R_g22",
    "e1_PSF",
    "e2_PSF",
]
for key in add_cols_post:
    add_cols_data[key] = cat_gal[key]

# %%
# Other additional columns
add_cols_data["e1_leak_corrected"] = e1_leak_corrected
add_cols_data["e2_leak_corrected"] = e2_leak_corrected

# %%
# Add information to FITS header

# Generate new header
header = fits.Header()

# Add general and metacal config information to FITS header
obj.add_params_to_FITS_header(header, cm=cm)

# %%
# Add mask information to FITS header
for my_mask in masks:
    my_mask.add_summary_to_FITS_header(header)

# %%
output_shape_cat_path = obj._params["input_path"].replace("comprehensive", "cut")
output_shape_cat_path = output_shape_cat_path.replace("hdf5", "fits")

cat.write_shape_catalog(
    output_shape_cat_path,
    ra,
    dec,
    cat_gal["w_des"] if gal_metacal._legacy_response else w,
    mag=mag,
    snr=cat_gal["snr"],
    g=g_corr_mc,
    g1_uncal=g_uncorr[0],
    g2_uncal=g_uncorr[1],
    R=gal_metacal.R,
    R_shear=gal_metacal.R_shear_global,
    R_select=gal_metacal.R_selection,
    c=c,
    c_err=c_err,
    w_type=gal_metacal.weight_scheme or "des",
    add_cols=add_cols_data,
    add_cols_format=add_cols_format,
    add_header=header,
)

# %%
groups = (
    np.asarray(dat["TILE_ID"][mask_combined._mask])
    if "TILE_ID" in dat.dtype.names
    else None
)
calibration.save_response_product(output_shape_cat_path, gal_metacal, groups)

# Write mask information to ASCII file
with open("masks.txt", "w") as f_out:
    for my_mask in masks:
        my_mask.print_summary(f_out)


# %%

if not obj._params["cmatrices"]:
    print("Skipping cmatric calculations")
    sys.exit(0)

# %%

all_masks = masks[:-3]

r_val, r_cl = sp_joint.correlation_matrix(all_masks)

n = len(all_masks)
keys = [my_mask._label for my_mask in all_masks]

plt.imshow(r_val, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(np.arange(n), keys)
plt.yticks(np.arange(n), keys)
plt.xticks(rotation=90)
plt.colorbar()
plt.savefig("correlation_matrix.png")

# %%
n_key = len(all_masks)
cms = np.zeros((n_key, n_key, 2, 2))
for idx in range(n_key):
    for jdx in range(n_key):
        if idx == jdx:
            continue

        print(idx, jdx)
        res = sp_joint.confusion_matrix(masks[idx]._mask, masks[jdx]._mask)
        cms[idx][jdx] = res["cmn"]

# %%
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


obj.close_cat()

# %%
