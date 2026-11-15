# %%
# Calibrate comprehensive catalogue
#
# Reads config_mask.yaml from the working directory: the input catalogue
# (params.input_path), the cuts, and the metacal block of
# sp_validation.response.calibrate_catalogue.

# %%
import json
import sys

import healpy as hp
import matplotlib.pylab as plt
import numpy as np
from astropy.io import fits
from cs_util import cat as cs_cat

import sp_validation.catalog as cat
from sp_validation import catalog_builders as sp_joint
from sp_validation import response

# %%
# Initialize calibration class instance
obj = sp_joint.CalibrateCat()

# %%
# Read configuration file and set parameters
config = obj.read_config_set_params("config_mask.yaml")


# %%
# Get data. Set load_into_memory to False for very large files
dat = obj.read_cat(load_into_memory=False)


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
#
# One pass of the response core: per-tile sums of the five metacal branches,
# the weight table, the response R and additive bias c_e with their tile
# jackknife, and the calibrated shapes of the selected galaxies.

cm = config["metacal"]
tiles = (
    np.char.strip(np.asarray(dat["TILE_ID"]).astype(str))
    if "TILE_ID" in dat.dtype.names
    else None
)
cal = response.calibrate_catalogue(
    dat, cm, preselection=mask_combined._mask, groups=tiles
)
scheme = cm["weight_scheme"]
index = cal["index"]
print("R =", cal["result"]["total"], "c_e =", cal["result"]["c_e"])

num_ok = len(index)
num_obj = len(dat)
sp_joint.Mask.print_strings(
    "metacal", "gal selection", str(num_ok), f"{num_ok / num_obj:10.2%}"
)

# %%
# Per-object columns of the selected galaxies

cat_gal = {
    f"w_{scheme}": cal["w"],
    "e1_uncal": cal["g"][0],
    "e2_uncal": cal["g"][1],
    "NGMIX_T_NOSHEAR": np.asarray(dat["NGMIX_T_NOSHEAR"])[index],
    "NGMIX_T_PSF_RECONV_NOSHEAR": np.asarray(dat["NGMIX_T_PSF_RECONV_NOSHEAR"])[index],
}
cat_gal["snr"] = (
    np.asarray(dat["NGMIX_FLUX_NOSHEAR"])[index]
    / np.asarray(dat["NGMIX_FLUX_ERR_NOSHEAR"])[index]
)
for i in range(2):
    for j in range(2):
        cat_gal[f"R_g{i + 1}{j + 1}"] = cal["R_g"][i, j]

# %%
# Correct for PSF leakage

alpha_1, alpha_2 = sp_joint.compute_PSF_leakage(
    cat_gal, cal["e"], dat, index, num_bins=20, weight_type=scheme
)

# Compute leakage-corrected ellipticities
e1_leak_corrected = cal["e"][0] - alpha_1 * cat_gal["e1_PSF"]
e2_leak_corrected = cal["e"][1] - alpha_2 * cat_gal["e2_PSF"]

# %%
# Additional quantities
ra = np.asarray(dat["RA"])[index]
dec = np.asarray(dat["Dec"])[index]
mag = np.asarray(dat["mag"])[index]


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
add_cols_data = {
    key: np.asarray(dat[key])[index]
    for key in (
        "FLUX_RADIUS",
        "MAGERR_AUTO",
        "NGMIX_T_NOSHEAR",
        "NGMIX_T_PSF_RECONV_NOSHEAR",
        "fwhm_PSF",
    )
}
add_cols_format = {}
if tiles is not None:
    add_cols_data["TILE_ID"] = tiles[index]
    add_cols_format["TILE_ID"] = f"{max(map(len, tiles[index]))}A"
for key in ("R_g11", "R_g12", "R_g21", "R_g22", "e1_PSF", "e2_PSF"):
    add_cols_data[key] = cat_gal[key]
add_cols_data["e1_leak_corrected"] = e1_leak_corrected
add_cols_data["e2_leak_corrected"] = e2_leak_corrected

# %%
# FITS header: the configuration, the masks, and the calibration contract
# (weight scheme, response applied, additive bias subtracted).
header = fits.Header()
obj.add_params_to_FITS_header(header, cm=cm)
for my_mask in masks:
    my_mask.add_summary_to_FITS_header(header)
header["WGHTSCH"] = (scheme, "weight scheme of the response and w column")
header["RAPPLIED"] = (True, "e1, e2 carry the inverse response")
header["MEANSUB"] = (
    cm.get("additive_correction", True),
    "e1, e2 have c_e subtracted",
)

# %%
output_shape_cat_path = obj._params["input_path"].replace("comprehensive", "cut")
output_shape_cat_path = output_shape_cat_path.replace("hdf5", "fits")

cat.write_shape_catalog(
    output_shape_cat_path,
    ra,
    dec,
    cal["w"],
    mag=mag,
    snr=cat_gal["snr"],
    g=cal["e"],
    g1_uncal=cal["g"][0],
    g2_uncal=cal["g"][1],
    w_type=scheme,
    add_cols=add_cols_data,
    add_cols_format=add_cols_format,
    add_header=header,
)

# %%
# Products beside the catalogue: the response record (with its tile-jackknife
# covariance), the per-tile sums the image-simulation m reads, and the DES
# weight table.
stem = output_shape_cat_path.removesuffix(".fits")
covariance = cal["jackknife"][2] if "jackknife" in cal else None
with open(f"{stem}_response.json", "w") as f:
    json.dump(response.record(cal["result"], covariance), f, indent=2)
cal["sums"].save(f"{stem}_sums.npz")
if scheme == "des":
    response.write_table(f"{stem}_des_weights.fits", cal["table"])

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
