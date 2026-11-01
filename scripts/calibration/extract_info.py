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

# # Extract information
#
#
# from:
#
# shapepipe output final weak-lensing galaxy shape catalogue
#
# for:
#
# postprocessing (merging, object selection, cuts, calibration).
#
# This was formerly the series of notebooks / validation.py.

# %reload_ext autoreload
# %autoreload 2

# General library imports
import os
import sys

import h5py

# from sp_validation.catalog import *
from sp_validation import catalog as spv_cat
from sp_validation import galaxy, response
from sp_validation.calibration import metacal
from sp_validation.galaxy import T_to_fwhm, classification_galaxy_overlap_ra_dec
from sp_validation.io import open_stats_file, print_stats, read_catalogue
from sp_validation.survey import get_area, missing_tiles

# ## 1. Set-up

# Load parameters
sys.path.insert(0, os.getcwd())
from params import *

# ### Create and open output files and directories

os.makedirs(output_dir, exist_ok=True)
stats_file = open_stats_file(output_dir, stats_file_name)

output_shape_cat_stem = output_shape_cat_base
output_ext = output_format

# ## 2. Load data
#
# ### Load merged (final) galaxy catalogue

# +
dd = read_catalogue(
    galaxy_cat_path,
    columns=spv_cat.read_param_file(param_list_path) if param_list_path else None,
    column_map=galaxy_column_map,
    verbose=verbose,
)

n_obj = len(dd)
print_stats(
    f"Read {n_obj} objects from file {galaxy_cat_path}", stats_file, verbose=verbose
)
# -

# #### Print some quantities to check nothing obvious is wrong with catalogue

# PSF keys
key_PSF_g1 = "NGMIX_G1_PSF_ORIG_NOSHEAR"
key_PSF_g2 = "NGMIX_G2_PSF_ORIG_NOSHEAR"
key_PSF_size = "NGMIX_T_PSF_ORIG_NOSHEAR"
size_to_fwhm = T_to_fwhm

print_stats("Galaxies:", stats_file, verbose=verbose)
n_tot = spv_cat.print_some_quantities(dd, stats_file, verbose=verbose)
spv_cat.print_mean_ellipticity(
    dd,
    ["NGMIX_G1_NOSHEAR", "NGMIX_G2_NOSHEAR"],
    1,
    n_tot,
    stats_file,
    invalid=-10,
    verbose=verbose,
)

# #### Survey area and potential missing tiles
# The approximate observed area is the number of tiles $\times$ 0.25 deg$^2$ (ignoring overlaps and masking).

area_deg2, area_amin2, tile_IDs = get_area(dd, area_tile, verbose=verbose)

# Identify missing tiles by comparing tile ID from catalogue to external input tile ID file.

n_found, n_missing = missing_tiles(
    tile_IDs, path_tile_ID, path_found_ID, path_missing_ID, verbose=verbose
)

print_stats(f"Tiles in input catalogue: {n_found}", stats_file, verbose=verbose)

# ### Load star catalogue

if star_cat_path:
    d_star = read_catalogue(
        star_cat_path,
        hdu=hdu_star_cat,
        column_map=star_column_map,
        key_column="EXPID",
        verbose=verbose,
    )

if star_cat_path:
    print_stats("Stars:", stats_file, verbose=verbose)
    n_tot = spv_cat.print_some_quantities(d_star, stats_file, verbose=verbose)
    spv_cat.print_mean_ellipticity(
        d_star,
        ["HSM_G1_PSF", "HSM_G2_PSF"],
        1,
        n_tot,
        stats_file,
        invalid=-10,
        verbose=verbose,
    )

# ### 3. Matching of stars

# ### Matching of star catalogues
# Match the star catalogue `d_star` (selected on individual exposures using size-magnitude diagram) to catalogue from tile. Uses some simple criteria to select stars from tile catalogue.
#
# This is mainly for testing, this match will not be used later.

# #### Match to all objects

if star_cat_path:
    ind_star, n_star_tot = spv_cat.check_matching(
        d_star,
        dd,
        ["RA", "DEC"],
        [col_name_ra, col_name_dec],
        thresh,
        stats_file,
        verbose=verbose,
    )

    # Flags to indicate valid star sample.
    # Star matching, PSF-catalogue output and star metacalibration are all
    # diagnostics that require an input star catalogue; the image-simulation
    # pipeline has none (star_cat_path is None), so every star-dependent block
    # below is guarded and simply skipped for the sims.

    m_star = (
        (dd["FLAGS"][ind_star] == 0)
        & galaxy.mask_cut(dd, mask_columns)[ind_star]
        & (dd["NGMIX_MCAL_FLAGS"][ind_star] == 0)
        & (dd["NGMIX_G1_PSF_ORIG_NOSHEAR"][ind_star] != -10)
    )

    ra_star, dec_star, g_star_psf = spv_cat.match_subsample(
        dd,
        ind_star,
        m_star,
        [col_name_ra, col_name_dec],
        key_PSF_g1,
        key_PSF_g2,
        n_star_tot,
        stats_file,
        verbose=verbose,
    )

    # ### Write PSF catalogue with multi-epoch shapes from shape measurement methods

    spv_cat.write_PSF_cat(
        f"{output_PSF_cat_base}_{shape}.fits",
        ra_star,
        dec_star,
        g_star_psf[0],
        g_star_psf[1],
    )

# ## Check for objects with invalid PSF

spv_cat.check_invalid(
    dd,
    [key_PSF_g1, "NGMIX_G1_NOSHEAR"],
    [-10, -10],
    stats_file,
    name=["`PSF", "galaxy ellipticity"],
    verbose=verbose,
)

# +
# Flag for duplicate objects in tile boundaries

cut_overlap = classification_galaxy_overlap_ra_dec(
    dd,
    ra_key=col_name_ra,
    dec_key=col_name_dec,
)

n_ok = sum(cut_overlap)
print_stats(
    f"Non-overlapping objects: {n_ok:10d}, {n_ok / n_obj:10.2%}",
    stats_file,
    verbose=verbose,
)

# -

# Get coordinates for all objects
ra_all = spv_cat.get_col(dd, col_name_ra)
dec_all = spv_cat.get_col(dd, col_name_dec)

# ### Write comprehensive shape catalogue (all objects, no cuts)

# +
# Add additional columns without cuts nor mask applied

ext_cols_pre_cal = {}

# Standard additional columns
if add_cols:
    for key in add_cols:
        ext_cols_pre_cal[key] = dd[key]

# Pre-calibration columns
if add_cols_pre_cal:
    for key in add_cols_pre_cal:
        ext_cols_pre_cal[key] = dd[key]

# Retain upstream branch errors when supplied, even with older params files.
# Historical HDF5 merges omitted them; that must not block DES extraction.
# Modern IV rejects an incomplete set at calibration, rather than freezing W.
ext_cols_pre_cal.update(response.branch_error_columns(dd))

# Flag to cut duplicate objects in overlapping region with neighbouring tiles
ext_cols_pre_cal["overlap"] = cut_overlap
add_cols_pre_cal_format["overlap"] = "I"

# Additional columns {e1, e2, size}_PSF
ext_cols_pre_cal["e1_PSF"] = dd[key_PSF_g1]
ext_cols_pre_cal["e2_PSF"] = dd[key_PSF_g2]
ext_cols_pre_cal["fwhm_PSF"] = size_to_fwhm(dd[key_PSF_size])

_, _, iv_w = metacal.get_variance_ivweights(dd, sigma_eps_prior, mask=None)

mag = spv_cat.get_col(dd, "MAG_AUTO", None, None)
snr = spv_cat.get_snr(shape, dd, None, None)
g1_uncal = dd["NGMIX_G1_NOSHEAR"]
g2_uncal = dd["NGMIX_G2_NOSHEAR"]

# Comprehensive catalogue without cuts nor mask applied
if verbose:
    print("Writing comprehensive catalogue...")

comprehensive_cat_path = f"{output_shape_cat_stem}_comprehensive_{shape}{output_ext}"
spv_cat.write_shape_catalog(
    comprehensive_cat_path,
    ra_all,
    dec_all,
    iv_w,
    mag=mag,
    snr=snr,
    g1_uncal=g1_uncal,
    g2_uncal=g2_uncal,
    add_cols=ext_cols_pre_cal,
    add_cols_format=add_cols_pre_cal_format,
)

# Write tile count to HDF5 attributes
if output_ext == ".hdf5":
    try:
        with h5py.File(comprehensive_cat_path, "a") as hf:
            hf.attrs["n_tiles"] = n_found
            if verbose:
                print(f"  Added n_tiles={n_found} to HDF5 attributes")
    except Exception as e:
        if verbose:
            print(f"  Warning: could not add n_tiles attribute: {e}")
# -

do_selection_calibration = False
if not do_selection_calibration:
    if verbose:
        print("No selection and calibration; exiting here")
    sys.exit(0)
else:
    raise ValueError(
        "Direct extraction calibration has no shared weight_scheme/selection "
        "configuration. Run calibrate_comprehensive_cat.py with mask_v2.0.yaml "
        "on the comprehensive five-branch product instead."
    )
