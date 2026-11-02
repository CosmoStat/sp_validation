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
from sp_validation import galaxy
from sp_validation.calibration import *
from sp_validation.calibration import metacal
from sp_validation.galaxy import *
from sp_validation.io import *
from sp_validation.io import read_catalogue
from sp_validation.survey import *

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

# v1.3 old-grammar compatibility shim.
# Pre-shapepipe#761 catalogues (e.g. v1.3.x) store ellipticity as a single
# combined `NGMIX_ELL_<suffix>` 2-vector rather than split E1/E2 scalars, and
# use `NGMIX_Tpsf_<suffix>`/`NGMIX_T_PSFo_NOSHEAR` rather than
# `NGMIX_T_PSF_RECONV_<suffix>`/`NGMIX_T_PSF_ORIG_NOSHEAR`. The rest of this
# script assumes the current (split) grammar; synthesize those field names
# here from their old-grammar counterparts when present, so nothing below
# needs to know which vintage of catalogue it's reading. No-op for
# catalogues that already have the current grammar natively.
# MKDEBUG: These changes are not to be merged in a newer version.
if hasattr(dd, "dtype") and dd.dtype.names is not None:
    import numpy.lib.recfunctions as _rfn

    _key_base = shape.upper()
    _new_names, _new_data = [], []

    def _shim_add(new_name, data):
        if new_name not in dd.dtype.names:
            _new_names.append(new_name)
            _new_data.append(data)

    for _suf in ("1M", "1P", "2M", "2P", "NOSHEAR"):
        _ell_col = f"{_key_base}_ELL_{_suf}"
        if _ell_col in dd.dtype.names:
            _shim_add(f"{_key_base}_G1_{_suf}", dd[_ell_col][:, 0])
            _shim_add(f"{_key_base}_G2_{_suf}", dd[_ell_col][:, 1])
        _tpsf_col = f"{_key_base}_Tpsf_{_suf}"
        if _tpsf_col in dd.dtype.names:
            _shim_add(f"{_key_base}_T_PSF_RECONV_{_suf}", dd[_tpsf_col])

    _psfo_col = f"{_key_base}_ELL_PSFo_NOSHEAR"
    if _psfo_col in dd.dtype.names:
        _shim_add(f"{_key_base}_G1_PSF_ORIG_NOSHEAR", dd[_psfo_col][:, 0])
        _shim_add(f"{_key_base}_G2_PSF_ORIG_NOSHEAR", dd[_psfo_col][:, 1])

    _tpsfo_col = f"{_key_base}_T_PSFo_NOSHEAR"
    if _tpsfo_col in dd.dtype.names:
        _shim_add(f"{_key_base}_T_PSF_ORIG_NOSHEAR", dd[_tpsfo_col])

    # Only the NOSHEAR branch error is real data in v1.3.x catalogues (the
    # per-suffix NGMIX_ELL_ERR_{1M,1P,2M,2P} columns exist in the hdf5 dtype
    # but were never populated by create_final_cat.py -- they're zero-filled
    # placeholders, not measured errors -- so deliberately not shimmed here).
    _ellerr_col = f"{_key_base}_ELL_ERR_NOSHEAR"
    if _ellerr_col in dd.dtype.names:
        _shim_add(f"{_key_base}_G1_ERR_NOSHEAR", dd[_ellerr_col][:, 0])
        _shim_add(f"{_key_base}_G2_ERR_NOSHEAR", dd[_ellerr_col][:, 1])

    if _new_names:
        dd = _rfn.append_fields(dd, _new_names, _new_data, usemask=False)

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

    # v1.3 old-grammar compatibility shim (star catalogue).
    # Pre-shapepipe#761 star catalogues name HSM columns with a trailing
    # `_HSM` suffix (e.g. `E1_PSF_HSM`) rather than the current leading
    # `HSM_` prefix (`HSM_E1_PSF`). Strip the astropy FITS_rec wrapper (its
    # own column-name metadata otherwise shadows a plain dtype rename) and
    # rename in place when the old-style names are present. No-op for
    # catalogues that already have the current grammar natively.
    if "HSM_E1_PSF" not in d_star.dtype.names and "E1_PSF_HSM" in d_star.dtype.names:
        import numpy.lib.recfunctions as _rfn

        d_star = np.array(d_star)
        d_star = _rfn.rename_fields(
            d_star,
            {
                "E1_PSF_HSM": "HSM_E1_PSF",
                "E2_PSF_HSM": "HSM_E2_PSF",
                "SIGMA_PSF_HSM": "HSM_SIGMA_PSF",
                "E1_STAR_HSM": "HSM_E1_STAR",
                "E2_STAR_HSM": "HSM_E2_STAR",
                "SIGMA_STAR_HSM": "HSM_SIGMA_STAR",
                "FLAG_PSF_HSM": "HSM_FLAG_PSF",
                "FLAG_STAR_HSM": "HSM_FLAG_STAR",
            },
        )

if star_cat_path:
    print_stats("Stars:", stats_file, verbose=verbose)
    n_tot = spv_cat.print_some_quantities(d_star, stats_file, verbose=verbose)
    spv_cat.print_mean_ellipticity(
        d_star,
        ["HSM_E1_PSF", "HSM_E2_PSF"],
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
        & (dd["NGMIX_E1_PSF_ORIG_NOSHEAR"][ind_star] != -10)
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
