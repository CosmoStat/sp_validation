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
import sys
import os
import numpy as np
from astropy.io import fits

from cs_util import canfar
from sp_validation.io import *
#from sp_validation.cat import *
from sp_validation import cat as spv_cat
from sp_validation.survey import *
from sp_validation.galaxy import *
from sp_validation.calibration import *
from sp_validation.basic import *

# ## 1. Set-up

# Load parameters
sys.path.insert(0, os.getcwd())
from params import *

# ### Create and open output files and directories

make_out_dirs(output_dir, plot_dir, [], verbose=verbose)
stats_file = open_stats_file(plot_dir, stats_file_name)

# ## 2. Load data
#
# ### Load merged (final) galaxy catalogue

# +
extension = os.path.splitext(galaxy_cat_path)[1]
if extension == ".fits":
    print("Loading galaxy .npy file...")
    dd = np.load(galaxy_cat_path, mmap_mode=mmap_mode)
else:
    print("Loading galaxy .hdf5 file...")
    dd = spv_cat.read_hdf5_file(galaxy_cat_path, name, stats_file, param_path=param_list_path)

n_obj = len(dd)
print_stats(f'Read {n_obj} objects from file {galaxy_cat_path}', stats_file, verbose=verbose)
# -

# #### Print some quantities to check nothing obvious is wrong with catalogue

# PSF keys
key_base = shape.upper()
key_PSF_ell = f'{key_base}_ELL_PSFo_NOSHEAR'
key_PSF_size = f'{key_base}_T_PSFo_NOSHEAR'
size_to_fwhm = T_to_fwhm

print_stats('Galaxies:', stats_file, verbose=verbose)
n_tot = spv_cat.print_some_quantities(dd, stats_file, verbose=verbose)
spv_cat.print_mean_ellipticity(
    dd,
    f'{key_base}_ELL_NOSHEAR',
    2, 
    n_tot,
    stats_file,
    invalid=-10,
    verbose=verbose
)

# #### Survey area and potential missing tiles
# The approximate observed area is the number of tiles $\times$ 0.25 deg$^2$ (ignoring overlaps and masking).

area_deg2, area_amin2, tile_IDs = get_area(dd, area_tile, verbose=verbose)

# Identify missing tiles by comparing tile ID from catalogue to external input tile ID file.

n_found, n_missing = missing_tiles(tile_IDs, path_tile_ID, path_found_ID, path_missing_ID, verbose=verbose)

# ### Load star catalogue

if star_cat_path:
    d_star = fits.getdata(star_cat_path, hdu_star_cat)

if star_cat_path:
    print_stats('Stars:', stats_file, verbose=verbose)
    n_tot = spv_cat.print_some_quantities(d_star, stats_file, verbose=verbose)
    spv_cat.print_mean_ellipticity(
        d_star, 
        ['E1_PSF_HSM', 'E2_PSF_HSM'],
        1,
        n_tot,
        stats_file, 
        invalid=-10,
        verbose=verbose
    )

# ### 3. Matching of stars

# ### Matching of star catalogues
# Match the star catalogue `d_star` (selected on individual exposures using size-magnitude diagram) to catalogue from tile. Uses some simple criteria to select stars from tile catalogue such as SPREAD_CLASS.
#
# This is mainly for testing, this match will not be used later.

# #### Match to all objects

if star_cat_path:
    ind_star, mask_area_tiles, n_star_tot = spv_cat.check_matching(
        d_star,
        dd,
        ['RA', 'DEC'],
        [col_name_ra, col_name_dec],
        thresh,
        stats_file,
        name=None,
        verbose=verbose
    )

# #### Refine: Match to valid, unflagged galaxy sample

# +
# Flags to indicate valid star sample

m_star = (
    (dd['FLAGS'][ind_star] == 0)
    & (dd['IMAFLAGS_ISO'][ind_star] == 0)
    & (dd['NGMIX_MCAL_FLAGS'][ind_star] == 0)
    & (dd['NGMIX_ELL_PSFo_NOSHEAR'][:,0][ind_star] != -10)
)

ra_star, dec_star, g_star_psf = spv_cat.match_subsample(
    dd,
    ind_star,
    m_star,
    [col_name_ra, col_name_dec],
    key_PSF_ell,
    n_star_tot,
    stats_file,
    verbose=verbose
)
# -

#### Refine: Match to SPREAD_CLASS samples
if "SPREAD_CLASS" in dd.dtype.names:
    spv_cat.match_spread_class(dd, ind_star, m_star, stats_file, len(ra_star), verbose=verbose)
else:
    print_stats("No SPREAD_CLASS in input, skipping star-gal matching", stats_file, verbose=verbose)

# ## Check for objects with invalid PSF

spv_cat.check_invalid(
    dd,
    [key_PSF_ell, f'{key_base}_ELL_NOSHEAR'],
    [0, 0],
    [-10, -10],
    stats_file,
    name=['`PSF', 'galaxy ellipticity'],
    verbose=verbose
)

# +
# Flag for duplicate objects in tile boundaries

cut_overlap = classification_galaxy_overlap_ra_dec(
    dd,
    ra_key=col_name_ra,
    dec_key=col_name_dec,
)

n_ok = sum(cut_overlap)
print_stats(f"Non-overlapping objects: {n_ok:10d}, {n_ok/n_obj:10.2%}", stats_file, verbose=verbose)

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
ext_cols_pre_cal['e1_PSF'] = dd[key_PSF_ell][:,0]
ext_cols_pre_cal['e2_PSF'] = dd[key_PSF_ell][:,1]
ext_cols_pre_cal['fwhm_PSF'] = size_to_fwhm(dd[key_PSF_size])

_, _, iv_w = metacal.get_variance_ivweights(dd, sigma_eps_prior, mask=None, col_2d=True)

mag = spv_cat.get_col(dd, "MAG_AUTO", None, None)
snr = spv_cat.get_snr(shape, dd, None, None)
g1_uncal = dd[f"{key_base}_ELL_NOSHEAR"][:, 0]
g2_uncal = dd[f"{key_base}_ELL_NOSHEAR"][:, 1]
    
# Comprehensive catalogue without cuts nor mask applied
if verbose:
    print("Writing comprehensive catalogue...")

spv_cat.write_shape_catalog(
    f'{output_shape_cat_base}_comprehensive_{shape}.fits',
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
# -

if not do_selection_calibration:
    if verbose:
        print("No selection and calibration; exiting here")
    sys.exit(0)
else:
    if verbose:
        print("Continuing with selection and calibration")

# ## 4. Select galaxies

# #### Common flags and cuts
# First, set cuts common to ngmix and galsim:
#   - spread model: select objects well larger than the PSF
#   - magnitude: cut galaxies that are too faint (= too noisy, likely to be
#     artefacts), and too bright (might be too large for postage stamp)
#   - flags: cut objects that were flagged as invalid or masked
#   - n_epoch: select objects observed on at leatst one epoch (for safety,
#     to avoid potential errors with empty data)

# +
cut_common = classification_galaxy_base(
    dd,
    cut_overlap,
    gal_mag_bright=gal_mag_bright,
    gal_mag_faint=gal_mag_faint,
    flags_keep=flags_keep,
    n_epoch_min=n_epoch_min,
    do_spread_model=do_spread_model,
)
if shape == "ngmix":
    m_gal = classification_galaxy_ngmix(
        dd,
        cut_common,
        stats_file,
        verbose=verbose,
    )
else:
    raise ValueError(f"Invalid shape measurement method {shape}")

n_ok = sum(cut_common)
print_stats(f"objects after common cut: {n_ok:10d}, {n_ok/n_obj:3.2%}", stats_file, verbose=verbose)

# MKDEBUG for debugging calibrate_comprehensive
n_ok = sum(m_gal)
print_stats(f"common & ngmix = galaxy selection: {n_ok:10d}, {n_ok/n_obj:3.2%}", stats_file, verbose=verbose)
# -

# ### Metacal global

# +
import os
from uncertainties import ufloat

from lenspack.geometry.projections.gnom import radec2xy

from cs_util.plots import plot_histograms
# -

from sp_validation.survey import *
from sp_validation.util import *
from sp_validation.basic import *
from sp_validation.plots import *
from sp_validation.plot_style import *
from sp_validation.calibration import *

# ## metacalibration for galaxies

# +
gal_metacal = metacal(
    dd,
    m_gal,
    prefix=key_base,
    snr_min=gal_snr_min,
    snr_max=gal_snr_max,
    rel_size_min=gal_rel_size_min,
    rel_size_max=gal_rel_size_max,
    size_corr_ell=gal_size_corr_ell,
    sigma_eps=sigma_eps_prior,
    verbose=verbose
)

print_stats(
    f"Number of objects on metal input = {gal_metacal._n_input}",
    stats_file,
    verbose=verbose,
)
print_stats(
    f"Number of objects after galaxy selection masking = {gal_metacal._n_after_gal_mask}",
    stats_file,
    verbose=verbose,
)
# -

# #### Extract quantities after calibration and cuts (in metacal)

# +
g_corr, g_uncorr, w, mask = (
    get_calibrated_quantities(gal_metacal)
)

n_before_cuts = len(gal_metacal.ns['g1'])
n_after_cuts = len(w)
print_stats(
    f"Number of objects before cuts = {n_before_cuts}",
    stats_file,
    verbose=verbose
)
print_stats(
    f"Number of objects after cuts = {n_after_cuts}",
    stats_file,
    verbose=verbose
)
    
# coordinates
ra = spv_cat.get_col(dd, col_name_ra, m_gal, mask) 
dec = spv_cat.get_col(dd, col_name_dec, m_gal, mask)

# Modify R.A. for plots if R.A. = 0 in area
if wrap_ra != 0:
    ra_wrap = (ra + wrap_ra) % 360 - wrap_ra + 360
else:
    ra_wrap = ra

ra_mean = np.mean(ra_wrap)
dec_mean = np.mean(dec)
    
print_stats(
    f'Mean coordinates (ra, dec) ='
    + f' ({ra_mean:.3f}, {dec_mean:.3f}) deg,'
    + f' wrap_ra={wrap_ra} deg',
    stats_file,
    verbose=verbose
)

# magnitude, from SExtractor
mag = spv_cat.get_col(dd, "MAG_AUTO", m_gal, mask)
    
# Keep tile ID if external mask
if mask_external_path:
    tile_ID = spv_cat.get_col(dd, "TILE_ID", m_gal, mask)

snr = spv_cat.get_snr(shape, dd, m_gal, mask)

# +
# Add additional columns with metacal mask

if add_cols:
    add_cols_data = {}
    for key in add_cols:
        add_cols_data[key] = dd[key][m_gal][mask]
else:
    add_cols_data = None                               

# +
#### Compute coordinates for projection and spatial binning (needed later)
# -

# Project all objects from spherical to Cartesian coordinates
x, y =  radec2xy(ra_mean, dec_mean, ra_wrap, dec)

# #### Compute field size

# +
# Define mix, max and size
min_x = np.min(x)
max_x = np.max(x)
min_y = np.min(y)
max_y = np.max(y)

size_x_deg = np.rad2deg(max_x - min_x)
size_y_deg = np.rad2deg(max_y - min_y)

print_stats(
    f'Field size in projected coordinates is (x, y) '
    + f'= ({size_x_deg:.2f}, {size_y_deg:.2f}) deg',
    stats_file,
    verbose=verbose
)

# +
# Number density
n_gal = len(w)
n_gal_sm = len(np.where(m_gal)[0])

print_stats(
    f'Number of galaxies after metacal = {n_gal}/{n_gal_sm} '
    + f'= {n_gal / n_gal_sm * 100:.1f}%',
    stats_file,
    verbose=verbose
)
print_stats(
    f'Galaxy density (ignoring masks+overlaps)= {n_gal / area_amin2:.2f} gal/arcmin2',
    stats_file,
    verbose=verbose
)
# -

# Write number of galaxies for each tile to file
fname = f'{output_dir}/tile_id_gal_counts_{shape}.txt'
detection_IDs = dd['TILE_ID']
galaxy_IDs = detection_IDs[m_gal]
shape_IDs = galaxy_IDs[mask]
write_tile_id_gal_counts(detection_IDs, galaxy_IDs, shape_IDs, fname) 

# +
# Add all weights (for combining weighted averages of subpatches)

w_tot = np.sum(w)
    
print_stats(f'Sum of weights = {w_tot:.1f}', stats_file, verbose=verbose)

# +
# Effective sample size ESS = 1/sum(w_n^2)
# The inverse sum over squared normalised weights
# Range [1; N]
    
# normalised weights
wn = w / w_tot
s = np.sum(wn**2)
ess = 1/s
    
print_stats(f'effective sample size, ESS/N = {ess:.1f}/{n_gal} = {ess/n_gal:.3g}',
            stats_file, verbose=verbose)
# -

# #### Plot spatial distribution of objects

x_label = 'R.A. [deg]'
y_label = 'DEC [deg]'
cbar_label_base = 'Density [$A_{\\rm pix}^{-1}$]'

len(dec)

# +
# Galaxies
Apix = 1 # [arcmin^2]
cbar_label = '{}, $A_{{\\rm pix}} \\approx {:.1g}$ arcmin$^2$'.format(cbar_label_base, Apix)
n_grid = int(np.sqrt(area_amin2) / Apix)
if verbose:
    print('Number of pixels = {}^2'.format(n_grid))

title = f'Galaxies ({shape})'
out_path = f'{plot_dir}/galaxy_number_count_{shape}'
plot_spatial_density(
    ra_wrap,
    dec,
    title,
    x_label,
    y_label,
    cbar_label,
    out_path,
    n_grid=n_grid,
    verbose=verbose
)

# All objects without overlap, useful for position-only
# setting (no shapes)
title = f'Galaxies (all, no overlap)'
out_path = f'{plot_dir}/galaxy_number_count_all_nooverlap'
plot_spatial_density(
    ra_wrap_all[cut_overlap],
    dec_all[cut_overlap],
    title,
    x_label,
    y_label,
    cbar_label,
    out_path,
    n_grid=n_grid,
    verbose=verbose
)   
# -

# #### Plot galaxy signal-to-noise distribution

# +
x_label = 'SNR'
y_label = 'Frequency'
density = True
x_range = (0, 200)
n_bin = 500
x_cut = gal_snr_min

labels = []
if shape == 'ngmix':
# Do not apply `mask_ns`, so use all galaxies
    xs = [
        dd['NGMIX_FLUX_NOSHEAR'][m_gal] / dd['NGMIX_FLUX_ERR_NOSHEAR'][m_gal],
        dd['SNR_WIN'][m_gal]
    ]
    labels.append([f'$F/\\sigma(F)$'])

else:
    raise ValueError(f"Unknown shape measurement method {shape}")
    
labels.append(f'SExtractor SNR')

title = f'Galaxies'

out_name = f'hist_SNR_{shape}.pdf'
out_path = os.path.join(plot_dir, out_name)

plot_histograms(
    xs,
    labels,
    title,
    x_label,
    y_label,
    x_range,
    n_bin,
    out_path,
    vline_x=[x_cut],
    vline_lab=[f'SNR = {x_cut}']
)
# -

# ### Plot galaxy relative size distribution

# +
x_label = r'$R^2_{\rm gal} / R^2_{\rm PSF}$'
y_label = 'Frequency'
density = True
x_range = (0, 1.5)
n_bin = 500
x_cut = gal_rel_size_min

labels = []
if shape == 'ngmix':
    # Do not apply `mask_ns`, so use all galaxies
    xs = [
        dd['NGMIX_T_NOSHEAR'][m_gal] / dd['NGMIX_Tpsf_NOSHEAR'][m_gal]
    ]
    labels.append(f'size ratio')

else:
    raise ValueError(f"Unknown shape measurement method {shape}")

title = f'Galaxies'

out_name = f'hist_rel_size_{shape}.pdf'
out_path = os.path.join(plot_dir, out_name)

plot_histograms(
    xs,
    labels,
    title,
    x_label,
    y_label,
    x_range,
    n_bin,
    out_path,
    vline_x=[x_cut],
    vline_lab=[f'rel_size = {x_cut}']
)
# -

# ## Metacalibration for stars

star_metacal = metacal(
    dd[ind_star],
    m_star,
    masking_type='star',
    verbose=verbose
)

# #### Number density

# +
# mask for 'no shear' images

mask_ns_stars = star_metacal.mask_dict['ns']
n_star = len(star_metacal.ns['g1'][mask_ns_stars])

print_stats(f'Number of stars = {n_star}', stats_file, verbose=verbose)
print_stats('Star density = {:.2f} stars/deg2'.format(n_star / area_deg2), stats_file, verbose=verbose)
# -

# ## Additive bias
# Use raw, uncorrected ellipticities.

print_stats('additive bias', stats_file, verbose=verbose)

# +
# Compute mean, weighted mean, and (Poisson) error
    
c = np.zeros(2)
c_err = np.zeros(2)
cw = np.zeros(2)
cw_err = np.zeros(2)

for comp in (0, 1):
    c[comp] = np.average(g_uncorr[comp])
    c_err[comp] = np.std(g_uncorr[comp])
        
    cw[comp] = np.average(g_uncorr[comp], weights=w)
    variance = np.average((g_uncorr[comp] - cw[comp])**2, weights=w)
    cw_err[comp] = np.sqrt(variance)

for comp in (0, 1):
    print_stats(f'c_{comp+1} = {c[comp]:.3g}', stats_file, verbose=verbose)        
    print_stats(f'cw_{comp+1} = {cw[comp]:.3g}', stats_file, verbose=verbose)
for comp in (0, 1):
    print_stats(f'dc_{comp+1} = {c_err[comp]:.3g}', stats_file, verbose=verbose)        
    print_stats(f'dcw_{comp+1} = {cw_err[comp]:.3g}', stats_file, verbose=verbose)

# Error of mean: divide by sqrt(N) (TBC whether this is correct)
for comp in (0, 1):
    print_stats(f'dmc_{comp+1} = {c_err[comp]/np.sqrt(n_gal):.3e}', stats_file, verbose=verbose)
    print_stats(
        f'dmcw_{comp+1} = {cw_err[comp]/np.sqrt(n_gal):.3e}',
        stats_file,
        verbose=verbose
    )

# +
# Compute jackknife mean and errors

remove_size = 0.05
    
cjk = np.zeros(2) * -1
cjk_err = np.zeros(2) * -1

for comp in (0, 1):
    if n_jack > 0:
        cjk[comp], cjk_err[comp] = jackknif_weighted_average(
            g_uncorr[comp],
            w,
            remove_size=remove_size,
            n_realization=n_jack
        )           
    cjk_dc = ufloat(cjk[comp], cjk_err[comp])
    print_stats(f'cjk_{comp+1} = {cjk_dc:.3eP}', stats_file, verbose=verbose)
# -

# ## Get quantities calibrated for both multiplicative and additive bias

g_corr_mc = np.zeros_like(g_corr)
c_corr = np.linalg.inv(gal_metacal.R).dot(c)
for comp in (0, 1):
    g_corr_mc[comp] = g_corr[comp] - c_corr[comp]

# ## Response matrix

# ### Mean

# +
print_stats('total response matrix:', stats_file, verbose=verbose)
rs = np.array2string(gal_metacal.R)
print_stats(rs, stats_file, verbose=verbose)

print_stats('shear response matrix:', stats_file, verbose=verbose)
R_shear = np.mean(gal_metacal.R_shear, 2)
rs = np.array2string(R_shear)
print_stats(rs, stats_file, verbose=verbose)

print_stats('selection response matrix:', stats_file, verbose=verbose)
rs = np.array2string(gal_metacal.R_selection)
print_stats(rs, stats_file, verbose=verbose)

# +
print_stats("stars:", stats_file, verbose=verbose)

print_stats('total response matrix:', stats_file, verbose=verbose)
rs = np.array2string(star_metacal.R)
print_stats(rs, stats_file, verbose=verbose)

print_stats('shear response matrix:', stats_file, verbose=verbose)
R_shear_stars = np.mean(star_metacal.R_shear, 2)
rs = np.array2string(R_shear_stars)
print_stats(rs, stats_file, verbose=verbose)

print_stats('selection response matrix:', stats_file, verbose=verbose)
rs = np.array2string(star_metacal.R_selection)
print_stats(rs, stats_file, verbose=verbose)
# -

# ### Plot distribution of response matrix elements

x_label = 'response matrix element'
y_label = 'Frequency'
x_range = (-3, 3)
n_bin = 500

colors = ['blue', 'red','blue', 'red']
linestyles = ['-', '-', ':', ':']

# +
labels = [
    '$R_{11}$ galaxies',
    '$R_{22}$ galaxies',
    '$R_{11}$ stars',
    '$R_{22}$ stars'
]

xs = [
    gal_metacal.R_shear[0,0],
    gal_metacal.R_shear[1,1],
    star_metacal.R_shear[0,0],
    star_metacal.R_shear[1,1]
]
title = shape
    
out_name = f'R_{shape}_diag.pdf'
out_path = os.path.join(plot_dir, out_name)
    
plot_histograms(
    xs,
    labels,
    title,
    x_label,
    y_label,
    x_range,
    n_bin,
    out_path,
    colors=colors,
    linestyles=linestyles
)

# +
labels = [
    '$R_{12}$ galaxies',
    '$R_{21}$ galaxies',
    '$R_{12}$ stars',
    '$R_{21}$ stars'
]

xs = [gal_metacal.R_shear[0,1],
      gal_metacal.R_shear[1,0],
      star_metacal.R_shear[0,1],
      star_metacal.R_shear[1,0]
     ]
title = shape
out_name = f'R_{shape}_offdiag.pdf'
out_path = os.path.join(plot_dir, out_name)

plot_histograms(
    xs,
    labels,
    title,
    x_label,
    y_label,
    x_range, 
    n_bin,
    out_path,
    colors=colors,
    linestyles=linestyles
)
# -

# ## Ellipticities

# +
x_label = 'ellipticity'
y_label = 'Frequency'
x_range = (-1, 1)
n_bin = 500

labels = ['$e_1$', '$e_2$']
colors = ['blue', 'red']
linestyles = ['-', '-'] 

# +
xs = [g_corr[0], g_corr[1]]
weights = [w] * 2

title = f'{shape} galaxies'
out_name = f'ell_gal_{shape}.pdf'
out_path = os.path.join(plot_dir, out_name)

plot_histograms(
    xs, 
    labels, 
    title, 
    x_label, 
    y_label, 
    x_range, 
    n_bin,
    out_path,
    weights=weights, 
    colors=colors, 
    linestyles=linestyles
)

# +
xs = [star_metacal.ns['g1'][mask_ns_stars], star_metacal.ns['g2'][mask_ns_stars]]
weights = [star_metacal.ns['w'][mask_ns_stars]] * 2

title = "stars"
out_name = f'ell_stars_{shape}.pdf'
out_path = os.path.join(plot_dir, out_name)

plot_histograms(
    xs, 
    labels, 
    title, 
    x_label, 
    y_label, 
    x_range, 
    n_bin, 
    out_path,
    weights=weights, 
    colors=colors, 
    linestyles=linestyles
)
# -

x_range = (-0.15, 0.15)
n_bin = 250

# +
key = key_PSF_ell
xs = [
    dd[key][:,0][mask_ns_stars],
    dd[key][:,1][mask_ns_stars]
]
title = "PSF"
out_name = f'ell_PSF_{shape}.pdf'
out_path = os.path.join(plot_dir, out_name)

plot_histograms(
    xs, 
    labels, 
    title, 
    x_label, 
    y_label, 
    x_range, 
    n_bin, 
    out_path,      
    colors=colors, 
    linestyles=linestyles
)
# -

# ## Magnitudes

# +
x_label = '$r$-band magnitude'
y_label = 'Frequency'
x_range = (gal_mag_bright + 1, gal_mag_faint - 1)
n_bin = 500

colors = ['blue', 'red']
linestyles = ['-', '-']

title = 'galaxies'
out_name = 'mag_gal.pdf'
out_path = os.path.join(plot_dir, out_name)

# +
labels = [shape]
xs = [dd['MAG_AUTO'][m_gal][mask]]

plot_histograms(
    xs, 
    labels, 
    title, 
    x_label, 
    y_label, 
    x_range, 
    n_bin, 
    out_path,             
    colors=colors, 
    linestyles=linestyles
)
# -

# ## Ellipticity dispersion

sig_eps = np.sqrt(np.var(g_corr[0]) + np.var(g_corr[1]))
print_stats('Dispersion of complex ellipticity = {:.3f}' \
            ''.format(sig_eps), stats_file, verbose=verbose)
print_stats('Dispersion of (average) single-component ellipticity = {:.3f} = {:.3f} / sqrt(2)' \
            ''.format(sig_eps /  np.sqrt(2), sig_eps), stats_file, verbose=verbose)

# ## Write catalogues

import os

from sp_validation.util import *
from sp_validation.cat import *
from sp_validation.basic import metacal

# Shear response per galaxy
R_shear_ind = gal_metacal.R_shear

# ### Write basic shape catalogue

spv_cat.write_shape_catalog(
    f'{output_shape_cat_base}_{shape}.fits',
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
    add_cols=add_cols_data,
)

# ### Write extended shape catalogue

ext_cols = {}
if add_cols:
    ext_cols = add_cols_data
else:
    ext_cols = {}

# Optional: Create flag from external mask
if mask_external_path:
    m_extern = mask_overlap(ra, dec, tile_ID, mask_external_path)

# +
# Additional columns:
# {e1, e2, size}_PSF
ext_cols['e1_PSF'] = dd[key_PSF_ell][:,0][m_gal][mask]
ext_cols['e2_PSF'] = dd[key_PSF_ell][:,1][m_gal][mask]
ext_cols['fwhm_PSF'] = size_to_fwhm(dd[key_PSF_size][m_gal][mask])
if mask_external_path:
    ext_cols['mask_extern'] = m_extern

# Extended catalogue with SNR, individual R matrices, ext_cols
spv_cat.write_shape_catalog(
    f'{output_shape_cat_base}_extended_{shape}.fits',
    ra,
    dec,
    w,
    mag=mag,
    snr=snr,
    g=g_corr_mc,
    g1_uncal=g_uncorr[0],
    g2_uncal=g_uncorr[1],
    R_g11=R_shear_ind[0, 0],
    R_g12=R_shear_ind[0, 1],
    R_g21=R_shear_ind[1, 0],
    R_g22=R_shear_ind[1, 1],       
    R=gal_metacal.R,
    R_shear=R_shear,
    R_select=gal_metacal.R_selection,
    c=c,
    c_err=c_err,
    add_cols=ext_cols,
 )
# -

# ### Write galaxy (or random) position catalogue

if shape == "":
    print('writing random cat (hack)')
        
    ra = dd['RA'][cut_overlap]
    dec = dd['DEC'][cut_overlap]
    tile_id = dd['TILE_ID'][cut_overlap]
    write_galaxy_cat(f'{output_shape_cat_base}.fits', ra, dec, tile_id)

# ### Write PSF catalogue with multi-epoch shapes from shape measurement methods

write_PSF_cat(                                         
    f'{output_PSF_cat_base}_{shape}.fits',
    ra_star,
    dec_star,
    g_star_psf[0],
    g_star_psf[1],
) 
