"""
  
:Name: params.py

:Description: This script contains parameters to run the validation notebook.

:Author: Martin Kilbinger

:Date: 2021

:Package: sp_validation

"""

import numpy as np


# Control

## Verbose output
verbose = True

## Math output
np.set_printoptions(precision=3, formatter={'float': '{: .3g}'.format})


# Survey parameters

## Field or patch name. Put None if n/a
name = 'P7'
print('Field name = {}'.format(name))

## Area of a tile in deg^2
area_tile = 0.25

## Pixel size in arcsec
pixel_size = 0.187

## Shape measurement method, implemented is
##  'ngix': multi-epoch model fitting
##  'galsim': stacked-image moments (experimental)
shape = 'ngmix'

# Paths

## Input paths

### Input data directory
data_dir = '.'

### Tile IDs
path_tile_ID = f'{data_dir}/tiles_{name}.txt'

### Weak-lensing galaxy catalog name
galaxy_cat_path = f'{data_dir}/final_cat_{name}.hdf5'
print(f'Galaxy catalogue = {galaxy_cat_path}')

## Parameter list; optional, set to `None` if not required
param_list_path = f"{data_dir}/cfis/final_cat.param"

### Star and PSF catalog name; optional, set to `None` if not required
star_cat_path = f'{data_dir}/full_starcat-0000000.fits'

# HDU number of star and PSF catalogue
hdu_star_cat = 1

### External mask; optional, set to `None` if not required
mask_external_path = None

## Output paths

### Output base directory
output_dir = f'{data_dir}/sp_output'

### Galaxy shape catalogue base name.
### Will be appended by
### - '_{sh}.fits' for the basic catalogue
### - 'extended_{sh}.fits' for the extended catalogue
output_shape_cat_base= f'{output_dir}/shape_catalog'

### PSF output catalogue base name.
### Will be appended by '_{sh}.fits'
output_PSF_cat_base = f'{output_dir}/psf_catalog'

### File for found tile IDs
path_found_ID = f'{output_dir}/found_ID.txt'

### File for missing tile IDs
path_missing_ID = f'{output_dir}/missing_ID.txt'

### Plot directory and subdirs
plot_dir = f'{output_dir}/plots/'

### Statistics text file
stats_file_name = 'stats_file.txt'

# Other IO options

## Input

### Coordinate column names
col_name_ra = 'XWIN_WORLD'
col_name_dec = 'YWIN_WORLD'

### Memory mode, set to None unless very large file
mmap_mode = None

## Output

### Additional output columns
add_cols = ["FLUX_RADIUS", "FWHM_IMAGE", "FWHM_WORLD", "MAGERR_AUTO", "MAG_WIN", "MAGERR_WIN", "FLUX_AUTO", "FLUXERR_AUTO", "FLUX_APER", "FLUXERR_APER", "NGMIX_T_NOSHEAR", "NGMIX_Tpsf_NOSHEAR"]

## Pre-calibration catalogue, including masked objects and mask flags
add_cols_pre_cal = ["TILE_ID", "NUMBER", "IMAFLAGS_ISO", "FLAGS", "NGMIX_MCAL_FLAGS", "NGMIX_MOM_FAIL", "N_EPOCH", "NGMIX_N_EPOCH", "NGMIX_ELL_PSFo_NOSHEAR", "NGMIX_ELL_ERR_NOSHEAR"]

### Set flag columns as integer format
add_cols_pre_cal_format = {}
for key in ("NUMBER", "IMAFLAGS_ISO", "FLAGS", "NGMIX_MCAL_FLAGS", "NGMIX_MOM_FAIL", "N_EPOCH", "NGMIX_N_EPOCH"):
    add_cols_pre_cal_format[key] = "I"
    
add_cols_pre_cal_format["TILE_ID"] = "A7"
add_cols_pre_cal_format["NUMBER"] = "J"

# Create key names for metacal information
prefix = "NGMIX"
suffixes = ["1M", "1P", "2M", "2P", "NOSHEAR"]
centers = ["FLAGS", "ELL", "FLUX", "FLUX_ERR", "T", "T_ERR", "Tpsf"]
for center in centers:
    for suffix in suffixes:
        add_cols_pre_cal.append(f"{prefix}_{center}_{suffix}")
        
for suffix in suffixes:
    add_cols_pre_cal_format[f"FLAGS_{suffix}"] = "I"


# Catalog parameters

## Star matching threshold [deg]
thresh = 0.0002

## Number of jackknife resamples for additive bias
## (0: no jackknife computation).
## If < 2000 the jackknife mean fluctuates a lot. 
n_jack = 0


## Galaxy selection

# Flag to output selected and calibrated galaxy catalogue (<= SP v1.4.1).       
# If False, only output comprehensive catalogue.                                
do_selection_calibration = False 

## Magnitude limits
gal_mag_bright = 15
gal_mag_faint = 30

### Spread-model
do_spread_model = False

### SExtractor flags to keep in addition to FLAGS=0
### (bit-coded; list of powers of 2);
### Empty list if no flags
flags_keep = []

## Minimum number of epochs
n_epoch_min = 2

### Signal-to-noise (selection within metacal)
#### minimum to cut noisy objects
gal_snr_min = 10
#### maximum to cut too bright objects, potentially too large for the postage stamp
gal_snr_max = 500

### Relative size, T_gal / T_psf (selection within metacal)
### to select objects that are not too small compared to the PSF, thus not likely to be point-like,
### or to big as they seem to bias the correlation functions
gal_rel_size_min = 0.5
gal_rel_size_max = 3.

### Correct galaxy size for ellipticity
gal_size_corr_ell = False

### prior ellipticity dispersion (one component), *only* used for galaxy weight
sigma_eps_prior = 0.34


## Wrap coordinates around this value [deg], set to != 0 if ra=0 is within coordinate range
wrap_ra = 0
