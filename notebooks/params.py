"""
  
:Name: params.py

:Description: This script contains parameters to run the validation notebook.

:Author: Martin Kilbinger

:Date: 2021

:Package: sp_validation

"""

import os
import numpy as np


# Control

## Verbose output
verbose = True

## Math output
np.set_printoptions(precision=3, formatter={'float': '{: .3g}'.format})


# Survey parameters

## Field or patch name. Put None if n/a
name = 'P3'
print('Field name = {}'.format(name))

## Area of a tile in deg^2
area_tile = 0.25

## Pixel size in arcsec
pixel_size = 0.187

## Shape measurement method list, implemented are
##  'ngix': multi-epoch model fitting
##  'galsim': stacked-image moments (experimental)
shapes = ['ngmix']
print('Shape measurement method(s):', shapes)

# Paths

## Input paths

### Input data directory
#data_dir = f'{os.environ["HOME"]}/data_WL'
data_dir = '.'

### Tile IDs
path_tile_ID = f'{data_dir}/tiles_{name}.txt'

### Weak-lensing galaxy catalog name
galaxy_cat_path = f'{data_dir}/final_cat.npy'
print(f'Galaxy catalogue = {galaxy_cat_path}')

### Star and PSF catalog name; optional, set to `None` if not required
star_cat_path = f'{data_dir}/full_starcat-0000000.fits'

# HDU number of star and PSF catalogue
hdu_star_cat = 1

### External mask; optional, set to `None` if not required
#mask_external_path = f'{data_dir}/../LensFitMisc/CFIS3500_THELI_{name}_tiles.reg'
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
plot_subdirs = []
for sh in shapes:
    plot_subdirs.append(f'psf_leak_{sh}')
    plot_subdirs.append(f'local_cal_{sh}')

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


# Catalog parameters

## Star matching threshold [deg]
thresh = 0.0002

## Number of jackknife resamples for additive bias
## (0: no jackknife computation).
## If < 2000 the jackknife mean fluctuates a lot. 
n_jack = 0

## Galaxy selection

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


# Correlation parameters

## Minimum and maximum angular scales, in arcmin
theta_min_amin = 1
theta_max_amin = 350

## Number of bins
n_theta = 20


# Plotting parameters

## Wrap coordinates around this value [deg], set to != 0 if ra=0 is within coordinate range
wrap_ra = 0

## PSF leakage y-axis limits
leakage_alpha_ylim = [-0.1, 0.1]
leakage_xi_sys_ylim = [-4e-5, 5e-5]
leakage_xi_sys_log_ylim = [2e-13, 5e-5]

# Maps parameters

## Pixel size of ellipticty maps in arc minutes
pixel_size_emap_amin = 0.4

## Pixel size of smoothed convergence map, in pixels
## of size pixel_size_emap_amin
smoothing_scale_pix = 20

# cutout map around specific coordinates, optional                              
map_cut_coords = [112, 154, 41, 31]

## Sign of shear components, to correct for left-handed
## coordinate system
g1_sign = +1
g2_sign = -1

# Cosmology

## Basic cosmological parameters
Om = 0.3153
sig8 = 0.8111
ns = 0.9649
Ob = 0.0493
h = 0.6736
