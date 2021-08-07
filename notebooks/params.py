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
name = 'W3'
print('Field name = {}'.format(name))

## Area of a tile in deg^2
area_tile = 0.25

## Pixel size in arcsec
pixel_size = 0.187


# Paths

## Input paths

### Input data directory
data_dir = f'{os.environ["HOME"]}/data_WL'
#data_dir = '.'

### Tile IDs
path_tile_ID = f'{data_dir}/tiles_{name}.txt'

### Weak-lensing galaxy catalog name
galaxy_cat_path = f'{data_dir}/final_cat.npy'

### Star and PSF catalog name
star_cat_path = f'{data_dir}/psf_validation_merged/psf_cat_full.fits'

## Output paths

### Output base directory
output_dir = f'{data_dir}/sp_output'

### Galaxy shape catalogue name
output_shape_cat_path = f'{output_dir}/shape_catalog.fits'

### File for missing tile ID, can be used as input for re-run
path_missing_ID = f'{output_dir}/missing_ID.txt'

### Plot directory and subdirs
plot_dir = f'{output_dir}/plots/'
plot_subdirs = ['psf_leak_ngmix', 'psf_leak_galsim', 'local_cal_ngmix']

### Statistics text file
stats_file_name = 'stats_file.txt'

# Other IO options

## Memory mode, set to None unless very large file
mmap_mode = None


# Catalog parameters

## Star matching threshold [deg]
thresh = 0.0002

## Galaxy selection for metacal

### Signal-to-noise
gal_snr_min = 10
gal_snr_max = 500

### Relative size, T_gal / T_psf
gal_rel_size_min = 0.5

# Plotting parameters

## PSF leakage limits
leakage_alpha_ylim = [-0.1, 0.065]
leakage_xi_sys_ylim = [-4e-5, 5e-5]
leakage_xi_sys_log_ylim = [2e-13, 5e-5]

# Maps parameters

## Pixel size of ellipticty maps in arc minutes
pixel_size_emap_amin = 0.4

## Pixel size of smoothed convergence map, in pixels
## of size pixel_size_emap_amin
smoothing_scale_pix = 20

## Sign of shear components, to correct for lef-handed
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
