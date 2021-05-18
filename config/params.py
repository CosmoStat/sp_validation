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


# Paths

## Shapepipe path
BASE = '{}/astro/repositories/github'.format(os.environ['HOME']) 
SP_BASE = '{}/shapepipe'.format(BASE)

## Input paths

### Tile IDs
path_tile_ID = '{}/aux/CFIS/tiles_202007/tiles_W3.txt'.format(SP_BASE)

### Weak-lensing galaxy catalog name
galaxy_cat_path = './final_cat.npy'

### Star and PSF catalog name
star_cat_path = './psf_validation_merged/psf_cat_full.fits'

## Output paths

### Galaxy shape catalogue name
output_shape_cat_path = './shape_catalog.fits'

### File for missing tile ID, can be used as input for re-run
path_missing_ID = './missing_ID.txt'

### Plot directory and subdirs
plot_dir = './plots/'
plot_subdirs = ['psf_leak_ngmix', 'psf_leak_galsim']

### Statistics text file
stats_file_name = 'stats_file.txt'

# Other IO options

## Memory mode, set to None unless very large file
mmap_mode = None


# Survey parameters

## Field or patch name. Put None if n/a
name = 'W3'
print('Field name = {}'.format(name))

## Area of a tile in deg^2
area_tile = 0.25

## Pixel size in arcsec
pixel_size = 0.187


# Catalog parameters

## Matching threshold [deg]
thresh = 0.0002
