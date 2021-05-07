"""
  
:Name: params.py

:Description: This script contains parameters to run the validation notebook.

:Author: Martin Kilbinger

:Date: 2021

:Package: sp_validation

"""

import os

# Control

## Verbose output
verbose = True


# Paths

## Shapepipe path
BASE = '{}/astro/repositories/github'.format(os.environ['HOME']) 
SP_BASE = '{}/shapepipe'.format(BASE)

## Input tile IDs
path_tile_ID = '{}/aux/CFIS/tiles_202007/tiles_W3.txt'.format(SP_BASE)

## Output file for missing tile ID, can be used as input for re-run
path_missing_ID = './missing_ID.txt'

## Weak-lensing galaxy catalog name
galaxy_cat_path = './final_cat.npy'

## Output plot directory and subdirs
plot_dir = './plots/'
plot_subdirs = ['psf_leak_ngmix', 'psf_leak_galsim']

## Output statistics text file
stats_file_name = 'stats_file.txt'

# Other IO options

## Memory mode, set to None unless very large file
mmap_mode = None


# Survey parameters

## Area of a tile in deg^2
area_tile = 0.25
