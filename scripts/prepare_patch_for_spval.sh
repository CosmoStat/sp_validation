#!/usr/bin/env bash

patch=$1

spdir=$HOME/astro/repositories/github/sp_validation

# Galaxy catalogue
cp ~/psfex/final_cat_${patch}.hdf5 .

# Parameter file, to avoid read errors for hdf5 file
ln -sf ~/shapepipe/example/cfis/final_cat.param

# Star catalogue
ln -sf $HOME/psfex/P3/output/run_sp_Ms/merge_starcat_runner/output/full_starcat-0000000.fits

# Parameter file
cp $spdir/notebooks/params.py .

# nb/python script
ln -sf $spdir/notebooks/validation.py

# Tile number list
ln -sf ~/shapepipe/auxdir/CFIS/tiles_202106/tiles_${patch}.txt

