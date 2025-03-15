#!/usr/bin/env bash

patch=$1

spdir=$HOME/astro/repositories/github/sp_validation

# Galaxy catalogue
#cp ~/psfex/final_cat_${patch}.hdf5 .
ln -s ~/psfex/final_cat_${patch}.hdf5

# Parameter file, to avoid read errors for hdf5 file
ln -sf ~/shapepipe/example/cfis/final_cat.param

# Star catalogue
## Ellipticities in pixel coordinates, MCCD output
# ln -sf $HOME/psfex/${patch}/output/run_sp_Ms/merge_starcat_runner/output/full_starcat-0000000.fits

## Projected back to world coordinates
ln -sf $HOME/psfex/star_cat/${patch}/output/run_sp_Ms/merge_starcat_runner/output/full_starcat-0000000.fits

# Tile number list
ln -sf ~/shapepipe/auxdir/CFIS/tiles_202106/tiles_${patch}.txt

# Parameter file
#cp $spdir/notebooks/params.py .
echo "Diff:"
diff $spdir/notebooks/params.py params.py
echo "Run?"
echo "cp $spdir/notebooks/params.py params.py"
echo "Run?"
echo "ipython ~/sp_validation/notebooks/validation.py"
