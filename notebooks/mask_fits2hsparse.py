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

# +
import os
import numpy as np
import healpy as hp
import healsparse as hs
import glob
import matplotlib.pylab as plt
from astropy.io import fits
from astropy import wcs
from astropy import units
import tqdm
from timeit import default_timer as timer

from IPython.display import display, clear_output

from sp_validation import plot_style
# -

import warnings
warnings.filterwarnings('ignore')

# List of FITS mask files
directory = f"{os.environ['HOME']}/arc/projects/unions/catalogues/unions/GAaP_photometry/extfinalmask_UNIONS5000"
#directory = "."
fits_mask_files = glob.glob(f"{directory}/UNIONS.*_extfinalmask.fits")

# Define HealSparse parameters
nside_coverage = 32  # Define the nside of the coverage map (adjust as needed)
nside_sparse = 4096  # Define the sparse map resolution

# +
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline

def interpol(ra_step, dec_step, x, y, x_step, y_step):

    # Flatten the coarse grid and create pairs of coordinates
    points = np.array([x_step.flatten(), y_step.flatten()]).T
    ra_values = ra_step.flatten()
    dec_values = dec_step.flatten()
    
    # Interpolate RA and Dec values to the fine grid
    ra = griddata(points, ra_values, (x, y), method='cubic')
    dec = griddata(points, dec_values, (x, y), method='cubic')
    
    return ra, dec
    
def process(fits_mask_file, nside_sparse, nside_coverage, step=1):

    # Get data
    hdu_list = fits.open(fits_mask_file)
    WCS = wcs.WCS(hdu_list[0].header)

    # Mask values
    mask = hdu_list[0].data

    # Coordinates

    nx, ny = mask.shape
    x = np.arange(nx)
    y = np.arange(ny)
    x_mesh, y_mesh = np.meshgrid(x, y)

    if step > 1:

        # Note: Use nx+1, ny+1 to add additional point at x_ar=nx, y_ar=ny.
        # Required to
        # reduce extrapolation errors towards large x, y.
        x_ar = np.arange(0, nx + 1, step)
        y_ar = np.arange(0, ny + 1, step)
        x_step, y_step = np.meshgrid(x_ar, y_ar, indexing="ij")

        #start = timer()
        ra_step, dec_step = WCS.wcs_pix2world(x_step, y_step, 0)
        #end = timer()
        #print(f"WCS pix2 world course {end - start:.1f}s")

        #start = timer()
        ra_interp = RectBivariateSpline(y_ar, x_ar, ra_step, ky=2, kx=2)
        dec_interp = RectBivariateSpline(y_ar, x_ar, dec_step, ky=2, kx=2)
        #end = timer()
        #print(f"create spline {end - start:.1f}s")

        #start = timer()
        ra = ra_interp(y, x).T
        dec = dec_interp(y, x).T
        #end = timer()
        #print(f"interpolate {end - start:.1f}s")

    else:
        #start = timer()
        ra, dec = WCS.wcs_pix2world(x_mesh, y_mesh, 0)
        #end = timer()
        #print(f"WCS pix2 world {end - start:.1f}s")

    # coordinates in deg
    return ra, dec, mask

def update_map(hs_map, ra, dec, mask):
    
    # Add pixels to healsparse map
    #start = timer()
    hs_map.update_values_pos(ra, dec, mask, lonlat=True, operation="or")
    #end = timer()
    #print(f"update map {end - start:.1f}s")


# -

def flatten(ra, dec, mask):
    ra_flatten = np.ravel(np.radians(ra))
    dec_flatten = np.ravel(np.radians(dec))
    mask_flatten = np.ravel(mask)

    return ra_flatten, dec_flatten, mask_flatten


# Initialise map (first time)
hs_map = hs.HealSparseMap.make_empty(nside_coverage, nside_sparse, dtype="int16", sentinel=0)

do_plot = False
do_gif = True

# +
step = 10

batch_size = 10

batch_ra, batch_dec, batch_mask = [], [], []

# Loop over mask files
idx = 0
for fits_mask_file in tqdm.tqdm(fits_mask_files, total=len(fits_mask_files), disable=False):

    # Check for zero file size
    if os.stat(fits_mask_file).st_size == 0:
        print(f"File {fits_mask_file} has zero size, continuing")    

    # Get coordinates of pixel grid
    ra, dec, mask = process(fits_mask_files[0], nside_sparse, nside_coverage, step=step)

    ra_flatten, dec_flatten, mask_flatten = flatten(ra, dec, mask)


    # Append to batch
    batch_ra.append(ra_flatten)
    batch_dec.append(dec_flatten)
    batch_mask.append(mask_flatten)

    # Process batch update every `batch_size` files or at the end
    if (idx + 1) % batch_size == 0 or idx == len(fits_mask_files) - 1:
        start = timer()

        # Concatenate arrays for batch processing
        batch_ra = np.concatenate(batch_ra)
        batch_dec = np.concatenate(batch_dec)
        batch_mask = np.concatenate(batch_mask)
    
        update_map(hs_map, batch_ra, batch_dec, batch_mask)

        # Clear batch storage
        batch_ra, batch_dec, batch_mask = [], [], []

        if do_gif:

            map_lowres = hs_map.generate_healpix_map(nside=256)
            hp.mollview(map_lowres)
            fname = f"map_lowres_{idx:04d}.jpg"
            print(f"Saving figure {fname}")

            plt.savefig(fname)

    idx += 1
    
    if do_plot:
        clear_output(wait=True)

        # Create the figure
        plt.figure(figsize=(8, 5))
    
        hp.mollview(hs_map.coverage_mask)
        plt.show()
# -

hs_map.write('mask_all.fits', clobber=False)






































