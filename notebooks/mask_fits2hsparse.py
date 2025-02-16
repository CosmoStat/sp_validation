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
# -

import warnings
warnings.filterwarnings('ignore')

# List of FITS mask files
directory = "/arc/projects/unions/catalogues/unions/GAaP_photometry/extfinalmask_UNIONS5000"
directory = "."
fits_mask_files = glob.glob(f"{directory}/UNIONS.*_extfinalmask.fits")

# Define HealSparse parameters
nside_coverage = 32  # Define the nside of the coverage map (adjust as needed)
nside_sparse = 4096  # Define the sparse map resolution

# +
#from scipy.interpolate import griddata
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
    mask_flatten = np.ravel(mask)

    # Coordinates
    start = timer()

    nx, ny = mask.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))

    if step > 1:
        x_ar = np.arange(0, nx, step)
        y_ar = np.arange(0, ny, step)
        x_step, y_step = np.meshgrid(x_ar, y_ar)
        ra_step, dec_step = WCS.wcs_pix2world(x_step, y_step, 0)
        #ra, dec = interpol(ra_step, dec_step, x, y, x_step, y_step)
        
        ra_interp = RectBivariateSpline(y_ar, x_ar, ra_step)
        ra = ra_interp(y, x)
        dec_interp = RectBivariateSpline(y_ar, x_ar, dec_step)
        dec = dec_interp(y, x)
    else:
        ra, dec = WCS.wcs_pix2world(x, y, 0)

    ra_flatten = np.ravel(np.radians(ra))
    dec_flatten = np.ravel(np.radians(dec))

    end = timer()
    print(f"WCS pix2 world {end - start:.1f}s")

    return ra_flatten, dec_flatten, mask_flatten

def update_map(hs_map, ra, dec, mask):
    
    # Add pixels to healsparse map
    start = timer()
    hs_map.update_values_pos(ra, dec, mask, lonlat=True, operation="or")
    end = timer()
    print(f"update map {end - start:.1f}s")


# -

# Initialise map (first time)
hs_map = hs.HealSparseMap.make_empty(nside_coverage, nside_sparse, dtype="int16", sentinel=0)

do_plot = False
do_gif = True

# +
step = 10

# Loop over mask files
idx = 0
for fits_mask_file in tqdm.tqdm(fits_mask_files, total=len(fits_mask_files), disable=True):

    # Check for zero file size
    if os.stat(fits_mask_file).st_size == 0:
        print(f"File {fits_mask_file} has zero size, continuing")    

    # Get coordinates of pixel grid
    ra, dec, mask = process(fits_mask_files[0], nside_sparse, nside_coverage, step=step)

    update_map(hs_map, ra, dec, mask)

    if do_plot:
        clear_output(wait=True)

        # Create the figure
        plt.figure(figsize=(8, 5))
    
        hp.mollview(hs_map.coverage_mask)
        plt.show()

    if do_gif:
        idx += 1

        if idx % 5 == 0:
            map_lowres = hs_map.generate_healpix_map(nside=256)
            hp.mollview(map_lowres)
            fname = f"map_lowres_{idx:04d}.jpg"
            print(f"Saving figure {fname}")
            plt.savefig(fname)
# -

# Save the final HealSparse mask
hs_map.write("combined_healsparse_mask.hs")

map = hs_map.generate_healpix_map(nside=64)
hp.mollview(map)
plt.savefig("map.jpg")


