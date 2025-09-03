# %%
from IPython import get_ipython                                                  

# %%
# enable autoreload for interactive sessions                                     
ipython = get_ipython()                                                          
if ipython is not None:                                                          
    ipython.run_line_magic("load_ext", "autoreload")                             
    ipython.run_line_magic("autoreload", "2")
    ipython.run_line_magic("load_ext", "log_cell_time")

# %%
import sys
import os
import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt
import healpy as hp
import healsparse as hsp

from sp_validation import run_joint_cat as sp_joint
from sp_validation import util
from sp_validation.basic import metacal
from sp_validation import calibration
import sp_validation.cat as cat

# %%
# Initialize calibration class instance
obj = sp_joint.CalibrateCat()

# %%
# Read configuration file and set parameters
config = obj.read_config_set_params("config_mask.yaml")

# %%
# Get data. Set load_into_memory to False for very large files
dat, dat_ext = obj.read_cat(load_into_memory=False)
# %%
key_ra = "RA"
key_dec = "Dec"

# %%
# Create healsparse mask instance
hsp_obj = sp_joint.ApplyHspMasks()

# %% Mask directory
hsp_obj._params["mask_dir"] = f"{os.environ['HOME']}/masks"
# %%
# Masks to use

# Bits
hsp_obj._params["bits"] = 64

# Names
masks_to_apply = [
    "64_r",
]

masks, labels = sp_joint.get_masks_from_config(
    config,
    dat,
    dat_ext,
    masks_to_apply=masks_to_apply,
    verbose=True
)

# %%
def get_areas(hsp_obj, masks):
    
    areas_deg2 = []


    # Path to mask file(s)
    paths = hsp_obj.get_paths_bit_masks()

    for key, mask in zip(paths, masks):
        print(f"Reading hsp mask file {paths[key]}...")
        hsp_mask = hsp.HealSparseMap.read(paths[key])

        print(f"sentinel = {hsp_mask.sentinel}")

        # Pixel area
        pix_area_deg2 = hp.nside2pixarea(hsp_mask.nside_sparse, degrees=True)

        # Check total area
        Npix = hp.nside2npix(hsp_mask.nside_sparse)
        A_tot_deg2 = Npix * pix_area_deg2
        print(f"Total area: {A_tot_deg2:.3f} deg^2")

        nside_frac = hsp_mask.nside_coverage * 2

        fracdet = hsp_mask.fracdet_map(nside_frac)
        vals = fracdet.get_values_pix(fracdet.valid_pixels).astype(float)
        pix_area_deg2_frac = hp.nside2pixarea(nside_frac, degrees=True)
        area_unmasked_deg2 = vals.sum() * pix_area_deg2_frac
        area_footprint_deg2 = (vals > 0).sum() * pix_area_deg2_frac
        print(f"Unmasked ≈ {area_unmasked_deg2:.1f} deg²")
        print(f"Footprint ≈ {area_footprint_deg2:.1f} deg²")

        cov_mask = hsp_mask.coverage_mask
        npop_pix = np.count_nonzero(cov_mask)
        area_deg2 = npop_pix * pix_area_deg2
        print(f"{mask._col_name}: Area of coverage: {area_deg2:.3f} deg^2")

        # Get valid (active) pixels and convert to array
        vals = hsp_mask.get_values_pix(hsp_mask.valid_pixels)
        n_valid = len(vals)
        area_deg2 = n_valid * pix_area_deg2
        print(f"{mask._col_name}: Area of valid pixels: {area_deg2:.3f} deg^2")


        # TODO: Use class Mask evaluation and config file entries
        #if mask._value == True:
            #n_unmasked = int(np.count_nonzero(vals))
        #elif mask._value == False:
            #n_unmasked = int(np.count_nonzero(vals == False))
        #else:
            #print("Not implemented yet:", mask._value)


        # Testing
        for value in (True, False):
            n_unmasked = int(np.count_nonzero(vals == value))

            # Footprint area
            area_deg2 = n_unmasked * pix_area_deg2

            print(f"{mask._col_name}: Area with value={value}: {area_deg2:.3f} deg^2")

        areas_deg2.append(area_deg2)
        
    return areas_deg2

# %%
areas_deg2 = get_areas(hsp_obj, masks)
# %%
# Bin data
for mask in masks:
    mask.apply(dat_ext)
    

# %%
def get_binned_area(ra, dec, nside=512):
    
    # Pixel list of input data
    ipix = hp.ang2pix(nside, ra, dec, lonlat=True)
    
    # Number of occupied pixels
    Nocc  = np.unique(ipix).size
    
    # Pixel area
    pix_area_deg2 = hp.nside2pixarea(nside, degrees=True)

    # Footprint area
    area_deg2 = Nocc * pix_area_deg2
    

    return area_deg2

# %%
for mask in masks:
    print("Mask:", mask._col_name)
    ra = dat[key_ra][mask._mask]
    dec = dat[key_dec][mask._mask]
    
    for nside in [256, 512, 1024]:
       
        area_binned_deg2 = get_binned_area(ra, dec, nside=nside)
        print(f"binned area (nside={nside}) ≈ {area_binned_deg2:.3f} deg^2")
# %%
print(areas_deg2)
# %%
