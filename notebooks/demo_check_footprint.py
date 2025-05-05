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
#     name: python3
# ---

# # Demo notebook to check footprint

# %reload_ext autoreload
# %autoreload 2

# +
import os
import numpy as np
import tracemalloc
import healpy as hp
import healsparse as hsp

import cs_util.cat as cs_cat

from astropy.io import fits
from astropy.table import Table

from sp_validation import run_joint_cat as sp_joint
# -

# Create instance of ApplyHspMasks object
obj = sp_joint.ApplyHspMasks()

# +
# Set parameters
obj._params["input_path"] = "erass1cl_primary_v3.2.fits"
obj._params["output_path"] = "eRosita_in_footprint.fits"
obj._params["verbose"] = True

footprint_path = "footprint_v1.4.4.hsp"
key_ra = "RA"
key_dec = "Dec"
good_value = False
hdu = 1
# -

# ## Run

# +
# Check parameter validity
obj.check_params()

# Update parameters (here: strings to list)
obj.update_params()
# -

# Read catalogue
#dat = obj.read_cat(load_into_memory=False, mode="r")
hdu_list = fits.open(obj._params["input_path"])
dat = hdu_list[hdu].data


footprint = obj.get_mask(footprint_path)

nside_coverage = footprint.nside_coverage

# Get mask pixel numbers of coordinates                                      
ipix = hp.ang2pix(nside_coverage, dat[key_ra], dat[key_dec], lonlat=True)

## Get pixels in footprint, where mask is "good_value"                                  
in_footprint = (footprint[ipix] == good_value)

# Get numbers of pixels in footprint                                         
ipix_in_footprint = ipix[in_footprint]                                       

# Get indices of coordinates in footprint                                    
idx_np = np.where(in_footprint)[0]                                           

if obj._params['verbose']:                                                        
    n_in_footprint = len(idx_np)                                             
    print(                                                                   
        f'{n_in_footprint}/{len(dat[key_ra])} ='                                      
        + f' {n_in_footprint/len(dat[key_ra]):.2%} objects in footprint'              
    )                                                                        

# Restrict data to footprint                                                 
dat_in_footprint = dat[idx_np]    

#  Write data in footprint to disk                                            
t = Table(dat_in_footprint)                                                  
if obj._params['verbose']:                                                        
    print(f'Writing objects in footprint to {obj._params["output_path"]}')        
cols = []                                                                    
for col_ind,key in enumerate(t.keys()):                                      
    cols.append(fits.Column(name=key, array=t[key], format=dat_in_footprint.columns[col_ind].format))
cs_cat.write_fits_BinTable_file(cols, obj._params["output_path"])
