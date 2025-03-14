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

# # Plot footprints

# !pip install skymapper

import numpy as np

import healpy as hp
import skymapper as skm
import os
from astropy.io import fits

versions = ["v1.4.2"]
base_dir = f"{os.environ['HOME']}/v1.4.x/"
base_fname = "unions_shapepipe_cut_struc_2024_"

cats = {}
for ver in versions:
    path = f"{base_dir}/{base_fname}{ver}.fits"
    cats[ver] = fits.getdata(path)

cats


# +
# Create map projection
## image of (0 0)

def get_map(ra, dec):

    lon_0 = np.mean(cat["RA"])
    lat_0 = np.mean(cat["Dec"])

    ## the two standard latitudes
    lat_1 = 0
    lat_2 = 30

    ## Projection
    proj = skm.Albers(lon_0, lat_0, lat_1, lat_2)

    ## Create map
    return skm.Map(proj)


# +
ver = versions[0]

ra = cat[ver]["RA"]
dec = cat[ver]["Dec"]
# -

map = get_map(ra, dec)


