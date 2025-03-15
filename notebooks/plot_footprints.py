# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Plot footprints

# %matplotlib inline

# +
import numpy as np
import matplotlib.pylab as plt
import healpy as hp
import healsparse as hsp
from collections import Counter


import skyproj

import os
from astropy.io import fits
# -

versions = ["v1.4.2"]
base_dir = f"{os.environ['HOME']}/v1.4.x/"
base_fname = "unions_shapepipe_cut_struc_2024_"

cats = {}
for ver in versions:
    path = f"{base_dir}/{ver}/{base_fname}{ver}.fits"
    cats[ver] = fits.getdata(path)

# +
ver = versions[0]

ra = cats[ver]["RA"]
dec = cats[ver]["Dec"]

# +
nside_coverage = 128
nside_map = 4096


sp_map = hsp.HealSparseMap.make_empty(nside_coverage, nside_map, dtype=np.float32, sentinel=np.nan)

# Get pixel list corresponding to coordinates
hpix = hp.ang2pix(nside_map, ra, dec, nest=True, lonlat=True)

# Get count of objects per pixel
pixel_counts = Counter(hpix)

# List of unique pixels
unique_hpix = np.array(list(pixel_counts.keys()))

# Number of objects
values = np.array(list(pixel_counts.values()), dtype=np.float32)  # Use float32 to match dtype

# Create maps with numbers per pixel
sp_map[unique_hpix] = values

# +
ra_0 = 180
ralo = 270
rahi = 120
declo = 29
dechi = 70

fig, ax = plt.subplots(figsize=(10, 10))
sp = skyproj.McBrydeSkyproj(ax=ax, lon_0=ra_0, extent=[ralo, rahi, declo, dechi], autorescale=True)
_ = sp.draw_hspmap(sp_map, lon_range=[ralo, rahi], lat_range=[declo, dechi])
# -


