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

# # Plot footprints

# %matplotlib inline

# +
import numpy as np
import re
import matplotlib.pylab as plt
import healpy as hp
import healsparse as hsp
from collections import Counter

import skyproj

import os
from astropy.io import fits
# -

versions = ["v1.4.2", "v1.5.4"]
base_dir = f"{os.environ['HOME']}"
base_fname = "unions_shapepipe_cut_struc_2024_"

cats = {}
for ver in versions:
    subdir_ver = re.sub(r"(\d+)$", "x", ver)
    path = f"{base_dir}/{subdir_ver}/{ver}/{base_fname}{ver}.fits"
    cats[ver] = fits.getdata(path)


def create_sp_map(ra, dec, nside_coverage=32, nside_map=2048):

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
    
    return sp_map


def plot_area(
    hsp_map,
    ra_0=0,
    ra_low=120,
    ra_high=270,
    dec_low=29,
    dec_high=70,
    vmax=60,
    ax=None,
):
    
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 10))

    extend = [ra_low, ra_high, dec_low, dec_high]

    projection = skyproj.McBrydeSkyproj(
        ax=ax,
        lon_0=ra_0,
        extent=extend,
        autorescale=True,
        vmax=vmax
    )

    _ = projection.draw_hspmap(
        hsp_map, lon_range=extend[0:2],
        lat_range=extend[2:]
    )


# +
for ver in versions:

    ra = cats[ver]["RA"]
    dec = cats[ver]["Dec"]

    sp_map = create_sp_map(ra, dec)

    ra_0 = 180
    ra_low = 270
    ra_high = 120
    dec_low = 30
    dec_high = 70

    vmax = 60


    plt.clf()
    
    plot_area(
        sp_map,
        ra_0=ra_0,
        ra_low=ra_low,
        ra_high=ra_high,
        dec_low=dec_low,
        dec_high=dec_high,
        vmax=vmax,
    )
    plt.savefig(f"footprint_{ver}.png")
    

# -


