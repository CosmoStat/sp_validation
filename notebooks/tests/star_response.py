# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# # Star response tests
# 2023

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# +
import sys
import os
import numpy as np
from astropy.io import fits

import matplotlib.pylab as plt
import healpy as hp

from cs_util import canfar
from sp_validation.io import *
from sp_validation.cat import *
from sp_validation.survey import *
from sp_validation.galaxy import *
from sp_validation.basic import *

from cs_util.plots import plot_histograms
from cs_util import args

from unions_wl import catalogue as wl_cat

print("star_response.py start")

# +
# Parameters: TODO move to parameter file, use cs_util.args (not working so fat)

galaxy_cat_path = "final_cat.npy"
mmap_mode = None
col_name_ra = "XWIN_WORLD"
col_name_dec = "YWIN_WORLD"
sh = "ngmix"
stats_file_name = "stats.txt"
plot_dir = "."
verbose = True
gal_mag_bright = 15
gal_mag_faint = 30
flags_keep = [1]
n_epoch_min = 2
do_spread_model = False

star_cat_path = "full_starcat-0000000.fits"
star_cat_path_psfex = "/home/mkilbing/astro/Runs/shapepipe/CFIS/big_run/W3/psf_validation_merged/psf_cat_full.fits"
print("Opening star cat")
d_star = fits.getdata(star_cat_path, 1)
d_star_psfex = fits.getdata(star_cat_path_psfex, 2)
print("Done reading star cat")
thresh = 0.0002

input_mask = f"{os.environ['HOME']}/astro/data/CFIS/v1.0/Lensfit/masks/CFIS3500_THELI_mask_hp_4096.fits"
# -

dd = np.load(galaxy_cat_path, mmap_mode=mmap_mode)

cut_overlap = classification_galaxy_overlap_ra_dec(
    dd, ra_key=col_name_ra, dec_key=col_name_dec
)

stats_file = open_stats_file(plot_dir, stats_file_name)

# Remove duplicates due to overlapping tiles
ddc = dd[cut_overlap]

xlim = [-0.05, 0.75]
ylim = [23, 17]
plt.plot(
    ddc["NGMIX_T_NOSHEAR"] / ddc["NGMIX_Tpsf_NOSHEAR"],
    ddc["MAG_AUTO"],
    ".",
    markersize=0.01,
)
plt.ylim(ylim)
plt.xlim(xlim)
plt.xlabel(r"$T / T_{\rm psf}$")
plt.ylabel("$r$")
plt.axvline(x=0.3, color="k", linewidth=1)
plt.axvline(x=0.0, color="g", linewidth=1)
plt.savefig("size_mag_all.png")

# Apply THELI mask
mask, nest, nside = wl_cat.read_hp_mask(input_mask, verbose=True)

# Get mask pixel numbers of coordinates
ipix = hp.ang2pix(nside, ddc[col_name_ra], ddc[col_name_dec], nest=nest, lonlat=True)

# Get pixels in footprint, where mask is marked "good"
in_footprint = (mask[ipix] == 0)

# Get numbers of pixels in footprint
ipix_in_footprint = ipix[in_footprint] 

# Get indices of coordinates in footprint
idx_np = np.where(in_footprint)[0]

n_in_footprint = len(idx_np)                                             
print(                                                                   
    f'{n_in_footprint}/{len(ddc[col_name_ra])} ='                                      
    + f' {n_in_footprint/len(ddc[col_name_ra]):.2%} objects in footprint'              
)

# Restrict data to footprint
ddcm = ddc[idx_np]

xlim = [-0.05, 0.75]
plt.plot(
    ddcm["NGMIX_T_NOSHEAR"] / ddcm["NGMIX_Tpsf_NOSHEAR"],
    ddcm["MAG_AUTO"],
    ".",
    markersize=0.01,
)
plt.ylim(ylim)
plt.xlim(xlim)
plt.xlabel(r"$T / T_{\rm psf}$")
plt.ylabel("$r$")
plt.axvline(x=0.3, color="k", linewidth=1)
plt.axvline(x=0.01, color="g", linewidth=1)
plt.savefig("relsize_mag_infootprint.png")


# ### Star sample

def create_mask_stars(ddx):
    """
    Define "star" classes: all, point sources, and resolved objects.
    Return corresponding masks.
    
    """
    # Magnitude range for star selection
    mask_mag = (ddx["MAG_AUTO"] <= 22) & (ddx["MAG_AUTO"] >= 18)
    
    mask_stars = {}
    stars = {}
    mask_stars["all"] = (
        ddx["NGMIX_T_NOSHEAR"] / ddx["NGMIX_Tpsf_NOSHEAR"] < 0.3
    ) & mask_mag
    
    mask_stars["point"] = (
        ddx["NGMIX_T_NOSHEAR"] / ddx["NGMIX_Tpsf_NOSHEAR"] < 0.01
    ) & mask_mag

    mask_stars["resol"] = (
        (ddx["NGMIX_T_NOSHEAR"] / ddx["NGMIX_Tpsf_NOSHEAR"] >= 0.01)
        & (ddx["NGMIX_T_NOSHEAR"] / ddx["NGMIX_Tpsf_NOSHEAR"] <= 0.3)
        & mask_mag
    )

    for key in mask_stars:
        stars[key] = ddx[mask_stars[key]]

    return mask_stars, stars


# Get masks for all and THELI-restricted data
mask_stars_nm, stars_nm = create_mask_stars(ddc)
mask_stars, stars = create_mask_stars(ddcm)

# +
# Print some stats
print("All objects")
for key in stars_nm:
    print_stats(f"{key} {len(stars_nm[key])}", stats_file, verbose=True)

for key in stars:
    print_stats(f"{key} {len(stars[key])}", stats_file, verbose=True)

print(f'{len(stars_nm["resol"]) / len(stars_nm["point"]):.2f}')
print(f'{len(stars["resol"]) / len(stars["point"]):.2f}')

# +
# Plot zoom-in relative size-magnitude diagram

xlim = [-0.05, 0.75]
plt.plot(
    stars["all"]["NGMIX_T_NOSHEAR"] / stars["all"]["NGMIX_Tpsf_NOSHEAR"],
    stars["all"]["MAG_AUTO"],
    "k.",
    markersize=0.02,
    label="all non-gals",
)
plt.plot(
    stars["point"]["NGMIX_T_NOSHEAR"] / stars["point"]["NGMIX_Tpsf_NOSHEAR"],
    stars["point"]["MAG_AUTO"],
    "g.",
    markersize=0.02,
    label="point-like non-gals",
)
plt.plot(
    stars["resol"]["NGMIX_T_NOSHEAR"] / stars["resol"]["NGMIX_Tpsf_NOSHEAR"],
    stars["resol"]["MAG_AUTO"],
    "r.",
    markersize=0.02,
    label="resolved non-gals",
)
plt.plot(
    ddc["NGMIX_T_NOSHEAR"] / ddc["NGMIX_Tpsf_NOSHEAR"],
    ddc["MAG_AUTO"],
    "b.",
    markersize=0.005,
    label="all objects",
)
plt.ylim(ylim)
plt.xlim(xlim)
plt.xlabel(r"$T / T_{\rm{psf}}$")
plt.ylabel("$r$")
plt.legend()
_ = plt.savefig("size_mag_zoom.png")
# -

# Call metacal to get R_shear
stars_cal = {}
for key in stars:
    mask = [True] * len(stars[key])
    stars_cal[key] = metacal(
        stars[key],
        mask,
        prefix="NGMIX",
        snr_min=0,
        snr_max=10000,
        rel_size_min=0,
        size_corr_ell=0,
        sigma_eps=0.34,
        verbose=True,
    )

print_stats("Shear response", stats_file, verbose=True)
for key in stars_cal:
    print_stats(key, stats_file, verbose=True)
    rs = np.array2string(stars_cal[key].R)
    print_stats(rs, stats_file, verbose=True)

# +
# Plot response matrix distribution

y_label = "frequency"
n_bin = 100
colors = ["blue", "green", "red"]
linestyles = ["-"] * 3
title = "Shear response"
x_range = [-2, 2]

for idx in (0, 1):
    for jdx in (0, 1):
        x_label = f"$R_{{{idx}{jdx}}}$"
        out_path = f"hist_R_{idx}_{jdx}.pdf"

        xs = []
        labels = []
        for key in stars_cal:
            xs.append(stars_cal[key].R_shear[idx, jdx])
            labels.append(key)

        plot_histograms(
            xs,
            labels,
            title,
            x_label,
            y_label,
            x_range,
            n_bin,
            out_path,
            colors=colors,
            linestyles=linestyles,
        )

# +
# Signal-to-noise distribution
x_label = f"SNR"
y_label = "frequency"
n_bin = 100
out_path = f"hist_SNR.pdf"
colors = ["blue", "green", "red"]
linestyles = ["-"] * 3
title = "Signal-to-noise ratio"
x_range = [0, 1000]

xs = []
labels = []
for key in stars_cal:
    xs.append(stars[key]["SNR_WIN"])
    labels.append(key)

_ = plot_histograms(
    xs,
    labels,
    title,
    x_label,
    y_label,
    x_range,
    n_bin,
    out_path=out_path,
    colors=colors,
    linestyles=linestyles,
    close_fig=False,
)
plt.show()
# -
# ## Match to star catalogue



# sp_validation matching: matches to multiple galaxies are removed.
print("MCCD:")
ind_star, mask_area_tiles, n_star_tot = check_matching(
    d_star,
    ddcm,
    ["RA", "DEC"],
    [col_name_ra, col_name_dec],
    thresh,
    stats_file,
    name=None,
    verbose=True,
)
print("PSFEx")
_ = check_matching(
    d_star_psfex,
    ddcm,
    ["RA", "DEC"],
    [col_name_ra, col_name_dec],
    thresh,
    stats_file,
    name=None,
    verbose=True,
)

# Get star classes for matched catalogue
mask_stars_matched, stars_matched = create_mask_stars(ddcm[ind_star])

for key in mask_stars_matched:
    print_stats(f"{key} {len(stars_matched[key])}", stats_file, verbose=True)

# Plot relative size - magnitude diagram for matched stars
xlim = [-0.05, 0.75]
plt.plot(
    ddcm[ind_star]["NGMIX_T_NOSHEAR"] / ddcm[ind_star]["NGMIX_Tpsf_NOSHEAR"],
    ddcm[ind_star]["MAG_AUTO"],
    "m.",
    markersize=0.02,
    label="matched PSF stars",
)
plt.ylim(ylim)
plt.xlim(xlim)
plt.xlabel(r"$T / T_{\rm{psf}}$")
plt.ylabel("$r$")
plt.legend()
plt.savefig("relsize_mag_matched.png")

# +
xlim = [-0.05, 0.75]

plt.plot(
    stars_matched["point"]["NGMIX_T_NOSHEAR"]
    / stars_matched["point"]["NGMIX_Tpsf_NOSHEAR"],
    stars_matched["point"]["MAG_AUTO"],
    "g.",
    markersize=0.02,
    label="matched point-like stars",
)
plt.plot(
    stars_matched["resol"]["NGMIX_T_NOSHEAR"]
    / stars_matched["resol"]["NGMIX_Tpsf_NOSHEAR"],
    stars_matched["resol"]["MAG_AUTO"],
    "r.",
    markersize=0.02,
    label="matched resolved stars",
)

plt.ylim(ylim)
plt.xlim(xlim)
plt.xlabel(r"$T / T_{\rm{psf}}$")
plt.ylabel("$r$")
plt.legend()
_ = plt.savefig("size_mag_matched_zoom.png")


# -

# From sp_validation, added ind_gal
def match_stars2(ra_gal, dec_gal, ra_star, dec_star, thresh=0.0002):                                                                        
    gal_coord = coords.SkyCoord(ra=ra_gal * u.degree, dec=dec_gal * u.degree)    
    star_coord = coords.SkyCoord(                                                
        ra=ra_star * u.degree,                                                   
        dec=dec_star * u.degree,                                                 
    )                                                                            
                                                                                 
    idx, d2d, d3d = coords.match_coordinates_sky(star_coord, gal_coord)              
    sep_constraint =  d2d.value < thresh                                                              
    ind_stars = idx[sep_constraint]
    ind_gal = sep_constraint
                                                                                 
    return ind_stars, ind_gal


idx, idg = match_stars2(ddcm[col_name_ra], ddcm[col_name_dec], d_star["RA"], d_star["DEC"])
idx_psfex, idg_psfex = match_stars2(ddcm[col_name_ra], ddcm[col_name_dec], d_star_psfex["RA"], d_star_psfex["DEC"])

len(ind_star)

# + active=""
# # We have not removed matches to multiple galaxies
# len(idx)
# -

# Test whether matching worked and we understood the indexing
print(ddcm[idx]["XWIN_WORLD"][:3], d_star[idg]["RA"][:3], ddcm["XWIN_WORLD"][:3])

# +
# Create combined dictionary of galaxy and star information
comb = {}
for key in ddcm.dtype.names:
    comb[key] = ddcm[idx][key]

for key in d_star.dtype.names:
    comb[key] = d_star[idg][key]
# -

# Test: use new index instead of duplicate-removed one
xlim = [-0.05, 0.75]
plt.plot(
    ddcm[idx]["NGMIX_T_NOSHEAR"] / ddcm[idx]["NGMIX_Tpsf_NOSHEAR"],
    ddcm[idx]["MAG_AUTO"],
    "m.",
    markersize=0.02,
    label="matched point-like stars",
)
plt.ylim(ylim)
plt.xlim(xlim)
plt.xlabel(r"$T / T_{\rm{psf}}$")
plt.ylabel("$r$")
plt.legend()

# Redo mathing to point- and resolved sources using new inmask_starsdex
mask_stars_matched, stars_matched = create_mask_stars(ddcm[idx])

for key in mask_stars_matched:
    stars_matched[key] = ddcm[idx][mask_stars_matched[key]]
    print_stats(f"{key} {len(stars_matched[key])}", stats_file, verbose=True)

xlim = [-0.05, 0.75]
plt.plot(
    comb["NGMIX_T_NOSHEAR"][mask_stars_matched["point"]] / comb["NGMIX_Tpsf_NOSHEAR"][mask_stars_matched["point"]],
    comb["MAG_AUTO"][mask_stars_matched["point"]],
    "g.",
    markersize=0.02,
    label="matched point-like stars",
)
plt.plot(
    comb["NGMIX_T_NOSHEAR"][mask_stars_matched["resol"]] / comb["NGMIX_Tpsf_NOSHEAR"][mask_stars_matched["resol"]],
    comb["MAG_AUTO"][mask_stars_matched["resol"]],
    "r.",
    markersize=0.02,
    label="matched resolved stars",
)
plt.ylim(ylim)
plt.xlim(xlim)
plt.xlabel(r"$T / T_{\rm{psf}}$")
plt.ylabel("$r$")
plt.legend()
plt.savefig("relsize_mag_matched_resol_point.png")

# +
# Look at HSM quantities for point- and resolved stars
xlim = [1, 2]
ylim = [23, 17]
plt.plot(
    comb["SIGMA_PSF_HSM"][mask_stars_matched["point"]],
    comb["MAG_AUTO"][mask_stars_matched["point"]],
    "g.",
    markersize=0.5,
    label="matched point-like stars",
)
plt.plot(
    comb["SIGMA_PSF_HSM"][mask_stars_matched["resol"]],
    comb["MAG_AUTO"][mask_stars_matched["resol"]],
    "r.",
    markersize=0.3,
    label="matched resolved stars",
)

plt.ylim(ylim)
plt.xlim(xlim)
plt.xlabel(r"$\sigma$")
plt.ylabel("$r$")
plt.legend()
plt.savefig("relsize_hsm_mag_matched_resol_point.png")

print("sample mean std: PSF sizes")
for key in mask_stars_matched:
    mu = np.mean(comb["SIGMA_PSF_HSM"][mask_stars_matched[key]])
    std = np.std(comb["SIGMA_PSF_HSM"][mask_stars_matched[key]])
    print(key, mu, std)

print("sample mean std: star sizes")
for key in mask_stars_matched:
    mu = np.mean(comb["SIGMA_STAR_HSM"][mask_stars_matched[key]])
    std = np.std(comb["SIGMA_STAR_HSM"][mask_stars_matched[key]])
    print(key, mu, std)

# +
xlim = [-0.2, 0.2]

plt.plot(
    comb["SIGMA_PSF_HSM"][mask_stars_matched["point"]] - comb["SIGMA_STAR_HSM"][mask_stars_matched["point"]],
    comb["MAG_AUTO"][mask_stars_matched["point"]],
    "g.",
    markersize=0.5,
    label="matched point-like stars",
)
plt.plot(
    comb["SIGMA_PSF_HSM"][mask_stars_matched["resol"]] - comb["SIGMA_STAR_HSM"][mask_stars_matched["resol"]],
    comb["MAG_AUTO"][mask_stars_matched["resol"]],
    "r.",
    markersize=0.3,
    label="matched resolved stars",
)

plt.ylim(ylim)
plt.xlim(xlim)
plt.xlabel(r"$\sigma$")
plt.ylabel("$r$")
plt.legend()
plt.savefig("dsize_hsm_mag_matched_resol_point.png")

print("sample mean std")
for key in mask_stars_matched:
    mu = np.mean(comb["SIGMA_PSF_HSM"][mask_stars_matched[key]] - comb["SIGMA_STAR_HSM"][mask_stars_matched[key]])
    std = np.std(comb["SIGMA_PSF_HSM"][mask_stars_matched[key]] - comb["SIGMA_STAR_HSM"][mask_stars_matched[key]])
    print(key, mu, std)

# The point-source stars are larger than the PSF, therefore it seems that they get easily deconvolved to zero size.
# The resolved stars have a close-to-zero size residual, which is good in principle. However, they do not
# get deconvolved to zero size, the ngmix fit gives them a finite size. Something must go wrong in the fit.

# +
xlim = [-0.1, 0.1]
ylim = [23, 17]
plt.plot(
    comb["E1_PSF_HSM"][mask_stars_matched["point"]] - comb["E1_STAR_HSM"][mask_stars_matched["point"]],
    comb["MAG_AUTO"][mask_stars_matched["point"]],
    "g.",
    markersize=0.5,
    label="matched point-like stars",
)
plt.plot(
    comb["E1_PSF_HSM"][mask_stars_matched["resol"]] - comb["E1_STAR_HSM"][mask_stars_matched["resol"]],
    comb["MAG_AUTO"][mask_stars_matched["resol"]],
    "r.",
    markersize=0.3,
    label="matched resolved stars",
)

plt.ylim(ylim)
plt.xlim(xlim)
plt.xlabel(r"$e_1$")
plt.ylabel("$r$")
plt.legend()
plt.savefig("dell1_hsm_mag_matched_resol_point.png")

print("sample mean std")
for key in mask_stars_matched:
    mu = np.mean(comb["E1_PSF_HSM"][mask_stars_matched[key]] - comb["E1_STAR_HSM"][mask_stars_matched[key]])
    std = np.std(comb["E1_PSF_HSM"][mask_stars_matched[key]] - comb["E1_STAR_HSM"][mask_stars_matched[key]])
    print(key, mu, std)

# +
ylim = [23, 17]
plt.plot(
    comb["E2_PSF_HSM"][mask_stars_matched["point"]] - comb["E2_STAR_HSM"][mask_stars_matched["point"]],
    comb["MAG_AUTO"][mask_stars_matched["point"]],
    "g.",
    markersize=0.5,
    label="matched point-like stars",
)
plt.plot(
    comb["E2_PSF_HSM"][mask_stars_matched["resol"]] - comb["E2_STAR_HSM"][mask_stars_matched["resol"]],
    comb["MAG_AUTO"][mask_stars_matched["resol"]],
    "r.",
    markersize=0.3,
    label="matched resolved stars",
)

plt.ylim(ylim)
plt.xlim(xlim)
plt.xlabel(r"$e_2$")
plt.ylabel("$r$")
plt.legend()
plt.savefig("dell2_hsm_mag_matched_resol_point.png")

print("sample mean std")
for key in mask_stars_matched:
    mu = np.mean(comb["E2_PSF_HSM"][mask_stars_matched[key]] - comb["E2_STAR_HSM"][mask_stars_matched[key]])
    std = np.std(comb["E2_PSF_HSM"][mask_stars_matched[key]] - comb["E2_STAR_HSM"][mask_stars_matched[key]])
    print(key, mu, std)

# +
plt.plot(
    comb["X"][mask_stars_matched["point"]],
    comb["Y"][mask_stars_matched["point"]],
    "g.",
    markersize=0.5,
    label="matched point-like stars",
)
plt.plot(
    comb["X"][mask_stars_matched["resol"]],
    comb["Y"][mask_stars_matched["resol"]],
    "r.",
    markersize=0.5,
    label="matched resolved stars",
)

#plt.ylim(ylim)
#plt.xlim(xlim)
plt.xlabel(r"x [pix]")
plt.ylabel(r"y [pix]")
plt.legend(bbox_to_anchor=(0.25, 1.01))
plt.savefig("stars_point_resol_x_y.png")

# +
xlim = -0.01, 0.005
plt.plot(
    comb["SPREAD_MODEL"][mask_stars_matched["point"]],
    comb["MAG_AUTO"][mask_stars_matched["point"]],
    "g.",
    markersize=0.5,
    label="matched point-like stars",
)
plt.plot(
    comb["SPREAD_MODEL"][mask_stars_matched["resol"]],
    comb["MAG_AUTO"][mask_stars_matched["resol"]],
    "r.",
    markersize=0.3,
    label="matched resolved stars",
)

plt.xlim(xlim)
plt.xlabel(r"SPREAD MODEL")
plt.ylabel("$r$")
plt.legend()

print("sample mean std: SPREAD_MODEL")
for key in mask_stars_matched:
    mu = np.mean(comb["SPREAD_MODEL"][mask_stars_matched[key]])
    std = np.std(comb["SPREAD_MODEL"][mask_stars_matched[key]])
    print(key, mu, std)
# -

for key in comb: print(key)


print("star_response.py end")
