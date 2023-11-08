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
#     name: sp_validation
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
import seaborn as sns

from cs_util import canfar
from sp_validation.io import *
from sp_validation.cat import *
from sp_validation.survey import *
from sp_validation.galaxy import *
from sp_validation.basic import *

from cs_util.plots import plot_histograms

# +
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
thresh = 0.0002

# + endofcell="--"
# Read run-specific parameters

# Set up "dummy" object
import types

obj = types.SimpleNamespace()
obj._params = {}

params_in_path = "params_star_response.py"
if os.path.exists(params_in_path):
    print(f"Reading configuration script {params_in_path}")

    with open(params_in_path) as f:
        exec(f.read())

    # Set instance parameters, copy from above
    for key in params_in:
        obj._params[key] = params_in[key]

else:
    print(f"Configuration script {params_in_path} not found, asking for user input")

    for key in obj._params:
        msg = f"{key}? [{obj._params[key]}] "
        val_user = input(msg)
        if val_user != "":
            obj._params[key] = val_user
# -

print(obj._params)
# --
dd = np.load(galaxy_cat_path, mmap_mode=mmap_mode)

cut_overlap = classification_galaxy_overlap_ra_dec(
    dd, ra_key=col_name_ra, dec_key=col_name_dec
)

classification_method = classification_galaxy_ngmix

m_gal = {}

stats_file = open_stats_file(plot_dir, stats_file_name)

cut_common = classification_galaxy_base(
    dd,
    cut_overlap,
    gal_mag_bright=gal_mag_bright,
    gal_mag_faint=gal_mag_faint,
    flags_keep=flags_keep,
    n_epoch_min=n_epoch_min,
    do_spread_model=do_spread_model,
)

m_gal[sh] = classification_method(
    dd,
    cut_common,
    stats_file,
    verbose=verbose,
)

print(dd.dtype.names)

ddc = dd[cut_overlap]

xlim = [0, 4]
ylim = [27, 16]

plt.plot(ddc["NGMIX_T_NOSHEAR"], ddc["MAG_AUTO"], ".", markersize=0.01)
plt.ylim(ylim)
plt.xlim(xlim)
plt.xlabel("T")
plt.ylabel("r")

xlim = [-0.05, 0.75]
plt.plot(
    ddc["NGMIX_T_NOSHEAR"] / ddc["NGMIX_Tpsf_NOSHEAR"],
    ddc["MAG_AUTO"],
    ".",
    markersize=0.01,
)
plt.ylim(ylim)
plt.xlim(xlim)
plt.xlabel("T")
plt.ylabel("r")
plt.axvline(x=0.3, color="k", linewidth=1)
plt.axvline(x=0.01, color="g", linewidth=1)

mask_mag = (ddc["MAG_AUTO"] <= 22) & (ddc["MAG_AUTO"] >= 18)

mask_stars = {}
stars = {}
mask_stars["all"] = (
    ddc["NGMIX_T_NOSHEAR"] / ddc["NGMIX_Tpsf_NOSHEAR"] < 0.3
) & mask_mag

mask_stars["point"] = (
    ddc["NGMIX_T_NOSHEAR"] / ddc["NGMIX_Tpsf_NOSHEAR"] < 0.01
) & mask_mag

mask_stars["resol"] = (
    (ddc["NGMIX_T_NOSHEAR"] / ddc["NGMIX_Tpsf_NOSHEAR"] >= 0.01)
    & (ddc["NGMIX_T_NOSHEAR"] / ddc["NGMIX_Tpsf_NOSHEAR"] <= 0.3)
    & mask_mag
)

for key in mask_stars:
    stars[key] = ddc[mask_stars[key]]
    print_stats(f"{key} {len(stars[key])}", stats_file, verbose=True)

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


d_star = fits.getdata(star_cat_path, 2)

ind_star, mask_area_tiles, n_star_tot = check_matching(
    d_star,
    ddc,
    ["RA", "DEC"],
    [col_name_ra, col_name_dec],
    thresh,
    stats_file,
    name=None,
    verbose=True,
)

xlim = [-0.05, 0.75]
plt.plot(
    ddc[ind_star]["NGMIX_T_NOSHEAR"] / ddc[ind_star]["NGMIX_Tpsf_NOSHEAR"],
    ddc[ind_star]["MAG_AUTO"],
    "m.",
    markersize=0.02,
    label="matched PSF stars",
)
plt.ylim(ylim)
plt.xlim(xlim)
plt.xlabel(r"$T / T_{\rm{psf}}$")
plt.ylabel("$r$")
plt.legend()

# +
mask_stars_matched = {}
stars_matched = {}

mask_mag_matched = (ddc[ind_star]["MAG_AUTO"] <= 22) & (ddc[ind_star]["MAG_AUTO"] >= 18)
mask_stars_matched["point"] = (
    ddc[ind_star]["NGMIX_T_NOSHEAR"] / ddc[ind_star]["NGMIX_Tpsf_NOSHEAR"] < 0.01
) & mask_mag_matched
stars_matched["point"] = ddc[ind_star][mask_stars_matched["point"]]

mask_stars_matched["resol"] = (
    (ddc[ind_star]["NGMIX_T_NOSHEAR"] / ddc[ind_star]["NGMIX_Tpsf_NOSHEAR"] >= 0.01)
    & (ddc[ind_star]["NGMIX_T_NOSHEAR"] / ddc[ind_star]["NGMIX_Tpsf_NOSHEAR"] <= 0.3)
    & mask_mag_matched
)

for key in mask_stars_matched:
    stars_matched[key] = ddc[ind_star][mask_stars_matched[key]]
    print_stats(f"{key} {len(stars_matched[key])}", stats_file, verbose=True)

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
