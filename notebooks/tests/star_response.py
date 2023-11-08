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
# -

galaxy_cat_path = "final_cat.npy"
mmap_mode = None
col_name_ra = 'XWIN_WORLD'                                                       
col_name_dec = 'YWIN_WORLD'
sh = "ngmix"
stats_file_name = "stats.txt"
plot_dir = "."
verbose = True
gal_mag_bright = 15
gal_mag_faint = 30
flags_keep = [1]
n_epoch_min = 2
do_spread_model = False

dd = np.load(galaxy_cat_path, mmap_mode=mmap_mode)

cut_overlap = classification_galaxy_overlap_ra_dec(                              
    dd,                                                                          
    ra_key=col_name_ra,                                                          
    dec_key=col_name_dec                                                         
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

plt.plot(ddc['NGMIX_T_NOSHEAR'], ddc['MAG_AUTO'], '.', markersize=0.01)
plt.ylim(ylim)
plt.xlim(xlim)
plt.xlabel("T")
plt.ylabel("r")

xlim = [-0.05, 0.75]
plt.plot(ddc['NGMIX_T_NOSHEAR'] / ddc['NGMIX_Tpsf_NOSHEAR'], ddc['MAG_AUTO'], '.', markersize=0.01)
plt.ylim(ylim)
plt.xlim(xlim)
plt.xlabel("T")
plt.ylabel("r")
plt.axvline(x=0.3, color='k', linewidth=1)
plt.axvline(x=0.01, color='g', linewidth=1)

mask_mag = (
    (ddc["MAG_AUTO"] <= 22)
    & (ddc["MAG_AUTO"] >= 18)
)

mask_stars = {}
stars = {}
mask_stars["all"] = (
    (ddc['NGMIX_T_NOSHEAR'] / ddc['NGMIX_Tpsf_NOSHEAR'] < 0.3)
    & mask_mag
)

mask_stars["point"] = (
    (ddc['NGMIX_T_NOSHEAR'] / ddc['NGMIX_Tpsf_NOSHEAR'] < 0.01)
    & mask_mag
)

mask_stars["resol"] = (
    (ddc['NGMIX_T_NOSHEAR'] / ddc['NGMIX_Tpsf_NOSHEAR'] >= 0.01)
    & (ddc['NGMIX_T_NOSHEAR'] / ddc['NGMIX_Tpsf_NOSHEAR'] <= 0.3)
    & mask_mag
)

for key in mask_stars:
    print(key)
    stars[key] = ddc[mask_stars[key]]

xlim = [-0.05, 0.75]
plt.plot(
    stars["all"]['NGMIX_T_NOSHEAR'] / stars["all"]['NGMIX_Tpsf_NOSHEAR'], stars["all"]['MAG_AUTO'],
    'k.',
    markersize=0.02,
    label="all stars"
)
plt.plot(
    stars["point"]['NGMIX_T_NOSHEAR'] / stars["point"]['NGMIX_Tpsf_NOSHEAR'], stars["point"]['MAG_AUTO'],
    'g.',
    markersize=0.02,
    label="point-like stars"
)
plt.plot(
    stars["resol"]['NGMIX_T_NOSHEAR'] / stars["resol"]['NGMIX_Tpsf_NOSHEAR'], stars["resol"]['MAG_AUTO'],
    'r.',
    markersize=0.02,
    label="resolved stars"
)
plt.plot(
    ddc['NGMIX_T_NOSHEAR'] / ddc['NGMIX_Tpsf_NOSHEAR'], ddc['MAG_AUTO'],
    'b.',
    markersize=0.005,
    label="all objects"
)
plt.ylim(ylim)
plt.xlim(xlim)
plt.xlabel("T")
plt.ylabel("r")
plt.axvline(x=0.3, color='k', linewidth=1)
plt.axvline(x=0.01, color='g', linewidth=1)
plt.legend()

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

for key in stars_cal:
    print(key)
    print(stars_cal[key].R)

# +
idx = 0
jdx = 0
x_label = f"$R_{{{idx}{jdx}}}$"
y_label = "frequency"
n_bin = 100
out_path = f"hist_R_{idx}_{jdx}.pdf"
colors = ["blue", "green", "red"]
linestyles = ["-"] * 3
title = "Shear response"
x_range = [-2, 2]

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
        linestyles=linestyles                                                    
)
# -


