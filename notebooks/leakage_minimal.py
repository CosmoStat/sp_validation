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

# # Leakage of minimal catalogue

# %reload_ext autoreload
# %autoreload 2

# +
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from cs_util import plots as cs_plots

from sp_validation import run_joint_cat as sp_joint
from sp_validation import cat as sp_cat
from sp_validation.basic import metacal
from sp_validation import calibration
# -


# Initialize calibration class instance
obj = sp_joint.CalibrateCat()

config = obj.read_config_set_params("config_minimal.yaml")

obj._params

# Get data. Set load_into_memory to False for very large files
dat, _ = obj.read_cat(load_into_memory=False)

applied_masks = obj._hd5file["applied_masks"]

applied_masks["desc"]

# ## Masking

masks_to_apply = [
    "N_EPOCH",
    "FLAGS",
    "4_Stars",
    "npoint3",
]

# Check that mask has not already been applied
for mask in masks_to_apply:
    if mask in applied_masks["desc"]:
        print(f"Warning: Mask {mask} has already been applied")
    else:
        pass

# Gather mask information for above list from config
masks, labels = sp_joint.get_masks_from_config(config, dat, dat, masks_to_apply=masks_to_apply, verbose=obj._params["verbose"])

# Initialise combined mask
label = "comb"
my_mask = sp_joint.Mask(label, label, kind="combined", value=None)

# Combine masks
mask_combined = sp_joint.Mask.from_list(
    masks,
    label="combined",
    verbose=obj._params["verbose"],
)

# Output some mask statistics
sp_joint.print_mask_stats(dat.shape[0], masks, mask_combined)

# +
# For testing!!
#mask_combined._mask[10000:] = False

# +
# Call metacal

cm = config["metacal"]

gal_metacal = metacal(
    dat,
    mask_combined._mask,
    snr_min=cm["gal_snr_min"],
    snr_max=cm["gal_snr_max"],
    rel_size_min=cm["gal_rel_size_min"],
    rel_size_max=cm["gal_rel_size_max"],
    size_corr_ell=cm["gal_size_corr_ell"],
    sigma_eps=cm["sigma_eps_prior"],
    col_2d=False,
    verbose=True,
)
# -

# Get metacal outputs; here mask is needed
g_corr_mc, g_uncorr, w, mask_metacal, c, c_err = calibration.get_calibrated_m_c(gal_metacal)

# Apply masks to data
datm = dat[mask_combined._mask][mask_metacal]

dx = len(datm) - len(gal_metacal.R11)
print(dx, len(datm))

# +
cat_gal = {}

calibration.fill_cat_gal(cat_gal, dat, g_uncorr, gal_metacal, mask_combined, mask_metacal)

# +

num_bins = 20

R, bin_edges = calibration.get_response_binned(cat_gal, num_bins)

# Write to ascii file

# Save edges
for key in bin_edges:
    np.savetxt(f"bin_edges_{key}.txt", bin_edges[key])

# Flatten R to save
R_flat = R.reshape(-1, 1)
np.savetxt("R.txt", R_flat)

# To read:
#for key in bin_edges:
    #bin_edges[key] = np.loadtxt(f"bin_edges_{key}.txt")
# R_flat = np.loadtxt("R.txt")
# R = R_flat.reshape(20, 20, 2, 2)


# +
fig = plt.figure(figsize=(8, 8))

extent = [bin_edges["snr"][0], bin_edges["snr"][-1], bin_edges["size"][0], bin_edges["size"][-1]]

for idx in (0, 1):
    for jdx in (0, 1):
        ax = plt.subplot2grid((2, 2), (idx, jdx))
        
        plt.imshow(R[:,:,idx,jdx], vmin=-1, vmax=1, origin="lower", extent=extent, aspect="auto")
        plt.xscale('log')
        plt.yscale("log")
        plt.title(f"$R_{{{idx+1}{jdx+1}}}$")
        plt.xlabel("SNR")
        plt.ylabel(r"$r / r_{\rm psf}$")

        plt.colorbar()        
plt.tight_layout()
# -

cs_plots.savefig("R_binned.png")
