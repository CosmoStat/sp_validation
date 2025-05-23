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
#
# Run with Python 3.11 in sp_validation apptainer.

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

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


# %pwd

# Initialize calibration class instance
obj = sp_joint.CalibrateCat()

config = obj.read_config_set_params("config_minimal.yaml")

obj._params

# Get data. Set load_into_memory to False for very large files
dat, _ = obj.read_cat(load_into_memory=False)

# +
# Use only some columns in dat for efficiency

use_all_columns = True

if not use_all_columns:
    
    required_columns = set()
    for section, mask_list in config_data.items():
        for mask_params in mask_list:
            required_columns.add(mask_params["col_name"])

    #user_columns = ["NGMIX_T_NOSHEAR", "NGMIX_Tpsf_NOSHEAR", "NGMIX_FLUX_NOSHEAR", "NGMIX_FLUX_ERR_NOSHEAR"]      
    #required_columns.update(user_columns)

    print(f"{len(dat.dtype.names)} -> {len(required_columns)}")

    # Unsolved error here
    #dat = {col: obj._hd5file['data'][col][:] for col in required_columns if col in obj._hd5file['data']}
# -

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

# +
# Apply masks to data
#datm = dat[mask_combined._mask][mask_metacal]

# +
cat_gal = {}

calibration.fill_cat_gal(cat_gal, dat, g_uncorr, gal_metacal, mask_combined._mask, mask_metacal, purpose="leakage")
# -

df = calibration.build_df(cat_gal)

# +

num_bins_x = 4
num_bins_y = 4

quantities, bin_edges = calibration.get_quantities_binned(cat_gal, num_bins_x, num_bins_y)


# +
def plot_binned_one(ax, quantity, bin_edges_x, bin_edges_y, vmin=None, vmax=None, title=None, xlabel=None, ylabel=None):
    
    # Note: transpose R slice to match (y, x) shape required by pcolormesh
    pcm = ax.pcolormesh(bin_edges_x, bin_edges_y, quantity, vmin=vmin, vmax=vmax, shading="auto")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    plt.colorbar(pcm, ax=ax)
    

def plot_binned(quantities, key, bin_edges_x, bin_edges_y, title_base, vmin=None, vmax=None, xlabel=None, ylabel=None):
                
    len_shape = len(quantities[key].shape)
    
    fig_size = 2 * len_shape
    fig = plt.figure(figsize=(fig_size, fig_size))
    
    if len_shape == 2:

        ax = plt.subplot2grid((1, 1), (0, 0))
        plot_binned_one(ax, quantities[key].T, bin_edges_x, bin_edges_y, vmin=vmin, vmax=vmax, title=title_base, xlabel=xlabel, ylabel=ylabel)
 
    elif len_shape == 4:
        for idx in (0, 1):
            for jdx in (0, 1):
                ax = plt.subplot2grid((2, 2), (idx, jdx))
        
                if vmin is not None and vmax is not None:
                    if idx == jdx:
                        my_vmin = vmin["diag"]
                        my_vmax = vmax["diag"]
                    else:
                        my_vmin = vmin["offdiag"]
                        my_vmax = vmax["offdiag"]
                else:
                    my_vmin = None
                    my_vmax = None
                
                plot_binned_one(ax, quantities[key][:,:, idx, jdx].T, bin_edges_x, bin_edges_y, vmin=my_vmin, vmax=my_vmax, title=f"${title_base}_{{{idx+1}{jdx+1}}}$", xlabel=xlabel, ylabel=ylabel)


    plt.tight_layout()
    cs_plots.savefig(f"{key}_binned.png")

# +
vmin = {"diag": -0.2, "offdiag": -0.2}
vmax =  {"diag": 1.2, "offdiag": 0.2}

plot_binned(quantities, "response", bin_edges["snr"], bin_edges["size_ratio"], "R", vmin=vmin, vmax=vmax, xlabel="SNR", ylabel=r"$r / r_{\rm psf}$")
# -

plot_binned(quantities, "number", bin_edges["snr"], bin_edges["size_ratio"], "R", vmin=1, vmax=np.nanmax(quantities["number"]), xlabel="SNR", ylabel=r"$r / r_{\rm psf}$")

# +
vmin = {"diag": -0.2, "offdiag": -0.2}
vmax = {"diag": 0.2, "offdiag": 0.2}

plot_binned(quantities, "leakage", bin_edges["snr"], bin_edges["size_ratio"], r"\alpha", vmin=vmin, vmax=vmax, xlabel="SNR", ylabel=r"$r / r_{\rm psf}$")
# -

# %pwd


