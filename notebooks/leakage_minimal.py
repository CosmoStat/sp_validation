# %%
# Plot binned quantites, see leakage_minima.py

# %%                                                                             
from IPython import get_ipython                                                  

ipython = get_ipython()                                                          

# enable autoreload for interactive sessions                                     
if ipython is not None:                                                          
    ipython.run_line_magic("load_ext", "autoreload")                             
    ipython.run_line_magic("autoreload", "2")
    ipython.run_line_magic("load_ext", "log_cell_time")

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from cs_util import plots as cs_plots

from sp_validation import run_joint_cat as sp_joint
from sp_validation import cat as sp_cat
from sp_validation.basic import metacal
from sp_validation import calibration
from sp_validation import io
from sp_validation import plots as sp_plots

# %%
os.getcwd()

# %%
# Initialize calibration class instance
obj = sp_joint.CalibrateCat()

config = obj.read_config_set_params("config_minimal.yaml")


# Get data. Set load_into_memory to False for very large files
dat, _ = obj.read_cat(load_into_memory=False)

# %%
n_test = -1
#n_test = 1_000_000
if n_test > 0:
    print(f"MKDEBUG testing only first {n_test} objects")
    dat = dat[:n_test]

# %%
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

# %%
# Look for masks that were already applied to data if this 
# information is in header
if "applied_masks" in obj._hd5file:
    applied_masks = obj._hd5file["applied_masks"]
    applied_masks["desc"]
else:
    applied_masks = None

# ## Masking
masks_to_apply = [
    "N_EPOCH",
    "FLAGS",
    "4_Stars",
    "npoint3",
]

# %%
# Check that mask has not already been applied
for mask in masks_to_apply:
    if applied_masks and mask in applied_masks["desc"]:
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

# %%
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

# Get metacal outputs; here mask is needed
g_corr_mc, g_uncorr, w, mask_metacal, c, c_err = calibration.get_calibrated_m_c(gal_metacal)

# %%
cat_gal = {}

calibration.fill_cat_gal(cat_gal, dat, g_uncorr, gal_metacal, mask_combined._mask, mask_metacal, purpose="leakage")


df = calibration.build_df(cat_gal)

# %

num_bins_x = 12
num_bins_y = 12

quantities, bin_edges = calibration.get_quantities_binned(cat_gal, num_bins_x, num_bins_y)

# Save binned matrices to files
for key in quantities:
    io.write_binned_quantity(quantities[key], key, bin_edges)

# %%
vmin = {"diag": -0.2, "offdiag": -0.2}
vmax =  {"diag": 1.2, "offdiag": 0.2}

sp_plots.plot_binned(quantities, "response", bin_edges["snr"], bin_edges["size_ratio"], "R", vmin=vmin, vmax=vmax, xlabel="SNR", ylabel=r"$r / r_{\rm psf}$")

# %%
sp_plots.plot_binned(quantities, "number", bin_edges["snr"], bin_edges["size_ratio"], "R", vmin=1, vmax=np.nanmax(quantities["number"]), xlabel="SNR", ylabel=r"$r / r_{\rm psf}$")

# %%
vmin = {"diag": -0.2, "offdiag": -0.2}
vmax = {"diag": 0.2, "offdiag": 0.2}

sp_plots.plot_binned(quantities, "leakage", bin_edges["snr"], bin_edges["size_ratio"], r"\alpha", vmin=vmin, vmax=vmax, xlabel="SNR", ylabel=r"$r / r_{\rm psf}$")
