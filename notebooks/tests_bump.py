# %%
# Tests of rhe bump in xi_+

# %%                                                                             
from IPython import get_ipython                                                  

ipython = get_ipython()                                                          

# enable autoreload for interactive sessions                                     
if ipython is not None:                                                          
    ipython.run_line_magic("load_ext", "autoreload")                             
    ipython.run_line_magic("autoreload", "2")
    ipython.run_line_magic("load_ext", "log_cell_time")
    ipython.run_line_magic("matplotlib", "inline")

# %%
import os
import numpy as np
import pandas as pd
import treecorr

import matplotlib.pyplot as plt

from cs_util import plots as cs_plots

from sp_validation import run_joint_cat as sp_joint
from sp_validation import cat as cat
from sp_validation.basic import metacal
from sp_validation import calibration
from sp_validation import io
from sp_validation import plots as sp_plots

# %%
print("cwd = ", os.getcwd())

# %%
# Initialize calibration class instance
obj = sp_joint.CalibrateCat()
config = obj.read_config_set_params("config_minimal.yaml")

# %%
# Get data. Set load_into_memory to False for very large files
dat, _ = obj.read_cat(load_into_memory=False)

# %%
n_test = -1
#n_test = 1_000_000
if n_test > 0:
    print(f"MKDEBUG testing only first {n_test} objects")
    dat = dat[:n_test]

# %%
def get_combined_mask(config, dat, masks_to_apply, verbose=False):

    # Gather mask information for above list from config
    masks, labels = sp_joint.get_masks_from_config(
        config,
        dat,
        dat,
        masks_to_apply=masks_to_apply,
        verbose=verbose,
    )

    # Initialise combined mask
    label = "comb"
    my_mask = sp_joint.Mask(label, label, kind="combined", value=None)

    # Combine masks
    mask_combined = sp_joint.Mask.from_list(
        masks,
        label="combined",
        verbose=verbose,
    )

    if True or verbose:
        # Output some mask statistics
        sp_joint.print_mask_stats(dat.shape[0], masks, mask_combined)

    return mask_combined

# %%
def xi_from_dat(dat, mask_combined, cm, config_treecorr, n_cpu):

    # Call metacal
    gal_metacal = metacal(
        dat,
        mask_combined._mask,
        snr_min=cm["gal_snr_min"],
        snr_max=cm["gal_snr_max"],
        rel_size_min=cm["gal_rel_size_min"],
        rel_size_max=cm["gal_rel_size_max"],
        size_corr_ell=cm["gal_size_corr_ell"],
        sigma_eps=cm["sigma_eps_prior"],
        global_R_weight=cm["global_R_weight"],
        col_2d=False,
        verbose=True,
    )

    # Get metacal outputs; here mask is needed
    g_corr_mc, g_uncorr, w, mask_metacal, c, c_err = (
        calibration.get_calibrated_m_c(gal_metacal)
    )

    # Compute DES weights
    cat_gal = {}

    sp_joint.compute_weights_gatti(
        cat_gal,
        g_uncorr,
        gal_metacal,
        dat,
        mask_combined,
        mask_metacal,
        num_bins=20,
    )

    # Correct for PSF leakage
    alpha_1, alpha_2 = sp_joint.compute_PSF_leakage(
        cat_gal,
        g_corr_mc,
        dat,
        mask_combined,
        mask_metacal,
        num_bins=20,    
    )

    # Compute leakage-corrected ellipticities
    e1_leak_corrected = g_corr_mc[0] - alpha_1 * cat_gal["e1_PSF"]
    e2_leak_corrected = g_corr_mc[1] - alpha_2 * cat_gal["e2_PSF"]


    ra = cat.get_col(dat, "RA", mask_combined._mask, mask_metacal)
    dec = cat.get_col(dat, "Dec", mask_combined._mask, mask_metacal)

    gg = treecorr.GGCorrelation(config_treecorr, var_method='jackknife')

    npatch = 20
    catalog = treecorr.Catalog(                                                         
        ra=ra,                                               
        dec=dec,                                                    
        g1=e1_leak_corrected,                                                              
        g2=e2_leak_corrected,                                                       
        w=cat_gal["w_des"],                                                           
        ra_units="deg",                                        
        dec_units="deg",
        npatch=npatch,                                
    )

    gg.process(catalog, catalog, num_threads=n_cpu)

    return gg

# %%
# Set up correlation
# %%
minsep = 1
maxsep = 50
nbins = 10
n_cpu = 24
config_treecorr = {
    "ra_units": "deg",
    "dec_units": "deg",
    "min_sep": minsep,
    "max_sep": maxsep,
    "nbins": nbins,
    "num_threads": n_cpu,
    "sep_units": "arcmin",
    "var_method": "jackknife",
    "bin_type": "Log",
}
cm = config["metacal"]


# %%
# Basic masking
masks_to_apply = []

masks_to_add = [
    "4_Stars",
    "1024_Maximask",
    "N_EPOCH",
    "2_Bright_star_halos",
    "1_Faint_star_halos",
    "npoint3",
]

# %%
# Main loop
ggs = []
for mask in masks_to_add:
    
    print(f"Adding mask {mask}...")
    
    masks_to_apply.append(mask)

    # Get combined mask
    mask_combined = get_combined_mask(
        config,
        dat,
        masks_to_apply,
        verbose=obj._params["verbose"],
    )

    # Compute xi_+ from data
    print("Compute xi")
    gg = xi_from_dat(dat, mask_combined, cm, config_treecorr, n_cpu)
    
    xi_out_name = f"xi_{mask}.txt"
    gg.write(xi_out_name)
    ggs.append(gg)
# %%
