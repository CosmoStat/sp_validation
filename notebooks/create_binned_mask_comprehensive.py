# create_binned_mask_comprehensive.py

# Create healpy mask file from binning the comprehensive catalogue

# %%
from IPython import get_ipython

# %%
# enable autoreload for interactive sessions
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
    ipython.run_line_magic("load_ext", "log_cell_time")
# %%
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# %%
import sys
import os
import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt
import healpy as hp
import healsparse as hsp

from cs_util import plots as cs_plots
from cs_util import cat as cs_cat

from sp_validation import run_joint_cat as sp_joint
from sp_validation import util
from sp_validation.basic import metacal
from sp_validation import calibration
import sp_validation.cat as cat

# %%
# Initialize calibration class instance (for config and data)
obj = sp_joint.CalibrateCat()

# %%
# Read configuration file and set parameters
config = obj.read_config_set_params("config_for_compr_mask.yaml")

# %%
# Get data. Set load_into_memory to False for very large files
dat, dat_ext = obj.read_cat(load_into_memory=False)

# %%
n_test = -1
# n_test = 100_000
if n_test > 0:
    print(f"MKDEBUG testing only first {n_test} objects")
    dat = dat[:n_test]
    dat_ext = dat_ext[:n_test]
    test = "_test"
else:
    test = ""

# %%
# Additional parameters
key_ra = "RA"
key_dec = "Dec"

# %%
# Create healsparse mask instance
hsp_obj = sp_joint.ApplyHspMasks()

# %% Mask directory and aux mask file(s)
hsp_obj._params["mask_dir"] = f"{os.environ['HOME']}/masks"
hsp_obj._params["aux_mask_files"] = (
    f"{hsp_obj._params['mask_dir']}/mask_r_nside131072_npoint.hsp"
)
hsp_obj._params["aux_mask_labels"] = "npoint3"
hsp_obj._params["verbose"] = True

# %%
# Load and initialise masks
masks, labels = sp_joint.get_masks_from_config(
    config, dat, dat_ext, verbose=True
)

mask_combined = sp_joint.Mask.from_list(
    masks,
    label="combined",
    verbose=obj._params["verbose"],
)

# Apply mask to positions
m = mask_combined._mask
ra = dat[key_ra][m]
dec = dat[key_dec][m]

cbm = config["binned_mask"]
nside = cbm["nside"]
area_deg2, pix = cs_cat.get_binned_area(ra, dec, nside=nside, return_pix=True)
print(f"binned area (nside={nside}) ≈ {area_deg2:.3f} deg^2")

healpix_map = np.full(hp.nside2npix(nside), not cbm["good"])
healpix_map[pix] = cbm["good"]
output_path = f"{cbm['output_base']}_{nside}{test}.fits"
print(f"Writing binned mask to file {output_path}...")

header = fits.Header()
for my_mask in masks:
    my_mask.add_summary_to_FITS_header(header)

# Transform in list form
extra_header = [
    (key, header[key], header.comments[key]) for key in header.keys()
]

hp.write_map(
    output_path,
    healpix_map,
    overwrite=True,
    extra_header=extra_header,
)


# %%
def save_area(area_deg2, filename):
    """Save area to a ASCII file.

    Parameters
    ----------
    areas_deg2 : float
        mask area in square degrees
    filename : str
        Output filename.
    """
    with open(filename, "w") as f:
        # Write header
        f.write("# nside area [deg^2]\n")
        f.write(f"{nside} {area_deg2:.6f}\n")

    print(f"Wrote areas to {filename}")


# %%
save_area(area_deg2, f"area{test}.txt")
