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
import os

import healsparse as hsp
import matplotlib.pylab as plt
import numpy as np
from cs_util import cat as cs_cat
from cs_util import plots as cs_plots

from sp_validation import run_joint_cat as sp_joint

# %%
# Initialize calibration class instance (for config and data)
obj = sp_joint.CalibrateCat()

# %%
# Read configuration file and set parameters
config = obj.read_config_set_params("config_mask.yaml")

# %%
# Get data. Set load_into_memory to False for very large files
dat, dat_ext = obj.read_cat(load_into_memory=False)
# %%
# Additional parameters
key_ra = "RA"
key_dec = "Dec"

nside_min = 512
nside_max = 16384

# %%
# Create healsparse mask instance
hsp_obj = sp_joint.ApplyHspMasks()

# %% Mask directory and aux mask file(s)
hsp_obj._params["mask_dir"] = f"{os.environ['HOME']}/masks"
hsp_obj._params["aux_mask_files"] = f"{hsp_obj._params['mask_dir']}/mask_r_nside131072_npoint.hsp"
hsp_obj._params["aux_mask_labels"] = "npoint3"
hsp_obj._params["verbose"] = True

# %%
# Masks to use

# Bits
hsp_obj._params["bits"] = 1+2+4+8+64+1024

# Index of base mask (e.g., coverage mask)
label_base_mask = "npoint3"
#label_base_mask = ""

# Names
masks_to_apply = [
    "npoint3",
    "1_Faint_star_halos",
    "2_Bright_star_halos",
    "4_Stars",
    "8_Manual",
    "64_r",
    "1024_Maximask",
]

# Load and initialise masks
masks, labels = sp_joint.get_masks_from_config(
    config,
    dat,
    dat_ext,
    masks_to_apply=masks_to_apply,
    verbose=True
)

# %%
# if base mask: apply to all other masks
if label_base_mask != "":

    print(f"Applying base mask {label_base_mask} to all other masks:")

    # Find base mask
    for mask in masks:
        if mask._col_name == label_base_mask:
            base_mask = mask._mask
            break

    # Mask all others
    for mask in masks:
        if mask._col_name != label_base_mask:
            mask._mask = mask._mask & base_mask
            print(mask._col_name, "#True = ", sum(mask._mask), f"({100*sum(mask._mask)/len(mask._mask):.2f}%)")
else:
    print("No base mask applied.")

# %%
# Bin data

# Create array of nsides between min and max with powers of two
nsides = np.array([2**i for i in range(int(np.log2(nside_min)), int(np.log2(nside_max))+1)])

# %%
areas_deg2 = {}
for mask in masks:

    print(f"Mask: {mask._col_name}")
    areas_deg2[mask._col_name] = {}

    # Apply mask to positions
    m = mask._mask

    ra = dat[key_ra][m]
    dec = dat[key_dec][m]

    for nside in nsides:

            area_deg2 = cs_cat.get_binned_area(ra, dec, nside=nside)
            print(f"binned area (nside={nside}) ≈ {area_deg2:.3f} deg^2")
            areas_deg2[mask._col_name][nside] = area_deg2

# %%
def save_areas(areas_deg2, filename):
    """Save areas to a ASCII file.

    Parameters
    ----------
    areas_deg2 : dict
        Dictionary with mask names as keys and another dictionary as values.
        The inner dictionary has nsides as keys and areas in deg² as values.
    filename : str
        Output filename.
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("# Mask_name nside area_deg2\n")
        for mask_name, area_dict in areas_deg2.items():
            for nside, area in area_dict.items():
                f.write(f"{mask_name} {nside} {area:.6f}\n")

    print(f"Wrote areas to {filename}")

# %%

# %%
# Get area of hsp masks
area_wcov_deg2 = {}
if True:
    if label_base_mask != "":
        # Coverage = read aux file mask
        coverage = hsp.HealSparseMap.read(hsp_obj._params["aux_mask_files"])
        area_coverage = coverage.get_valid_area(degrees=True)
        area_wcov_deg2["npoint3"] = area_coverage

    # Loop over other, bit-coded masks
    paths = hsp_obj.get_paths_bit_masks()
    for bit, path in paths.items():
        print(bit, path)
        hsp_mask = hsp.HealSparseMap.read(path)

        if label_base_mask != "":
            # Apply coverage mask
            hsp_mask_wcov = coverage & (~hsp_mask)
        else:
            hsp_mask_wcov = ~hsp_mask

        key = hsp_obj.get_mask_col_name(bit)
        area_wcov_deg2[key] = hsp_mask_wcov.get_valid_area(degrees=True)

# %%
# Plot areas
markers = ['o', 's', '^', 'v', 'D', 'x', '*']
ls = ['dashed', 'dotted', 'dashdot', 'solid', (0, (3, 1, 1, 1)), (0, (5, 10)), (0, (1, 10))]
fig, ax = plt.subplots(figsize=(8,8))
plt.xticks(nsides, nsides)
for idx, mask in enumerate(masks):
    label = mask._col_name
    area_dict = areas_deg2[label]
    nsides = np.array(list(area_dict.keys()))
    dx = cs_plots.dx(idx, len(masks))
    areas = np.array(list(area_dict.values()))
    ax.plot(nsides * dx, areas, marker=markers[idx], label=label, linestyle=ls[idx])
    ax.hlines(
        area_wcov_deg2[label],
        nsides[0]*0.8,
        nsides[-1]*1.1,
        colors='C'+str(idx),
        linestyles=ls[idx],
    )
plt.legend()
plt.xlabel("nside")
plt.ylabel("binned area [deg²]")
_ = plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
# %%
cs_plots.savefig("binned_hsp_areas.png")
# %%
# Add hsp values to dict of binned catalogue with new "hsp" key
for key in areas_deg2:
    areas_deg2[key]["hsp"] = area_wcov_deg2[key]
# %%
# Write to disk
save_areas(areas_deg2, "areas.txt")
