# %%                                                                             
# hist_mag.py
#
# Plot magnitude histogram for various cuts and selection criteria

# %%
import matplotlib
import matplotlib.pylab as plt

# enable autoreload for interactive sessions                                     
from IPython import get_ipython                                                  
ipython = get_ipython()                                                          
if ipython is not None:                                                          
    ipython.run_line_magic("matplotlib", "inline")
    ipython.run_line_magic("reload_ext", "autoreload")                             
    ipython.run_line_magic("autoreload", "2")                                    
    ipython.run_line_magic("reload_ext", "log_cell_time")                          

# %%
import sys
import os
import re
import numpy as np
from astropy.io import fits
from io import StringIO

from sp_validation import run_joint_cat as sp_joint
from sp_validation import util
from sp_validation.basic import metacal
from sp_validation import calibration
import sp_validation.cat as cat

# %%
# Initialize calibration class instance
obj = sp_joint.CalibrateCat()

config = obj.read_config_set_params("config_mask.yaml")

hist_data_path = "magnitude_histograms_data.npz"

test_only = False

# %%
def get_data(obj, test_only=False):


    # Get data. Set load_into_memory to False for very large files
    dat, dat_ext = obj.read_cat(load_into_memory=False)

    if test_only:
        n_max = 1_000_000
        print(f"MKDEBUG testing only first {n_max} objects")
        dat = dat[:n_max]
        dat_ext = dat_ext[:n_max]

    return dat, dat_ext


def read_hist_data(hist_data_path):
    """
    Read histogram data from npz file.

    Parameters
    ----------
    hist_data_path : str
        Path to the npz file containing histogram data

    Returns
    -------
    hist_data : dict
        Dictionary with keys for each selection criterion containing:
        - 'counts': histogram counts
        - 'bins': bin edges
        - 'label': label for the histogram
    """
    loaded = np.load(hist_data_path, allow_pickle=True)
    hist_data = {}

    for key in loaded.files:
        data = loaded[key]
        hist_data[key] = {
            'counts': data[0],
            'bins': data[1],
            'label': str(data[2])
        }

    return hist_data



def get_mask(masks, col_name):

    # Get mask fomr masks with col_name = col_name
    for mask in masks:
        if mask._col_name == col_name:
            return mask


def compute_hist(masks, col_name, mask_cumul, mag, bins):
    """
    Compute histogram for given mask and magnitude data.

    Parameters
    ----------
    masks : list
        List of mask objects
    col_name : str
        Column name to identify the mask
    mask_cumul : array or None
        Cumulative mask array
    mag : array
        Magnitude data
    bins : array
        Bin edges for histogram

    Returns
    -------
    counts : array
        Histogram counts
    bins : array
        Bin edges
    label : str
        Label for the histogram
    n_valid : int
        Number of valid data points after masking
    mask_cumul : array
        Updated cumulative mask array
    """
    this_mask = get_mask(masks, col_name)

    # First time:
    if mask_cumul is None:
        print("Init mask_cumul")
        mask_cumul = this_mask._mask
    else:
        mask_cumul &= this_mask._mask

    # Data values
    my_mag = mag[mask_cumul]

    # Data count
    n_valid = np.sum(mask_cumul)

    # Label
    string_buffer = StringIO()
    this_mask.print_condition(string_buffer)
    label = string_buffer.getvalue().strip()
    if label == "":
        label = col_name

    counts, bin_edges = np.histogram(my_mag, bins=bins)

    return counts, bin_edges, label, n_valid, mask_cumul


def plot_hist(counts, bins, label, alpha=1, ax=None, color=None):
    """
    Plot histogram from counts and bins using bar plot.

    Parameters
    ----------
    counts : array
        Histogram counts
    bins : array
        Bin edges
    label : str
        Label for the histogram
    alpha : float
        Transparency for plot
    ax : matplotlib axis
        Axis to plot on
    color : str or None
        Color for the histogram

    """
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    ax.bar(
        bin_centers,
        counts,
        width=np.diff(bins),
        alpha=alpha,
        label=label,
        align='center',
        color=color
    )

# %%

if os.path.exists(hist_data_path):
    print(f"Histogram data file {hist_data_path} found.")
    print("Reading and plotting.")

    dat = dat_ext = None
    hist_data = read_hist_data(hist_data_path)

else:
    print(f"Histogram data file {hist_data_path} not found.")
    print("Reading UNIONS cat and computing.")

    dat, dat_ext = get_data(obj, test_only=test_only)
    hist_data = None
    

# ## Masking
# %%
# Get all masks, with or without dat, dat_ext
masks, labels = sp_joint.get_masks_from_config(
    config,
    dat,
    dat_ext,
    verbose=obj._params["verbose"],
)

# %%
# List of basic masks to apply to all cases
masks_labels_basic = [
    "FLAGS",
    "overlap",
    "IMAFLAGS_ISO",
    "mag",
    "NGMIX_MOM_FAIL",
    "NGMIX_ELL_PSFo_NOSHEAR_0",
    "NGMIX_ELL_PSFo_NOSHEAR_1",
    "4_Stars",
    "8_Manual",
    "64_r",
    "1024_Maximask",
]

# %%
masks_basic = []
for mask in masks:
    if mask._col_name in masks_labels_basic:
        masks_basic.append(mask)

if dat is not None:
    print("Creating combined basic mask")
    masks_basic_combined = sp_joint.Mask.from_list(
        masks_basic,
        label="combined",
        verbose=obj._params["verbose"],
    )
else:
    # Create dummy combined mask (for plot label)
    print("Creating dummy combined basic mask")
    masks_basic_combined = sp_joint.Mask(
        "combined",
        "combined",
        kind="none",
    )

masks.append(masks_basic_combined)

# %%
# Metacal mask (cuts)
mask_tmp = sp_joint.Mask(
    "metacal",
    "metacal",
    kind="none",
)

# %%
if dat is not None:
    cm = config["metacal"]
    gal_metacal = metacal(                                                           
        dat,                                                                         
        masks_basic_combined._mask,                                                         
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

    g_corr_mc, g_uncorr, w, mask_metacal, c, c_err = (
        calibration.get_calibrated_m_c(gal_metacal)
    )

    # Convert index array to boolean mask
    mask_metacal_bool = np.zeros(len(dat), dtype=bool)
    mask_metacal_bool[mask_metacal] = True

    mask_tmp._mask = mask_metacal_bool

masks.append(mask_tmp)


# %%
# Plot magnitude histograms for various selection criteria

# Define magnitude bins
mag_bins = np.arange(15, 30, 0.05)
mag_centers = 0.5 * (mag_bins[:-1] + mag_bins[1:])


# %%
# Create figure with multiple subplots
figsize = 10
alpha = 0.5

col_names = ["combined", "N_EPOCH", "npoint3", "metacal"]

# Define explicit colors for each histogram
colors = ['C0', 'C1', 'C2', 'C3']  # Use matplotlib default color cycle
color_map = dict(zip(col_names, colors))


# %%
# If hist_data not loaded, compute it
if hist_data is None:
    hist_data = {}

if dat is not None:
    # Get magnitude column
    mag = dat['mag']

    mask_cumul = None
    for col_name in col_names:
        counts, bins, label, n_valid, mask_cumul = compute_hist(
            masks=masks,
            col_name=col_name,
            mask_cumul=mask_cumul,
            mag=mag,
            bins=mag_bins
        )
        hist_data[col_name] = {
            'counts': counts,
            'bins': bins,
            'label': label,
            'n_valid': n_valid,
        }

# Plot histogram data
plt.figure()
fig, (ax) = plt.subplots(1, 1, figsize=(figsize, figsize))

for col_name in col_names:
    if col_name in hist_data:
        data = hist_data[col_name]
        plot_hist(
            data['counts'],
            data['bins'],
            data['label'],
            alpha=alpha,
            ax=ax,
            color=color_map[col_name]
        )
        #print(f"{col_name}: n_valid = {data['n_valid']}")

ax.set_xlabel('$r$')
ax.set_ylabel('Number')
ax.set_xlim(17.5, 26.5)
ax.legend()

plt.tight_layout()
plt.savefig('magnitude_histograms.png', dpi=150, bbox_inches='tight')

# Save histogram data to file (only if we computed it)
if dat is not None:
    np.savez(
        hist_data_path,
        **{
            key: np.array(
                [
                    val['counts'],
                    val['bins'],
                    val['label'],
                    val["n_valid"],
                ],
                dtype=object
            )
            for key, val in hist_data.items()
        }
    )
    print(f"Histogram data saved to {hist_data_path}")

# %%
if dat is not None:
    obj.close_hd5()
# %%
