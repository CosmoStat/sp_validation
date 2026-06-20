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

from sp_validation import catalog_builders as sp_joint
from sp_validation import format
from sp_validation.calibration import metacal
from sp_validation import calibration
import sp_validation.cat as cat

# %%
# Initialize calibration class instance
obj = sp_joint.CalibrateCat()

config = obj.read_config_set_params("config_mask.yaml")

test_only = True

# %%
# Funcitons
def get_data(obj, test_only=False):
    """Get Data.
    
    Returns catalogue.
    
    Parameters
    ----------
    obj : CalibrateCat instance
        Instance of CalibrateCat class
    test_only : bool, optional 
        If True, only load a subset of data for testing;
        default is False.
    
    """
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
    Read Hist Data.

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
    """Get Mask.
    
    Returns mask corresponding to col_name.
    
    Parameters
    ----------
    masks : list
        List of mask objects
    col_name : str
        Column name to identify the mask
    Returns
    -------
    list
        Mask object
    integer
        Mask position in list
    
    """
    # Get mask fomr masks with col_name = col_name
    for idx, mask in enumerate(masks):
        if mask._col_name == col_name:
            return mask, idx


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
    this_mask, _ = get_mask(masks, col_name)

    # First time:
    if mask_cumul is None:
        mask_cumul = this_mask._mask
    else:
        mask_cumul &= this_mask._mask

    # Data values
    my_mag = mag[mask_cumul]

    # Data count
    n_valid = np.sum(mask_cumul)

    # Label
    string_buffer = StringIO()
    this_mask.print_condition(string_buffer, latex=True)
    label = string_buffer.getvalue().strip()
    if label == "":
        label = col_name
    print("MKDEBUG", label)

    counts, bin_edges = np.histogram(my_mag, bins=bins)

    return counts, bin_edges, rf"{label}", n_valid, mask_cumul


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


def plot_all_hists(
    hist_data,
    col_names,
    figsize=10,
    alpha=0.5,
    color_map=None,
    fraction=False,
    ax=None,
    out_path=None,
):

    if ax is None:
        plt.figure()
        fig, (ax) = plt.subplots(
            1,
            1,
            figsize=(figsize, figsize)
        )

    counts0 = None
    for col_name in col_names:
        if col_name in hist_data:

            data = hist_data[col_name]
            if fraction:
                if counts0 is None:
                    counts0 = data["counts"]
                counts = data["counts"] / counts0
            else:
                counts = data["counts"]

            plot_hist(
                counts,
                data['bins'],
                data['label'],
                alpha=alpha,
                ax=ax,
                color=color_map[col_name]
            )
            #print(f"{col_name}: n_valid = {data['n_valid']}")

    ax.set_xlabel('$r$')
    ylabel = "fraction" if fraction else "number"
    ax.set_ylabel(ylabel)
    ax.set_xlim(17.5, 26.5)
    if not fraction:
        ax.legend()

    if out_path:
        plt.tight_layout()
        plt.savefig(
            out_path,
            dpi=150,
            bbox_inches='tight'
        )


# %%
# Main program
scenario = 1
hist_data_path = f"magnitude_histograms_data_scenario-{scenario}.npz"

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
    

# %%
# Masking
# Get all masks, with or without dat, dat_ext
masks, labels = sp_joint.get_masks_from_config(
    config,
    dat,
    dat_ext,
    verbose=obj._params["verbose"],
)

# %%
# Combine mask according to scenario
# List of basic masks to apply to all cases

masks_labels_basic = ["overlap", "mag", "64_r"]
col_names = ["basic masks"]

if scenario == 0:
    masks_labels_basic.extend([
        "FLAGS",
        "IMAFLAGS_ISO",
        "NGMIX_MOM_FAIL",
        "NGMIX_ELL_PSFo_NOSHEAR_0",
        "NGMIX_ELL_PSFo_NOSHEAR_1",
        "4_Stars",
        "8_Manual",
        "1024_Maximask",
    ])

    col_names.extend(["N_EPOCH", "npoint3", "metacal"])

elif scenario == 1:
    
    col_names.extend([
        "IMAFLAGS_ISO",
        "FLAGS",
        "NGMIX_MOM_FAIL",
        "NGMIX_ELL_PSFo_NOSHEAR_0",
        "NGMIX_ELL_PSFo_NOSHEAR_1",
        "4_Stars",
        "8_Manual",
        "1024_Maximask",
        "N_EPOCH",
        "npoint3",
        "metacal",
    ])
    
    combine_cols = {
        "ngmix failures": [
            "NGMIX_MOM_FAIL",
            "NGMIX_ELL_PSFo_NOSHEAR_0",
            "NGMIX_ELL_PSFo_NOSHEAR_1",
        ]
    }
    
# %%
# Combine columns if specified.
# Remove old columns after combining.
if combine_cols is not None:
    for new_col, old_cols in combine_cols.items():
        print(f"Combining columns {old_cols} into {new_col}")
        # Create combined mask
        old_masks = []
        idx_first = None
        for col in old_cols:
            mask, idx = get_mask(masks, col)
            old_masks.append(mask)
            if idx_first is None:
                idx_first = idx
        if dat is not None:
            print(f"Creating combined mask for {new_col}")
            masks_combined = sp_joint.Mask.from_list(
                old_masks,
                label=new_col,
                verbose=obj._params["verbose"],
            )
        else:
            print(f"Creating dummy mask for {new_col} (for plot label)")
            masks_combined = sp_joint.Mask(
                new_col,
                new_col,
                kind="none",
            )
        masks.insert(idx, masks_combined)
        col_names.insert(idx, new_col)
        
        for old_mask, old_col in zip(old_masks, old_cols):
            masks.remove(old_mask)
            col_names.remove(old_col)

    print("After combining: masks =", [mask._col_name for mask in masks])

# %%
# Createe list of masks
masks_basic = []
for mask in masks:
    if mask._col_name in masks_labels_basic:
        masks_basic.append(mask)

if dat is not None:
    print("Creating combined basic mask")
    masks_basic_combined = sp_joint.Mask.from_list(
        masks_basic,
        label="basic masks",
        verbose=obj._params["verbose"],
    )
else:
    # Create dummy combined mask (for plot label)
    print("Creating dummy combined basic mask")
    masks_basic_combined = sp_joint.Mask(
        "basic masks",
        "basic masks",
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
def get_info_for_metacal_masking(dat, mask, prefix = "NGMIX", name_shear = "NOSHEAR"):

    res = {}

    res["flag"] = dat[mask][f"{prefix}_FLAGS_{name_shear}"]

    for key in ("flux", "flux_err", "T"):
        res[key] = dat[mask][f"{prefix}_{key.upper()}_{name_shear}"]
    res["Tpsf"] = dat[mask][f"{prefix}_Tpsf_{name_shear}"]
    
    return res

# %%
if dat is not None:
    cm = config["metacal"]



# %%
# Call metacal if data is available
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
# Define magnitude bins
mag_bins = np.arange(15, 30, 0.05)
mag_centers = 0.5 * (mag_bins[:-1] + mag_bins[1:])

# %%
# Create figure with multiple subplots
figsize = 10
alpha = 0.5

# Define explicit colors for each histogram
colors = [f'C{i}' for i in range(len(col_names))]  # Use matplotlib default color cycle
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
# %%
# Create plots
fig, axes = plt.subplots(
    1,
    2,
    figsize=(2 * figsize, figsize)
)
# Plot histogram data
plot_all_hists(
    hist_data,
    col_names,
    alpha=alpha,
    color_map=color_map,
    ax=axes[0],
)
plot_all_hists(
    hist_data,
    col_names,
    alpha=alpha,
    color_map=color_map,
    fraction=True,
    ax=axes[1],
)
plt.tight_layout()
out_path = f"magnitude_histograms_scenario-{scenario}.png"
plt.savefig(
    out_path,
    dpi=150,
    bbox_inches='tight'
)

# %%
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
for mask in masks:
    mask.print_condition(sys.stdout, latex=True)

# %%
# print number of valid objects and name
for data in hist_data

# %%
