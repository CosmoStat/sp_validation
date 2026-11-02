"""
Script to calculate the sample properties and generate a latex table for the paper.
Author: Sacha Guerrini
"""

# %%
import os
import sys
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits

try:
    from IPython import get_ipython

    ipython = get_ipython()
except Exception:
    ipython = None

# enable autoreload for interactive session
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
else:
    # Force unbuffered stdout and stderr
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# Load matplotlib plot style
plt.style.use("./matplotlib_config/paper.mplstyle")

# Load seaborn color palette
sns.set_palette("Dark2")

# %%
psfex_path = (
    "/n17data/UNIONS/WL/v1.4.x/v1.4.6.3/unions_shapepipe_cut_struc_2024_v1.4.6.3.fits"
)
mccd_path = (
    "/n17data/UNIONS/WL/v1.3.x/v1.3.6.3/unions_shapepipe_cut_struc_2022_v1.3.6.3.fits"
)
psfex_mask_path = "/home/guerrini/sp_validation/cosmo_inference/data/mask/mask_map_footprint_nside_4096.fits"
mccd_mask_path = "/n17data/cdaley/unions/analyses/shear_2d/bmodes_2d/mccd_v1363_inputs/handoff/v1.3.6.3_blind_A/mask_map_footprint_v1.3.6.3_nside_4096.fits"


# Load the catalogs
psfex_data = fits.getdata(psfex_path)
mccd_data = fits.getdata(mccd_path)
psfex_mask = hp.read_map(psfex_mask_path)
mccd_mask = hp.read_map(mccd_mask_path)


# %%
area_psfex = np.sum(psfex_mask > 0) * hp.nside2pixarea(
    hp.get_nside(psfex_mask), degrees=True
)
area_mccd = np.sum(mccd_mask > 0) * hp.nside2pixarea(
    hp.get_nside(mccd_mask), degrees=True
)


# %%
def number_density(mask, data):
    """
    Calculate the number density of sources in a given mask and data.
    """
    area = np.sum(mask > 0) * hp.nside2pixarea(hp.get_nside(mask), degrees=True)
    return 1 / (area * 3600) * np.sum(data["w_des"]) ** 2 / np.sum(data["w_des"] ** 2)


def shape_noise(data):
    """
    Calculate the shape noise of a given data.
    """
    e1 = data["e1_leak_corrected"]
    e2 = data["e2_leak_corrected"]
    w = data["w_des"]
    return float(
        np.sqrt(
            0.5 * (np.average(e1**2, weights=w**2) + np.average(e2**2, weights=w**2))
        )
    )


# %%
n_eff_psfex = number_density(psfex_mask, psfex_data)
n_eff_mccd = number_density(mccd_mask, mccd_data)

# %%
sigma_e_psfex = shape_noise(psfex_data)
sigma_e_mccd = shape_noise(mccd_data)

# %%
# Write a tex file containing a Table summarising the results
outpath_table = Path("Tables/sample_properties.tex")
with open(outpath_table, "w") as f:
    f.write("\\begin{table}[h!]\n")
    f.write("\\centering\n")
    f.write("\\begin{tabular}{lcc}\n")
    f.write("\\hline\n")
    f.write("\\hline\n")
    f.write(" & \\textbf{PSFEx} & \\textbf{MCCD} \\\\\n")
    f.write("\\hline\n")
    f.write(f"Area (deg$^2$) & {area_psfex:.0f} & {area_mccd:.0f} \\\\\n")
    f.write(
        f"$n_{{\\rm eff}}$ (arcmin$^{{-2}}$) & {n_eff_psfex:.3f} & {n_eff_mccd:.3f} \\\\\n"
    )
    f.write(f"$\\sigma_e$ & {sigma_e_psfex:.3f} & {sigma_e_mccd:.3f} \\\\\n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write(
        "\\caption{Properties of the weak lensing samples for the PSFEx and MCCD catalogs.}\n"
    )
    f.write("\\label{tab:sample_properties}\n")
    f.write("\\end{table}\n")
# %%
