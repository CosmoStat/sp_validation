# %%
import os
import configparser
import subprocess
import sys

# Append any useful folder in the path
sys.path.append(
    "/home/guerrini/sp_validation/cosmo_inference/scripts/"
)
sys.path.append(
    "/home/guerrini/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_unblinding/"
)

from getdist import plots, loadMCSamples
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.stats as stats
from IPython.display import Markdown, display
import healpy as hp
import matplotlib.scale as mscale
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
import seaborn as sns

plt.style.use(
    "/home/guerrini/sp_validation/cosmo_inference/notebooks/2D_harmonic_space_cosmic_shear_plots/matplotlib_config/paper.mplstyle"
)

plt.rcParams["text.usetex"] = True

sns.set_palette("husl")

from sp_validation.rho_tau import SquareRootScale
mscale.register_scale(SquareRootScale)

import IPython
ipython = IPython.get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

import chain_postprocessing as cp
import utils

g = plots.get_subplot_plotter(width_inch=30)
g.settings.axes_fontsize=30
g.settings.axes_labelsize=30
g.settings.alpha_filled_add = 0.7
g.settings.legend_fontsize = 40

# Directory where the chains are located
root_dir = "/n09data/guerrini/output_chains"

# THE BLIND TO USE FOR THE PLOTS
blind = "B" # Options are "A", "B", or "C"
catalog_version = "SP_v1.4.6.3"
fiducial_root_cell = f"SP_v1.4.6.3_leak_corr_{blind}"
label_fiducial_cell = r"UNIONS $C_{\ell}$"
fiducial_root_xi = f"SP_v1.4.6.3_{blind}_fiducial_config"
label_fiducial_xi = r"UNIONS $\xi_{\pm}$"

# Path to the ini files used
path_ini_files = '/home/guerrini/sp_validation/cosmo_inference/cosmosis_config'
path_datavectors = '/home/guerrini/sp_validation/cosmo_inference/data/'
path_output_chains = "/n09data/guerrini/output_chains/"

# %%
# 0. Do a funny print with emojis for the unblinding
display(
    Markdown(
        "## 🎉 Let the Unblinding Party begin 🎉"
    )
)
# %%
# 1. Plot the datavectors without best-fit
display(
    Markdown(
        "### 1. Plot the datavectors without best-fit"
    )
)

# Plot Cells EE
data = fits.open(
    os.path.join(
        path_datavectors,
        f"{fiducial_root_cell}/cosmosis_{fiducial_root_cell}.fits"
    )
)
cell_ee = data['CELL_EE'].data
cov_mat = data['COVMAT'].data

# Plot hyperparameter
loc_legend = "lower center"
bbox_to_anchor = (0.685, 0.70)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ell, cell = cell_ee['ANG'], cell_ee['VALUE']
ax.errorbar(ell, ell*cell, yerr=ell*np.sqrt(np.diag(cov_mat)), fmt='o', label=r"UNIONS $C_{\ell}$ data", color='black', capsize=2)

# Plot the scale cuts for different k_max
ax.axvline(x=1800, color='black', linestyle='--', alpha=0.5)
ax.axvline(x=2048, color='black', linestyle='--', alpha=1.0)
ax.axvline(x=500, color='black', linestyle='--', alpha=0.3)

ymin = ax.get_ylim()[0]
ymax = ax.get_ylim()[1]
# Shadowing cut scaled
ax.fill_betweenx(y=[ymin, ymax], x1=0, x2=300, color='gray', alpha=0.2, label=r'$B$-mode informed scale cut')
ax.fill_betweenx(y=[ymin, ymax], x1=1600, x2=2048, color='gray', alpha=0.2)

ax.set_ylim(ymin, ymax)

# Add labels directly under the tick
ax.text(1740, 0.90,
        r"$k_\mathrm{max} = 3 h$ Mpc$^{-1}$",
        transform=ax.get_xaxis_transform(),
        ha='center', va='top', fontsize=10, rotation=90)

ax.text(1978, 0.90,
        r"$k_\mathrm{max} = 5 h$ Mpc$^{-1}$",
        transform=ax.get_xaxis_transform(),
        ha='center', va='top', fontsize=10, rotation=90)

ax.text(470,  0.90,
        r"$k_\mathrm{max} = 1 h$ Mpc$^{-1}$",
        transform=ax.get_xaxis_transform(),
        ha='center', va='top', fontsize=10, rotation=90)

ell, cell = cell_ee['ANG'], cell_ee['VALUE']
ax.set_ylabel('$\ell C_\ell$', fontsize=16)
ax.set_xlabel('$\ell$', fontsize=16)
ax.set_xlim(ell.min()-10, ell.max()+100)
ax.set_xscale('squareroot')
ax.set_xticks(np.array([100, 400, 900, 1600]))
ax.minorticks_on()
ax.tick_params(axis="x", which="minor", length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
ax.xaxis.set_ticks(minor_ticks, minor=True)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=10)
ax.yaxis.get_offset_text().set_fontsize(14)

plt.legend(loc=loc_legend, bbox_to_anchor=bbox_to_anchor, fontsize=12)

plt.show()

# Plots xi_+ and xi_-

#TODO: add the plot for xi_+ and xi_-

# %%
# 2. Plot the best-fit datavectors
display(
    Markdown(
        "### 2. Plot the best-fit datavectors"
    )
)

# Perform the computation for the fiducial of Cell
path_samples_fiducial_cell = os.path.join(
    path_output_chains,
    fiducial_root_cell,
    fiducial_root_cell,
    f"samples_{fiducial_root_cell}_cell.txt"
)
path_gd_fiducial_cell = os.path.join(
    path_output_chains,
    fiducial_root_cell,
    fiducial_root_cell,
    f"getdist_{fiducial_root_cell}_cell"
)
cp.load_samples_and_write_paramnames(path_samples_fiducial_cell, path_gd_fiducial_cell+".paramnames")
cp.write_samples_getdist_format(path_samples_fiducial_cell, path_gd_fiducial_cell+".txt", chain_type='polychord')

chain_fiducial_cell = cp.load_chain(path_gd_fiducial_cell, smoothing_scale=0.3)

best_fit_params_fiducial_cell = cp.extract_best_fit_params(chain_fiducial_cell)

cp.compute_best_fit(
    path_ini_files,
    best_fit_params_fiducial_cell,
    fiducial_root_cell,
    is_harmonic=True,
    blind=blind
)

# Perform the computation for the fiducial of xi
path_samples_fiducial_xi = os.path.join(
    path_output_chains,
    fiducial_root_xi,
    f"samples_{fiducial_root_xi}.txt"
)
path_gd_fiducial_xi = os.path.join(
    path_output_chains,
    fiducial_root_xi,
    f"getdist_{fiducial_root_xi}"
)
cp.load_samples_and_write_paramnames(path_samples_fiducial_xi, path_gd_fiducial_xi+".paramnames")
cp.write_samples_getdist_format(path_samples_fiducial_xi, path_gd_fiducial_xi+".txt", chain_type='polychord')

chain_fiducial_xi = cp.load_chain(path_gd_fiducial_xi, smoothing_scale=0.3)

best_fit_params_fiducial_xi = cp.extract_best_fit_params(chain_fiducial_xi)

ini_file_root = os.path.join(
    path_ini_files,
    f'config_space_v1.4.6.3_fiducial/pipeline/blind_{blind}/fiducial.ini'
)
cp.compute_best_fit(
    path_ini_files,
    best_fit_params_fiducial_xi,
    fiducial_root_xi,
    is_harmonic=False,
    blind=blind,
    ini_file_root=ini_file_root
)

# %%
# Make the plot for the best-fit datavector for Cell EE
root_to_plot = [
    fiducial_root_cell,
    fiducial_root_xi
]

labels = [
    r"UNIONS $C_\ell$",
    r"UNIONS $\xi_\pm(\vartheta)$"
]

line_args = [
    {'color': 'royalblue', 'linestyle': '-'},
    {'color': 'orange', 'linestyle': '-'}
]

properties = {}

properties = utils.update_properties_w_roots(properties, fiducial_root_cell, path_ini_files, with_configuration=False)
properties = utils.update_properties_w_roots(properties, fiducial_root_xi, path_ini_files, with_configuration=True, path_to_this_ini=ini_file_root)

utils.plot_best_fit(fiducial_root_cell, root_to_plot, path_output_chains, line_args, savefile=None, labels=labels, loc_legend=loc_legend, bbox_to_anchor=bbox_to_anchor, properties=properties)

# TODO: add the plot for xi

# %%
# 3. Do a whisker plot with external experiments and our constraints

# %%
# 4. Make a contour plots