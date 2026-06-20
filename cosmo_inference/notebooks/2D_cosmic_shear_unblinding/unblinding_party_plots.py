# %%
import os
import configparser
import subprocess
import sys
import warnings

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

plt.style.use(
    "/home/guerrini/sp_validation/papers/harmonic/matplotlib_config/paper.mplstyle"
)

plt.rcParams["text.usetex"] = True

sns.set_palette("husl")

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
fiducial_root_xi_data = f"SP_v1.4.6.3_leak_corr_{blind}_masked"
fiducial_root_xi_chains = f"SP_v1.4.6.3_{blind}_fiducial_config"
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
        "### 1.a. Plot the datavectors without best-fit"
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
display(
    Markdown(
        r"### 1.b. Plot the datavectors without best-fit ($\xi_\pm$)"
    )
)

# Plot xi_pm's
data = fits.open(
    os.path.join(
        path_datavectors,
        f"SP_v1.4.6.3_config/SP_v1.4.6.3_{blind}/cosmosis_{fiducial_root_xi_data}.fits"
    )
)
xi_p_data = data['XI_PLUS'].data
xi_m_data = data['XI_MINUS'].data
cov_mat = data['COVMAT'].data

# Plot hyperparameter
loc_legend = "lower center"
bbox_to_anchor_xip = (0.685, 0.03)
bbox_to_anchor_xim = (0.3, 0.65)

fig, [ax,ax2] = plt.subplots(2, 1, figsize=(8, 9))

theta, xi_p = xi_p_data['ANG'], xi_p_data['VALUE']
ax.errorbar(theta, theta*xi_p, yerr=theta*np.sqrt(np.diag(cov_mat[:len(theta),:len(theta)])), fmt='o', label=r"UNIONS $\xi_+$ data", color='black', capsize=2)

# Plot the scale cuts for different k_max
ax.axvline(x=3.2, color='black', linestyle='--', alpha=0.3)

ymin = ax.get_ylim()[0]
ymax = ax.get_ylim()[1]
# Shadowing cut scaled
ax.fill_betweenx(y=[ymin, ymax], x1=0, x2=12, color='gray', alpha=0.2, label=r'$B$-mode informed scale cut')
ax.fill_betweenx(y=[ymin, ymax], x1=83, x2=250, color='gray', alpha=0.2)

ax.set_ylim(ymin, ymax)

# Add labels directly under the tick
ax.text(3,  1e-4,
        r"$k_\mathrm{max} = 1 h$ Mpc$^{-1}$",
        # transform=ax.get_xaxis_transform(),
        ha='center', va='top', fontsize=10, rotation=90)

ax.set_ylabel(r'$\theta \xi_+$', fontsize=16)
# ax.set_xlabel('$\theta$', fontsize=16)
# ax.set_xlim([theta.min()-0.1, theta.max()+20])
ax.set_xscale('log')
ax.set_xticks(np.array([1, 10, 100]))
ax.tick_params(axis="x", which="minor", length=2, width=0.8)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=10)
ax.yaxis.get_offset_text().set_fontsize(14)
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.legend(loc=loc_legend, bbox_to_anchor=bbox_to_anchor_xip, fontsize=12)

theta, xi_m = xi_p_data['ANG'], xi_m_data['VALUE']
ax2.errorbar(theta, theta*xi_m, yerr=theta*np.sqrt(np.diag(cov_mat[len(theta):2*len(theta),len(theta):2*len(theta)])), fmt='o', label=r"UNIONS $\xi_-$ data", color='black', capsize=2)

# Plot the scale cuts for different k_max
ax2.axvline(x=24, color='black', linestyle='--', alpha=0.3)

ymin = ax2.get_ylim()[0]
ymax = ax2.get_ylim()[1]
# Shadowing cut scaled
ax2.fill_betweenx(y=[ymin, ymax], x1=0, x2=12, color='gray', alpha=0.2, label=r'$B$-mode informed scale cut')
ax2.fill_betweenx(y=[ymin, ymax], x1=83, x2=250, color='gray', alpha=0.2)

ax2.set_ylim(ymin, ymax)

# Add labels directly under the tick
ax2.text(22.3,  9e-5,
        r"$k_\mathrm{max} = 1 h$ Mpc$^{-1}$",
        # transform=ax.get_xaxis_transform(),
        ha='center', va='top', fontsize=10, rotation=90)

ax2.set_ylabel(r'$\theta÷ \xi_-$', fontsize=16)
ax2.set_xlabel('$\theta$', fontsize=16)
ax2.set_xlim([theta.min()-0.1, theta.max()+20])
ax2.set_xscale('log')
ax2.set_xticks(np.array([1, 10, 100]))
ax2.tick_params(axis="x", which="minor", length=2, width=0.8)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='minor', labelsize=10)
ax2.yaxis.get_offset_text().set_fontsize(14)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax2.legend(loc=loc_legend, bbox_to_anchor=bbox_to_anchor_xim, fontsize=12)

plt.show()

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
    fiducial_root_xi_chains,
    f"samples_{fiducial_root_xi_chains}.txt"
)
path_gd_fiducial_xi = os.path.join(
    path_output_chains,
    fiducial_root_xi_chains,
    f"getdist_{fiducial_root_xi_chains}"
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
    fiducial_root_xi_chains,
    is_harmonic=False,
    blind=blind,
    ini_file_root=ini_file_root
)

# %%
# Make the plot for the best-fit datavector for Cell EE
root_to_plot = [
    fiducial_root_cell,
    fiducial_root_xi_chains
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
properties = utils.update_properties_w_roots(properties, fiducial_root_xi_chains, path_ini_files, with_configuration=True, path_to_this_ini=ini_file_root)

utils.plot_best_fit(fiducial_root_cell, root_to_plot, path_output_chains, line_args, savefile=None, labels=labels, loc_legend=loc_legend, bbox_to_anchor=bbox_to_anchor, properties=properties)

# TODO: add the plot for xi
# %%
# Plot best-fit xi_+ and xi_- (also from C_ell's)

path_best_fit_xi_theta = os.path.join(
    path_output_chains,
    fiducial_root_xi_chains,
    "best_fit/shear_xi_plus/"
    f"theta.txt"
)
theta_rad = np.loadtxt(path_best_fit_xi_theta)

cp.compute_best_fit_xi_from_cell(path_output_chains, fiducial_root_cell, best_fit_params_fiducial_cell, theta_rad)


xi_data_path = os.path.join(path_datavectors,f"SP_v1.4.6.3_config/SP_v1.4.6.3_{blind}/cosmosis_{fiducial_root_xi_data}.fits")
utils.plot_best_fit_config(xi_data_path, root_to_plot, path_output_chains, line_args, savefile=None, labels=labels, loc_legend=loc_legend, bbox_to_anchor_xip=bbox_to_anchor_xip, bbox_to_anchor_xim=bbox_to_anchor_xim, properties=properties)

# %%
# 3. Do a whisker plot with external experiments and our constraints
display(
    Markdown(
        "### 🥸 Time to look at the whisker plot 🥸"
    )
)
colour_blind = {
    "A": "royalblue",
    "B": "crimson",
    "C": "forestgreen"
}

roots = [
    f"SP_v1.4.6.3_leak_corr_{blind}",
    f"SP_v1.4.6.3_{blind}_fiducial_config",
    "Planck18",
    "DES_Y3",
    "DES_Y3_cell",
    "KiDS-1000",
    "KiDS-1000_cosebis",
    "KiDS-1000_bp",
    "DES+KiDS",
    "HSC_Y3",
    "HSC_Y3_cell",
]

legend_labels = [
    r"UNIONS $C_\ell$, unblind",
    r"UNIONS $\xi_\pm(\vartheta)$, unblind",
    r"\textit{Planck} 2018",
    r"DES Y3 $\xi_\pm(\vartheta)$",
    r"DES Y3 $C_\ell$",
    r"KiDS-1000 $\xi_\pm(\vartheta)$",
    r"KiDS-1000 $E_n$",
    r"KiDS-1000 $C_E$",
    r"DES Y3 + KiDS-1000 combined",
    r"HSC Y3 $\xi_\pm(\vartheta)$",
    r"HSC Y3 $C_\ell$",
]

colours = [
    colour_blind[blind],
    colour_blind[blind],
    "violet",
    "black",
    "black",
    "black",
    "black",
    "black",
    "black",
    "black",
    "black",
]

categories = [
    "harmonic",
    "configuration",
    "external",
    "external",
    "external",
    "external",
    "external_compute_sample",
    "external_compute_sample",
    "external",
    "external",
    "external_compute_sample"

]

for bl in ["A", "B", "C"]:
    if bl != blind:
        roots.append(
            f"SP_v1.4.6.3_leak_corr_{bl}"
        )
        roots.append(
            f"SP_v1.4.6.3_{bl}_fiducial_config"
        )
        legend_labels.append(
            fr"UNIONS $C_\ell$, Blind {bl}"
        )
        legend_labels.append(
            fr"UNIONS $\xi_\pm(\vartheta)$, Blind {bl}"
        )
        colours.append(
            colour_blind[bl]
        )
        colours.append(
            colour_blind[bl]
        )
        categories.append(
            "harmonic"
        )
        categories.append(
            "configuration"
        )

# Loop on all versions to load the chain
chains = []
for i, root in enumerate(roots):
    category = categories[i]
    if category != "external":
        if category == 'configuration':
            path_samples = os.path.join(
                path_output_chains,
                f"{root}/samples_{root}.txt"
            )
            path_getdist = os.path.join(
                path_output_chains,
                f"{root}/getdist_{root}"
            )
        elif category == 'harmonic':
            path_samples = os.path.join(
                path_output_chains,
                f"{root}/{root}/samples_{root}_cell.txt"
            )
            path_getdist = os.path.join(
                path_output_chains,
                f"{root}/{root}/getdist_{root}"
            )
        elif category == "external_compute_sample":
            path_samples = os.path.join(
                path_output_chains,
                f"ext_data/{root}/samples_{root}.txt"
            )
            path_getdist = os.path.join(
                path_output_chains,
                f"ext_data/{root}/getdist_{root}"
            )
        else:
            raise ValueError(f"The category, {category}, of {root} is not correct")
        
        cp.load_samples_and_write_paramnames(path_samples, path_getdist+".paramnames")
        cp.write_samples_getdist_format(path_samples, path_getdist+".txt")
        chains.append(
            cp.load_chain(path_getdist, smoothing_scale=0.5)
        )
    else:
        path_getdist = os.path.join(
            path_output_chains,
            f"ext_data/{root}/getdist_{root}"
        )
        chains.append(
            cp.load_chain(path_getdist)
        )

# Give labels for the chains
name_list = ['OMEGA_M','ombh2','h0','n_s','SIGMA_8','S_8','s_8_input', 'logt_agn','a','m1','bias_1']
label_list = [r'\Omega_{\rm m}', r'\omega_b h^2', r'h_0', r'n_s', r'\sigma_8', r'S_8', r'S_8', r'\log T_{\rm AGN}', r'A_{\rm IA}', r'm_1', r'\Delta z_1']

for i, chain in enumerate(chains):
    print(legend_labels[i])
    param_names = chain.getParamNames()
    for name, label in zip(name_list, label_list):
        try:
            param_names.parWithName(name).label = label
        except:
            warnings.warn(f"Parameter {name} not found in chain {roots[i]}.")

# Account for the missing parameter conventions
#OMEGA_M not in DES_Y3_cell
idx = roots.index('DES_Y3_cell')
cp.adjust_paramname_chain(chains[idx], 'omega_m', 'OMEGA_M', r'\Omega_{\rm m}')
cp.derive_parameter_S8(chains[idx])

#OMEGA_M not in KiDS-1000
idx = roots.index('KiDS-1000')
cp.adjust_paramname_chain(chains[idx], 'omega_m', 'OMEGA_M', r'\Omega_{\rm m}')

#OMEGA_M not in DES+KiDS
idx = roots.index('DES+KiDS')
cp.adjust_paramname_chain(chains[idx], 'omega_m', 'OMEGA_M', r'\Omega_{\rm m}')

#OMEGA_M not in HSC_Y3_cell
idx = roots.index('HSC_Y3_cell')
cp.adjust_paramname_chain(chains[idx], 'omega_m', 'OMEGA_M', r'\Omega_{\rm m}')

# Build an array containing the parameter values
param_values = np.array(["# Expt", "Colour", "S8_Mean", "S8_low", "S8_high",  "sigma_8_Mean", "sigma_8_low", "sigma_8_high", "Omega_m_Mean", "Omega_m_low", "Omega_m_high"])
escaped = np.char.replace(legend_labels, '\\', '\\\\')
for i, chain in enumerate(chains):
    print(chain.root)
    margestats = chain.getMargeStats()
    likestats = chain.getLikeStats()

    s8_stats = margestats.parWithName('S_8')
    sigma8_stats = margestats.parWithName('SIGMA_8')
    omegam_stats = margestats.parWithName('OMEGA_M')

    param_values = np.vstack((
        param_values,
        [
            escaped[i],
            colours[i],
            s8_stats.mean,
            s8_stats.mean-s8_stats.limits[0].lower,
            s8_stats.limits[0].upper-s8_stats.mean,
            sigma8_stats.mean,
            sigma8_stats.mean - sigma8_stats.limits[0].lower,
            sigma8_stats.limits[0].upper - sigma8_stats.mean,
            omegam_stats.mean,
            omegam_stats.mean - omegam_stats.limits[0].lower,
            omegam_stats.limits[0].upper - omegam_stats.mean,
        ]
    ))
print(param_values)
np.savetxt(f"./param_values.txt", param_values, fmt=['%s' for i in range(11)], delimiter=';')

# Reload the table
# Load the value of the parameters
cosmo = np.loadtxt(f"./param_values.txt",
            dtype={'names': ('Expt', 'colour', 's8_mean', 's8_low', 's8_high', 'sigma8_mean', 'sigma8_low', 'sigma8_high', 'omegam_mean', 'omegam_low', 'omegam_high'),
                   'formats': ('U250', 'U20', 'U20', 'U20', 'U20', 'U20', 'U20', 'U20', 'U20', 'U20', 'U20')}, skiprows=1, delimiter=';')
expt = np.char.replace(cosmo['Expt'], '\\\\', '\\')
colours =  cosmo['colour']
s8_mean = cosmo['s8_mean'].astype(np.float64)
s8_low = cosmo['s8_low'].astype(np.float64)
s8_high = cosmo['s8_high'].astype(np.float64)
sigma8_mean = cosmo['sigma8_mean'].astype(np.float64)
sigma8_low = cosmo['sigma8_low'].astype(np.float64)
sigma8_high = cosmo['sigma8_high'].astype(np.float64)
omegam_mean = cosmo['omegam_mean'].astype(np.float64)
omegam_low = cosmo['omegam_low'].astype(np.float64)
omegam_high = cosmo['omegam_high'].astype(np.float64)

# %%
# Perform the plot
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(10, 6))
gs = GridSpec(1, 3, width_ratios=[1, 0.5, 0.5])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharey=ax1)
ax3 = fig.add_subplot(gs[2], sharey=ax1)

axs = [ax1, ax2, ax3]

params = [
    (s8_mean, s8_low, s8_high, r"$S_8$"),
    (sigma8_mean, sigma8_low, sigma8_high, r"$\sigma_8$"),
    (omegam_mean, omegam_low, omegam_high, r"$\Omega_{\rm m}$"),
]
reference = "UNIONS $C_\ell$, unblind"
separation_after = [
    r"UNIONS $\xi_\pm(\vartheta)$, unblind",
    r"HSC Y3 $C_\ell$",
]
list_section_index = [
    r"(ii)",
    r"(iii)",
    r"(iv)",
    r"(v)",
    r"(vi)",
    r"(vii)"
]

preliminary_watermark = False
blind_axes = False
row_spacing = 0.1

index_ref = np.where(expt == reference)[0][0]

y = np.arange(len(expt))
for ax, param in zip(axs, params):
    means, lows, highs, label = param
    for i, mean, low, high, color in zip(y, means, lows, highs, colours):
        ax.errorbar(mean, 0.05+i*row_spacing, xerr=np.array([low, high])[:, None], fmt='o', color=color, ecolor=color, elinewidth=2, capsize=3)
    ax.set_xlabel(label, fontsize=14)
        
    ax.grid(False)
    ax.tick_params(axis='y', left=False, labelleft=False)
    if label == r"$S_8$":
        ax.axvspan(s8_mean[index_ref] - s8_low[index_ref], s8_mean[index_ref] + s8_high[index_ref], color=colours[index_ref], alpha=0.2)
        ax.set_xlim(0.25, 1.1)
        if blind_axes:
            ref_tick = np.mean(s8_mean[:4])
            ax.set_xticks(
                [ref_tick + i*0.1 for i in range(-5, 5)], labels=[]
            )
    elif label == r"$\sigma_8$":
        ax.axvspan(sigma8_mean[index_ref] - sigma8_low[index_ref], sigma8_mean[index_ref] + sigma8_high[index_ref], color=colours[index_ref], alpha=0.2)
        ax.set_xlim(0.5, 1.2)
        if blind_axes:
            ref_tick = np.mean(sigma8_mean[:4])
            ax.set_xticks(
                [ref_tick + i*0.2 for i in range(-2, 2)], labels=[]
            )
    elif label == r"$\Omega_{\rm m}$":
        ax.axvspan(omegam_mean[index_ref] - omegam_low[index_ref], omegam_mean[index_ref] + omegam_high[index_ref], color=colours[index_ref], alpha=0.2)
        ax.set_xlim(0.1, 0.5)
        if blind_axes:
            ref_tick = np.mean(omegam_mean[:4])
            ax.set_xticks(
                [ref_tick + i*0.1 for i in range(-2, 3)], labels=[]
            )


axs[0].set_yticks(0.05+y*row_spacing)
axs[0].set_yticklabels([])
for label, color in zip(expt, colours):
    axs[0].text(0.26, 0.05 + row_spacing * np.where(expt == label)[0][0], label, fontsize=12, ha='left', va='center', color=color)
    if label != reference:
        index = np.where(expt == label)[0][0]
        s8_tension = cp.get_sigma_tension(
            s8_mean[index], s8_low[index], s8_high[index],
            s8_mean[index_ref], s8_low[index_ref], s8_high[index_ref]
        )
        sign_str = "+" if s8_tension > 0 else "-"
        axs[0].text(1.095, 0.05 + row_spacing * index, rf"${sign_str}{np.abs(s8_tension):.2f}" + r"\, \sigma$", fontsize=10, ha='right', va='center', color=color)
# Add separation lines
for i, sep in enumerate(separation_after):
    index_sep = np.where(expt == sep)[0][0]
    for ax in axs:
        ax.axhline(row_spacing * (index_sep + 1), color='black', linestyle='dotted', linewidth=1)
        axs[0].text(0.25, 0.05 + row_spacing * (index_sep + 1), 
            list_section_index[i], fontsize=14, fontweight='bold', va='center', ha='right')


# --- Add section labels (i), (ii)) ---
axs[0].text(0.25, 0.05, 
            r"(i)", fontsize=14, fontweight='bold', va='center', ha='right')

if preliminary_watermark:
    plt.figtext(0.5, 0.5, 'PRELIMINARY',
            fontsize=50, color='gray',
            ha='center', va='center',
            alpha=0.3, rotation=330)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()

# %%
# 4. Make a contour plots
display(
    Markdown(
        "### Here comes $S_8$ and $\Omega_m$"
    )
)
colours = [
    "royalblue",
    "orange",
    "violet"
]

filled = [
    True,
    True,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False
]

line_args = [
    dict(color=col, ls='solid') for col in colours
]

g = plots.get_single_plotter(width_inch=30)
g.settings.axes_fontsize=60
g.settings.axes_labelsize=60
g.settings.alpha_filled_add = 0.7
g.settings.legend_fontsize = 45
g.settings.figure_legend_ncol = 3
g.settings.legend_frame = False

g.plot_2d(chains[:-4],
          'OMEGA_M', 'S_8',
          filled=filled,
          line_args=line_args,
          contour_colors=colours,)

g.add_legend(
    legend_labels[:-4],
    legend_loc='upper center',
    bbox_to_anchor=(0.5, 1.20),   # moves legend above the axes
)

plt.show()
# %%
