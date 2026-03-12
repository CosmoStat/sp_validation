"""
Useful scripts to perform the plots for the unblinding party.
"""
import os
import configparser
import subprocess
import sys

# Append any useful folder in the path
sys.path.append(
    "/home/guerrini/sp_validation/cosmo_inference/scripts/"
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

def read_config(path_ini_files, root, thisfile=None):
    config = configparser.ConfigParser()
    config.optionxform = str
    if thisfile is not None:
        read_path = thisfile
    else:
        read_path = os.path.join(path_ini_files, f"{root}.ini")
    config.read(read_path)
    return config

def update_properties_w_roots(properties, root, path_ini_files, path_to_this_ini=None, with_configuration=False):
    config = read_config(path_ini_files, root, thisfile=path_to_this_ini)

    try:
        lower_bound_cell_ee, upper_bound_cell_ee = map(
            float, config["2pt_like"]["angle_range_CELL_EE_1_1"].split()
        )
        properties[root].update(
            {
                'lower_bound_cell_ee': lower_bound_cell_ee,
                'upper_bound_cell_ee': upper_bound_cell_ee
            }
        )
    except KeyError:
        properties[root] = {
            'lower_bound_cell_ee': 0.0,
            'upper_bound_cell_ee': 2048
        }

    if with_configuration:
        # Also save the scale cuts in theta for xi
        add_xi_sys = config["2pt_like"]["add_xi_sys"]
        add_xi_sys = add_xi_sys == 'T'
        lower_bound_xi_plus, upper_bound_xi_plus = map(float, config["2pt_like"]["angle_range_XI_PLUS_1_1"].split())
        lower_bound_xi_minus, upper_bound_xi_minus = map(float, config["2pt_like"]["angle_range_XI_MINUS_1_1"].split())

        properties[root].update({
            'add_xi_sys': add_xi_sys,
            'lower_bound_xi_plus': lower_bound_xi_plus,
            'upper_bound_xi_plus': upper_bound_xi_plus,
            'lower_bound_xi_minus': lower_bound_xi_minus,
            'upper_bound_xi_minus': upper_bound_xi_minus
        })
    return properties


def plot_best_fit(data_points, root_to_plot, output_folder, line_args, savefile, ell_min=10.0, ell_max=2048.0, multiply_ell=True, loc_legend="best", bbox_to_anchor=None, label_data="Fiducial data", labels=None, properties=None, paths_to_bestfit=None):
    data = fits.open(f'/home/guerrini/sp_validation/cosmo_inference/data/{data_points}/cosmosis_{data_points}.fits')
    cell_ee = data['CELL_EE'].data
    cov_mat = data['COVMAT'].data

    if labels is None:
        labels = root_to_plot

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ell, cell = cell_ee['ANG'], cell_ee['VALUE']
    ax.errorbar(ell, ell*cell, yerr=ell*np.sqrt(np.diag(cov_mat)), fmt='o', label=label_data, color='black', capsize=2)
    
    for idx, (label, root) in enumerate(zip(labels, root_to_plot)):
        lower_bound_cell_ee = properties[root]['lower_bound_cell_ee']
        upper_bound_cell_ee = properties[root]['upper_bound_cell_ee']
        
        #Read the results
        if paths_to_bestfit is None:
            ell = np.loadtxt(output_folder + '{}/best_fit/shear_cl/ell.txt'.format(root, root))
            shear_cl = np.loadtxt(output_folder + '{}/best_fit/shear_cl/bin_1_1.txt'.format(root, root))
        else:
            ell = np.loadtxt(paths_to_bestfit[idx] + 'best_fit/shear_cl/ell.txt')
            shear_cl = np.loadtxt(paths_to_bestfit[idx] + 'best_fit/shear_cl/bin_1_1.txt')

        mask = (ell > ell_min) & (ell < ell_max)

        ax.plot(ell[mask], ell[mask]*shear_cl[mask] if multiply_ell else shear_cl[mask], label=label, **line_args[idx])
    
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
            ha='center', va='top', fontsize=14, rotation=90)

    ax.text(1978, 0.90,
            r"$k_\mathrm{max} = 5 h$ Mpc$^{-1}$",
            transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=14, rotation=90)

    ax.text(470,  0.90,
            r"$k_\mathrm{max} = 1 h$ Mpc$^{-1}$",
            transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=14, rotation=90)

    ell, cell = cell_ee['ANG'], cell_ee['VALUE']
    ax.set_ylabel(r'$\ell C_\ell \times 10^{-7}$', fontsize=20)
    ax.set_xlabel(r'Multipole $\ell$', fontsize=20)
    ax.set_xlim(ell.min()-10, ell.max()+100)
    ax.set_xscale('squareroot')
    ax.set_xticks(np.array([100, 400, 900, 1600]))
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", length=2, width=0.8)
    minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
    ax.xaxis.set_ticks(minor_ticks, minor=True)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    ax.yaxis.get_offset_text().set_visible(False)


    plt.legend(loc=loc_legend, bbox_to_anchor=bbox_to_anchor, fontsize=11)

    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight')

    plt.show()