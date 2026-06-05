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

def plot_best_fit_config(data, root_to_plot, output_folder, line_args, savefile, theta_min=1.0, theta_max=250.0, multiply_theta=True, loc_legend="best", bbox_to_anchor_xip=None, bbox_to_anchor_xim=None, label_data="Fiducial data", labels=None, properties=None, paths_to_bestfit=None):
    
    data = fits.open(data)
    
    xi_p_data = data['XI_PLUS'].data
    xi_m_data = data['XI_MINUS'].data
    cov_mat = data['COVMAT'].data

    # Plot hyperparameter
    loc_legend = "lower center"

    fig, [ax,ax2] = plt.subplots(2, 1, figsize=(8, 9))

    theta, xi_p, xi_m = xi_p_data['ANG'], xi_p_data['VALUE'], xi_m_data['VALUE']
    ax.errorbar(theta, theta*xi_p, yerr=theta*np.sqrt(np.diag(cov_mat[:len(theta),:len(theta)])), fmt='o', label=r"UNIONS $\xi_+$ data", color='black', capsize=2)
    ax2.errorbar(theta, theta*xi_m, yerr=theta*np.sqrt(np.diag(cov_mat[len(theta):2*len(theta),len(theta):2*len(theta)])), fmt='o', label=r"UNIONS $\xi_-$ data", color='black', capsize=2)
    
    for idx, (label, root) in enumerate(zip(labels, root_to_plot)):
        #Read the results
        if paths_to_bestfit is None:
            theta = (np.loadtxt(output_folder + '{}/best_fit/shear_xi_plus/theta.txt'.format(root))) * 180/np.pi * 60
            xi_plus = np.loadtxt(output_folder + '{}/best_fit/shear_xi_plus/bin_1_1.txt'.format(root))
            xi_minus = np.loadtxt(output_folder + '{}/best_fit/shear_xi_minus/bin_1_1.txt'.format(root))
            if '$C_\ell$' not in label:
                xi_sys_plus = np.loadtxt(output_folder + '{}/best_fit/xi_sys/shear_xi_plus.txt'.format(root))
                xi_sys_minus = np.loadtxt(output_folder + '{}/best_fit/xi_sys/shear_xi_minus.txt'.format(root))
                theta_xi_sys = np.loadtxt(output_folder + '{}/best_fit/xi_sys/theta.txt'.format(root)) * 180/np.pi * 60
                xi_plus += np.interp(theta, theta_xi_sys, xi_sys_plus)
                xi_minus += np.interp(theta, theta_xi_sys, xi_sys_minus)
        else:
            theta = (np.loadtxt(paths_to_bestfit[idx] + 'best_fit/shear_xi_plus/theta.txt')) * 180/np.pi * 60
            xi_plus = np.loadtxt(paths_to_bestfit[idx] + 'best_fit/shear_xi_plus/bin_1_1.txt')
            xi_minus = np.loadtxt(paths_to_bestfit[idx] + 'best_fit/shear_xi_minus/bin_1_1.txt')
            if '$C_\ell$' not in label:
                xi_sys_plus = np.loadtxt(output_folder + '{}/best_fit/xi_sys/shear_xi_plus.txt'.format(root))
                xi_sys_minus = np.loadtxt(output_folder + '{}/best_fit/xi_sys/shear_xi_minus.txt'.format(root))
                theta_xi_sys = np.loadtxt(output_folder + '{}/best_fit/xi_sys/theta.txt'.format(root)) * 180/np.pi * 60
                xi_plus += np.interp(theta, theta_xi_sys, xi_sys_plus)
                xi_minus += np.interp(theta, theta_xi_sys, xi_sys_minus)

        mask = (theta > theta_min) & (theta < theta_max)
        theta = theta[mask]
        ax.plot(theta, theta*xi_plus[mask] if multiply_theta else xi_plus[mask], label=label, **line_args[idx])
        ax2.plot(theta, theta*xi_minus[mask] if multiply_theta else xi_minus[mask], label=label, **line_args[idx])
    
    # XI PLUS PLOT SETTINGS
    
    # Plot the scale cuts for different k_max
    ax.axvline(x=3.2, color='black', linestyle='--', alpha=0.7)

    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]
    # Shadowing cut scaled
    ax.fill_betweenx(y=[ymin, ymax], x1=0, x2=12, color='gray', alpha=0.2, label=r'$B$-mode informed scale cut')
    ax.fill_betweenx(y=[ymin, ymax], x1=83, x2=250, color='gray', alpha=0.2)

    ax.set_ylim(ymin, ymax)

    # Add labels directly under the tick
    ax.text(2.9,  1.23e-4,
            r"$k_\mathrm{max} = 1 h$ Mpc$^{-1}$",
            ha='center', va='top', fontsize=14, rotation=90)

    # ax.set_ylabel('$\theta \xi_+$', fontsize=16)
    # ax.set_xlabel('$\theta$', fontsize=16)
    ax.set_xlim([theta.min()-0.1, theta.max()+20])
    ax.set_xscale('log')
    ax.set_xticks(np.array([1, 10, 100]))
    ax.tick_params(axis="x", which="minor", length=2, width=0.8)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    ax.yaxis.get_offset_text().set_fontsize(14)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.legend(loc=loc_legend, bbox_to_anchor=bbox_to_anchor_xip, fontsize=12)

    # XI_MINUS PLOT SETTINGS
    
    # Plot the scale cuts for different k_max
    ax2.axvline(x=24, color='black', linestyle='--', alpha=0.7)

    ymin = ax2.get_ylim()[0]
    ymax = ax2.get_ylim()[1]
    # Shadowing cut scaled
    ax2.fill_betweenx(y=[ymin, ymax], x1=0, x2=12, color='gray', alpha=0.2, label=r'$B$-mode informed scale cut')
    ax2.fill_betweenx(y=[ymin, ymax], x1=83, x2=250, color='gray', alpha=0.2)

    ax2.set_ylim(ymin, ymax)

    # Add labels directly under the tick
    ax2.text(21.8,  1.15e-4,
            r"$k_\mathrm{max} = 1 h$ Mpc$^{-1}$",
            ha='center', va='top', fontsize=14, rotation=90)

    ax2.set_ylabel(r'$\theta \xi_-$', fontsize=16)
    ax2.set_xlabel(r'$\theta$', fontsize=16)
    ax2.set_xlim([theta.min()-0.1, theta.max()+20])
    ax2.set_xscale('log')
    ax2.set_xticks(np.array([1, 10, 100]))
    ax2.tick_params(axis="x", which="minor", length=2, width=0.8)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='minor', labelsize=10)
    ax2.yaxis.get_offset_text().set_fontsize(14)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax2.legend(loc=loc_legend, bbox_to_anchor=bbox_to_anchor_xim, fontsize=12)

    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight')

    plt.show()