"""
Scripts to postprocess the CosmoSIS chains
Author: Sacha Guerrini
"""
import os
import configparser
import subprocess

import numpy as np
from getdist import plots, MCSamples
from astropy.io import fits
import matplotlib.pyplot as plt

# Mapping for CosmoSIS ini files section
section_map = {
    'omch2': 'cosmological_parameters',
    'ombh2': 'cosmological_parameters',
    'h0': 'cosmological_parameters',
    'n_s': 'cosmological_parameters',
    's_8_input': 'cosmological_parameters',
    'logt_agn': 'halo_model_parameters',
    'a': 'intrinsic_alignment_parameters',
    'm1': 'shear_calibration_parameters',
    'bias_1': 'nofz_shifts',
    'alpha': 'psf_leakage_parameters',
    'beta': 'psf_leakage_parameters',
}

# Utils functions
def compute_average(chain, param_name):
    """
    Compute the average of a parameter from a CosmoSIS chain
    """
    margestats = chain.getMargeStats()
    param_stats = margestats.parWithName(param_name)
    return param_stats.mean

def compute_map_1D(chain, param_name, num_bins=1000):
    """
    Compute the MAP value of a parameter from a CosmoSIS chain using 1D KDE
    """
    param_names_getdist = chain.getParamNames()
    par = param_names_getdist.parWithName(param_name)
    kde = chain.get1DDensity(par, num_bins=num_bins)
    kde_map = kde.x[np.argmax(kde.P)]
    return kde_map

def compute_map_2D(chain, param_name_x, param_name_y, num_bins=1000):
    """
    Compute the MAP value of two parameters from a CosmoSIS chain using 2D KDE
    """
    param_names_getdist = chain.getParamNames()
    par_x = param_names_getdist.parWithName(param_name_x)
    par_y = param_names_getdist.parWithName(param_name_y)
    kde = chain.get2DDensity(par_x, par_y, fine_bins_2D=num_bins)
    kde_map_index = np.unravel_index(np.argmax(kde.P), kde.P.shape)
    return kde.x[kde_map_index[1]], kde.y[kde_map_index[0]]

def compute_limits(chain, param_name):
    """
    Compute the 68% and 95% confidence limits of a parameter from a CosmoSIS chain.
    """
    margestats = chain.getMargeStats()
    param_stats = margestats.parWithName(param_name)
    return param_stats.limits[0].upper, param_stats.limits[0].lower, param_stats.limits[1].upper, param_stats.limits[1].lower

def load_samples_and_write_paramnames(path_samples, path_paramnames, chain_type='polychord'):
    """
    Load the samples from a CosmoSIS chain and write the parameter names to a file
    """
    with open(path_samples, 'r') as file:
        if chain_type == 'nautilus':
            params = file.readline()[1:].split('\t')[:-3]
        else:
            params = file.readline()[1:].split('\t')[:-4]
        file.close()

    with open(path_paramnames, 'w') as file:
        for i in range(len(params)):
            if len(params[i].split('--')) > 1:
                param_name = params[i].split('--')[1]
                if 'Legacy' in path_paramnames and param_name not in ['OMEGA_M', 'SIGMA_8']:
                    continue
                file.write(param_name + '\n')
            else:
                param_name = params[i].split('--')[0]
                if 'Legacy' in path_paramnames and param_name not in ['OMEGA_M', 'SIGMA_8']:
                    continue
                file.write(param_name + '\n')
        file.close()
    return 0

def write_samples_getdist_format(path_samples, path_gd, chain_type='polychord'):
    """
    Load the samples from a CosmoSIS chain and write them in GetDist format
    """
    samples = np.loadtxt(
        path_samples
    )
    if chain_type == 'nautilus':
        if 'Legacy' in path_gd:
            samples = np.column_stack((np.exp(samples[:,-3]),samples[:,-2]-samples[:,-1], samples[:, 21], samples[:, 23]))
        else:
            samples = np.column_stack((np.exp(samples[:,-3]),samples[:,-2]-samples[:,-1],samples[:,0:-3]))
        mask = np.isfinite(samples[:, 1])
        samples = samples[mask]
    else:
        samples = np.column_stack((samples[:,-1],samples[:,-2],samples[:,0:-4]))
    np.savetxt(path_gd, samples)
    return 0

def load_chain(path_gd, smoothing_scale=0.3):
    g = plots.get_single_plotter()
    chain = g.samples_for_root(path_gd, cache=False, settings={'ignore_rows':0, 'smooth_scale_1D': smoothing_scale, 'smooth_scale_2D': smoothing_scale})
    return chain

def extract_best_fit_params(chain, best_fit_method='weighted_mean'):
    best_fit_params = {}
    margestats = chain.getMargeStats()
    likestats = chain.getLikeStats()
    for i, par in enumerate(likestats.names):
        if best_fit_method == 'weighted_mean':
            best_fit = compute_average(chain, par.name)
        elif best_fit_method == '1Dkde':
            best_fit = compute_map_1D(chain, par.name)
        elif best_fit_method == '2Dkde':
            # If the parameter is S8 or Omega_m, use the 2D KDE
            if par.name == 'S_8' or par.name == "s_8_input" or par.name == 'OMEGA_M':
                s8_map_2D, omega_m_map_2D = compute_map_2D(chain, 'S_8', 'OMEGA_M')
                best_fit = s8_map_2D if par.name == 'S_8' or par.name == "s_8_input" else omega_m_map_2D
            else:
                best_fit = compute_map_1D(chain, par.name)
        else:
            raise ValueError("Invalid best fit method. Choose one of: '2Dkde', '1Dkde', 'weighted_mean'")
        best_fit_params.update(
            {
                par.name: best_fit
            }
        )
    return best_fit_params

def compute_best_fit(path_ini_files, best_fit, root, is_harmonic, blind=None, ini_file_root=None):
    # Check if the values empty ini file exists
    if not os.path.exists(path_ini_files+'/values_empty.ini'):
        content = """[cosmological_parameters]

        tau          =  0.0544
        w            = -1.0
        mnu = 0.06
        omega_k      =  0.0
        wa           =  0.0

        [halo_model_parameters]

        [intrinsic_alignment_parameters]

        [shear_calibration_parameters]

        [nofz_shifts]

        [psf_leakage_parameters]
        """

        with open(path_ini_files+'/values_empty.ini', 'w') as f:
            f.write(content)
            f.close()

        print('File created successfully')


    # Load cosmosis in the library path
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/home/guerrini/.conda/envs/sp_validation/lib/python3.9/site-packages/cosmosis/datablock:" + env.get("LD_LIBRARY_PATH", "")

    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve case sensitivity of option names
    config.read(path_ini_files+'/values_empty.ini')
    for param, value in best_fit.items():
        section = section_map.get(param)
        if section is None:
            continue
        if section not in config:
            config.add_section(section)
        config[section][param] = str(value)

    with open(path_ini_files+'/values_empty.ini', 'w') as configfile:
        config.write(configfile)

    #Modify the ini file to run in test mode at the best fit
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve case sensitivity of option names
    if ini_file_root is None:
        # If the ini file root is not provided, we construct it based on the root and blind parameters
        if blind is not None:
            subdir = f'harmonic_space_fiducial_{blind}' if is_harmonic else f'' #TODO: add real space subdir if needed
        else:
            subdir=''
        ini_file_root = os.path.join(
            path_ini_files,
            subdir,
            f'cosmosis_pipeline_{root}_cell.ini'
        )
    config.read(ini_file_root)


    sampler = config['runtime']['sampler']
    config['runtime']['sampler'] = 'test'
    values = config['pipeline']['values']
    config['pipeline']['values'] = path_ini_files + '/values_empty.ini'

    with open(ini_file_root, 'w') as configfile:
        config.write(configfile)

    #Run cosmosis
    os.chdir('/home/guerrini/sp_validation/cosmo_inference')
    result = subprocess.run(
        ['cosmosis',  ini_file_root],
        env=env,
        capture_output=True,
        text=True
    )
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")

    #Modify the ini file to the previous one
    config['pipeline']['values'] = values
    config['runtime']['sampler'] = sampler

    with open(ini_file_root, 'w') as configfile:
        config.write(configfile)

def compute_best_fit_xi_from_cell(output_folder, root, best_fit_params, theta_rad):
    
    ell = np.loadtxt(output_folder + '{}/best_fit/shear_cl/ell.txt'.format(root))
    shear_cl = np.loadtxt(output_folder + '{}/best_fit/shear_cl/bin_1_1.txt'.format(root))
    
    import pyccl as ccl
    cosmo = ccl.Cosmology(Omega_c=best_fit_params['omch2']/(best_fit_params['h0']/100)**2, 
                          Omega_b=best_fit_params['ombh2']/(best_fit_params['h0']/100)**2, 
                          h=best_fit_params['h0']/100, 
                          n_s=best_fit_params['n_s'], 
                          sigma8=best_fit_params['SIGMA_8'],
                          baryonic_effects=None,
                          extra_parameters = {"camb": {"halofit_version": "mead2020_feedback",
                             "HMCode_logT_AGN": best_fit_params['logt_agn']}})

    theta_deg = np.rad2deg(theta_rad)
    xi_p = ccl.correlation(cosmo, ell=ell, C_ell=shear_cl, theta=theta_deg, type='GG+')
    xi_m = ccl.correlation(cosmo, ell=ell, C_ell=shear_cl, theta=theta_deg, type='GG-')
    
    os.makedirs(output_folder + '{}/best_fit/shear_xi_minus'.format(root), exist_ok=True)
    os.makedirs(output_folder + '{}/best_fit/shear_xi_plus'.format(root), exist_ok=True)
    
    np.savetxt(output_folder + '{}/best_fit/shear_xi_plus/bin_1_1.txt'.format(root), xi_p)
    np.savetxt(output_folder + '{}/best_fit/shear_xi_minus/bin_1_1.txt'.format(root), xi_m)
    np.savetxt(output_folder + '{}/best_fit/shear_xi_plus/theta.txt'.format(root), theta_rad)
    np.savetxt(output_folder + '{}/best_fit/shear_xi_minus/theta.txt'.format(root), theta_rad)
    
    print(f"Best fit xi+ and xi- from Cl's computed and saved in {output_folder + '{}/best_fit/shear_xi_plus'.format(root)} and {output_folder + '{}/best_fit/shear_xi_minus'.format(root)}")

def adjust_paramname_chain(chain, current_name, target_name, label):
    """
    Adjusts the parameter name and label in a GetDist chain.
    """
    param_names = chain.getParamNames()
    par = param_names.parWithName(current_name)

    par.label = label
    par.name = target_name

    chain.setParamNames(param_names)

    p = chain.paramNames.parWithName(target_name)

def derive_parameter_S8(chain):
    """
    Derives the S_8 parameter from Omega_m and Sigma_8 in a GetDist chain.
    S_8 = Sigma_8 * (Omega_m / 0.3) ** 0.5
    """
    omega_m = chain.getParams().OMEGA_M
    sigma_8 = chain.getParams().SIGMA_8

    s_8 = sigma_8 * (omega_m / 0.3) ** 0.5

    chain.addDerived(s_8, name='S_8', label=r'S_8')

    p = chain.paramNames.parWithName('S_8')

    return chain

def derive_parameter_Om(chain):
    """
    Derives the Omega_m parameter from omch2 and h0 in a GetDist chain.
    """
    omch2 = chain.getParams().omch2
    h0 = chain.getParams().h0

    omega_m = omch2 / (h0 / 100) ** 2

    chain.addDerived(omega_m, name='OMEGA_M', label=r'\Omega_{\rm m}')

    
    p = chain.paramNames.parWithName('OMEGA_M')

    return chain

def get_sigma_tension(mean1, low1, high1, mean2, low2, high2):
    sigma1 = 0.5 * (high1 + low1)
    sigma2 = 0.5 * (high2 + low2)
    delta_mean = np.abs(mean1 - mean2)
    sigma_tension = delta_mean / np.sqrt(sigma1**2 + sigma2**2)
    sign = 1 if mean1 > mean2 else -1
    return sigma_tension * sign

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