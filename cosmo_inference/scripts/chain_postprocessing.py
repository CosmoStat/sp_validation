"""
Scripts to postprocess the CosmoSIS chains
Author: Sacha Guerrini
"""
import os
import configparser
import subprocess

import numpy as np
from getdist import plots, MCSamples

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

def load_samples_and_write_paramnames(path_samples, path_paramnames):
    """
    Load the samples from a CosmoSIS chain and write the parameter names to a file
    """
    with open(path_samples, 'r') as file:
        params = file.readline()[1:].split('\t')[:-4]
        file.close()

    with open(path_paramnames, 'w') as file:
        for i in range(len(params)):
            if len(params[i].split('--')) > 1:
                file.write(params[i].split('--')[1] + '\n')
            else:
                file.write(params[i].split('--')[0] + '\n')
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
        samples = np.column_stack((np.exp(samples[:,-3]),samples[:,-1]-samples[:,-2],samples[:,0:-3]))
    else:
        samples = np.column_stack((samples[:,-1],samples[:,-2],samples[:,0:-4]))
    np.savetxt(path_gd, samples)
    return 0

def load_chain(path_gd, smoothing_scale=0.3):
    g = plots.get_single_plotter()
    chain = g.samples_for_root(path_gd, cache=False, settings={'ignore_rows':0, 'smooth_scale_1D': smoothing_scale, 'smooth_scale_2D': smoothing_scale})
    return chain

def extract_best_fit_params(chain):
    best_fit_params = {}
    margestats = chain.getMargeStats()
    likestats = chain.getLikeStats()
    for i, par in enumerate(likestats.names):
        best_fit_params.update(
            {
                par.name: np.average(chain.samples[:, i], weights=chain.weights) #Use the average as the default
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

