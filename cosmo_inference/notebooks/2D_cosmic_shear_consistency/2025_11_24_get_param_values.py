# %%
import IPython

ipython = IPython.get_ipython()

if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

import os
import copy
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import healpy as hp
import seaborn as sns
from astropy.io import fits

from getdist import plots, MCSamples

g = plots.get_subplot_plotter(width_inch=7)
g.settings.axes_fontsize=15
g.settings.axes_labelsize=15
g.settings.alpha_filled_add = 0.7
g.settings.legend_fontsize = 15

if os.path.exists("/home/guerrini/matplotlib_config/paper.mplstyle"):
    plt.style.use(
        "/home/guerrini/matplotlib_config/paper.mplstyle"
    )

# Set default palette - will be updated per plot as needed
sns.set_palette("husl")

if ipython is not None:
    ipython.run_line_magic('matplotlib', 'inline')

# %%
root_dir = '/n09data/guerrini/glass_mock_chains/'

failed_simulations = [
    
]

weird_simulations = [
    82, 97, 123
]

roots = [
    f"glass_mock_v0_{i+1:05d}" for i in range(350)
]

# %%
def load_samples_and_write_paramames(root_dir, root, chain_type="configuration"):
    assert chain_type in ["configuration", "harmonic"], "chain_type must be 'configuration' or 'harmonic'"

    if chain_type == "configuration":
        path_samples = root_dir + '{}/{}/samples_{}.txt'.format('/'+root, root, root)
        path_paramnames = root_dir + '{}/{}/getdist_{}.paramnames'.format('/'+root, root, root)
    else:
        path_samples = root_dir + '{}/{}/samples_{}_cell.txt'.format('/'+root, root, root)
        path_paramnames = root_dir + '{}/{}/getdist_{}_cell.paramnames'.format('/'+root, root, root)
    
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

def write_samples_getdist_format(root_dir, root, chain_type="configuration"):
    assert chain_type in ["configuration", "harmonic"], "chain_type must be 'configuration' or 'harmonic'"

    if chain_type == "configuration":
        path_samples = root_dir + '{}/{}/samples_{}.txt'.format('/'+root, root, root)
        path_gd_samples = root_dir + '{}/{}/getdist_{}.txt'.format('/'+root, root, root)
        path_gd = root_dir + '{}/{}/getdist_{}'.format(root,root,root)
    else:
        path_samples = root_dir + '{}/{}/samples_{}_cell.txt'.format('/'+root, root, root)
        path_gd_samples = root_dir + '{}/{}/getdist_{}_cell.txt'.format('/'+root, root, root)
        path_gd = root_dir + '{}/{}/getdist_{}_cell'.format(root,root,root)
    
    samples = np.loadtxt(
        path_samples,
    )
    if 'nautilus' in root:
        samples = np.column_stack((np.exp(samples[:,-3]),samples[:,-1]-samples[:,-2],samples[:,0:-3]))
    else:
        samples = np.column_stack((samples[:,-1],samples[:,-3],samples[:,0:-4]))
    np.savetxt(path_gd_samples, samples)

    chain = g.samples_for_root(
        path_gd,
        cache=False,
        settings={
            'ignore_rows': 0.,
            'smooth_scale_2D': 0.5,
            'smooth_scale_1D': 0.5
        }
    )

    return chain

def extract_param_chain(chain, param_names):
    margestats = chain.getMargeStats()
    likestats = chain.getLikeStats()

    param_values = {}
    for param_name in param_names:
        if param_name not in chain.getParamNames().list():
            raise ValueError(f"Parameter {param_name} not found in chain.")
        
        param_stats = margestats.parWithName(param_name)
        param_values[param_name] = {
            'mean': param_stats.mean,
            '1sigma_minus': param_stats.mean - param_stats.limits[0].lower,
            '1sigma_plus': param_stats.limits[0].upper - param_stats.mean,
            '2sigma_minus': param_stats.mean - param_stats.limits[1].lower,
            '2sigma_plus': param_stats.limits[1].upper - param_stats.mean,
        }

    return param_values

def concatenate_param_stats(name, param_values, verbose=False):
    output = [name]
    for key in param_values.keys():
        param_stat = param_values[key]
        if verbose:
            print(f"{name} - {key}: {param_stat['mean']:.4f} +{param_stat['1sigma_plus']:.4f}/-{param_stat['1sigma_minus']:.4f} (1σ), +{param_stat['2sigma_plus']:.4f}/-{param_stat['2sigma_minus']:.4f} (2σ)")

        param_list = [
            param_stat['mean'],
            param_stat['1sigma_minus'],
            param_stat['1sigma_plus'],
            param_stat['2sigma_minus'],
            param_stat['2sigma_plus']
        ]

        output += param_list

    return output

def merge_param_stats(params_configuration, params_harmonic):
    merged_params = {}
    for key in params_configuration.keys():
        if key in params_harmonic:
            merged_params[key] = {
                'configuration': params_configuration[key],
                'harmonic': params_harmonic[key]
            }
    return merged_params

def concatenate_merge_params(name, merged_params, verbose=False):
    output = [name]
    for key in merged_params.keys():
        param_config = merged_params[key]['configuration']
        param_harm = merged_params[key]['harmonic']

        if verbose:
            print(f"{name} - {key} (Configuration): {param_config['mean']:.4f} +{param_config['1sigma_plus']:.4f}/-{param_config['1sigma_minus']:.4f} (1σ), +{param_config['2sigma_plus']:.4f}/-{param_config['2sigma_minus']:.4f} (2σ)")
            print(f"{name} - {key} (Harmonic): {param_harm['mean']:.4f} +{param_harm['1sigma_plus']:.4f}/-{param_harm['1sigma_minus']:.4f} (1σ), +{param_harm['2sigma_plus']:.4f}/-{param_harm['2sigma_minus']:.4f} (2σ)")

        param_list = [
            param_config['mean'],
            param_config['1sigma_minus'],
            param_config['1sigma_plus'],
            param_config['2sigma_minus'],
            param_config['2sigma_plus'],
            param_harm['mean'],
            param_harm['1sigma_minus'],
            param_harm['1sigma_plus'],
            param_harm['2sigma_minus'],
            param_harm['2sigma_plus']
        ]

        output += param_list

    return output

# %%
chain_configuration = []
chain_harmonic = []
skip_weird = True

for i, root in enumerate(tqdm(roots)):
    if (i+1) in failed_simulations:
        print(f"Skipping failed simulation {i+1}")
        print(f"Add a flag 'ERROR' to the chains lists")
        chain_configuration.append('ERROR')
        chain_harmonic.append('ERROR')
        continue

    if skip_weird and (i+1) in weird_simulations:
        print(f"Skipping weird simulation {i+1}")
        print(f"Add a flag 'ERROR' to the chains lists")
        chain_configuration.append('ERROR')
        chain_harmonic.append('ERROR')
        continue

    # Load samples and write paramnames for configuration space
    load_samples_and_write_paramames(root_dir, root, chain_type="configuration")
    write_samples_getdist_format(root_dir, root, chain_type="configuration")
    chain_config = g.samples_for_root(
        root_dir + f'/{root}/{root}/getdist_{root}',
        cache=False,
        settings={
            'ignore_rows': 0.,
            'smooth_scale_2D': 0.5,
            'smooth_scale_1D': 0.5
        }
    )
    chain_configuration.append(chain_config)

    # Load samples and write paramnames for harmonic space
    load_samples_and_write_paramames(root_dir, root, chain_type="harmonic")
    write_samples_getdist_format(root_dir, root, chain_type="harmonic")
    chain_harm = g.samples_for_root(
        root_dir + f'/{root}/{root}/getdist_{root}_cell',
        cache=False,
        settings={
            'ignore_rows': 0.,
            'smooth_scale_2D': 0.5,
            'smooth_scale_1D': 0.5
        }
    )
    chain_harmonic.append(chain_harm)

# %%
param_names = ['S_8', 'OMEGA_M', 'SIGMA_8']

output_mocks_config = np.array([
    "Name", "S8_mean", "S8_1sigma_minus", "S8_1sigma_plus", "S8_2sigma_minus", "S8_2sigma_plus",
    "OMEGA_M_mean", "OMEGA_M_1sigma_minus", "OMEGA_M_1sigma_plus", "OMEGA_M_2sigma_minus", "OMEGA_M_2sigma_plus",
    "SIGMA_8_mean", "SIGMA_8_1sigma_minus", "SIGMA_8_1sigma_plus", "SIGMA_8_2sigma_minus", "SIGMA_8_2sigma_plus"
])



output_mocks_harm = copy.deepcopy(output_mocks_config)

output_mocks_merged = np.array([
    "Name",
    "S8_config_mean", "S8_config_1sigma_minus", "S8_config_1sigma_plus", "S8_config_2sigma_minus", "S8_config_2sigma_plus",
    "S8_harm_mean", "S8_harm_1sigma_minus", "S8_harm_1sigma_plus", "S8_harm_2sigma_minus", "S8_harm_2sigma_plus",
    "OMEGA_M_config_mean", "OMEGA_M_config_1sigma_minus", "OMEGA_M_config_1sigma_plus", "OMEGA_M_config_2sigma_minus", "OMEGA_M_config_2sigma_plus",
    "OMEGA_M_harm_mean", "OMEGA_M_harm_1sigma_minus", "OMEGA_M_harm_1sigma_plus", "OMEGA_M_harm_2sigma_minus", "OMEGA_M_harm_2sigma_plus",
    "SIGMA_8_config_mean", "SIGMA_8_config_1sigma_minus", "SIGMA_8_config_1sigma_plus", "SIGMA_8_config_2sigma_minus", "SIGMA_8_config_2sigma_plus",
    "SIGMA_8_harm_mean", "SIGMA_8_harm_1sigma_minus", "SIGMA_8_harm_1sigma_plus", "SIGMA_8_harm_2sigma_minus", "SIGMA_8_harm_2sigma_plus"
])

for i, root in enumerate(tqdm(roots)):
    if chain_configuration[i] == 'ERROR' or chain_harmonic[i] == 'ERROR':
        param = [
            root ] + [ np.nan for _ in range(len(param_names)*5)
        ]
        param_merged = [
            root ] + [ np.nan for _ in range(len(param_names)*10)
        ]
        output_mocks_merged = np.vstack((output_mocks_merged, param_merged))
        output_mocks_config = np.vstack((output_mocks_config, param))
        output_mocks_harm = np.vstack((output_mocks_harm, param))
        continue

    param_values_config = extract_param_chain(chain_configuration[i], param_names)
    param_values_harm = extract_param_chain(chain_harmonic[i], param_names)
    param_merged = merge_param_stats(param_values_config, param_values_harm)

    param_config = concatenate_param_stats(root, param_values_config, verbose=False)
    param_harm = concatenate_param_stats(root, param_values_harm, verbose=False)
    param_merged = concatenate_merge_params(root, param_merged, verbose=False)
    
    output_mocks_config = np.vstack((output_mocks_config, param_config))
    output_mocks_harm = np.vstack((output_mocks_harm, param_harm))
    output_mocks_merged = np.vstack((output_mocks_merged, param_merged))

np.savetxt(f"{root_dir}/summary_parameter_constraints_configuration_space.txt", output_mocks_config, fmt='%s', delimiter=';')
np.savetxt(f"{root_dir}/summary_parameter_constraints_harmonic_space.txt", output_mocks_harm, fmt='%s', delimiter=';')
np.savetxt(f"{root_dir}/summary_parameter_constraints_merged.txt", output_mocks_merged, fmt='%s', delimiter=';')

# %%
