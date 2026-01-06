# %%
import os
import configparser
import subprocess

from IPython import get_ipython

ipython = get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import numpy as np
from astropy.io import fits
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from getdist import plots, MCSamples

plt.style.use(
    './matplotlib_config/paper.mplstyle'
)

sns.set_palette("husl")

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

g = plots.get_subplot_plotter(width_inch=30)
g.settings.axes_fontsize=30
g.settings.axes_labelsize=30
g.settings.alpha_filled_add = 0.7
g.settings.legend_fontsize = 40

# %%
root_dir = "/n09data/guerrini/glass_mock_chains/"
chain_version = "v2" # choose v0 or v1
assert chain_version in ["v0", "v1", "v2"], "Invalid chain version"

#Path to the ini config files
path_ini_files = f'/home/guerrini/sp_validation/cosmo_inference/cosmosis_config/glass_mocks_{chain_version}/'

# Create the list of mocks
max_sim = 8
roots = [f"glass_mock_{chain_version}_{str(i).zfill(5)}" for i in range(1, max_sim + 1)]
lower_boud_cell_ee = 300.0
upper_bound_cell_ee = 1600.0

run_best_fit_type = "map" # should be in ["map", "average"]
assert run_best_fit_type in ["map", "average"], "Invalid best fit type"

# %%
# Retrieve the chains
for root in roots:
    with open(root_dir + '{}/{}/samples_{}_cell.txt'.format(root, root, root)) as file:
        params = file.readline()[1:].split('\t')[:-4]
        file.close()

    with open(root_dir + '{}/{}/getdist_{}_cell.paramnames'.format(root, root, root), "w") as file:
        for i in range(len(params)):
            if len(params[i].split('--')) > 1:
                file.write(params[i].split('--')[1] + '\n')
            else:
                file.write(params[i].split('--')[0] + '\n')
        file.close()

# Read chain
chains=[]

for root in roots:

    samples = np.loadtxt(root_dir + '{}/{}/samples_{}_cell.txt'.format(root,root,root))
    print(len(samples))
    if 'nautilus' in root:
        samples = np.column_stack((np.exp(samples[:,-3]),samples[:,-1]-samples[:,-2],samples[:,0:-3]))
    else:
        samples = np.column_stack((samples[:,-1],samples[:,-3],samples[:,0:-4]))
    np.savetxt(root_dir + '{}/{}/getdist_{}_cell.txt'.format(root,root,root), samples)

    chain = g.samples_for_root(root_dir + '{}/{}/getdist_{}_cell'.format(root,root,root),
                               cache=False,
                               settings={'ignore_rows':0,
                                         'smooth_scale_2D':0.3,
                                         'smooth_scale_1D':0.3})

    chains.append(chain)

# %%
name_list = ['OMEGA_M','ombh2','h0','n_s','SIGMA_8','s_8_input', 'logt_agn','a','m1','bias_1']
label_list = ['\Omega_m', '\omega_b h^2', 'h_0', 'n_s', '\sigma_8', 'S_8', 'log T_{AGN}', 'A_{IA}', 'm_1', '\Delta z_1']

for chain in chains:
    param_names = chain.getParamNames()
    for name, label in zip(name_list, label_list):
        param_names.parWithName(name).label = label

# %%
#Extract the best fit parameters
best_fit = {}

for root, chain in zip(roots, chains):
    print(root)
    likestats = chain.getLikeStats()
    bestfit_idx = np.argmax(chain.loglikes)
    maxlike = chain.loglikes[bestfit_idx]
    print(f"Maximum Likelihood: {maxlike:.5g}")
    best_fit[root] = {
        'likelihood': maxlike
    }
    margestats = chain.getMargeStats()
    s8_stats = margestats.parWithName('S_8')
    sigma8_stats = margestats.parWithName('SIGMA_8')
    omegam_stats = margestats.parWithName('OMEGA_M')
    a_ia_stats = margestats.parWithName('a')

    best_fit[root].update({
        'S_8_mean': s8_stats.mean,
        'S_8_lower': s8_stats.mean - s8_stats.limits[0].lower,
        'S_8_upper': s8_stats.limits[0].upper - s8_stats.mean,
        'sigma_8_mean': sigma8_stats.mean,
        'sigma_8_lower': sigma8_stats.mean - sigma8_stats.limits[0].lower,
        'sigma_8_upper': sigma8_stats.limits[0].upper - sigma8_stats.mean,
        'omega_m_mean': omegam_stats.mean,
        'omega_m_lower': omegam_stats.mean - omegam_stats.limits[0].lower,
        'omega_m_upper': omegam_stats.limits[0].upper - omegam_stats.mean,
        'A_IA_mean': a_ia_stats.mean,
        'A_IA_lower': a_ia_stats.mean - a_ia_stats.limits[0].lower,
        'A_IA_upper': a_ia_stats.limits[0].upper - a_ia_stats.mean
    })
    try:
        t_agn_stats = margestats.parWithName('logt_agn')
        best_fit[root].update({
            'logt_agn_mean': t_agn_stats.mean,
            'logt_agn_lower': t_agn_stats.mean - t_agn_stats.limits[0].lower,
            'logt_agn_upper': t_agn_stats.limits[0].upper - t_agn_stats.mean
        })
    except:
        pass
    for i, par in enumerate(likestats.names):
        if run_best_fit_type == "average":
            best_fit[root].update({par.name: np.average(chain.samples[:, i], weights=chain.weights)})
        elif run_best_fit_type == "map":
            best_fit[root].update({par.name: chain.samples[:, i][bestfit_idx]})
        else:
            raise ValueError("Invalid run_best_fit_type")



# %%
# Run CosmoSis in test mode to get the data vectors

if not os.path.exists(path_ini_files+'/values_empty.ini'):
    content = """[cosmological_parameters]

tau          =  0.0544
w            = -1.0
massive_nu   =  1
massless_nu  =  2.046
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

env = os.environ.copy()
env["LD_LIBRARY_PATH"] = "/home/guerrini/.conda/envs/sp_validation_3.11/lib/python3.11/site-packages/cosmosis/datablock:" + env.get("LD_LIBRARY_PATH", "")

os.chdir('/home/guerrini/sp_validation/cosmo_inference/')

for idx, root in enumerate(roots):
    print(root)
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve case sensitivity of option names
    config.read(path_ini_files+'/values_empty.ini')
    for param, value in best_fit[root].items():
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
    config_file_path = path_ini_files+f'/cosmosis_pipeline_glass_mocks_{chain_version}_glass_mock_{str(idx+1).zfill(5)}_cell.ini'
    config.read(config_file_path)
    sampler = config['runtime']['sampler']
    config['runtime']['sampler'] = 'test'
    values = config['pipeline']['values']
    config['pipeline']['values'] = path_ini_files + '/values_empty.ini'
    if run_best_fit_type == "map":
        config['test']['save_dir'] = f"%(SCRATCH)s/best_fit/{root}_cell_map"
    elif run_best_fit_type == "average":
        config['test']['save_dir'] = f"%(SCRATCH)s/best_fit/{root}_cell"
    else:
        raise ValueError("Invalid run_best_fit_type")


    with open(config_file_path, 'w') as configfile:
        config.write(configfile)

    #Run cosmosis
    result = subprocess.run(
        ['cosmosis',  config_file_path],
        env=env,
        capture_output=True,
        text=True
    )
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")

    #Modify the ini file to the previous one
    config['pipeline']['values'] = values
    config['runtime']['sampler'] = sampler
    if run_best_fit_type == "map":
        config['test']['save_dir'] = f"%(SCRATCH)s/best_fit/{root}_cell"

    with open(config_file_path, 'w') as configfile:
        config.write(configfile)
# %%
