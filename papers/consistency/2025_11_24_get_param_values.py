# %%
import IPython

ipython = IPython.get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import copy
import os
import sys

from tqdm import tqdm

sys.path.append("/home/guerrini/sp_validation/cosmo_inference/scripts/")

import chain_postprocessing as cpp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from getdist import plots

g = plots.get_subplot_plotter(width_inch=7)
g.settings.axes_fontsize = 15
g.settings.axes_labelsize = 15
g.settings.alpha_filled_add = 0.7
g.settings.legend_fontsize = 15

if os.path.exists("/home/guerrini/matplotlib_config/paper.mplstyle"):
    plt.style.use("/home/guerrini/matplotlib_config/paper.mplstyle")

# Set default palette - will be updated per plot as needed
sns.set_palette("husl")

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# %%
root_dir = "/n09data/guerrini/glass_mock_chains/"

failed_simulations = [82, 83, 281, 282, 283, 284, 285, 286, 287]

weird_simulations = [82, 97, 123]

weird_simulations_v1 = [16, 97, 120, 123]

weird_simulations_v2 = []
version = "v6"
roots = [f"glass_mock_{version}_{i + 1:05d}" for i in range(350)]


# %%
def load_samples_and_write_paramames(root_dir, root, chain_type="configuration"):
    assert chain_type in ["configuration", "harmonic"], (
        "chain_type must be 'configuration' or 'harmonic'"
    )

    if chain_type == "configuration":
        path_samples = root_dir + "{}/{}/samples_{}.txt".format("/" + root, root, root)
        path_paramnames = root_dir + "{}/{}/getdist_{}.paramnames".format(
            "/" + root, root, root
        )
    else:
        path_samples = root_dir + "{}/{}/samples_{}_cell.txt".format(
            "/" + root, root, root
        )
        path_paramnames = root_dir + "{}/{}/getdist_{}_cell.paramnames".format(
            "/" + root, root, root
        )

    cpp.load_samples_and_write_paramnames(path_samples, path_paramnames)


def write_samples_getdist_format(root_dir, root, chain_type="configuration"):
    assert chain_type in ["configuration", "harmonic"], (
        "chain_type must be 'configuration' or 'harmonic'"
    )

    if chain_type == "configuration":
        path_samples = root_dir + "{}/{}/samples_{}.txt".format("/" + root, root, root)
        path_gd_samples = root_dir + "{}/{}/getdist_{}.txt".format(
            "/" + root, root, root
        )
        path_gd = root_dir + "{}/{}/getdist_{}".format(root, root, root)
    else:
        path_samples = root_dir + "{}/{}/samples_{}_cell.txt".format(
            "/" + root, root, root
        )
        path_gd_samples = root_dir + "{}/{}/getdist_{}_cell.txt".format(
            "/" + root, root, root
        )
        path_gd = root_dir + "{}/{}/getdist_{}_cell".format(root, root, root)

    sampler_type = "nautilus" if "nautilus" in root else "polychord"
    cpp.write_samples_getdist_format(
        path_samples, path_gd_samples, chain_type=sampler_type
    )

    chain = g.samples_for_root(
        path_gd,
        cache=False,
        settings={"ignore_rows": 0.0, "smooth_scale_2D": 0.5, "smooth_scale_1D": 0.5},
    )

    return chain


def extract_param_chain(chain, param_names):
    param_values = {}
    for param_name in param_names:
        if param_name not in chain.getParamNames().list():
            raise ValueError(f"Parameter {param_name} not found in chain.")

        mean = cpp.compute_average(chain, param_name)
        limi_68_upper, limi_68_lower, limi_95_upper, limi_95_lower = cpp.compute_limits(
            chain, param_name
        )
        map_1D = cpp.compute_map_1D(chain, param_name)
        param_values[param_name] = {
            "mean": mean,
            "1sigma_minus": limi_68_lower,
            "1sigma_plus": limi_68_upper,
            "2sigma_minus": limi_95_lower,
            "2sigma_plus": limi_95_upper,
            "map_1D": map_1D,
        }
        if param_name == "S_8" or param_name == "OMEGA_M":
            s8_map_2D, omega_m_map_2D = cpp.compute_map_2D(chain, "S_8", "OMEGA_M")
            param_values[param_name]["map_2D"] = (
                s8_map_2D if param_name == "S_8" else omega_m_map_2D
            )

    return param_values


def concatenate_param_stats(name, param_values, verbose=False):
    output = [name]
    for key in param_values.keys():
        param_stat = param_values[key]
        if verbose:
            print(
                f"{name} - {key}: {param_stat['mean']:.4f} +{param_stat['1sigma_plus']:.4f}/-{param_stat['1sigma_minus']:.4f} (1σ), +{param_stat['2sigma_plus']:.4f}/-{param_stat['2sigma_minus']:.4f} (2σ)"
            )

        param_list = [
            param_stat["mean"],
            param_stat["1sigma_minus"],
            param_stat["1sigma_plus"],
            param_stat["2sigma_minus"],
            param_stat["2sigma_plus"],
            param_stat["map_1D"],
        ]

        if key == "S_8" or key == "OMEGA_M":
            param_list.append(param_stat["map_2D"])

        output += param_list

    return output


def merge_param_stats(params_configuration, params_harmonic):
    merged_params = {}
    for key in params_configuration.keys():
        if key in params_harmonic:
            merged_params[key] = {
                "configuration": params_configuration[key],
                "harmonic": params_harmonic[key],
            }
    return merged_params


def concatenate_merge_params(name, merged_params, verbose=False):
    output = [name]
    for key in merged_params.keys():
        param_config = merged_params[key]["configuration"]
        param_harm = merged_params[key]["harmonic"]

        if verbose:
            print(
                f"{name} - {key} (Configuration): {param_config['mean']:.4f} +{param_config['1sigma_plus']:.4f}/-{param_config['1sigma_minus']:.4f} (1σ), +{param_config['2sigma_plus']:.4f}/-{param_config['2sigma_minus']:.4f} (2σ)"
            )
            print(
                f"{name} - {key} (Harmonic): {param_harm['mean']:.4f} +{param_harm['1sigma_plus']:.4f}/-{param_harm['1sigma_minus']:.4f} (1σ), +{param_harm['2sigma_plus']:.4f}/-{param_harm['2sigma_minus']:.4f} (2σ)"
            )

        param_list = [
            param_config["mean"],
            param_config["1sigma_minus"],
            param_config["1sigma_plus"],
            param_config["2sigma_minus"],
            param_config["2sigma_plus"],
            param_config["map_1D"],
        ]

        if key == "S_8" or key == "OMEGA_M":
            param_list.append(param_config["map_2D"])

        param_list += [
            param_harm["mean"],
            param_harm["1sigma_minus"],
            param_harm["1sigma_plus"],
            param_harm["2sigma_minus"],
            param_harm["2sigma_plus"],
            param_harm["map_1D"],
        ]

        if key == "S_8" or key == "OMEGA_M":
            param_list.append(param_harm["map_2D"])

        output += param_list

    return output


# %%
chain_configuration = []
chain_harmonic = []
skip_weird = True

for i, root in enumerate(tqdm(roots)):
    if (i + 1) in failed_simulations:
        print(f"Skipping failed simulation {i + 1}")
        print("Add a flag 'ERROR' to the chains lists")
        chain_configuration.append("ERROR")
        chain_harmonic.append("ERROR")
        continue

    if skip_weird and (i + 1) in weird_simulations_v2:
        print(f"Skipping weird simulation {i + 1}")
        print("Add a flag 'ERROR' to the chains lists")
        chain_configuration.append("ERROR")
        chain_harmonic.append("ERROR")
        continue

    # Load samples and write paramnames for configuration space
    load_samples_and_write_paramames(root_dir, root, chain_type="configuration")
    write_samples_getdist_format(root_dir, root, chain_type="configuration")
    chain_config = g.samples_for_root(
        root_dir + f"/{root}/{root}/getdist_{root}",
        cache=False,
        settings={"ignore_rows": 0.0, "smooth_scale_2D": 0.5, "smooth_scale_1D": 0.5},
    )
    chain_configuration.append(chain_config)

    # Load samples and write paramnames for harmonic space
    load_samples_and_write_paramames(root_dir, root, chain_type="harmonic")
    write_samples_getdist_format(root_dir, root, chain_type="harmonic")
    chain_harm = g.samples_for_root(
        root_dir + f"/{root}/{root}/getdist_{root}_cell",
        cache=False,
        settings={"ignore_rows": 0.0, "smooth_scale_2D": 0.5, "smooth_scale_1D": 0.5},
    )
    chain_harmonic.append(chain_harm)

# %%
param_names = ["S_8", "OMEGA_M", "SIGMA_8"]

output_mocks_config = np.array(
    [
        "Name",
        "S8_mean",
        "S8_1sigma_minus",
        "S8_1sigma_plus",
        "S8_2sigma_minus",
        "S8_2sigma_plus",
        "S8_map_1D",
        "S8_map_2D",
        "OMEGA_M_mean",
        "OMEGA_M_1sigma_minus",
        "OMEGA_M_1sigma_plus",
        "OMEGA_M_2sigma_minus",
        "OMEGA_M_2sigma_plus",
        "OMEGA_M_map_1D",
        "OMEGA_M_map_2D",
        "SIGMA_8_mean",
        "SIGMA_8_1sigma_minus",
        "SIGMA_8_1sigma_plus",
        "SIGMA_8_2sigma_minus",
        "SIGMA_8_2sigma_plus",
        "SIGMA_8_map_1D",
    ]
)


output_mocks_harm = copy.deepcopy(output_mocks_config)

output_mocks_merged = np.array(
    [
        "Name",
        "S8_config_mean",
        "S8_config_1sigma_minus",
        "S8_config_1sigma_plus",
        "S8_config_2sigma_minus",
        "S8_config_2sigma_plus",
        "S8_config_map_1D",
        "S8_config_map_2D",
        "S8_harm_mean",
        "S8_harm_1sigma_minus",
        "S8_harm_1sigma_plus",
        "S8_harm_2sigma_minus",
        "S8_harm_2sigma_plus",
        "S8_harm_map_1D",
        "S8_harm_map_2D",
        "OMEGA_M_config_mean",
        "OMEGA_M_config_1sigma_minus",
        "OMEGA_M_config_1sigma_plus",
        "OMEGA_M_config_2sigma_minus",
        "OMEGA_M_config_2sigma_plus",
        "OMEGA_M_config_map_1D",
        "OMEGA_M_config_map_2D",
        "OMEGA_M_harm_mean",
        "OMEGA_M_harm_1sigma_minus",
        "OMEGA_M_harm_1sigma_plus",
        "OMEGA_M_harm_2sigma_minus",
        "OMEGA_M_harm_2sigma_plus",
        "OMEGA_M_harm_map_1D",
        "OMEGA_M_harm_map_2D",
        "SIGMA_8_config_mean",
        "SIGMA_8_config_1sigma_minus",
        "SIGMA_8_config_1sigma_plus",
        "SIGMA_8_config_2sigma_minus",
        "SIGMA_8_config_2sigma_plus",
        "SIGMA_8_config_map_1D",
        "SIGMA_8_harm_mean",
        "SIGMA_8_harm_1sigma_minus",
        "SIGMA_8_harm_1sigma_plus",
        "SIGMA_8_harm_2sigma_minus",
        "SIGMA_8_harm_2sigma_plus",
        "SIGMA_8_harm_map_1D",
    ]
)

for i, root in enumerate(tqdm(roots)):
    if chain_configuration[i] == "ERROR" or chain_harmonic[i] == "ERROR":
        param = [root] + [np.nan for _ in range(output_mocks_config.shape[1] - 1)]
        param_merged = [root] + [
            np.nan for _ in range(output_mocks_merged.shape[1] - 1)
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

np.savetxt(
    f"{root_dir}/summary_parameter_constraints_configuration_space_{version}.txt",
    output_mocks_config,
    fmt="%s",
    delimiter=";",
)
np.savetxt(
    f"{root_dir}/summary_parameter_constraints_harmonic_space_{version}.txt",
    output_mocks_harm,
    fmt="%s",
    delimiter=";",
)
np.savetxt(
    f"{root_dir}/summary_parameter_constraints_merged_{version}.txt",
    output_mocks_merged,
    fmt="%s",
    delimiter=";",
)

# %%
