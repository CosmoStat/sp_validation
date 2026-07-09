import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from getdist import plots
from tqdm import tqdm

g = plots.get_subplot_plotter(width_inch=7)
g.settings.axes_fontsize = 15
g.settings.axes_labelsize = 15
g.settings.alpha_filled_add = 0.7
g.settings.legend_fontsize = 15

if os.path.exists("/home/guerrini/matplotlib_config/paper.mplstyle"):
    plt.style.use("/home/guerrini/matplotlib_config/paper.mplstyle")

# Set default palette - will be updated per plot as needed
sns.set_palette("husl")

root_dir = "/n09data/guerrini/glass_mock_chains/"
chain_version = "v6"
num_sims = 350

roots = [f"glass_mock_{chain_version}_{i + 1:05d}" for i in range(num_sims)]


#
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

    with open(path_samples, "r") as file:
        params = file.readline()[1:].split("\t")[:-4]
        file.close()

    with open(path_paramnames, "w") as file:
        for i in range(len(params)):
            if len(params[i].split("--")) > 1:
                file.write(params[i].split("--")[1] + "\n")
            else:
                file.write(params[i].split("--")[0] + "\n")
        file.close()


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

    samples = np.loadtxt(
        path_samples,
    )
    if "nautilus" in root:
        samples = np.column_stack(
            (np.exp(samples[:, -3]), samples[:, -1] - samples[:, -2], samples[:, 0:-3])
        )
    else:
        samples = np.column_stack((samples[:, -1], samples[:, -2], samples[:, 0:-4]))
    np.savetxt(path_gd_samples, samples)

    chain = g.samples_for_root(
        path_gd,
        cache=False,
        settings={"ignore_rows": 0.0, "smooth_scale_2D": 0.5, "smooth_scale_1D": 0.5},
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
            "mean": param_stats.mean,
            "1sigma_minus": param_stats.mean - param_stats.limits[0].lower,
            "1sigma_plus": param_stats.limits[0].upper - param_stats.mean,
            "2sigma_minus": param_stats.mean - param_stats.limits[1].lower,
            "2sigma_plus": param_stats.limits[1].upper - param_stats.mean,
        }

        param_stats = likestats.parWithName(param_name)
        param_names_getdist = chain.getParamNames()
        par = param_names_getdist.parWithName(param_name)
        kde = chain.get1DDensity(par, num_bins=1000)
        kde_map = kde.x[np.argmax(kde.P)]
        param_values[param_name].update(
            {
                "MAP": kde_map,
            }
        )

    par = chain.getParamNames().parWithName("S_8")
    par_om = chain.getParamNames().parWithName("OMEGA_M")
    kde = chain.get2DDensity(par, par_om, fine_bins_2D=1000)
    s8_kde_map = kde.x[np.unravel_index(np.argmax(kde.P), kde.P.shape)[1]]
    om_kde_map = kde.y[np.unravel_index(np.argmax(kde.P), kde.P.shape)[0]]
    param_values["S_8"].update(
        {
            "MAP_2D": s8_kde_map,
        }
    )
    param_values["OMEGA_M"].update(
        {
            "MAP_2D": om_kde_map,
        }
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
            param_stat["MAP"],
        ]

        if key == "S_8":
            param_list.append(param_stat["MAP_2D"])

        if key == "OMEGA_M":
            param_list.append(param_stat["MAP_2D"])

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
            param_config["MAP"],
            param_harm["mean"],
            param_harm["1sigma_minus"],
            param_harm["1sigma_plus"],
            param_harm["2sigma_minus"],
            param_harm["2sigma_plus"],
            param_harm["MAP"],
        ]

        output += param_list

    return output


chain_harmonic = []
chain_config = []

for i, root in enumerate(tqdm(roots)):
    if os.path.isfile(f"{root_dir}/{root}/{root}/getdist_{root}.txt"):
        # Load samples and write paramnames for harmonic space
        load_samples_and_write_paramames(root_dir, root, chain_type="harmonic")
        write_samples_getdist_format(root_dir, root, chain_type="harmonic")
        chain_harm = g.samples_for_root(
            root_dir + f"/{root}/{root}/getdist_{root}_cell",
            cache=False,
            settings={
                "ignore_rows": 0.0,
                "smooth_scale_2D": 0.5,
                "smooth_scale_1D": 0.5,
            },
        )
        chain_harmonic.append(chain_harm)

        # Load samples and write paramnames for harmonic space
        load_samples_and_write_paramames(root_dir, root, chain_type="configuration")
        write_samples_getdist_format(root_dir, root, chain_type="configuration")
        chain_conf = g.samples_for_root(
            root_dir + f"/{root}/{root}/getdist_{root}",
            cache=False,
            settings={
                "ignore_rows": 0.0,
                "smooth_scale_2D": 0.5,
                "smooth_scale_1D": 0.5,
            },
        )
        chain_config.append(chain_conf)
#
param_names = ["S_8", "OMEGA_M", "SIGMA_8", "a"]

output_mocks_harm = np.array(
    [
        "Name",
        "S8_mean",
        "S8_1sigma_minus",
        "S8_1sigma_plus",
        "S8_2sigma_minus",
        "S8_2sigma_plus",
        "S8_MAP",
        "S8_MAP_2D",
        "OMEGA_M_mean",
        "OMEGA_M_1sigma_minus",
        "OMEGA_M_1sigma_plus",
        "OMEGA_M_2sigma_minus",
        "OMEGA_M_2sigma_plus",
        "OMEGA_M_MAP",
        "OMEGA_M_MAP_2D",
        "SIGMA_8_mean",
        "SIGMA_8_1sigma_minus",
        "SIGMA_8_1sigma_plus",
        "SIGMA_8_2sigma_minus",
        "SIGMA_8_2sigma_plus",
        "SIGMA_8_MAP",
        "a_mean",
        "a_1sigma_minus",
        "a_1sigma_plus",
        "a_2sigma_minus",
        "a_2sigma_plus",
        "a_MAP",
    ]
)

output_mocks_config = np.array(
    [
        "Name",
        "S8_mean",
        "S8_1sigma_minus",
        "S8_1sigma_plus",
        "S8_2sigma_minus",
        "S8_2sigma_plus",
        "S8_MAP",
        "S8_MAP_2D",
        "OMEGA_M_mean",
        "OMEGA_M_1sigma_minus",
        "OMEGA_M_1sigma_plus",
        "OMEGA_M_2sigma_minus",
        "OMEGA_M_2sigma_plus",
        "OMEGA_M_MAP",
        "OMEGA_M_MAP_2D",
        "SIGMA_8_mean",
        "SIGMA_8_1sigma_minus",
        "SIGMA_8_1sigma_plus",
        "SIGMA_8_2sigma_minus",
        "SIGMA_8_2sigma_plus",
        "SIGMA_8_MAP",
        "a_mean",
        "a_1sigma_minus",
        "a_1sigma_plus",
        "a_2sigma_minus",
        "a_2sigma_plus",
        "a_MAP",
    ]
)

for i, root in enumerate(tqdm(roots[:-1])):
    param_values_harm = extract_param_chain(chain_harmonic[i], param_names)

    param_harm = concatenate_param_stats(root, param_values_harm, verbose=False)

    output_mocks_harm = np.vstack((output_mocks_harm, param_harm))

    param_values_config = extract_param_chain(chain_config[i], param_names)

    param_config = concatenate_param_stats(root, param_values_config, verbose=False)

    output_mocks_config = np.vstack((output_mocks_config, param_config))

np.savetxt(
    f"summary_parameter_constraints_harmonic_space_{chain_version}.txt",
    output_mocks_harm,
    fmt="%s",
    delimiter=";",
)
np.savetxt(
    f"summary_parameter_constraints_configuration_space_{chain_version}.txt",
    output_mocks_config,
    fmt="%s",
    delimiter=";",
)
print(
    f"Saved summary of parameter constraints for harmonic space in summary_parameter_constraints_harmonic_space_{chain_version}.txt"
)
print(
    f"Saved summary of parameter constraints for configuration space in summary_parameter_constraints_configuration_space_{chain_version}.txt"
)


import pandas as pd

output_df_harm = pd.read_csv(
    f"summary_parameter_constraints_harmonic_space_{chain_version}.txt",
    delimiter=";",
    skiprows=1,
    names=output_mocks_harm[0],
)

output_df_config = pd.read_csv(
    f"summary_parameter_constraints_configuration_space_{chain_version}.txt",
    delimiter=";",
    skiprows=1,
    names=output_mocks_config[0],
)


# Define the true value of the parameters
from astropy.cosmology import Planck18 as planck

Omega_m_fid = planck.Om0
sigma_8_fid = 0.8102
s8_fid = sigma_8_fid * (Omega_m_fid / 0.3) ** 0.5
h = planck.h
Omega_b_fig = planck.Ob0
n_s_fid = 0.9665
print(
    f"Fiducial values: Omega_m = {Omega_m_fid}, sigma_8 = {sigma_8_fid}, S_8 = {s8_fid}"
)


sns.histplot(
    output_df_harm["S8_mean"] - output_df_config["S8_mean"],
    kde=True,
    bins=30,
    label="Mean",
)
# sns.histplot(
#     output_df_harm["S8_MAP"]-output_df_config["S8_MAP"],
#     kde=True,
#     bins=20,
#     label="MAP",
# )
sns.histplot(
    output_df_harm["S8_MAP_2D"] - output_df_config["S8_MAP_2D"],
    kde=True,
    bins=30,
    label="2D Mode",
    alpha=0.5,
)
plt.axvline(0, color="black", linestyle="--")
plt.legend(fontsize=12)

plt.xlabel(r"$\Delta S_8$")
plt.savefig(
    "./../../results/S8_comparison_harmonic_vs_configuration.pdf",
    bbox_inches="tight",
)


output_df_config["S8_MAP_2D"].shape
output_df_harm["S8_MAP_2D"].shape


# Create JointGrid
g = sns.JointGrid(
    x=output_df_config["OMEGA_M_MAP_2D"],
    y=output_df_config["S8_MAP_2D"],
    height=7,
    ratio=5,
    space=0,
)

# Main 2D histogram
sns.histplot(
    x=output_df_config["OMEGA_M_MAP_2D"],
    y=output_df_config["S8_MAP_2D"],
    bins=25,
    cmap="Greens",
    cbar=False,
    ax=g.ax_joint,
)

# Marginal histograms
sns.histplot(
    x=output_df_config["OMEGA_M_MAP_2D"], bins=25, color="#2ca25f", ax=g.ax_marg_x
)
sns.histplot(y=output_df_config["S8_MAP_2D"], bins=25, color="#2ca25f", ax=g.ax_marg_y)

# Add dashed reference lines
g.ax_joint.axvline(Omega_m_fid, color="k", linestyle="--")
g.ax_joint.axhline(s8_fid, color="k", linestyle="--")

# Labels
g.set_axis_labels(
    r"$\Omega_m$ estimated from mocks (Configuration space)",
    r"$S_8$ estimated from mocks (Configuration space)",
)

# Optional styling tweaks
g.ax_joint.tick_params(labelsize=12)
plt.savefig(
    "./../../results/S8_vs_OmegaM_configuration_space_mocks.pdf",
    bbox_inches="tight",
)
