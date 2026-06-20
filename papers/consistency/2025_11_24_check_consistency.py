# %%
# Trick to plot with tex
import os
import sys

os.environ["LD_LIBRARY_PATH"] = ""
os.environ["CONDA_PREFIX"] = "/home/guerrini/.conda/envs/sp_validation_3.11"


sys.path.append("/home/guerrini/sp_validation/cosmo_inference/scripts/")

import IPython

ipython = IPython.get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import os
import copy
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import healpy as hp
import seaborn as sns
from astropy.io import fits
import pandas as pd
from scipy.stats import norm
import chain_postprocessing as cp

from getdist import plots, MCSamples


if os.path.exists("/home/guerrini/matplotlib_config/paper.mplstyle"):
    plt.style.use("/home/guerrini/matplotlib_config/paper.mplstyle")

# Set default palette - will be updated per plot as needed
sns.set_palette("husl")

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# %%
output_fig_path = "./plots"
if not os.path.exists(output_fig_path):
    os.makedirs(output_fig_path)

chain_version = "v6"

# Load the output of the simulations
simulation_output = pd.read_csv(
    f"/n09data/guerrini/glass_mock_chains/summary_parameter_constraints_merged_{chain_version}.txt",
    delimiter=";",
)

# %%
simulation_output.info()

# %%
simulation_output.head()

# %%
# Define the true value of the parameters
from astropy.cosmology import Planck18 as planck

Omega_m_fid = planck.Om0
sigma_8_fid = 0.8102
s8_fid = sigma_8_fid * (Omega_m_fid / 0.3) ** 0.5
h = planck.h
Omega_b_fig = planck.Ob0
n_s_fid = 0.965
print(
    f"Fiducial values: Omega_m = {Omega_m_fid}, sigma_8 = {sigma_8_fid}, S_8 = {s8_fid}"
)

# %%
sns.histplot(
    simulation_output["S8_config_mean"], kde=True, bins=20, label="Configuration space"
)
sns.histplot(
    simulation_output["S8_harm_mean"],
    kde=True,
    label="Harmonic space",
    color="blue",
    bins=20,
    alpha=0.3,
)
plt.axvline(s8_fid, color="black", linestyle="--", label="Fiducial S8")
plt.legend(fontsize=12)
plt.title("Using the weighted average")

plt.xlabel(r"$S_8$ estimated from mocks")
plt.savefig(f"{output_fig_path}/S8_comparison_config_harm.png", dpi=300)
plt.show()
# %%
# Same plot for Omega_m and sigma_8
sns.histplot(
    simulation_output["OMEGA_M_config_mean"],
    kde=True,
    bins=20,
    label="Configuration space",
)
sns.histplot(
    simulation_output["OMEGA_M_harm_mean"],
    kde=True,
    label="Harmonic space",
    color="blue",
    bins=20,
    alpha=0.3,
)
plt.axvline(Omega_m_fid, color="black", linestyle="--", label="Fiducial Omega_m")
plt.legend(fontsize=12)
plt.title("Using the weighted average")

plt.xlabel(r"$\Omega_m$ estimated from mocks")
plt.savefig(f"{output_fig_path}/Omega_m_comparison_config_harm.png", dpi=300)
plt.show()

# %%
sns.histplot(
    simulation_output["SIGMA_8_config_mean"],
    kde=True,
    bins=20,
    label="Configuration space",
)
sns.histplot(
    simulation_output["SIGMA_8_harm_mean"],
    kde=True,
    label="Harmonic space",
    color="blue",
    bins=20,
    alpha=0.3,
)
plt.axvline(sigma_8_fid, color="black", linestyle="--", label="Fiducial sigma_8")
plt.legend(fontsize=12)
plt.title("Using the weighted average")

plt.xlabel(r"$\sigma_8$ estimated from mocks")
plt.savefig(f"{output_fig_path}/sigma_8_comparison_config_harm.png", dpi=300)
plt.show()
# %%
sns.histplot(
    simulation_output["S8_config_mean"] - simulation_output["S8_harm_mean"],
    kde=False,
    bins=20,
    stat="density",
    label="Difference (Config - Harm)",
)
plt.xlabel(r"$\Delta S_8$ estimated from mocks")
plt.legend(fontsize=12)
plt.title("Using the weighted average")
plt.savefig(f"{output_fig_path}/S8_difference_config_harm.png", dpi=300)
# Save PDF
plt.savefig(f"{output_fig_path}/S8_difference_config_harm.pdf")
plt.show()

# %%
# Make the plots using the map 1D
sns.histplot(
    simulation_output["S8_config_map_1D"],
    kde=True,
    bins=20,
    label="Configuration space",
)
sns.histplot(
    simulation_output["S8_harm_map_1D"],
    kde=True,
    label="Harmonic space",
    color="blue",
    bins=20,
    alpha=0.3,
)
plt.axvline(s8_fid, color="black", linestyle="--", label="Fiducial S8")
plt.legend(fontsize=12)
plt.title("Using the MAP (KDE 1D)")

plt.xlabel(r"$S_8$ estimated from mocks")
plt.savefig(f"{output_fig_path}/S8_comparison_config_harm_map.png", dpi=300)
plt.show()
# %%
# Same plot for Omega_m and sigma_8
sns.histplot(
    simulation_output["OMEGA_M_config_map_1D"],
    kde=True,
    bins=20,
    label="Configuration space",
)
sns.histplot(
    simulation_output["OMEGA_M_harm_map_1D"],
    kde=True,
    label="Harmonic space",
    color="blue",
    bins=20,
    alpha=0.3,
)
plt.axvline(Omega_m_fid, color="black", linestyle="--", label="Fiducial Omega_m")
plt.legend(fontsize=12)
plt.title("Using the MAP (KDE 1D)")

plt.xlabel(r"$\Omega_m$ estimated from mocks")
plt.savefig(f"{output_fig_path}/Omega_m_comparison_config_harm_map.png", dpi=300)
plt.show()

# %%
sns.histplot(
    simulation_output["SIGMA_8_config_map_1D"],
    kde=True,
    bins=20,
    label="Configuration space",
)
sns.histplot(
    simulation_output["SIGMA_8_harm_map_1D"],
    kde=True,
    label="Harmonic space",
    color="blue",
    bins=20,
    alpha=0.3,
)
plt.axvline(sigma_8_fid, color="black", linestyle="--", label="Fiducial sigma_8")
plt.legend(fontsize=12)
plt.title("Using the MAP (KDE 1D)")

plt.xlabel(r"$\sigma_8$ estimated from mocks")
plt.savefig(f"{output_fig_path}/sigma_8_comparison_config_harm_map.png", dpi=300)
plt.show()
# %%
sns.histplot(
    simulation_output["S8_config_map_1D"] - simulation_output["S8_harm_map_1D"],
    kde=False,
    bins=20,
    stat="density",
    label="Difference (Config - Harm)",
)
plt.xlabel(r"$\Delta S_8$ estimated from mocks")
plt.legend(fontsize=12)
plt.title("Using the MAP (KDE 1D)")
plt.savefig(f"{output_fig_path}/S8_difference_config_harm_map.png", dpi=300)
# Save PDF
plt.savefig(f"{output_fig_path}/S8_difference_config_harm_map.pdf")
plt.show()

# %%
# Use the MAP 2D for Omega_m and S8
sns.histplot(
    simulation_output["OMEGA_M_config_map_2D"],
    kde=True,
    bins=20,
    label="Configuration space",
)
sns.histplot(
    simulation_output["OMEGA_M_harm_map_2D"],
    kde=True,
    label="Harmonic space",
    color="blue",
    bins=20,
    alpha=0.3,
)
plt.axvline(Omega_m_fid, color="black", linestyle="--", label="Fiducial Omega_m")
plt.legend(fontsize=12)
plt.title("Using the MAP (KDE 2D)")

plt.xlabel(r"$\Omega_m$ estimated from mocks")
plt.savefig(f"{output_fig_path}/Omega_m_comparison_config_harm_map_2D.png", dpi=300)
plt.show()

# %%
sns.histplot(
    simulation_output["S8_config_map_2D"],
    kde=True,
    bins=20,
    label="Configuration space",
)
sns.histplot(
    simulation_output["S8_harm_map_2D"],
    kde=True,
    label="Harmonic space",
    color="blue",
    bins=20,
    alpha=0.3,
)
plt.axvline(s8_fid, color="black", linestyle="--", label="Fiducial S8")
plt.legend(fontsize=12)
plt.title("Using the MAP (KDE 2D)")

plt.xlabel(r"$S_8$ estimated from mocks")
plt.savefig(f"{output_fig_path}/S8_comparison_config_harm_map_2D.png", dpi=300)
plt.show()

# %%
# Plot the S8 difference using the MAP 2D
sns.histplot(
    simulation_output["S8_config_map_2D"] - simulation_output["S8_harm_map_2D"],
    kde=False,
    bins=20,
    stat="density",
    label="Difference (Config - Harm)",
)
plt.xlabel(r"$\Delta S_8$ estimated from mocks")
plt.legend(fontsize=12)
plt.title("Using the MAP (KDE 2D)")
plt.savefig(f"{output_fig_path}/S8_difference_config_harm_map_2D.png", dpi=300)
# Save PDF
plt.savefig(f"{output_fig_path}/S8_difference_config_harm_map_2D.pdf")
plt.show()

# %%
sns.histplot(
    simulation_output,
    x="S8_config_mean",
    y="S8_harm_mean",
    bins=25,
    cbar="seismic",
    cbar_kws={"label": "Prob."},
)
plt.plot(
    [
        simulation_output["S8_config_mean"].min(),
        simulation_output["S8_config_mean"].max(),
    ],
    [
        simulation_output["S8_config_mean"].min(),
        simulation_output["S8_config_mean"].max(),
    ],
    color="grey",
    linestyle="--",
    alpha=1.0,
)
plt.axvline(s8_fid, color="black", linestyle="--", label="Fiducial S8")
plt.xlabel(r"$S_8$ estimated from mocks (Configuration space)")
plt.ylabel(r"$S_8$ estimated from mocks (Harmonic space)")
plt.savefig(f"{output_fig_path}/S8_scatter_config_harm.png", dpi=300)
plt.show()


# %%
sns.histplot(
    simulation_output,
    x="OMEGA_M_config_mean",
    y="SIGMA_8_config_mean",
    bins=25,
    cbar="seismic",
    cbar_kws={"label": "Counts"},
)

plt.axvline(Omega_m_fid, color="black", linestyle="--", label="Fiducial Omega_m")
plt.axhline(sigma_8_fid, color="black", linestyle="--", label="Fiducial sigma_8")
plt.xlabel(r"$\Omega_m$ estimated from mocks (Configuration space)")
plt.ylabel(r"$\sigma_8$ estimated from mocks (Configuration space)")
plt.savefig(f"{output_fig_path}/Omega_m_sigma_8_scatter_config.png", dpi=300)
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)

sns.histplot(
    simulation_output,
    x="OMEGA_M_config_mean",
    y="SIGMA_8_config_mean",
    bins=25,
    cmap="mako",
    cbar=True,
    ax=axes[0],
)
axes[0].axvline(Omega_m_fid, color="black", linestyle="--", label="Fiducial Omega_m")
axes[0].axhline(sigma_8_fid, color="black", linestyle="--", label="Fiducial sigma_8")
axes[0].set_xlabel(r"$\Omega_m$ estimated from mocks (Configuration space)")
axes[0].set_ylabel(r"$\sigma_8$ estimated from mocks (Configuration space)")

sns.histplot(
    simulation_output,
    x="OMEGA_M_harm_mean",
    y="SIGMA_8_harm_mean",
    bins=25,
    cmap="mako",
    cbar=True,
    ax=axes[1],
    cbar_kws={"label": "Counts"},
)
axes[1].axvline(Omega_m_fid, color="black", linestyle="--", label="Fiducial Omega_m")
axes[1].axhline(sigma_8_fid, color="black", linestyle="--", label="Fiducial sigma_8")
axes[1].set_xlabel(r"$\Omega_m$ estimated from mocks (Harmonic space)")
axes[1].set_ylabel(r"$\sigma_8$ estimated from mocks (Harmonic space)")


plt.tight_layout()
plt.show()

# %%
g = sns.JointGrid(
    data=simulation_output, x="OMEGA_M_config_mean", y="SIGMA_8_config_mean", space=0
)

g.plot_joint(
    sns.histplot,
    fill=True,
    bins=25,
)
g.plot_marginals(sns.histplot, bins=25)

g.ax_joint.axvline(Omega_m_fid, color="black", linestyle="--", label="Fiducial Omega_m")
g.ax_joint.axhline(sigma_8_fid, color="black", linestyle="--", label="Fiducial sigma_8")

g.ax_joint.set_xlabel(r"$\Omega_m$ estimated from mocks (Configuration space)")
g.ax_joint.set_ylabel(r"$\sigma_8$ estimated from mocks (Configuration space)")

plt.savefig(f"{output_fig_path}/Omega_m_sigma_8_joint_config.png", dpi=300)
plt.show()

# %%
g = sns.JointGrid(
    data=simulation_output, x="OMEGA_M_harm_mean", y="SIGMA_8_harm_mean", space=0
)

g.plot_joint(
    sns.histplot,
    fill=True,
    bins=25,
)
g.plot_marginals(sns.histplot, bins=25)

g.ax_joint.axvline(Omega_m_fid, color="black", linestyle="--", label="Fiducial Omega_m")
g.ax_joint.axhline(sigma_8_fid, color="black", linestyle="--", label="Fiducial sigma_8")
g.ax_joint.set_xlabel(r"$\Omega_m$ estimated from mocks (Harmonic space)")
g.ax_joint.set_ylabel(r"$\sigma_8$ estimated from mocks (Harmonic space)")

plt.savefig(f"{output_fig_path}/Omega_m_sigma_8_joint_harm.png", dpi=300)
plt.show()


# %%
g = sns.JointGrid(
    data=simulation_output, x="S8_config_map_2D", y="S8_harm_map_2D", space=0
)

g.plot_joint(
    sns.histplot,
    fill=True,
    bins=25,
)
g.plot_marginals(sns.histplot, bins=25, kde=True)

g.ax_joint.axvline(s8_fid, color="black", linestyle="--", label="Fiducial S8")
g.ax_joint.axhline(s8_fid, color="black", linestyle="--", label="Fiducial S8")
g.ax_joint.set_xlabel(r"$S_8$ estimated from mocks (Configuration space)")
g.ax_joint.set_ylabel(r"$S_8$ estimated from mocks (Harmonic space)")

g.ax_joint.plot(
    [
        simulation_output["S8_config_mean"].min(),
        simulation_output["S8_config_mean"].max(),
    ],
    [
        simulation_output["S8_config_mean"].min(),
        simulation_output["S8_config_mean"].max(),
    ],
    color="royalblue",
    linestyle="-",
    alpha=0.7,
)

plt.savefig(f"{output_fig_path}/S8_joint_config_harm.png", dpi=300)
# Save PDF
plt.savefig(f"{output_fig_path}/S8_joint_config_harm.pdf")
plt.show()


# %%
# Get p-value
blind = "B"
best_fit_method = "map_2D"
assert best_fit_method in ["weighted_mean", "map_1D", "map_2D"], (
    "Invalid best fit method. Choose from 'weighted_mean', 'map_1D', 'map_2D'."
)

g = plots.get_subplot_plotter(width_inch=7)

root_harmonic = f"SP_v1.4.6.3_leak_corr_{blind}"
root_configuration = f"SP_v1.4.6.3_{blind}_fiducial_config"

# Load configuration space result
path_configuration = (
    f"/n09data/guerrini/output_chains/{root_configuration}/getdist_{root_configuration}"
)
chain_configuration = g.samples_for_root(
    path_configuration,
    cache=False,
    settings={"ignore_rows": 0.0, "smooth_scale_2D": 0.5, "smooth_scale_1D": 0.5},
)

if best_fit_method == "weighted_mean":
    S8_configuration_analysis = cp.compute_average(chain_configuration, "S_8")
elif best_fit_method == "map_1D":
    S8_configuration_analysis = cp.compute_map_1D(chain_configuration, "S_8")
elif best_fit_method == "map_2D":
    S8_configuration_analysis, _ = cp.compute_map_2D(
        chain_configuration, "S_8", "OMEGA_M"
    )

# Load harmonic space result
path_harmonic = f"/n09data/guerrini/output_chains/{root_harmonic}/{root_harmonic}/getdist_{root_harmonic}"
chain_harmonic = g.samples_for_root(
    path_harmonic,
    cache=False,
    settings={"ignore_rows": 0.0, "smooth_scale_2D": 0.5, "smooth_scale_1D": 0.5},
)

if best_fit_method == "weighted_mean":
    S8_harmonic_analysis = cp.compute_average(chain_harmonic, "S_8")
elif best_fit_method == "map_1D":
    S8_harmonic_analysis = cp.compute_map_1D(chain_harmonic, "S_8")
elif best_fit_method == "map_2D":
    S8_harmonic_analysis, _ = cp.compute_map_2D(chain_harmonic, "S_8", "OMEGA_M")

delta_S8_analysis = S8_configuration_analysis - S8_harmonic_analysis
print(f"Delta S8 from analysis: {delta_S8_analysis}")

# Select glass mocks without nan values
if best_fit_method == "weighted_mean":
    selection = ~np.isnan(simulation_output["S8_config_mean"]) & ~np.isnan(
        simulation_output["S8_harm_mean"]
    )
    delta_S8 = (
        simulation_output["S8_config_mean"][selection]
        - simulation_output["S8_harm_mean"][selection]
    ).values
elif best_fit_method == "map_1D":
    selection = ~np.isnan(simulation_output["S8_config_map_1D"]) & ~np.isnan(
        simulation_output["S8_harm_map_1D"]
    )
    delta_S8 = (
        simulation_output["S8_config_map_1D"][selection]
        - simulation_output["S8_harm_map_1D"][selection]
    ).values
elif best_fit_method == "map_2D":
    selection = ~np.isnan(simulation_output["S8_config_map_2D"]) & ~np.isnan(
        simulation_output["S8_harm_map_2D"]
    )
    delta_S8 = (
        simulation_output["S8_config_map_2D"][selection]
        - simulation_output["S8_harm_map_2D"][selection]
    ).values

counts, bin_edges = np.histogram(delta_S8, bins=20, density=True)

sns.histplot(
    delta_S8,
    kde=False,
    bins=bin_edges,
    stat="density",
    label=r"$\Delta S_8$ in \texttt{GLASS} mocks",
    color="blue",
    alpha=0.3,
)

# Compute the p-value

# 1. Get in which bin the delta_S8_analysis falls
bin_index = np.digitize(delta_S8_analysis, bin_edges)

# 2. Compute the p-value as the integral of the histogram from that bin to the tails
if delta_S8_analysis < 0:
    p_value = np.sum(counts[:bin_index]) * (bin_edges[1] - bin_edges[0])
else:
    p_value = np.sum(counts[bin_index:]) * (bin_edges[1] - bin_edges[0])

print(f"P-value for delta S8 = {delta_S8_analysis}: {p_value}")

plt.axvline(
    delta_S8_analysis,
    color="red",
    linestyle="--",
    label=r"$\Delta S_8$ in the analysis",
)

mantissa, exponent = f"{p_value:.1e}".split("e")
exponent = int(exponent)

pte_string = rf"${{\rm PTE}} = {mantissa} \times 10^{{{exponent}}}$"
if best_fit_method == "weighted_mean":
    x_text = -0.045
    y_text = 15
elif best_fit_method == "map_1D":
    x_text = -0.045
    y_text = 17
elif best_fit_method == "map_2D":
    x_text = 0.05
    y_text = 10

plt.text(
    x_text,
    y_text,
    pte_string,
    color="black",
    fontsize=12,
)

n_sigma = norm.isf(p_value)
sign = "-" if delta_S8_analysis < 0 else "+"
print(f"Number of sigma corresponding to the p-value: {n_sigma}")
if best_fit_method == "weighted_mean":
    x_text = -0.045
    y_text = 13
elif best_fit_method == "map_1D":
    x_text = -0.045
    y_text = 15
elif best_fit_method == "map_2D":
    x_text = 0.05
    y_text = 9
plt.text(
    x_text,
    y_text,
    rf"$N_\sigma = {sign}{n_sigma:.2f}\sigma$",
    color="black",
    fontsize=12,
)
plt.xlabel(r"$\Delta S_8 =S_{8, {\rm config}} - S_{8, {\rm harm}}$")
plt.legend(fontsize=11, framealpha=1.0)
plt.savefig(
    f"{output_fig_path}/S8_difference_config_harm_{best_fit_method}.png", dpi=300
)
# Save PDF
plt.savefig(f"{output_fig_path}/S8_difference_config_harm_{best_fit_method}.pdf")
plt.show()
# %%
