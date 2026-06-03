# %%
# Trick to plot with tex
import os
os.environ["LD_LIBRARY_PATH"] = ""
os.environ["CONDA_PREFIX"] = "/home/guerrini/.conda/envs/sp_validation_3.11"

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
import pandas as pd
from scipy.stats import norm

from getdist import plots, MCSamples


if os.path.exists("/home/guerrini/matplotlib_config/paper.mplstyle"):
    plt.style.use(
        "/home/guerrini/matplotlib_config/paper.mplstyle"
    )

# Set default palette - will be updated per plot as needed
sns.set_palette("husl")

if ipython is not None:
    ipython.run_line_magic('matplotlib', 'inline')

# %%
output_fig_path = "./plots"
if not os.path.exists(output_fig_path):
    os.makedirs(output_fig_path)

chain_version = "v2"
assert chain_version in [f"v{i}" for i in range(5)], "The chain run version is not correct."

# Load the output of the simulations
simulation_output = pd.read_csv(
    f"/n09data/guerrini/glass_mock_chains/summary_parameter_constraints_harmonic_space_{chain_version}.txt", delimiter=";"
)

# %%
simulation_output.info()

# %%
simulation_output.head()

# %%
# Define the true value of the parameters
from astropy.cosmology import Planck18 as planck

Omega_m_fid = planck.Om0
sigma_8_fid = 0.8054
s8_fid = sigma_8_fid * (Omega_m_fid / 0.3)**0.5
h = planck.h
Omega_b_fig = planck.Ob0
n_s_fid = 0.965
print(f"Fiducial values: Omega_m = {Omega_m_fid}, sigma_8 = {sigma_8_fid}, S_8 = {s8_fid}")

# %%
sns.histplot(
    simulation_output["S8_mean"],
    kde=True,
    label="Harmonic space",
    color='blue',
    bins=20,
    alpha = 0.3
)
plt.axvline(s8_fid, color="black", linestyle="--", label="Fiducial S8")
plt.legend(fontsize=12)

plt.xlabel(r"$S_8$ estimated from mocks")
plt.title("Sample average")
plt.savefig(f"{output_fig_path}/S8_comparison_config_harm.png", dpi=300)
plt.show()

# %%
sns.histplot(
    simulation_output["S8_MAP"],
    kde=True,
    bins=20,
    label="Configuration space"
)
plt.axvline(s8_fid, color="black", linestyle="--", label="Fiducial S8")
plt.legend(fontsize=12)

plt.xlabel(r"$S_8$ estimated from mocks")
plt.title("MAP")
plt.savefig(f"{output_fig_path}/S8_comparison_config_harm_MAP.png", dpi=300)
plt.show()
# %%
# Same plot for Omega_m and sigma_8
sns.histplot(
    simulation_output["OMEGA_M_config_mean"],
    kde=True,
    bins=20,
    label="Configuration space"
)
sns.histplot(
    simulation_output["OMEGA_M_harm_mean"],
    kde=True,
    label="Harmonic space",
    color='blue',
    bins=20,
    alpha = 0.3
)
plt.axvline(Omega_m_fid, color="black", linestyle="--", label="Fiducial Omega_m")
plt.legend(fontsize=12) 

plt.xlabel(r"$\Omega_m$ estimated from mocks")
plt.title("Sample average")
plt.savefig(f"{output_fig_path}/Omega_m_comparison_config_harm.png", dpi=300)
plt.show()

# %%
# Same plot for Omega_m and sigma_8
sns.histplot(
    simulation_output["OMEGA_M_config_MAP"],
    kde=True,
    bins=20,
    label="Configuration space"
)
sns.histplot(
    simulation_output["OMEGA_M_harm_MAP"],
    kde=True,
    label="Harmonic space",
    color='blue',
    bins=20,
    alpha = 0.3
)
plt.axvline(Omega_m_fid, color="black", linestyle="--", label="Fiducial Omega_m")
plt.legend(fontsize=12) 

plt.xlabel(r"$\Omega_m$ estimated from mocks")
plt.title("MAP")
plt.savefig(f"{output_fig_path}/Omega_m_comparison_config_harm_MAP.png", dpi=300)
plt.show()

# %%
sns.histplot(
    simulation_output["SIGMA_8_config_mean"],
    kde=True,
    bins=20,
    label="Configuration space"
)
sns.histplot(
    simulation_output["SIGMA_8_harm_mean"],
    kde=True,
    label="Harmonic space",
    color='blue',
    bins=20,
    alpha = 0.3
)
plt.axvline(sigma_8_fid, color="black", linestyle="--", label="Fiducial sigma_8")
plt.legend(fontsize=12) 

plt.xlabel(r"$\sigma_8$ estimated from mocks")
plt.title("Sample average")
plt.savefig(f"{output_fig_path}/sigma_8_comparison_config_harm.png", dpi=300)
plt.show()

# %%
# %%
sns.histplot(
    simulation_output["SIGMA_8_config_MAP"],
    kde=True,
    bins=20,
    label="Configuration space"
)
sns.histplot(
    simulation_output["SIGMA_8_harm_MAP"],
    kde=True,
    label="Harmonic space",
    color='blue',
    bins=20,
    alpha = 0.3
)
plt.axvline(sigma_8_fid, color="black", linestyle="--", label="Fiducial sigma_8")
plt.legend(fontsize=12) 

plt.xlabel(r"$\sigma_8$ estimated from mocks")
plt.title("MAP")
plt.savefig(f"{output_fig_path}/sigma_8_comparison_config_harm_MAP.png", dpi=300)
plt.show()

# %%
sns.histplot(
    simulation_output["a_config_MAP"],
    kde=True,
    bins=20,
    label="Configuration space"
)
sns.histplot(
    simulation_output["a_harm_MAP"],
    kde=True,
    label="Harmonic space",
    color='blue',
    bins=20,
    alpha = 0.3
)
plt.axvline(0., color="black", linestyle="--", label=r"Fiducial $A_{\rm IA}$")
plt.legend(fontsize=12)

plt.xlabel(r"$A_{\rm IA}$ estimated from mocks")
plt.title("MAP")
plt.savefig(f"{output_fig_path}/a_ia_comparison_config_harm_MAP.png", dpi=300)
plt.show()

# %%
sns.histplot(
    simulation_output["a_config_mean"],
    kde=True,
    bins=20,
    label="Configuration space"
)
sns.histplot(
    simulation_output["a_harm_mean"],
    kde=True,
    label="Harmonic space",
    color='blue',
    bins=20,
    alpha = 0.3
)
plt.axvline(0., color="black", linestyle="--", label=r"Fiducial $A_{\rm IA}$")
plt.legend(fontsize=12)

plt.xlabel(r"$A_{\rm IA}$ estimated from mocks")
plt.title("Sample average")
plt.savefig(f"{output_fig_path}/a_ia_comparison_config_harm_mean.png", dpi=300)
plt.show()

# %%
sns.histplot(
    simulation_output["S8_config_mean"] - simulation_output["S8_harm_mean"],
    kde=False,
    bins=20,
    stat='density',
    label="Difference (Config - Harm)"
)
plt.xlabel(r"$\Delta S_8$ estimated from mocks")
plt.legend(fontsize=12)
plt.savefig(f"{output_fig_path}/S8_difference_config_harm.png", dpi=300)
#Save PDF
plt.savefig(f"{output_fig_path}/S8_difference_config_harm.pdf")
plt.show()

# %%
sns.histplot(
    simulation_output["S8_config_MAP"] - simulation_output["S8_harm_MAP"],
    kde=False,
    bins=20,
    stat='density',
    label="Difference (Config - Harm)"
)
plt.xlabel(r"$\Delta S_8$ estimated from mocks")
plt.legend(fontsize=12)
plt.savefig(f"{output_fig_path}/S8_difference_config_harm_map.png", dpi=300)
#Save PDF
plt.savefig(f"{output_fig_path}/S8_difference_config_harm_map.pdf")
plt.show()

# %%
sns.histplot(
    simulation_output,
    x="S8_config_mean",
    y="S8_harm_mean",
    bins=25,
    cbar='seismic',
    cbar_kws={'label': 'Prob.'},
)
plt.plot(
    [simulation_output["S8_config_mean"].min(), simulation_output["S8_config_mean"].max()],
    [simulation_output["S8_config_mean"].min(), simulation_output["S8_config_mean"].max()],
    color='grey',
    linestyle='--',
    alpha=1.0
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
    cbar='seismic',
    cbar_kws={'label': 'Counts'}
)

plt.axvline(Omega_m_fid, color="black", linestyle="--", label="Fiducial Omega_m")
plt.axhline(sigma_8_fid, color="black", linestyle="--", label="Fiducial sigma_8")
plt.xlabel(r"$\Omega_m$ estimated from mocks (Configuration space)")
plt.ylabel(r"$\sigma_8$ estimated from mocks (Configuration space)")
plt.savefig(f"{output_fig_path}/Omega_m_sigma_8_scatter_config.png", dpi=300)
plt.show()

# %%
g = sns.JointGrid(data=simulation_output, x="OMEGA_M_config_mean", y="SIGMA_8_config_mean", space=0)

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
g = sns.JointGrid(data=simulation_output, x="OMEGA_M_harm_mean", y="SIGMA_8_harm_mean", space=0)

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
g = sns.JointGrid(data=simulation_output, x="OMEGA_M_harm_MAP", y="SIGMA_8_harm_MAP", space=0)

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

plt.savefig(f"{output_fig_path}/Omega_m_sigma_8_joint_harm_MAP.png", dpi=300)
plt.show()

# %%
g = sns.JointGrid(data=simulation_output, x="S8_harm_mean", y="a_harm_mean", space=0)

g.plot_joint(
    sns.histplot,
    fill=True,
    bins=25,
)
g.plot_marginals(sns.histplot, bins=25)

g.ax_joint.axvline(s8_fid, color="black", linestyle="--", label=r"Fiducial $S_8$")
g.ax_joint.axhline(0, color="black", linestyle="--", label=r"Fiducial $A_{\rm IA}$")    
g.ax_joint.set_xlabel(r"$S_8$ estimated from mocks (Harmonic space)")
g.ax_joint.set_ylabel(r"$A_{\rm IA}$ estimated from mocks (Harmonic space)")

plt.savefig(f"{output_fig_path}/S8_AIA_joint_harm.png", dpi=300)
plt.show()

# %%
g = sns.JointGrid(data=simulation_output, x="S8_harm_MAP", y="a_harm_MAP", space=0)

g.plot_joint(
    sns.histplot,
    fill=True,
    bins=25,
)
g.plot_marginals(sns.histplot, bins=25)

g.ax_joint.axvline(s8_fid, color="black", linestyle="--", label=r"Fiducial $S_8$")
g.ax_joint.axhline(0, color="black", linestyle="--", label=r"Fiducial $A_{\rm IA}$")    
g.ax_joint.set_xlabel(r"$S_8$ estimated from mocks (Harmonic space)")
g.ax_joint.set_ylabel(r"$A_{\rm IA}$ estimated from mocks (Harmonic space)")

plt.savefig(f"{output_fig_path}/S8_AIA_joint_harm_MAP.png", dpi=300)
plt.show()


# %%
g = sns.JointGrid(data=simulation_output, x="S8_config_MAP", y="S8_harm_mean", space=0)

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
    [simulation_output["S8_config_mean"].min(), simulation_output["S8_config_mean"].max()],
    [simulation_output["S8_config_mean"].min(), simulation_output["S8_config_mean"].max()],
    color='royalblue',
    linestyle='-',
    alpha=0.7
)

plt.savefig(f"{output_fig_path}/S8_joint_config_harm.png", dpi=300)
#Save PDF
plt.savefig(f"{output_fig_path}/S8_joint_config_harm.pdf")
plt.show()


# %%
# Get p-value
g = plots.get_subplot_plotter(width_inch=7)

root_harmonic = 'SP_v1.4.6_leak_corr_A_lmin=300_lmax=1600_cell'
root_configuration = 'SP_v1.4.6_leak_corr_A_10_80'

# Load configuration space result
path_configuration = f"/n09data/guerrini/output_chains/{root_configuration}/getdist_{root_configuration}"
chain_configuration = g.samples_for_root(
        path_configuration,
        cache=False,
        settings={
            'ignore_rows': 0.,
            'smooth_scale_2D': 0.5,
            'smooth_scale_1D': 0.5
        }
)
margestats = chain_configuration.getMargeStats()
likestats = chain_configuration.getLikeStats()

param_stats = margestats.parWithName('S_8')
S8_configuration_analysis = param_stats.mean

# Load harmonic space result
path_harmonic = f"/n09data/guerrini/output_chains/{root_harmonic}/getdist_{root_harmonic}"
chain_harmonic = g.samples_for_root(
        path_harmonic,
        cache=False,
        settings={
            'ignore_rows': 0.,
            'smooth_scale_2D': 0.5,
            'smooth_scale_1D': 0.5
        }
)
margestats = chain_harmonic.getMargeStats()
likestats = chain_harmonic.getLikeStats()

param_stats = margestats.parWithName('S_8')
S8_harmonic_analysis = param_stats.mean

delta_S8_analysis = S8_configuration_analysis - S8_harmonic_analysis
print(f"Delta S8 from analysis: {delta_S8_analysis}")

# Select glass mocks without nan values
selection = ~np.isnan(simulation_output["S8_config_mean"]) & ~np.isnan(simulation_output["S8_harm_mean"])
delta_S8 = (simulation_output["S8_config_mean"][selection] - simulation_output["S8_harm_mean"][selection]).values
counts, bin_edges = np.histogram(
    delta_S8,
    bins=20,
    density=True
)

sns.histplot(
    delta_S8,
    kde=False,
    bins=bin_edges,
    stat='density',
    label="Difference (Config - Harm)",
    color='blue',
    alpha=0.3
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

plt.axvline(delta_S8_analysis, color="red", linestyle="--", label="Difference in the analysis")

mantissa, exponent = f"{p_value:.1e}".split("e")
exponent = int(exponent)

pte_string = rf"${{\rm PTE}} = {mantissa} \times 10^{{{exponent}}}$"
plt.text(
    -0.065, 15,
    pte_string,
    color='black',
    fontsize=12,
)

n_sigma = norm.isf(p_value)
sign = '-' if delta_S8_analysis < 0 else '+'
print(f"Number of sigma corresponding to the p-value: {n_sigma}")
plt.text(
    -0.065, 13,
    rf"$N_\sigma = {sign}{n_sigma:.2f}\sigma$",
    color='black',
    fontsize=12,
)
plt.xlabel(r"$\Delta S_8$ estimated from mocks")
plt.legend(fontsize=12, framealpha=1.0)
plt.savefig(f"{output_fig_path}/S8_difference_config_harm.png", dpi=300)
#Save PDF
plt.savefig(f"{output_fig_path}/S8_difference_config_harm.pdf")
plt.show()

# %%
