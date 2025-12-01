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
import pandas as pd


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

# Load the output of the simulations
simulation_output = pd.read_csv(
    "/n09data/guerrini/glass_mock_chains/summary_parameter_constraints_merged.txt", delimiter=";"
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
    simulation_output["S8_config_mean"],
    kde=True,
    bins=20,
    label="Configuration space"
)
sns.histplot(
    simulation_output["S8_harm_mean"],
    kde=True,
    label="Harmonic space",
    color='blue',
    bins=20,
    alpha = 0.3
)
plt.axvline(s8_fid, color="black", linestyle="--", label="Fiducial S8")
plt.legend(fontsize=12)

plt.xlabel(r"$S_8$ estimated from mocks")
plt.savefig(f"{output_fig_path}/S8_comparison_config_harm.png", dpi=300)
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
plt.savefig(f"{output_fig_path}/Omega_m_comparison_config_harm.png", dpi=300)
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
plt.savefig(f"{output_fig_path}/sigma_8_comparison_config_harm.png", dpi=300)
plt.show()
# %%
sns.histplot(
    simulation_output["S8_config_mean"] - simulation_output["S8_harm_mean"],
    kde=True,
    bins=20,
    label="Difference (Config - Harm)"
)
plt.xlabel(r"$\Delta S_8$ estimated from mocks")
plt.legend(fontsize=12)
plt.savefig(f"{output_fig_path}/S8_difference_config_harm.png", dpi=300)
plt.show()

# %%
sns.histplot(
    simulation_output,
    x="S8_config_mean",
    y="S8_harm_mean",
    bins=25,
    cbar='seismic',
    cbar_kws={'label': 'Counts'}
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
fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)

sns.histplot(
    simulation_output,
    x="OMEGA_M_config_mean",
    y="SIGMA_8_config_mean",
    bins=25,
    cmap='mako',
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
    cmap='mako',
    cbar=True,
    ax=axes[1],
    cbar_kws={'label': 'Counts'}
)
axes[1].axvline(Omega_m_fid, color="black", linestyle="--", label="Fiducial Omega_m")
axes[1].axhline(sigma_8_fid, color="black", linestyle="--", label="Fiducial sigma_8")
axes[1].set_xlabel(r"$\Omega_m$ estimated from mocks (Harmonic space)")
axes[1].set_ylabel(r"$\sigma_8$ estimated from mocks (Harmonic space)")


plt.tight_layout()
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
g = sns.JointGrid(data=simulation_output, x="S8_config_mean", y="S8_harm_mean", space=0)

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
    color='red',
    linestyle='--',
    alpha=0.7
)

plt.savefig(f"{output_fig_path}/S8_joint_config_harm.png", dpi=300)
plt.show()

# %%

# %%
