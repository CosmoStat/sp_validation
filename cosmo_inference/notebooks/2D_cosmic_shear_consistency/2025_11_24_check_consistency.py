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
