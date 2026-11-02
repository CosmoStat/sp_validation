# %%
from IPython import get_ipython

ipython = get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from getdist import plots
from tqdm import tqdm

plt.style.use("./matplotlib_config/paper.mplstyle")

sns.set_palette("husl")

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# %%
root_glass_chains = "/n09data/guerrini/glass_mock_chains"

# %%
num_sim = 1
root = f"glass_mock_v1_{str(num_sim).zfill(5)}"
path_sim = f"{root_glass_chains}/{root}/{root}/samples_{root}.txt"
with open(path_sim) as file:
    params = file.readline()[1:].split("\t")
    file.close()

print(params)

# %%
samples = np.loadtxt(path_sim)
print(samples)

# %%
# Iterate on all the simulations
n_mocks = 350
chi2_map_configuration = []
chi2_map_harmonic = []
for num_sim in tqdm(range(1, n_mocks + 1)):
    root = f"glass_mock_v1_{str(num_sim).zfill(5)}"
    path_sim = f"{root_glass_chains}/{root}/{root}/samples_{root}.txt"
    samples = np.loadtxt(path_sim)
    chi2_map_configuration.append(-2 * samples[-1, -3])

    path_sim = f"{root_glass_chains}/{root}/{root}/samples_{root}_cell.txt"
    samples = np.loadtxt(path_sim)
    chi2_map_harmonic.append(-2 * samples[-1, -3])

# %%
# Get p-value in harmonic space
g = plots.get_subplot_plotter(width_inch=7)

root_harmonic = "SP_v1.4.6_leak_corr_A_lmin=300_lmax=1600_cell"

# Load harmonic space result
path_harmonic = (
    f"/n09data/guerrini/output_chains/{root_harmonic}/samples_{root_harmonic}.txt"
)
samples_harmonic_fid = np.loadtxt(path_harmonic)
chi2_map_harmonic_fid = -2 * samples_harmonic_fid[-1, -3]

print(f"chi2_map_harmonic_fid: {chi2_map_harmonic_fid}")


plt.figure()

counts, bin_edges = np.histogram(chi2_map_harmonic, bins=20, density=True)

sns.histplot(
    chi2_map_harmonic,
    kde=False,
    bins=bin_edges,
    stat="density",
    label=r"$\chi^2({\rm MAP})$ GLASS mock",
    color="blue",
    alpha=0.3,
)
# plt.plot(x, chi2_stats, c='k')

plt.xlabel(r"$\chi^2({\rm MAP})$")
plt.ylabel("Prob.")

plt.savefig("./plots/chi2_map_harmonic_space.png", dpi=300)
plt.show()

# %%
root_configuration = "SP_v1.4.6_leak_corr_A_10_80"

# Load configuration space result
path_configuration = f"/n09data/guerrini/output_chains/{root_configuration}/samples_{root_configuration}.txt"
samples_configuration_fid = np.loadtxt(path_configuration)
chi2_map_configuration_fid = -2 * samples_configuration_fid[-1, -3]

print(f"chi2_map_configuration_fid: {chi2_map_configuration_fid}")

plt.figure()

sns.histplot(chi2_map_configuration, bins=20, kde=False, stat="density")

plt.show()
# %%
