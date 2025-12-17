# %%
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

plt.style.use(
    './matplotlib_config/paper.mplstyle'
)

sns.set_palette("husl")

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# %%
root_glass_chains = "/n09data/guerrini/glass_mock_chains"

# %%
num_sim = 1
root = f"glass_mock_v0_{str(num_sim).zfill(5)}"
path_sim = f"{root_glass_chains}/{root}/{root}/samples_{root}.txt"
with open(path_sim) as file:
    params = file.readline()[1:].split('\t')
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
for num_sim in tqdm(range(1, n_mocks+1)):
    root = f"glass_mock_v0_{str(num_sim).zfill(5)}"
    path_sim = f"{root_glass_chains}/{root}/{root}/samples_{root}.txt"
    samples = np.loadtxt(path_sim)
    chi2_map_configuration.append(samples[-1,-4])

    path_sim = f"{root_glass_chains}/{root}/{root}/samples_{root}_cell.txt"
    samples = np.loadtxt(path_sim) 
    chi2_map_harmonic.append(samples[-1, -4])

# %%
x = np.linspace(0, 25,  num=1000)
k = 16
chi2_stats = stats.chi2.pdf(x, df=k)

plt.figure()

n, bins, _ = plt.hist(chi2_map_harmonic, bins=20, density=True)
plt.plot(x, chi2_stats, c='k')

plt.xlabel(r'$\chi^2({\rm MAP})$')
plt.ylabel("Prob.")

plt.savefig("./plots/chi2_map_harmonic_space.png", dpi=300)
plt.show()

# %%
plt.figure()

plt.hist(chi2_map_configuration, bins=20, density=True)

plt.show()
# %%
