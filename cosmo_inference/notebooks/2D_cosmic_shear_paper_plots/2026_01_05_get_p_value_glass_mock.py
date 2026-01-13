# %%
import os

#Trick to use latex in the plots
os.environ["LD_LIBRARY_PATH"] = ""
os.environ["CONDA_PREFIX"] = "/home/guerrini/.conda/envs/phd_manuscript"

from IPython import get_ipython

ipython = get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import numpy as np
from astropy.io import fits
import scipy.stats as stats
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib import scale as mscale
import seaborn as sns
from tqdm import tqdm

from sp_validation.utils_cosmo_val import SquareRootScale

mscale.register_scale(SquareRootScale)

from getdist import plots, MCSamples

plt.style.use(
    './matplotlib_config/paper.mplstyle'
)

sns.set_palette("husl")

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# %%
root_glass_chains = "/n09data/guerrini/glass_mock_chains/"

# Version of the glass mock chain run
chain_version = "v2"
assert chain_version in ["v0", "v1", "v2"]

# Path to the glass mock data vectors
root_glass_dv = f"/home/guerrini/sp_validation/cosmo_inference/data/glass_mocks/{chain_version}/"

# Create the list of mocks
max_sim = 12
roots_glass_mock = [f"glass_mock_{chain_version}_{str(i).zfill(5)}" for i in range(1, max_sim + 1)]

lower_bound_xi_ee = 12.0
upper_bound_xi_ee = 83.0

# %%
def get_chi2_glass_mock(root, root_glass_chains, root_glass_dv, lower_bound_xi_plus, upper_bound_xi_plus, lower_bound_xi_minus, upper_bound_xi_minus):
    #Read the theory prediction at best fit
    theta = np.loadtxt(f"{root_glass_chains}/{root}/best_fit/{root}/shear_xi_plus/theta.txt")
    theta_arcmin = theta * 180 * 60 /np.pi
    shear_xi_plus = np.loadtxt(f"{root_glass_chains}/{root}/best_fit/{root}/shear_xi_plus/bin_1_1.txt")
    shear_xi_minus = np.loadtxt(f"{root_glass_chains}/{root}/best_fit/{root}/shear_xi_minus/bin_1_1.txt")
    xi_sys_plus = np.loadtxt(f"{root_glass_chains}/{root}/best_fit/{root}/xi_sys/shear_xi_plus.txt")
    xi_sys_minus = np.loadtxt(f"{root_glass_chains}/{root}/best_fit/{root}/xi_sys/shear_xi_minus.txt")

    #Read the data
    idx = root.replace(f"glass_mock_{chain_version}_", "")
    chain_version_data = "v0" if chain_version in ["v0", "v1"] else "v2"
    data = fits.open(f"{root_glass_dv}/glass_mock_{idx}/cosmosis_glass_mock_{chain_version_data}_{idx}.fits")

    theta_data = data["XI_PLUS"].data["ANG"]
    xi_plus_data = data["XI_PLUS"].data["VALUE"]
    xi_minus_data = data["XI_MINUS"].data["VALUE"]

    # Load the covariance
    cov = data["COVMAT"].data
    cov_xi = cov[0:2*len(xi_plus_data), 0:2*len(xi_plus_data)]

    # Interpolate the model
    interp_xi_plus = interp1d(theta_arcmin, shear_xi_plus, kind='cubic', fill_value='extrapolate')
    interp_xi_minus = interp1d(theta_arcmin, shear_xi_minus, kind='cubic', fill_value='extrapolate')

    xi_plus_model = interp_xi_plus(theta_data) + xi_sys_plus
    xi_minus_model = interp_xi_minus(theta_data) + xi_sys_minus

    xi_data = np.concatenate((xi_plus_data, xi_minus_data))
    xi_model = np.concatenate((xi_plus_model, xi_minus_model))

    # Apply scale cuts
    mask_xi_plus = (theta_data > lower_bound_xi_plus) & (theta_data < upper_bound_xi_plus)
    mask_xi_minus = (theta_data > lower_bound_xi_minus) & (theta_data < upper_bound_xi_minus)
    mask = np.concatenate((mask_xi_plus, mask_xi_minus))

    xi_data = xi_data[mask]
    xi_model = xi_model[mask]
    cov_xi = cov_xi[mask][:, mask]

    xi_chi2 = np.dot((xi_model - xi_data), np.dot(np.linalg.inv(cov_xi), (xi_model - xi_data)))

    return xi_chi2

def get_chi2_map_glass_mock(root, root_glass_chains):
    path_sim = f"{root_glass_chains}/{root}/{root}/samples_{root}.txt"
    samples = np.loadtxt(path_sim)
    xi_chi2_map = samples[-1, -3]

    return -2*xi_chi2_map

# %%
metrics = {}

for i, root in enumerate(roots_glass_mock):
    metrics[root] = {}
    metrics[root]["chi2"] = get_chi2_glass_mock(root, root_glass_chains, root_glass_dv, lower_bound_xi, upper_bound_xi, lower_bound_xi, upper_bound_xi)
    metrics[root]["chi2_map"] = get_chi2_map_glass_mock(root, root_glass_chains)

# %%
chi2_glass_mocks = np.array([metrics[root]["chi2"] for root in roots_glass_mock])
chi2_glass_mocks_map = np.array([metrics[root]["chi2_map"] for root in roots_glass_mock])

sns.histplot(
    chi2_glass_mocks,
    bins=25,
    kde=False,
    stat='density'
)

sns.histplot(
    chi2_glass_mocks_map,
    bins=25,
    kde=False,
    stat='density'
)

k = 8
x = np.linspace(8, 70)
chi2 = stats.chi2.pdf(x, k)

plt.plot(x, chi2, c='k')

plt.xlabel(r"$\chi^2$")

plt.show()
# %%
def plot_glass_mock_fit_root(root, root_glass_chains, root_glass_dv):
    #Read the theory prediction at best fit
    ell = np.loadtxt(f"{root_glass_chains}/{root}/best_fit/{root}_cell/shear_cl/ell.txt")
    shear_cl = np.loadtxt(f"{root_glass_chains}/{root}/best_fit/{root}_cell/shear_cl/bin_1_1.txt")

    ell_map = np.loadtxt(f"{root_glass_chains}/{root}/best_fit/{root}_cell_map/shear_cl/ell.txt")
    shear_cl_map = np.loadtxt(f"{root_glass_chains}/{root}/best_fit/{root}_cell_map/shear_cl/bin_1_1.txt")

    #Read the data
    idx = root.replace(f"glass_mock_{chain_version}_", "")
    chain_version_data = "v0" if chain_version in ["v0", "v1"] else "v2"
    data = fits.open(f"{root_glass_dv}/glass_mock_{idx}/cosmosis_glass_mock_{chain_version_data}_{idx}.fits")

    ell_data = data["CELL_EE"].data["ANG"]
    cell_data = data["CELL_EE"].data["VALUE"]

    fig = plt.figure()

    plt.errorbar(
        ell_data,
        ell_data*cell_data,
        yerr=ell_data*np.sqrt(data["COVMAT"].data.diagonal()[80:]),
        fmt='o',
        label='Data',
        alpha=0.5
    )

    plt.plot(
        ell,
        ell*shear_cl,
        label='Best-fit model',
        c='r'
    )

    plt.plot(
        ell_map,
        ell_map*shear_cl_map,
        label='Best-fit model (map)',
        c='orange'
    )

    plt.axvline(300)
    plt.axvline(1600)

    plt.xlim(0, 2048)
    plt.ylim(bottom=0.5e-7)
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$C_\ell^{EE}$")
    plt.xscale('squareroot')
    plt.legend()

    return fig

# %%
fig = plot_glass_mock_fit_root(roots_glass_mock[10], root_glass_chains, root_glass_dv)

plt.legend()

plt.show()
# %%
idx_sim = 1
idx_sim = str(idx_sim).zfill(5)

path_first_run = "/n09data/guerrini/glass_mock_v1.4.6/results/"
path_second_run = "/n09data/guerrini/glass_mock_v1.4.6_rerun/results/"
first_run = np.load(
    f"{path_first_run}/cl_glass_mock_{idx_sim}_4096.npy"
)
second_run = np.load(
    f"{path_second_run}/cl_glass_mock_{idx_sim}_4096.npy"
)

chain_version_data = "v0" if chain_version in ["v0", "v1"] else "v2"
data = fits.open(f"{root_glass_dv}/glass_mock_{idx_sim}/cosmosis_glass_mock_{chain_version_data}_{idx_sim}.fits")

plt.figure()

plt.errorbar(
    first_run[0],
    first_run[0] * first_run[1],
    yerr=first_run[0] * np.sqrt(data["COVMAT"].data.diagonal()[80:]),
    fmt='o',
    label='First run',
    alpha=0.5
)

plt.errorbar(
    second_run[0],
    second_run[0] * second_run[1],
    yerr=second_run[0] * np.sqrt(data["COVMAT"].data.diagonal()[80:]),
    fmt='o',
    label='Second run',
    alpha=0.5
)

plt.xscale('squareroot')
plt.legend()

plt.show()

# %%
first_run_cls = []
second_run_cls = []
for i in range(1, 351):
    idx_sim = str(i).zfill(5)

    path_first_run = "/n09data/guerrini/glass_mock_v1.4.6/results/"
    path_second_run = "/n09data/guerrini/glass_mock_v1.4.6_rerun/results/"
    first_run = np.load(
        f"{path_first_run}/cl_glass_mock_{idx_sim}_4096.npy"
    )
    second_run = np.load(
        f"{path_second_run}/cl_glass_mock_{idx_sim}_4096.npy"
    )

    first_run_cls.append(first_run[1])
    second_run_cls.append(second_run[1])
# %%
cov_first_run = np.cov(np.array(first_run_cls).T)
cov_second_run = np.cov(np.array(second_run_cls).T)

# %%
plt.figure()

plt.plot(np.sqrt(cov_first_run.diagonal()))
plt.plot(np.sqrt(cov_second_run.diagonal()))

plt.yscale('log')
plt.show()

# %%
