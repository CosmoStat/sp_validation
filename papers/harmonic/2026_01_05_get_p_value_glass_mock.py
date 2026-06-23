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

from sp_validation.rho_tau import SquareRootScale

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
chain_version = "v6"

# Path to the glass mock data vectors
root_glass_dv = f"/home/guerrini/sp_validation/cosmo_inference/data/glass_mocks/{chain_version}/"

# Choose the best-fit method
best_fit_method = "2Dkde"

# Create the list of mocks
max_sim = 350
failed_simulations = [
    82, 83, 281, 282, 283,284, 285, 286, 287
]
roots_glass_mock = [f"glass_mock_{chain_version}_{str(i).zfill(5)}" for i in range(1, max_sim + 1)]

lower_bound_cell_ee = 300.0
upper_bound_cell_ee = 1600.0

# %%
def get_chi2_glass_mock(root, root_glass_chains, root_glass_dv, lower_bound, upper_bound):
    #Read the theory prediction at best fit
    ell = np.loadtxt(f"{root_glass_chains}/{root}/best_fit/{root}_cell_{best_fit_method}/shear_cl/ell.txt")
    shear_cl = np.loadtxt(f"{root_glass_chains}/{root}/best_fit/{root}_cell_{best_fit_method}/shear_cl/bin_1_1.txt")

    #Read the data
    idx = root.replace(f"glass_mock_{chain_version}_", "")
    chain_version_data = "v0" if chain_version in ["v0", "v1"] else chain_version
    data = fits.open(f"{root_glass_dv}/glass_mock_{idx}/cosmosis_glass_mock_{chain_version_data}_{idx}.fits")

    ell_data = data["CELL_EE"].data["ANG"]
    cell_data = data["CELL_EE"].data["VALUE"]

    # Load the covariance
    cov = data["COVMAT"].data
    cov_cell = cov[80:, 80:]

    # Interpolate the model
    interp_cell_ee = interp1d(ell, shear_cl, kind='cubic', fill_value='extrapolate')

    cell_model = interp_cell_ee(ell_data)

    # Apply scale cuts
    mask_cell = (ell_data > lower_bound) & (ell_data < upper_bound)

    cell_model = cell_model[mask_cell]
    cell_data = cell_data[mask_cell]
    cov_cell = cov_cell[mask_cell][:, mask_cell]

    cell_chi2 = np.dot((cell_model - cell_data), np.dot(np.linalg.inv(cov_cell), (cell_model - cell_data)))

    return cell_chi2

def get_chi2_map_glass_mock(root, root_glass_chains):
    path_sim = f"{root_glass_chains}/{root}/{root}/samples_{root}_cell.txt"
    samples = np.loadtxt(path_sim)
    cell_chi2_map = samples[-1, -3]

    return -2*cell_chi2_map

# %%
metrics = {}

for i, root in enumerate(roots_glass_mock):
    if i + 1 in failed_simulations:
        continue
    metrics[root] = {}
    metrics[root]["chi2"] = get_chi2_glass_mock(root, root_glass_chains, root_glass_dv, lower_bound_cell_ee, upper_bound_cell_ee)
    metrics[root]["chi2_map"] = get_chi2_map_glass_mock(root, root_glass_chains)

# %%
chi2_glass_mocks = np.array([metrics[root]["chi2"] for i, root in enumerate(roots_glass_mock) if i + 1 not in failed_simulations])
chi2_glass_mocks_map = np.array([metrics[root]["chi2_map"] for i, root in enumerate(roots_glass_mock) if i + 1 not in failed_simulations])

sns.histplot(
    chi2_glass_mocks,
    bins=25,
    kde=False,
    stat='density',
    label=best_fit_method
)

sns.histplot(
    chi2_glass_mocks_map,
    bins=25,
    kde=False,
    stat='density',
    label='MLE'
)

k = 10
x = np.linspace(0, 25)
chi2 = stats.chi2.pdf(x, k)

plt.plot(x, chi2, c='k', label=rf"$n_{{\rm dof}}={k}$")

plt.xlabel(r"$\chi^2$")
plt.legend()
plt.show()
# %%
def plot_glass_mock_fit_root(root, root_glass_chains, root_glass_dv):
    #Read the theory prediction at best fit
    ell = np.loadtxt(f"{root_glass_chains}/{root}/best_fit/{root}_cell_{best_fit_method}/shear_cl/ell.txt")
    shear_cl = np.loadtxt(f"{root_glass_chains}/{root}/best_fit/{root}_cell_{best_fit_method}/shear_cl/bin_1_1.txt")

    #ell_map = np.loadtxt(f"{root_glass_chains}/{root}/best_fit/{root}_cell_map/{root}_cell_map/shear_cl/ell.txt")
    #shear_cl_map = np.loadtxt(f"{root_glass_chains}/{root}/best_fit/{root}_cell_map/{root}_cell_map/shear_cl/bin_1_1.txt")

    #Read the data
    idx = root.replace(f"glass_mock_{chain_version}_", "")
    chain_version_data = "v0" if chain_version in ["v0", "v1"] else chain_version
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

    """ plt.plot(
        ell_map,
        ell_map*shear_cl_map,
        label='Best-fit model (map)',
        c='orange'
    ) """

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
fig = plot_glass_mock_fit_root(roots_glass_mock[89], root_glass_chains, root_glass_dv)

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
# Paperplot: computation of the p-value for our fiducial
blind = "B"
root_fiducial = f"SP_v1.4.6.3_leak_corr_{blind}"

# Get the measured data vector and covariance for the fiducial
dv_fiducial = fits.open(
    f"/home/guerrini/sp_validation/cosmo_inference/data/SP_v1.4.6.3_leak_corr_{blind}/cosmosis_SP_v1.4.6.3_leak_corr_{blind}.fits"
)

cell_fiducial = dv_fiducial["CELL_EE"].data["VALUE"]
cov_fiducial = dv_fiducial["COVMAT"].data

low_ell_cut = 300
high_ell_cut = 1600

mask = (dv_fiducial["CELL_EE"].data["ANG"] > low_ell_cut) & (dv_fiducial["CELL_EE"].data["ANG"] < high_ell_cut)
cell_fiducial = cell_fiducial[mask]
cov_fiducial = cov_fiducial[mask][:, mask]

# Get the best-fit model for the fiducial
ell_bf = np.loadtxt(
    f"/n09data/guerrini/output_chains/{root_fiducial}/best_fit/shear_cl/ell.txt"
)
cell_bf = np.loadtxt(
    f"/n09data/guerrini/output_chains/{root_fiducial}/best_fit/shear_cl/bin_1_1.txt"
) 

# Interpolate the model
interp_cell_ee = interp1d(ell_bf, cell_bf, kind='cubic', fill_value='extrapolate')

cell_bf = interp_cell_ee(dv_fiducial["CELL_EE"].data["ANG"][mask])

chi2_fiducial = np.dot((cell_bf - cell_fiducial), np.dot(np.linalg.inv(cov_fiducial), (cell_bf - cell_fiducial)))
print(f"Chi2 for the fiducial: {chi2_fiducial}")


# %%
# Make the plot
output_fig_path = f"/home/guerrini/sp_validation/papers/harmonic/plots/"
counts, bin_edges = np.histogram(
    chi2_glass_mocks,
    bins=25,
    density=True
)

sns.histplot(
    chi2_glass_mocks,
    kde=False,
    bins=bin_edges,
    stat='density',
    label=r"$\chi^2$ for \texttt{GLASS} mocks best-fits",
    color='blue',
    alpha=0.3
)

# Compute the p-value

# 1. Get in which bin the chi2 of the fiducial falls
bin_index = np.digitize(chi2_fiducial, bin_edges)

# 2. Compute the p-value as the integral of the tail of the histogram
p_value = np.sum(counts[bin_index:]) * np.diff(bin_edges)[0]

print(f"P-value: {p_value}")

plt.axvline(chi2_fiducial, color='red', ls='--', label=r'$\chi^2$ of the fiducial')

mantissa, exponent = np.frexp(p_value)
pte_string = rf"${{\rm PTE}} = {mantissa:.2f} \times 10^{{{exponent}}}$" if exponent != 0 else rf"${{\rm PTE}} = {p_value:.2f}$"

x_text = 40
y_text = max(counts) * 0.6
plt.text(x_text, y_text, pte_string, fontsize=12)

chi2_string = rf"$\chi^2_{{\rm fid}} = {chi2_fiducial:.1f}$"
x_text = 40
y_text = max(counts) * 0.5
plt.text(
    x_text,
    y_text,
    chi2_string,
    fontsize=12
)

plt.xlabel(r"$\chi^2$")
plt.legend(fontsize=12)
plt.savefig(f"{output_fig_path}/chi2_glass_mocks_p_value.png")
# Save PDF
plt.savefig(f"{output_fig_path}/chi2_glass_mocks_p_value.pdf")
plt.show()

# %%
