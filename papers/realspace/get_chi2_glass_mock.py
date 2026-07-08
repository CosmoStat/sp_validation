

import configparser
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

# Make the plot
import seaborn as sns
from astropy.io import fits
from getdist import plots
from scipy.interpolate import interp1d
from scipy.stats import chi2

sys.path.append("/home/guerrini/sp_validation/cosmo_inference/scripts")

import chain_postprocessing


plt.style.use("/home/guerrini/matplotlib_config/paper.mplstyle")

plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18

plt.rcParams["text.usetex"] = True

g = plots.get_subplot_plotter(width_inch=30)
g.settings.axes_fontsize = 30
g.settings.axes_labelsize = 30
g.settings.alpha_filled_add = 0.7
g.settings.legend_fontsize = 40

# #SPECIFY DATA DIRECTORY AND DESIRED CHAINS TO ANALYSE

root_dir = "/n09data/guerrini/glass_mock_chains/"

# Version of the glass mock chain run
chain_version = "v6"

# Path to the glass mock data vectors
root_glass_dv = (
    f"/home/guerrini/sp_validation/cosmo_inference/data/glass_mocks/{chain_version}/"
)

# Choose the best-fit method
best_fit_method = "2Dkde"

# Create the list of mocks
max_sim = 350
failed_simulations = [82, 83, 281, 282, 283, 284, 285, 286, 287]
roots = [f"glass_mock_{chain_version}_{str(i).zfill(5)}" for i in range(1, max_sim + 1)]
roots = [root for root in roots if int(root.split("_")[-1]) not in failed_simulations]

catalog_versions = [
    "SP_v1.4.6.3_config/SP_v1.4.6.3_A",
]

output_folder_chains = "/n23data1/n06data/lgoh/scratch/temp/"
path_ini_files = "/home/guerrini/sp_validation/cosmo_inference/cosmosis_config/"
output_fig_path = (
    "/n23data1/n06data/lgoh/scratch/UNIONS/cosmo_inference/notebooks/Plots/"
)

ini_root = "blind_A/fiducial"

lower_bound_xi = 12
upper_bound_xi = 83

# ## Retrieve the chains


# READ CHAIN

chains = []
best_fit = {}

for i, root in enumerate(roots):
    burnin = 0

    if os.path.isfile(f"{root_dir}/{root}/{root}/getdist_{root}.txt") == True:
        chain = g.samples_for_root(
            f"{root_dir}/{root}/{root}/getdist_{root}",
            cache=False,
            settings={
                "ignore_rows": burnin,
                "smooth_scale_2D": 0.5,
                "smooth_scale_1D": 0.5,
            },
        )
        p = chain.getParams()

        best_fit[root] = chain_postprocessing.extract_best_fit_params(
            chain, best_fit_method="2Dkde"
        )


param_list = [
    "OMEGA_M",
    "ombh2",
    "h0",
    "n_s",
    "SIGMA_8",
    "s_8_input",
    "logt_agn",
    "a",
    "m1",
    "bias_1",
    "alpha",
    "beta",
    "omch2",
    "m",
    "a_planck",
]
label_list = [
    r"\Omega_m",
    r"\omega_b",
    "h_0",
    "n_s",
    r"\sigma_8",
    "S_8",
    "log T_{AGN}",
    "A_{IA}",
    "m_1",
    r"\Delta z_1",
    "\\alpha_{PSF}",
    "\\beta_{PSF}",
    r"\omega_c",
    "M",
    "A_{\rm Planck}",
]

 
# ## Run `Cosmosis` in test mode to get the data vectors


if not os.path.exists(path_ini_files + "/values_empty.ini"):
    content = """[cosmological_parameters]

tau          =  0.0544
w            = -1.0
mnu = 0.06
omega_k      =  0.0
wa           =  0.0

[halo_model_parameters]

[intrinsic_alignment_parameters]

[shear_calibration_parameters]

[nofz_shifts]

[psf_leakage_parameters]
"""

    with open(path_ini_files + "/values_empty.ini", "w") as f:
        f.write(content)
        f.close()

    print("File created successfully")


section_map = {
    "omch2": "cosmological_parameters",
    "ombh2": "cosmological_parameters",
    "h0": "cosmological_parameters",
    "n_s": "cosmological_parameters",
    "s_8_input": "cosmological_parameters",
    "logt_agn": "halo_model_parameters",
    "a": "intrinsic_alignment_parameters",
    "m1": "shear_calibration_parameters",
    "bias_1": "nofz_shifts",
    "alpha": "psf_leakage_parameters",
    "beta": "psf_leakage_parameters",
}


env = os.environ.copy()
env["LD_LIBRARY_PATH"] = (
    "/home/guerrini/.conda/envs/sp_validation/lib/python3.9/site-packages/cosmosis/datablock:"
    + env.get("LD_LIBRARY_PATH", "")
)
for i, root in enumerate(roots):
    print(root)
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve case sensitivity of option names

    for param, section in section_map.items():
        # Check if this parameter exists for the current root
        if param in best_fit[root]:
            value = best_fit[root][param]

            if section not in config:
                config.add_section(section)

            config[section][param] = str(value)

    with open(path_ini_files + "/values_empty.ini", "w") as configfile:
        config.write(configfile)

    # Modify the ini file to run in test mode at the best fit
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve case sensitivity of option names

    ini_file = (
        path_ini_files + f"config_space_v1.4.6.3_fiducial/pipeline/{ini_root}.ini"
    )
    config.read(ini_file)

    sampler = config["runtime"]["sampler"]
    config["runtime"]["sampler"] = "test"
    values = config["pipeline"]["values"]
    config["pipeline"]["values"] = path_ini_files + "/values_empty.ini"
    config["DEFAULT"]["FITS_FILE"] = (
        f"{root_glass_dv}/glass_mock_{root[-5:]}/cosmosis_glass_mock_v6_{root[-5:]}.fits"
    )
    config["test"]["save_dir"] = output_folder_chains + f"{root}/best_fit_config"

    with open(ini_file, "w") as configfile:
        config.write(configfile)

    # Run cosmosis
    result = subprocess.run(
        ["cosmosis", ini_file], env=env, capture_output=True, text=True
    )
    # print(f"STDOUT:\n{result.stdout}")
    # print(f"STDERR:\n{result.stderr}")

    # Modify the ini file to the previous one
    config["pipeline"]["values"] = values
    config["runtime"]["sampler"] = sampler

    with open(ini_file, "w") as configfile:
        config.write(configfile)


xi_plus_chi2s = np.array([])
xi_minus_chi2s = np.array([])
xi_chi2s = np.array([])
tau_chi2s = np.array([])
chi2_tots = np.array([])


for idx, root in enumerate(roots):
    print(root)

    data = fits.open(
        f"{root_glass_dv}/glass_mock_{root[-5:]}/cosmosis_glass_mock_v6_{root[-5:]}.fits"
    )

    tau_0_data = data["TAU_0_PLUS"].data["VALUE"]
    tau_2_data = data["TAU_2_PLUS"].data["VALUE"]

    theta_data = data["XI_PLUS"].data["ANG"]
    xi_plus_data = data["XI_PLUS"].data["VALUE"]
    xi_minus_data = data["XI_MINUS"].data["VALUE"]
    xi_data = np.concatenate((xi_plus_data, xi_minus_data))

    tau_data = np.concatenate((tau_0_data, tau_2_data))

    # Apply scale cuts
    mask_xi_plus = (theta_data > lower_bound_xi) & (theta_data < upper_bound_xi)
    mask_xi_minus = (theta_data > lower_bound_xi) & (theta_data < upper_bound_xi)
    mask = np.concatenate((mask_xi_plus, mask_xi_minus))
    # Load the covariance
    cov = data["COVMAT"].data
    cov_xi = cov[0 : 2 * len(xi_plus_data), 0 : 2 * len(xi_plus_data)]
    cov_tau = cov[
        2 * len(xi_plus_data) : 4 * len(xi_plus_data),
        2 * len(xi_plus_data) : 4 * len(xi_plus_data),
    ]
    xi_data = xi_data[mask]
    cov_xi = cov_xi[mask][:, mask]

    cov_xi_plus = cov[0 : len(xi_plus_data), 0 : len(xi_plus_data)]
    cov_xi_plus = cov_xi_plus[mask_xi_plus][:, mask_xi_plus]
    cov_xi_minus = cov[
        len(xi_plus_data) : 2 * len(xi_minus_data),
        len(xi_plus_data) : 2 * len(xi_minus_data),
    ]
    cov_xi_minus = cov_xi_minus[mask_xi_minus][:, mask_xi_minus]

    # Read the results
    theta = np.loadtxt(
        output_folder_chains + f"{root}/best_fit_config/shear_xi_plus/theta.txt"
    )
    theta_arcmin = theta * 180 * 60 / np.pi
    shear_xi_plus = np.loadtxt(
        output_folder_chains + f"{root}/best_fit_config/shear_xi_plus/bin_1_1.txt"
    )
    shear_xi_minus = np.loadtxt(
        output_folder_chains + f"{root}/best_fit_config/shear_xi_minus/bin_1_1.txt"
    )

    xi_sys_plus = np.loadtxt(
        output_folder_chains + f"{root}/best_fit_config/xi_sys/shear_xi_plus.txt"
    )
    xi_sys_minus = np.loadtxt(
        output_folder_chains + f"{root}/best_fit_config/xi_sys/shear_xi_minus.txt"
    )

    theta_tau = np.loadtxt(
        output_folder_chains + f"{root}/best_fit_config/tau_0_plus/theta.txt"
    )
    theta_tau_arcmin = theta_tau * 180 * 60 / np.pi
    tau_0_model = np.loadtxt(
        output_folder_chains + f"{root}/best_fit_config/tau_0_plus/bin_1_1.txt"
    )
    tau_2_model = np.loadtxt(
        output_folder_chains + f"{root}/best_fit_config/tau_2_plus/bin_1_1.txt"
    )

    # interpolate the model
    interp_xi_plus = interp1d(
        theta_arcmin, shear_xi_plus, kind="cubic", fill_value="extrapolate"
    )
    interp_xi_minus = interp1d(
        theta_arcmin, shear_xi_minus, kind="cubic", fill_value="extrapolate"
    )

    xi_plus_model = interp_xi_plus(theta_data)
    xi_plus_model += xi_sys_plus
    xi_minus_model = interp_xi_minus(theta_data)
    xi_minus_model += xi_sys_minus

    xi_model = np.concatenate((xi_plus_model, xi_minus_model))
    tau_model = np.concatenate((tau_0_model, tau_2_model))
    xi_model = xi_model[mask]

    xi_plus_chi2 = np.dot(
        (xi_plus_model[mask_xi_plus] - xi_plus_data[mask_xi_plus]),
        np.dot(
            np.linalg.inv(cov_xi_plus),
            (xi_plus_model[mask_xi_plus] - xi_plus_data[mask_xi_plus]),
        ),
    )
    xi_minus_chi2 = np.dot(
        (xi_minus_model[mask_xi_minus] - xi_minus_data[mask_xi_minus]),
        np.dot(
            np.linalg.inv(cov_xi_minus),
            (xi_minus_model[mask_xi_minus] - xi_minus_data[mask_xi_minus]),
        ),
    )
    xi_chi2 = np.dot(
        (xi_model - xi_data), np.dot(np.linalg.inv(cov_xi), (xi_model - xi_data))
    )
    tau_chi2 = np.dot(
        (tau_model - tau_data), np.dot(np.linalg.inv(cov_tau), (tau_model - tau_data))
    )
    chi2_tot = xi_plus_chi2 + xi_minus_chi2 + tau_chi2

    xi_plus_chi2s = np.append(xi_plus_chi2s, xi_plus_chi2)
    xi_minus_chi2s = np.append(xi_minus_chi2s, xi_minus_chi2)
    xi_chi2s = np.append(xi_chi2s, xi_chi2)
    tau_chi2s = np.append(tau_chi2s, tau_chi2)
    chi2_tots = np.append(chi2_tots, chi2_tot)


fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(7, 10))
chi2_fiducial = -2 * -37.560916821678894
dof, loc, scale = chi2.fit(chi2_tots, floc=0)

print(f"Best-fit dof: {dof:.3e}")
counts, bin_edges = np.histogram(chi2_tots, bins=25, density=True)

sns.histplot(
    chi2_tots,
    ax=ax1,
    kde=False,
    bins=bin_edges,
    stat="density",
    label=r"$\chi^2$ for \texttt{GLASS} mocks best-fits",
    color="green",
    alpha=0.3,
)

# Compute the p-value

# 1. Get in which bin the chi2 of the fiducial falls
bin_index = np.digitize(chi2_fiducial, bin_edges)

# 2. Compute the p-value as the integral of the tail of the histogram
p_value = np.sum(counts[bin_index:]) * np.diff(bin_edges)[0]

print(f"P-value: {p_value}")

ax1.axvline(chi2_fiducial, color="red", label=r"$\chi^2$ of the fiducial", lw=2)

mantissa, exponent = np.frexp(p_value)
pte_string = rf"${{\rm PTE}} = {p_value:.4f}$"
print(f"mantissa: {mantissa}, exponent: {exponent}")
x_text = 78
y_text = max(counts) * 0.95
ax1.text(
    x_text,
    y_text,
    pte_string,
    fontsize=15,
    bbox=dict(facecolor="wheat", alpha=0.8, edgecolor="black"),
)

chi2_string = rf"${{\rm Eff. dof}}= {dof:.1f}$"
y_text = max(counts) * 0.85
ax1.text(
    x_text,
    y_text,
    chi2_string,
    fontsize=15,
    bbox=dict(facecolor="wheat", alpha=0.8, edgecolor="black"),
)

ax1.set_xlabel(r"$\chi^2_{\rm tot}$")
ax1.set_ylabel("Density")

chi2_fiducial = 9.5
dof, loc, scale = chi2.fit(xi_chi2s, floc=0)

print(f"Best-fit dof: {dof:.3e}")
counts, bin_edges = np.histogram(xi_chi2s, bins=25, density=True)

sns.histplot(
    xi_chi2s,
    ax=ax2,
    kde=False,
    bins=bin_edges,
    stat="density",
    label=r"$\chi^2$ for \texttt{GLASS} mocks best-fits",
    color="pink",
    alpha=0.5,
)

# Compute the p-value

# 1. Get in which bin the chi2 of the fiducial falls
bin_index = np.digitize(chi2_fiducial, bin_edges)

# 2. Compute the p-value as the integral of the tail of the histogram
p_value = np.sum(counts[bin_index:]) * np.diff(bin_edges)[0]

print(f"P-value: {p_value}")

ax2.axvline(chi2_fiducial, color="red", label=r"$\chi^2$ of the fiducial", lw=2)

mantissa, exponent = np.frexp(p_value)
print(f"mantissa: {mantissa}, exponent: {exponent}")
pte_string = rf"${{\rm PTE}} = {p_value:.4f}$"
# rf"${{\rm PTE}} = {mantissa:.2f} \times 10^{{{exponent}}}$" if exponent != 0 else
x_text = 17.5
y_text = max(counts) * 0.95
ax2.text(
    x_text,
    y_text,
    pte_string,
    fontsize=15,
    bbox=dict(facecolor="wheat", alpha=0.8, edgecolor="black"),
)

chi2_string = rf"${{\rm Eff. dof}}= {dof:.1f}$"
y_text = max(counts) * 0.85
ax2.text(
    x_text,
    y_text,
    chi2_string,
    fontsize=15,
    bbox=dict(facecolor="wheat", alpha=0.8, edgecolor="black"),
)

ax2.set_xlabel(r"$\chi^2 (\xi_\pm)$")
ax2.set_ylabel("Density")
fig.savefig(f"{output_fig_path}/chi2_glass_mocks_p_value_xi_tau.pdf")






