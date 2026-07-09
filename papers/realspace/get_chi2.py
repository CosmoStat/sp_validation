import configparser
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from astropy.io import fits
from getdist import plots
from scipy.interpolate import interp1d

sys.path.append("/home/guerrini/sp_validation/cosmo_inference/scripts")

import chain_postprocessing

plt.rc("mathtext", fontset="stix")
plt.rc("font", family="sans-serif")

g = plots.get_subplot_plotter(width_inch=30)
g.settings.axes_fontsize = 30
g.settings.axes_labelsize = 30
g.settings.alpha_filled_add = 0.7
g.settings.legend_fontsize = 40

# #SPECIFY DATA DIRECTORY AND DESIRED CHAINS TO ANALYSE
root_dir = "/n09data/guerrini/output_chains/"
blind = "B"

roots = [
    f"SP_v1.4.6.3_{blind}_fiducial_config",
    f"SP_v1.4.6.3_{blind}_small_scales_config",
    f"SP_v1.4.6.3_{blind}_flat_alpha_beta_config",
    f"SP_v1.4.6.3_{blind}_no_xi_sys_config",
    f"SP_v1.4.6.3_{blind}_no_leak_corr_config",
    f"SP_v1.4.6.3_{blind}_flat_delta_z_config",
    f"SP_v1.4.6.3_{blind}_no_delta_z_config",
    f"SP_v1.4.6.3_{blind}_flat_ia_config",
    f"SP_v1.4.6.3_{blind}_no_ia_config",
    f"SP_v1.4.6.3_{blind}_no_m_bias_config",
    f"SP_v1.4.6.3_{blind}_unmasked_covmat_config",
    f"SP_v1.4.6.3_{blind}_halofit_config",
    f"SP_v1.4.6.3_{blind}_no_baryons_config",
    f"SP_v1.4.6.3_{blind}_nautilus_config",
    f"SP_v1.4.6.3_{blind}_planck_config",
    f"SP_v1.4.6.3_{blind}_planck_desi_config",
]

catalog_versions = [
    f"SP_v1.4.6.3_config/SP_v1.4.6.3_{blind}",
]

catalog_sub_versions = [
    f"SP_v1.4.6.3_leak_corr_{blind}_masked",
    f"SP_v1.4.6.3_leak_corr_{blind}_masked",
    f"SP_v1.4.6.3_leak_corr_{blind}_masked",
    f"SP_v1.4.6.3_leak_corr_{blind}_masked",
    f"SP_v1.4.6.3_{blind}_masked",
    f"SP_v1.4.6.3_leak_corr_{blind}_masked",
    f"SP_v1.4.6.3_leak_corr_{blind}_masked",
    f"SP_v1.4.6.3_leak_corr_{blind}_masked",
    f"SP_v1.4.6.3_leak_corr_{blind}_masked",
    f"SP_v1.4.6.3_leak_corr_{blind}_masked",
    f"SP_v1.4.6.3_leak_corr_{blind}",
    f"SP_v1.4.6.3_leak_corr_{blind}_masked",
    f"SP_v1.4.6.3_leak_corr_{blind}_masked",
    f"SP_v1.4.6.3_leak_corr_{blind}_masked",
    f"SP_v1.4.6.3_leak_corr_{blind}_masked",
    f"SP_v1.4.6.3_leak_corr_{blind}_masked",
]
output_folder = "/n09data/guerrini/output_chains/"

path_ini_files = "/home/guerrini/sp_validation/cosmo_inference/cosmosis_config/"


ini_roots = [
    f"blind_{blind}/fiducial",
    f"blind_{blind}/small_scales",
    f"blind_{blind}/flat_alpha_beta",
    f"blind_{blind}/no_xi_sys",
    f"blind_{blind}/no_leak_corr",
    f"blind_{blind}/flat_delta_z",
    f"blind_{blind}/no_delta_z",
    f"blind_{blind}/flat_ia",
    f"blind_{blind}/no_ia",
    f"blind_{blind}/no_m_bias",
    f"blind_{blind}/unmasked_covmat",
    f"blind_{blind}/halofit",
    f"blind_{blind}/no_baryons",
    f"blind_{blind}/nautilus",
    f"blind_{blind}/planck",
    f"blind_{blind}/planck_desi",
]

properties = {}

for i, root in enumerate(roots):
    print(root)
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve case sensitivity of option names
    config.read(
        path_ini_files
        + "config_space_v1.4.6.3_fiducial/pipeline/"
        + ini_roots[i]
        + ".ini"
    )
    add_xi_sys = config["2pt_like"]["add_xi_sys"]
    lower_bound_xi_plus, upper_bound_xi_plus = map(
        float, config["2pt_like"]["angle_range_XI_PLUS_1_1"].split()
    )
    lower_bound_xi_minus, upper_bound_xi_minus = map(
        float, config["2pt_like"]["angle_range_XI_MINUS_1_1"].split()
    )

    properties[root] = {
        "add_xi_sys": add_xi_sys,
        "lower_bound_xi_plus": lower_bound_xi_plus,
        "upper_bound_xi_plus": upper_bound_xi_plus,
        "lower_bound_xi_minus": lower_bound_xi_minus,
        "upper_bound_xi_minus": upper_bound_xi_minus,
    }


# ## Retrieve the chains


# READ CHAIN

chains = []

for i, root in enumerate(roots):
    burnin = 0

    if os.path.isfile(root_dir + "{}/getdist_{}.txt".format(root, root)) == False:
        samples = np.loadtxt(root_dir + "{}/samples_{}.txt".format(root, root))

        if "nautilus" in root:
            samples = np.column_stack(
                (
                    np.exp(samples[:, -3]),
                    samples[:, -1] - samples[:, -2],
                    samples[:, 0:-3],
                )
            )
        elif "mh" in root:
            samples = np.column_stack(
                (
                    np.ones_like(samples[:, -1]),
                    np.log(samples[:, -1]) - np.log(samples[:, -2]),
                    samples[:, 0:-2],
                )
            )
            burnin = 0.3
        else:
            samples = np.column_stack(
                (samples[:, -1], samples[:, -3], samples[:, 0:-4])
            )

        np.savetxt(root_dir + "{}/getdist_{}.txt".format(root, root), samples)

    chain = g.samples_for_root(
        root_dir + "{}/getdist_{}".format(root, root),
        cache=False,
        settings={
            "ignore_rows": burnin,
            "smooth_scale_2D": 0.5,
            "smooth_scale_1D": 0.5,
        },
    )
    p = chain.getParams()

    chains.append(chain)


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

for chain in chains:
    param_names = chain.getParamNames()
    for name, label in zip(param_list, label_list):
        if param_names.parWithName(name) is not None:
            param_names.parWithName(name).label = label


# ## Extract the best fit parameters


best_fit = {}

for root, chain in zip(roots, chains):
    print(root)
    p = chain.getParams()

    best_fit[root] = chain_postprocessing.extract_best_fit_params(
        chain, best_fit_method="2Dkde"
    )

    for param_name in best_fit[root].keys():
        high_68, low_68, high_95, low_95 = chain_postprocessing.compute_limits(
            chain, param_name
        )
        if param_name == "S_8":
            print(f"{best_fit[root][param_name]}")


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
    "tau": "cosmological_parameters",
    "s_8_input": "cosmological_parameters",
    "logt_agn": "halo_model_parameters",
    "a": "intrinsic_alignment_parameters",
    "m1": "shear_calibration_parameters",
    "bias_1": "nofz_shifts",
    "alpha": "psf_leakage_parameters",
    "beta": "psf_leakage_parameters",
    "m": "supernova_params",
    "a_planck": "planck",
}

best_fit["SP_v1.4.6.3_B_no_ia_config"]["a"] = 0


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

    ini_file = path_ini_files + "config_space_v1.4.6.3_fiducial/pipeline/{}.ini".format(
        ini_roots[i]
    )
    config.read(ini_file)

    sampler = config["runtime"]["sampler"]
    config["runtime"]["sampler"] = "test"
    values = config["pipeline"]["values"]
    config["pipeline"]["values"] = path_ini_files + "/values_empty.ini"
    config["DEFAULT"]["FITS_FILE"] = (
        f"/home/guerrini/sp_validation/cosmo_inference/data/{catalog_versions[0]}/cosmosis_{catalog_sub_versions[i]}.fits"
    )
    config["test"]["save_dir"] = root_dir + "{}/best_fit".format(root)

    with open(ini_file, "w") as configfile:
        config.write(configfile)

    # Run cosmosis
    result = subprocess.run(
        ["cosmosis", ini_file], env=env, capture_output=True, text=True
    )
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")

    # Modify the ini file to the previous one
    config["pipeline"]["values"] = values
    config["runtime"]["sampler"] = sampler

    with open(ini_file, "w") as configfile:
        config.write(configfile)


# ## Compute the $\chi^2$


metrics = {}

for idx, root in enumerate(roots):
    print(root)
    match = re.search(r"corr_([A-Za-z])", root)
    if match:
        blind = match.group(1)

    add_xi_sys = properties[root]["add_xi_sys"]
    print(f"add_xi_sys: {add_xi_sys}")
    lower_bound_xi_plus = properties[root]["lower_bound_xi_plus"]
    upper_bound_xi_plus = properties[root]["upper_bound_xi_plus"]
    lower_bound_xi_minus = properties[root]["lower_bound_xi_minus"]
    upper_bound_xi_minus = properties[root]["upper_bound_xi_minus"]

    # Read the results
    theta = np.loadtxt(
        output_folder + "{}/best_fit/shear_xi_plus/theta.txt".format(root)
    )
    theta_arcmin = theta * 180 * 60 / np.pi
    shear_xi_plus = np.loadtxt(
        output_folder + "{}/best_fit/shear_xi_plus/bin_1_1.txt".format(root)
    )
    shear_xi_minus = np.loadtxt(
        output_folder + "{}/best_fit/shear_xi_minus/bin_1_1.txt".format(root)
    )

    if add_xi_sys == "T":
        xi_sys_plus = np.loadtxt(
            output_folder + "{}/best_fit/xi_sys/shear_xi_plus.txt".format(root)
        )
        xi_sys_minus = np.loadtxt(
            output_folder + "{}/best_fit/xi_sys/shear_xi_minus.txt".format(root)
        )

        theta_tau = np.loadtxt(
            output_folder + "{}/best_fit/tau_0_plus/theta.txt".format(root)
        )
        theta_tau_arcmin = theta_tau * 180 * 60 / np.pi
        tau_0_model = np.loadtxt(
            output_folder + "{}/best_fit/tau_0_plus/bin_1_1.txt".format(root)
        )
        tau_2_model = np.loadtxt(
            output_folder + "{}/best_fit/tau_2_plus/bin_1_1.txt".format(root)
        )

    data = fits.open(
        f"/home/guerrini/sp_validation/cosmo_inference/data/{catalog_versions[0]}/cosmosis_{catalog_sub_versions[idx]}.fits"
    )

    tau_0_data = data["TAU_0_PLUS"].data["VALUE"]
    tau_2_data = data["TAU_2_PLUS"].data["VALUE"]

    theta_data = data["XI_PLUS"].data["ANG"]
    xi_plus_data = data["XI_PLUS"].data["VALUE"]
    xi_minus_data = data["XI_MINUS"].data["VALUE"]

    # Load the covariance
    cov = data["COVMAT"].data
    cov_xi = cov[0 : 2 * len(xi_plus_data), 0 : 2 * len(xi_plus_data)]
    cov_tau = cov[2 * len(xi_plus_data) :, 2 * len(xi_plus_data) :]

    # interpolate the model
    interp_xi_plus = interp1d(
        theta_arcmin, shear_xi_plus, kind="cubic", fill_value="extrapolate"
    )
    interp_xi_minus = interp1d(
        theta_arcmin, shear_xi_minus, kind="cubic", fill_value="extrapolate"
    )

    xi_plus_model = interp_xi_plus(theta_data)
    if add_xi_sys:
        xi_plus_model += xi_sys_plus
    xi_minus_model = interp_xi_minus(theta_data)
    if add_xi_sys:
        xi_minus_model += xi_sys_minus

    # Concatenate the data vector
    xi_data = np.concatenate((xi_plus_data, xi_minus_data))
    xi_model = np.concatenate((xi_plus_model, xi_minus_model))

    tau_data = np.concatenate((tau_0_data, tau_2_data))
    tau_model = np.concatenate((tau_0_model, tau_2_model))

    # Apply scale cuts
    mask_xi_plus = (theta_data > lower_bound_xi_plus) & (
        theta_data < upper_bound_xi_plus
    )
    mask_xi_minus = (theta_data > lower_bound_xi_minus) & (
        theta_data < upper_bound_xi_minus
    )
    mask = np.concatenate((mask_xi_plus, mask_xi_minus))

    xi_data = xi_data[mask]
    xi_model = xi_model[mask]
    cov_xi = cov_xi[mask][:, mask]

    cov_xi_plus = cov[0 : len(xi_plus_data), 0 : len(xi_plus_data)]
    cov_xi_plus = cov_xi_plus[mask_xi_plus][:, mask_xi_plus]
    cov_xi_minus = cov[
        len(xi_plus_data) : 2 * len(xi_minus_data),
        len(xi_plus_data) : 2 * len(xi_minus_data),
    ]
    cov_xi_minus = cov_xi_minus[mask_xi_minus][:, mask_xi_minus]

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
    n_dof_xi_plus = np.sum(mask_xi_plus)
    n_dof_xi_minus = np.sum(mask_xi_minus)
    n_dof_tau = len(tau_0_data) + len(tau_2_data)
    p_value_xi_plus = 1 - stats.chi2.cdf(xi_plus_chi2, n_dof_xi_plus)
    p_value_xi_minus = 1 - stats.chi2.cdf(xi_minus_chi2, n_dof_xi_minus)
    p_value_xi = 1 - stats.chi2.cdf(xi_chi2, n_dof_xi_plus + n_dof_xi_minus)
    p_value_tau = 1 - stats.chi2.cdf(tau_chi2, n_dof_tau)
    chi2_tot = xi_plus_chi2 + xi_minus_chi2 + tau_chi2
    n_dof_tot = n_dof_xi_plus + n_dof_xi_minus + n_dof_tau
    p_value_tot = 1 - stats.chi2.cdf(chi2_tot, n_dof_tot)

    metrics[root] = {
        "chi2_xi_plus": xi_plus_chi2,
        "n_dof_xi_plus": n_dof_xi_plus,
        "p_value_xi_plus": p_value_xi_plus,
        "chi2_xi_minus": xi_minus_chi2,
        "n_dof_xi_minus": n_dof_xi_minus,
        "p_value_xi_minus": p_value_xi_minus,
        "chi2_xi": xi_chi2,
        "p_value_xi": p_value_xi,
        "chi2_tau": tau_chi2,
        "n_dof_tau": n_dof_tau,
        "p_value_tau": p_value_tau,
        "chi2_tot": chi2_tot,
        "n_dof_tot": n_dof_tot,
        "p_value_tot": p_value_tot,
    }
    print("Done!")


def get_latex_table(metrics):
    latex_lines = [
        r"\begin{tabular}{lccc|ccc|ccc}",
        r"\hline",
        r"Root & $\chi^2_{\xi^+}$/dof & $p_{\xi^+}$ & $\chi^2_{\xi^-}$/dof & $p_{\xi^+}$ & $\chi^2_{\xi}$/dof  & $p_{\xi}$ &"
        r"$\chi^2_\tau$/dof & $p_\tau$ & $\chi^2_{\text{tot}}$/dof & $p_{\text{tot}}$ \\",
        r"\hline",
    ]

    for root, vals in metrics.items():
        escaped = root.replace("_", r"\_")
        line = (
            f"{escaped} & "
            f"{vals['chi2_xi_plus']:.2f}/{vals['n_dof_xi_plus']} & {vals['p_value_xi_plus']:.3g} & "
            f"{vals['chi2_xi_minus']:.2f}/{vals['n_dof_xi_minus']} & {vals['p_value_xi_minus']:.3g} & "
            f"{vals['chi2_xi']:.2f}/{vals['n_dof_xi_plus'] + vals['n_dof_xi_minus']} & {vals['p_value_xi']:.3g}  &"
            f"{vals['chi2_tau']:.2f}/{vals['n_dof_tau']} & {vals['p_value_tau']:.3g} & "
            f"{vals['chi2_tot']:.2f}/{vals['n_dof_tot']} & {vals['p_value_tot']:.3g} \\\\"
        )
        latex_lines.append(line)

    latex_lines.append(r"\hline")
    latex_lines.append(r"\end{tabular}")

    # Print LaTeX table
    print("\n".join(latex_lines))


get_latex_table(metrics)


def display_markdown(metrics):
    # Build Markdown table
    header = (
        "| Root | $\\chi^2$ (ξ⁺) / dof | p-val (ξ⁺) |$\\chi^2$ (ξ-) / dof | p-val (ξ-) |  $\\chi^2$ (ξ) / dof | p-val (ξ) | $\\chi^2$ (τ) / dof | p-val (τ) | $\\chi^2$ (tot) / dof | p-val (tot) |\n"
        "|------|----------------|------------|----------------|------------|------------|---------------|------------|------------|------------------|--------------|\n"
    )

    rows = []
    for root, vals in metrics.items():
        row = f"| `{root}` "
        row += f"| {vals['chi2_xi_plus']:.2f} / {vals['n_dof_xi_plus']} "
        row += f"| {vals['p_value_xi_plus']:.5f} "
        row += f"| {vals['chi2_xi_minus']:.2f} / {vals['n_dof_xi_minus']} "
        row += f"| {vals['p_value_xi_minus']:.5f} "
        row += f"| {vals['chi2_xi']:.2f} / {vals['n_dof_xi_minus'] + vals['n_dof_xi_plus']} "
        row += f"| {vals['p_value_xi']:.5f} "
        row += f"| {vals['chi2_tau']:.2f} / {vals['n_dof_tau']} "
        row += f"| {vals['p_value_tau']:.5f} "
        row += f"| {vals['chi2_tot']:.2f} / {vals['n_dof_tot']} "
        row += f"| {vals['p_value_tot']:.5f} |"
        rows.append(row)

    # Display in Jupyter
    return header + "\n".join(rows)


markdown_source = display_markdown(metrics)
