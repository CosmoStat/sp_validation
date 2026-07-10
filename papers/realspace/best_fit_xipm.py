import os
import sys

sys.path.append("/home/guerrini/sp_validation/cosmo_inference/scripts")

import chain_postprocessing as cp
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import numpy as np
import seaborn as sns
from astropy.io import fits
from getdist import plots

plt.style.use("/home/guerrini/matplotlib_config/paper.mplstyle")

from sp_validation.rho_tau import SquareRootScale

mscale.register_scale(SquareRootScale)

plt.rcParams["text.usetex"] = True

sns.set_palette("husl")

g = plots.get_subplot_plotter(width_inch=30)
g.settings.axes_fontsize = 40
g.settings.axes_labelsize = 40
g.settings.alpha_filled_add = 0.7
g.settings.legend_fontsize = 50

# Directory where the chains are located
root_dir = "/n09data/guerrini/output_chains"

# THE BLIND TO USE FOR THE PLOTS
blind = "B"
catalog_version = "SP_v1.4.6.3"
fiducial_root_cell = f"SP_v1.4.6.3_leak_corr_{blind}"
label_fiducial_cell = r"UNIONS $C_{\ell}$"
fiducial_root_xi_data = f"SP_v1.4.6.3_leak_corr_{blind}_masked"
fiducial_root_xi_chains = f"SP_v1.4.6.3_{blind}_fiducial_config"
label_fiducial_xi = r"UNIONS $\xi_{\pm}$"

# Path to the ini files used
path_ini_files = "/home/guerrini/sp_validation/cosmo_inference/cosmosis_config"
path_datavectors = "/home/guerrini/sp_validation/cosmo_inference/data/"
path_output_chains = "/n09data/guerrini/output_chains/"


data_cell = fits.open(
    os.path.join(
        path_datavectors, f"{fiducial_root_cell}/cosmosis_{fiducial_root_cell}.fits"
    )
)

data_xi = fits.open(
    os.path.join(
        path_datavectors,
        f"SP_v1.4.6.3_config/SP_v1.4.6.3_{blind}/cosmosis_{fiducial_root_xi_data}.fits",
    )
)

path_samples_fiducial_cell = os.path.join(
    path_output_chains,
    fiducial_root_cell,
    fiducial_root_cell,
    f"samples_{fiducial_root_cell}_cell.txt",
)
path_gd_fiducial_cell = os.path.join(
    path_output_chains,
    fiducial_root_cell,
    fiducial_root_cell,
    f"getdist_{fiducial_root_cell}_cell",
)
cp.load_samples_and_write_paramnames(
    path_samples_fiducial_cell, path_gd_fiducial_cell + ".paramnames"
)
cp.write_samples_getdist_format(
    path_samples_fiducial_cell, path_gd_fiducial_cell + ".txt", chain_type="polychord"
)

chain_fiducial_cell = cp.load_chain(path_gd_fiducial_cell, smoothing_scale=0.5)

best_fit_params_fiducial_cell = cp.extract_best_fit_params(
    chain_fiducial_cell, best_fit_method="2Dkde"
)

cp.compute_best_fit(
    path_ini_files,
    best_fit_params_fiducial_cell,
    fiducial_root_cell,
    is_harmonic=True,
    blind=blind,
)
path_samples_fiducial_xi = os.path.join(
    path_output_chains,
    fiducial_root_xi_chains,
    f"samples_{fiducial_root_xi_chains}.txt",
)

path_gd_fiducial_xi = os.path.join(
    path_output_chains, fiducial_root_xi_chains, f"getdist_{fiducial_root_xi_chains}"
)
cp.load_samples_and_write_paramnames(
    path_samples_fiducial_xi, path_gd_fiducial_xi + ".paramnames"
)
cp.write_samples_getdist_format(
    path_samples_fiducial_xi, path_gd_fiducial_xi + ".txt", chain_type="polychord"
)

chain_fiducial_xi = cp.load_chain(path_gd_fiducial_xi, smoothing_scale=0.5)

best_fit_params_fiducial_xi = cp.extract_best_fit_params(
    chain_fiducial_xi, best_fit_method="2Dkde"
)

ini_file_root = os.path.join(
    path_ini_files,
    f"config_space_v1.4.6.3_fiducial/pipeline/blind_{blind}/fiducial.ini",
)
cp.compute_best_fit(
    path_ini_files,
    best_fit_params_fiducial_xi,
    fiducial_root_xi_chains,
    is_harmonic=False,
    blind=blind,
    ini_file_root=ini_file_root,
)

root_to_plot = [
    fiducial_root_xi_chains,
    fiducial_root_cell,
]

labels = [
    r"UNIONS $\xi_\pm(\theta)$",
    r"UNIONS $C_\ell$",
]

line_args = [
    {"color": "royalblue", "linestyle": "-"},
    {"color": "orange", "linestyle": "-"},
]

properties = {}

properties = cp.update_properties_w_roots(
    properties, fiducial_root_cell, path_ini_files, with_configuration=False
)
properties = cp.update_properties_w_roots(
    properties,
    fiducial_root_xi_chains,
    path_ini_files,
    with_configuration=True,
    path_to_this_ini=ini_file_root,
)


root_to_plot = [fiducial_root_cell, fiducial_root_xi_chains]
labels = [r"Best fit $C_\ell$", r"Best fit $\xi_\pm(\theta)$"]
path_best_fit_xi_theta = os.path.join(
    path_output_chains, fiducial_root_xi_chains, "best_fit/shear_xi_plus/theta.txt"
)

theta_rad = np.loadtxt(path_best_fit_xi_theta)
theta_min = 1
theta_max = 250

cp.compute_best_fit_xi_from_cell(
    path_output_chains, fiducial_root_cell, best_fit_params_fiducial_cell, theta_rad
)

data = fits.open(
    os.path.join(
        path_datavectors,
        f"SP_v1.4.6.3_config/SP_v1.4.6.3_{blind}/cosmosis_{fiducial_root_xi_data}.fits",
    )
)
bbox_to_anchor_xip = (0.685, 0.09)
bbox_to_anchor_xim = (0.3, 0.65)
xi_p_data = data["XI_PLUS"].data
xi_m_data = data["XI_MINUS"].data
cov_mat = data["COVMAT"].data

# Plot hyperparameter
loc_legend = "lower center"

fig, [ax, ax2] = plt.subplots(1, 2, figsize=(20, 8))

theta, xi_p, xi_m = xi_p_data["ANG"], xi_p_data["VALUE"], xi_m_data["VALUE"]
ax.errorbar(
    theta,
    theta * xi_p,
    yerr=theta * np.sqrt(np.diag(cov_mat[: len(theta), : len(theta)])),
    fmt="o",
    label=r"UNIONS $\xi_+$ data",
    color="black",
    capsize=2,
)
ax2.errorbar(
    theta,
    theta * xi_m,
    yerr=theta
    * np.sqrt(
        np.diag(cov_mat[len(theta) : 2 * len(theta), len(theta) : 2 * len(theta)])
    ),
    fmt="o",
    label=r"UNIONS $\xi_-$ data",
    color="black",
    capsize=2,
)

for idx, (label, root) in enumerate(zip(labels, root_to_plot)):
    # Read the results
    theta = (
        (
            np.loadtxt(
                path_output_chains + "{}/best_fit/shear_xi_plus/theta.txt".format(root)
            )
        )
        * 180
        / np.pi
        * 60
    )
    xi_plus = np.loadtxt(
        path_output_chains + "{}/best_fit/shear_xi_plus/bin_1_1.txt".format(root)
    )
    xi_minus = np.loadtxt(
        path_output_chains + "{}/best_fit/shear_xi_minus/bin_1_1.txt".format(root)
    )
    if r"$C_\ell$" not in label:
        xi_sys_plus = np.loadtxt(
            path_output_chains + "{}/best_fit/xi_sys/shear_xi_plus.txt".format(root)
        )
        xi_sys_minus = np.loadtxt(
            path_output_chains + "{}/best_fit/xi_sys/shear_xi_minus.txt".format(root)
        )
        theta_xi_sys = (
            np.loadtxt(path_output_chains + "{}/best_fit/xi_sys/theta.txt".format(root))
            * 180
            / np.pi
            * 60
        )

        xi_sys_plus = np.interp(theta, theta_xi_sys, xi_sys_plus)
        xi_sys_minus = np.interp(theta, theta_xi_sys, xi_sys_minus)
        xi_plus += xi_sys_plus
        xi_minus += xi_sys_minus

        mask = (theta > theta_min) & (theta < theta_max)
        theta = theta[mask]
        ax.plot(
            theta,
            theta * xi_plus[mask],
            label=r"Best fit $\xi_+(\theta)$",
            **line_args[idx],
            lw=2.5,
        )
        ax.plot(
            theta,
            theta * xi_sys_plus[mask],
            label=r"Best fit $\xi^{\rm sys}_{+}(\theta)$",
            c="r",
        )
        ax2.plot(
            theta,
            theta * xi_minus[mask],
            label=r"Best fit $\xi_-(\theta)$",
            **line_args[idx],
            lw=2.5,
        )
        ax2.plot(
            theta,
            theta * xi_sys_minus[mask],
            label=r"Best fit $\xi^{\rm sys}_{-}(\theta)$",
            c="r",
        )

    else:
        mask = (theta > theta_min) & (theta < theta_max)
        theta = theta[mask]
        ax.plot(theta, theta * xi_plus[mask], label=label, **line_args[idx], lw=2.5)
        ax2.plot(theta, theta * xi_minus[mask], label=label, **line_args[idx], lw=2.5)

# XI PLUS PLOT SETTINGS

# Plot the scale cuts for different k_max
ax.axvline(x=5, color="gray", linestyle="--", alpha=0.7)
ax.axhline(y=0, color="black", linestyle="--", alpha=0.7)

ymin = ax.get_ylim()[0]
ymax = ax.get_ylim()[1]
# Shadowing cut scaled
ax.fill_betweenx(y=[ymin, ymax], x1=0, x2=12, color="gray", alpha=0.2)
ax.fill_betweenx(y=[ymin, ymax], x1=83, x2=250, color="gray", alpha=0.2)

ax.set_ylim(ymin, ymax)

# Add labels directly under the tick
ax.text(
    4.5,
    0.47e-4,
    r"$k_\mathrm{max} = 1 h$ Mpc$^{-1}$",
    ha="center",
    va="top",
    fontsize=20,
    rotation=90,
)

ax.set_ylabel(r"$\theta \xi_\pm$", fontsize=26)
ax.set_xlabel(r"$\theta$ (arcmin)", fontsize=26)
ax.set_xlim([theta.min() - 0.1, theta.max() + 20])
ax.set_title(r"$\xi_+(\theta)$", fontsize=26)
ax.set_xscale("log")
ax.set_xticks(np.array([1, 10, 100]))
ax.tick_params(axis="x", which="minor", length=2, width=0.8)
ax.tick_params(axis="both", which="major", labelsize=24)
ax.tick_params(axis="both", which="minor", labelsize=20)
ax.yaxis.get_offset_text().set_fontsize(24)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.legend(loc=loc_legend, bbox_to_anchor=bbox_to_anchor_xip, fontsize=20)

# XI_MINUS PLOT SETTINGS

# Plot the scale cuts for different k_max
ax2.axvline(x=50, color="gray", linestyle="--", alpha=0.7)
ax2.axhline(y=0, color="black", linestyle="--", alpha=0.7)

ymin = ax2.get_ylim()[0]
ymax = ax2.get_ylim()[1]
# Shadowing cut scaled
ax2.fill_betweenx(y=[ymin, ymax], x1=0, x2=12, color="gray", alpha=0.2)
ax2.fill_betweenx(y=[ymin, ymax], x1=83, x2=250, color="gray", alpha=0.2)

ax2.set_ylim(ymin, ymax)

# Add labels directly under the tick
ax2.text(
    45,
    1.15e-4,
    r"$k_\mathrm{max} = 1 h$ Mpc$^{-1}$",
    ha="center",
    va="top",
    fontsize=20,
    rotation=90,
)

# ax2.set_ylabel(r'$\theta \xi_-$', fontsize=16)
ax2.set_xlabel(r"$\theta$ (arcmin)", fontsize=26)
ax2.set_xlim([theta.min() - 0.1, theta.max() + 20])
ax2.set_xscale("log")
ax2.set_title(r"$\xi_-(\theta)$", fontsize=26)
ax2.set_xticks(np.array([1, 10, 100]))
ax2.tick_params(axis="x", which="minor", length=2, width=0.8)
ax2.tick_params(axis="both", which="major", labelsize=24)
ax2.tick_params(axis="both", which="minor", labelsize=20)
ax2.yaxis.get_offset_text().set_fontsize(24)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax2.legend(loc=loc_legend, bbox_to_anchor=bbox_to_anchor_xim, fontsize=20)

plt.savefig("./../../results/best_fit_xipm_SP_v1.4.6.3_B.pdf", bbox_inches="tight")


root_to_plot = [fiducial_root_xi_chains]
labels = [r"Best fit $\tau_{0,2}(\theta)$"]

bbox_to_anchor_xip = (0.285, 0.7)
bbox_to_anchor_xim = (0.3, 0.65)
tau0_data = data["TAU_0_PLUS"].data
tau2_data = data["TAU_2_PLUS"].data
cov_mat = data["COVMAT"].data

# Plot hyperparameter

fig, [ax, ax2] = plt.subplots(1, 2, figsize=(20, 8))

theta, tau0, tau2 = tau0_data["ANG"], tau0_data["VALUE"], tau2_data["VALUE"]
ax.errorbar(
    theta,
    theta * tau0,
    yerr=theta
    * np.sqrt(
        np.diag(
            cov_mat[2 * len(theta) : 3 * len(theta), 2 * len(theta) : 3 * len(theta)]
        )
    ),
    fmt="o",
    label=r"UNIONS $\tau_{0,+}$",
    color="black",
    capsize=2,
)
ax2.errorbar(
    theta,
    theta * tau2,
    yerr=theta
    * np.sqrt(
        np.diag(
            cov_mat[3 * len(theta) : 4 * len(theta), 3 * len(theta) : 4 * len(theta)]
        )
    ),
    fmt="o",
    label=r"UNIONS $\tau_{2,+}$",
    color="black",
    capsize=2,
)

for idx, (label, root) in enumerate(zip(labels, root_to_plot)):
    # Read the results
    theta = (
        (
            np.loadtxt(
                path_output_chains + "{}/best_fit/tau_0_plus/theta.txt".format(root)
            )
        )
        * 180
        / np.pi
        * 60
    )
    tau0_plus = np.loadtxt(
        path_output_chains + "{}/best_fit/tau_0_plus/bin_1_1.txt".format(root)
    )
    tau2_plus = np.loadtxt(
        path_output_chains + "{}/best_fit/tau_2_plus/bin_1_1.txt".format(root)
    )

    mask = (theta > theta_min) & (theta < theta_max)
    theta = theta[mask]
    ax.plot(
        theta,
        theta * tau0_plus[mask],
        label=r"Best fit $\tau_{0,+}(\theta)$",
        c="orange",
        lw=2.5,
    )
    ax2.plot(
        theta,
        theta * tau2_plus[mask],
        label=r"Best fit $\tau_{2,+}(\theta)$",
        c="orange",
        lw=2.5,
    )

# XI PLUS PLOT SETTINGS

# Plot the scale cuts for different k_max
ax.axhline(y=0, color="black", linestyle="--", alpha=0.7)

ymin = ax.get_ylim()[0]
ymax = ax.get_ylim()[1]

ax.set_ylim(ymin, ymax)

ax.set_ylabel(r"$\theta\tau_{0,2}$", fontsize=26)
ax.set_xlabel(r"$\theta$ (arcmin)", fontsize=26)
ax.set_xlim([theta.min() - 0.1, theta.max() + 20])
ax.set_title(r"$\tau_{0,+}(\theta)$", fontsize=26)
ax.set_xscale("log")
ax.set_xticks(np.array([1, 10, 100]))
ax.tick_params(axis="x", which="minor", length=2, width=0.8)
ax.tick_params(axis="both", which="major", labelsize=24)
ax.tick_params(axis="both", which="minor", labelsize=20)
ax.yaxis.get_offset_text().set_fontsize(24)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.legend(loc=loc_legend, bbox_to_anchor=bbox_to_anchor_xip, fontsize=20)

# XI_MINUS PLOT SETTINGS

# Plot the scale cuts for different k_max
ax2.axhline(y=0, color="black", linestyle="--", alpha=0.7)

ymin = ax2.get_ylim()[0]
ymax = ax2.get_ylim()[1]
# Shadowing cut scaled
ax2.fill_betweenx(
    y=[ymin, ymax],
    x1=0,
    x2=12,
    color="gray",
    alpha=0.2,
    label=r"$B$-mode informed scale cut",
)
ax2.fill_betweenx(y=[ymin, ymax], x1=83, x2=250, color="gray", alpha=0.2)

ax2.set_ylim(ymin, ymax)

# ax2.set_ylabel(r'$\theta \xi_-$', fontsize=16)
ax2.set_xlabel(r"$\theta$ (arcmin)", fontsize=26)
ax2.set_xlim([theta.min() - 0.1, theta.max() + 20])
ax2.set_xscale("log")
ax2.set_title(r"$\tau_{2,+}(\theta)$", fontsize=26)
ax2.set_xticks(np.array([1, 10, 100]))
ax2.tick_params(axis="x", which="minor", length=2, width=0.8)
ax2.tick_params(axis="both", which="major", labelsize=24)
ax2.tick_params(axis="both", which="minor", labelsize=20)
ax2.yaxis.get_offset_text().set_fontsize(24)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax2.legend(loc=loc_legend, bbox_to_anchor=bbox_to_anchor_xim, fontsize=20)

plt.savefig("./../../results/best_fit_tau_02_SP_v1.4.6.3_B.pdf", bbox_inches="tight")
