"""
Script to produce comparison plots between galaxy-PSF correlations
computed with MCCD and PSFEx
Author: Sacha Guerrini
"""

# %%
# if interactive
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits
from getdist import MCSamples, plots

# Your Import here
from shear_psf_leakage.rho_tau_stat import PSFErrorFit, RhoStat, TauStat

# %%
try:
    from IPython import get_ipython

    ipython = get_ipython()
except Exception:
    ipython = None

# enable autoreload for interactive session
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
else:
    # Force unbuffered stdout and stderr
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# Load matplotlib plot style
plt.style.use("./matplotlib_config/paper.mplstyle")

# Load seaborn color palette
sns.set_palette("Dark2")

# %%
versions = {
    "MCCD": {
        "version": "SP_v1.3.6.3",
        "minsep": 1.0,
        "maxsep": 250.0,
        "nbins": 20,
        "npatch": 1,
    },
    "PSFEx": {
        "version": "SP_v1.4.6.3_B",
        "minsep": 1.0,
        "maxsep": 250.0,
        "nbins": 20,
        "npatch": 1,
    },
}

data_path = "/home/guerrini/sp_validation/notebooks/cosmo_val/output/rho_tau_stats/"

# %%
# Load a plot the rho-statistics

# Title labels
e_psf = r"e^\mathrm{PSF}"
delta_e_psf = r"\delta e^\mathrm{PSF}"
delta_T_psf = r"\delta T^\mathrm{PSF}"

titles = [
    rf"$\langle {e_psf} {e_psf} \rangle$",
    rf"$\langle {delta_e_psf} {delta_e_psf} \rangle$",
    rf"$\langle {e_psf} {delta_e_psf} \rangle$",
    rf"$\langle {delta_T_psf} {delta_T_psf} \rangle$",
    rf"$\langle {delta_e_psf} {delta_T_psf} \rangle$",
    rf"$\langle {e_psf} {delta_T_psf} \rangle$",
]

fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex=True)

axs = axs.flatten()

for psf_method in ["MCCD", "PSFEx"]:
    # Read the rho_stat files
    path_rho_stats = f"{data_path}/rho_stats_{versions[psf_method]['version']}_minsep={versions[psf_method]['minsep']}_maxsep={versions[psf_method]['maxsep']}_nbins={versions[psf_method]['nbins']}_npatch={versions[psf_method]['npatch']}.fits"
    rho_stats = fits.getdata(path_rho_stats)
    path_cov_rho = f"{data_path}/cov_rho_{versions[psf_method]['version']}_minsep={versions[psf_method]['minsep']}_maxsep={versions[psf_method]['maxsep']}_nbins={versions[psf_method]['nbins']}_npatch={versions[psf_method]['npatch']}_jk.npy"
    cov_rho = np.load(path_cov_rho)
    n_bins = rho_stats["theta"].shape[0]
    label = f"{psf_method}"
    for i in range(6):
        axs[i].errorbar(
            rho_stats["theta"],
            np.abs(rho_stats[f"rho_{i}_p"]),
            yerr=np.sqrt(
                cov_rho[
                    i * n_bins : (i + 1) * n_bins, i * n_bins : (i + 1) * n_bins
                ].diagonal()
            ),
            fmt="o",
            linestyle="-",
            label=label,
            capsize=2,
        )

for i in range(6):
    axs[i].set_xscale("log")
    axs[i].set_yscale("log")
    axs[i].set_ylabel(rf"$|\rho_{i}(\vartheta)|$")
    if i in [3, 4, 5]:
        axs[i].set_xlabel(r"$\vartheta$ (arcmin)")
    axs[i].set_title(titles[i])
    axs[i].set_xlim(0.9, 300)

handles, labels = axs[4].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.05),  # centered under all axes
    ncol=2,
    frameon=False,
    fontsize=16,
)

plt.tight_layout()

plt.savefig("Figures/MCCD_vs_PSFEx_rho_stats.png", bbox_inches="tight")
plt.savefig("Figures/MCCD_vs_PSFEx_rho_stats.pdf", bbox_inches="tight")
plt.show()

# %%
# Inference of the xi_sys parameters
sep_units = "arcmin"
coord_units = "degrees"
theta_min = 0.1
theta_max = 250
nbins = 20


TreeCorrConfig_xi = {
    "ra_units": coord_units,
    "dec_units": coord_units,
    "min_sep": theta_min,
    "max_sep": theta_max,
    "sep_units": sep_units,
    "nbins": nbins,
    "var_method": "jackknife",
}

rho_stats_handler = RhoStat(output=".", treecorr_config=TreeCorrConfig_xi, verbose=True)

tau_stats_handler = TauStat(
    catalogs=rho_stats_handler.catalogs,
    output=".",
    treecorr_config=TreeCorrConfig_xi,
    verbose=True,
)

psf_fitter = PSFErrorFit(rho_stats_handler, tau_stats_handler, data_path)

samples = {}
for psf_method in ["MCCD", "PSFEx"]:
    for leak_corr in range(2):
        suffix = "_leak_corr" if leak_corr == 1 else ""
        version = f"{versions[psf_method]['version']}{suffix}"
        path_rho = f"rho_stats_{version}_minsep={versions[psf_method]['minsep']}_maxsep={versions[psf_method]['maxsep']}_nbins={versions[psf_method]['nbins']}_npatch={versions[psf_method]['npatch']}.fits"
        path_tau = f"tau_stats_{version}_minsep={versions[psf_method]['minsep']}_maxsep={versions[psf_method]['maxsep']}_nbins={versions[psf_method]['nbins']}_npatch={versions[psf_method]['npatch']}.fits"
        path_cov_tau = f"cov_tau_{version}_minsep={versions[psf_method]['minsep']}_maxsep={versions[psf_method]['maxsep']}_nbins={versions[psf_method]['nbins']}_npatch={versions[psf_method]['npatch']}_th.npy"
        path_cov_rho = f"cov_rho_{version}_minsep={versions[psf_method]['minsep']}_maxsep={versions[psf_method]['maxsep']}_nbins={versions[psf_method]['nbins']}_npatch={versions[psf_method]['npatch']}_jk.npy"

        # Load the data vectors and covariance matrices
        psf_fitter.load_rho_stat(path_rho)
        psf_fitter.load_tau_stat(path_tau)
        psf_fitter.load_covariance(path_cov_tau, cov_type="tau")
        psf_fitter.load_covariance(path_cov_rho, cov_type="rho")

        _samples, _, _ = psf_fitter.get_least_squares_params_samples(
            npatch=80, apply_debias=False
        )

        samples[f"{version}"] = _samples

# %%
# Plot the contours
g = plots.get_subplot_plotter()

line_args = {}
filled = {}
mcsamples = {}
legend_labels = {}
contour_colors = {}

for psf_method in ["MCCD", "PSFEx"]:
    for leak_corr in range(2):
        suffix = "_leak_corr" if leak_corr == 1 else ""
        version = f"{versions[psf_method]['version']}{suffix}"
        _samples = samples[version]
        label = f"{psf_method} {'w' if leak_corr else 'w/o'} leakage correction"
        mcsamples[version] = MCSamples(
            samples=_samples,
            names=["alpha", "beta", "eta"],
            labels=[r"\alpha", r"\beta", r"\eta"],
            label="",
        )

        legend_labels[version] = label
        line_args[version] = {
            "color": "C0" if psf_method == "MCCD" else "C1",
            "ls": "-" if leak_corr == 1 else "--",
        }
        filled[version] = leak_corr == 1
        contour_colors[version] = "C0" if psf_method == "MCCD" else "C1"

g.triangle_plot(
    list(mcsamples.values()),
    filled=list(filled.values()),
    legend_labels=list(legend_labels.values()),
    legend_loc="upper right",
    line_args=list(line_args.values()),
    contour_colors=list(contour_colors.values()),
)

g.export("Figures/MCCD_vs_PSFEx_contours.png")
g.export("Figures/MCCD_vs_PSFEx_contours.pdf")

plt.show()

# %%
# Make a plot of tau-statistics with the best-fit model

# Title and legend labels
e_obs = r"e^\mathrm{obs}"
e_psf = r"e^\mathrm{PSF}"
delta_e_psf = r"\delta e^\mathrm{PSF}"
delta_T_psf = r"\delta T^\mathrm{PSF}"

titles = [
    rf"$\langle {e_obs} {e_psf} \rangle$",
    rf"$\langle {e_obs} {delta_e_psf} \rangle$",
    rf"$\langle {e_obs} {delta_T_psf} \rangle$",
]

y_labels = [
    r"$\tau_0(\vartheta)$",
    r"$\vartheta \, \tau_2(\vartheta)$",
    r"$\vartheta \, \tau_5(\vartheta)$",
]

dict_index_tau = {
    0: "0",
    1: "2",
    2: "5",
}

fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True)

offset = 0.2
i_jitter = 0
for psf_method in ["MCCD", "PSFEx"]:
    for leak_corr in range(2):
        suffix = "_leak_corr" if leak_corr == 1 else ""
        version = f"{versions[psf_method]['version']}{suffix}"
        label = f"{psf_method} {'w' if leak_corr == 1 else 'w/o'} leakage correction"

        # Load the rho and tau stats in PSFErrorFit object
        path_rho_stats = f"rho_stats_{version}_minsep={versions[psf_method]['minsep']}_maxsep={versions[psf_method]['maxsep']}_nbins={versions[psf_method]['nbins']}_npatch={versions[psf_method]['npatch']}.fits"
        path_tau_stats = f"tau_stats_{version}_minsep={versions[psf_method]['minsep']}_maxsep={versions[psf_method]['maxsep']}_nbins={versions[psf_method]['nbins']}_npatch={versions[psf_method]['npatch']}.fits"
        path_cov_tau = f"cov_tau_{version}_minsep={versions[psf_method]['minsep']}_maxsep={versions[psf_method]['maxsep']}_nbins={versions[psf_method]['nbins']}_npatch={versions[psf_method]['npatch']}_th.npy"

        psf_fitter.load_rho_stat(path_rho_stats)
        psf_fitter.load_tau_stat(path_tau_stats)

        cov_tau = np.load(f"{data_path}/{path_cov_tau}")
        # Get the samples
        _samples = samples[version]
        # Build the rho matrix and tau vector
        rho_matrix = psf_fitter.build_rho_matrix()
        tau_true = psf_fitter.build_tau_vec().reshape((3, 20))
        theta = psf_fitter.rho_stat_handler.rho_stats["theta"]

        # Get the tau_samples
        tau_samples = np.reshape(rho_matrix @ _samples.T, (3, -1, 10000))
        best_fit_tau = np.quantile(tau_samples, [0.16, 0.5, 0.84], axis=2)

        # Compute the jittered theta values for better visualisation
        theta_widths = np.diff(theta)
        theta_widths = np.append(theta_widths, theta_widths[-1])
        jitter_fraction = (i_jitter - 3 / 2) * offset
        jittered_theta = theta + jitter_fraction * theta_widths
        i_jitter += 1
        for i in range(3):
            y_plot = tau_true[i]
            yerr = np.sqrt(
                cov_tau[i * 20 : (i + 1) * 20, i * 20 : (i + 1) * 20].diagonal()
            )
            if i > 0:
                y_plot *= jittered_theta
                yerr *= jittered_theta

            axs[i].errorbar(
                jittered_theta,
                y_plot,
                yerr=yerr,
                fmt="o",
                label=label,
                capsize=2,
            )

for i in range(3):
    axs[i].set_xscale("log")
    axs[i].set_xlabel(r"$\vartheta$ (arcmin)")
    axs[i].set_title(titles[i])
    axs[i].set_ylabel(y_labels[i])
    axs[i].set_xlim(0.9, 300)

fig.canvas.draw()  # Update the figure to get the correct legend handles
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.05),  # centered under all axes
    ncol=2,
    frameon=False,
    fontsize=16,
)

for i in range(3):
    text_offset = axs[i].yaxis.get_offset_text().get_text()
    if text_offset:
        axs[i].set_ylabel(axs[i].get_ylabel() + " " + text_offset)
        axs[i].yaxis.get_offset_text().set_visible(False)

plt.savefig("Figures/MCCD_vs_PSFEx_tau_stats.png", bbox_inches="tight")
plt.savefig("Figures/MCCD_vs_PSFEx_tau_stats.pdf", bbox_inches="tight")
plt.show()

# %%
# Plot the levels of systematic error
plt.figure()

for psf_method in ["MCCD", "PSFEx"]:
    for leak_corr in range(2):
        suffix = "_leak_corr" if leak_corr == 1 else ""
        version = f"{versions[psf_method]['version']}{suffix}"
        label = f"{psf_method} {'w' if leak_corr else 'w/o'} leakage correction"
        _samples = samples[version]

        path_rho_stat = f"rho_stats_{version}_minsep={versions[psf_method]['minsep']}_maxsep={versions[psf_method]['maxsep']}_nbins={versions[psf_method]['nbins']}_npatch={versions[psf_method]['npatch']}.fits"
        path_tau_stat = f"tau_stats_{version}_minsep={versions[psf_method]['minsep']}_maxsep={versions[psf_method]['maxsep']}_nbins={versions[psf_method]['nbins']}_npatch={versions[psf_method]['npatch']}.fits"

        # load the rho and tau stats in PSFErrorFit object
        psf_fitter.load_rho_stat(path_rho_stat)
        psf_fitter.load_tau_stat(path_tau_stat)

        nbins = psf_fitter.rho_stat_handler._treecorr_config["nbins"]
        xi_psf_sys_samples = np.array([]).reshape(0, nbins)
        for i in range(len(_samples)):
            xi_psf_sys = psf_fitter.compute_xi_psf_sys(_samples[i])
            xi_psf_sys_samples = np.vstack((xi_psf_sys_samples, xi_psf_sys))

        theta = psf_fitter.rho_stat_handler.rho_stats["theta"]
        xi_psf_sys_quant = np.quantile(xi_psf_sys_samples, [0.16, 0.5, 0.84], axis=0)
        plt.plot(
            theta,
            xi_psf_sys_quant[1],
            label=label,
        )
        plt.plot(
            theta,
            xi_psf_sys_quant[0],
            linestyle="--",
            color=plt.gca().lines[-1].get_color(),
        )
        plt.plot(
            theta,
            xi_psf_sys_quant[2],
            linestyle="--",
            color=plt.gca().lines[-1].get_color(),
        )
        plt.fill_between(
            theta,
            xi_psf_sys_quant[0],
            xi_psf_sys_quant[2],
            alpha=0.1,
            color=plt.gca().lines[-1].get_color(),
        )

plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\vartheta$ (arcmin)")
plt.ylabel(r"$\xi^\mathrm{sys}_+(\vartheta)$")

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),  # centered under all axes
    ncol=2,
    frameon=False,
    fontsize=12,
)

plt.savefig("Figures/MCCD_vs_PSFEx_xi_sys.png", bbox_inches="tight")
plt.savefig("Figures/MCCD_vs_PSFEx_xi_sys.pdf", bbox_inches="tight")
plt.show()

# %%
# Create a table for PSF error model best fit and chi2
output_path_table = "Tables/MCCD_vs_PSFEx_best_fit_params.tex"

with open(output_path_table, "w") as f:
    f.write("\\begin{table*}[h!]\n")
    f.write("\\centering\n")
    f.write("\\begin{tabular}{lcccc}\n")
    f.write("\\hline\n")
    f.write("\\hline\n")
    f.write(" & $\\alpha$ & $\\beta$ & $\\eta$ & $\\chi^2 / \\nu$ \\\\\n")
    f.write("\\hline\n")

    for psf_method in ["MCCD", "PSFEx"]:
        for leak_corr in range(2):
            suffix = "_leak_corr" if leak_corr == 1 else ""
            version = f"{versions[psf_method]['version']}{suffix}"
            _samples = samples[version]

            path_rho_stat = f"rho_stats_{version}_minsep={versions[psf_method]['minsep']}_maxsep={versions[psf_method]['maxsep']}_nbins={versions[psf_method]['nbins']}_npatch={versions[psf_method]['npatch']}.fits"
            path_tau_stat = f"tau_stats_{version}_minsep={versions[psf_method]['minsep']}_maxsep={versions[psf_method]['maxsep']}_nbins={versions[psf_method]['nbins']}_npatch={versions[psf_method]['npatch']}.fits"
            path_cov_tau = f"cov_tau_{version}_minsep={versions[psf_method]['minsep']}_maxsep={versions[psf_method]['maxsep']}_nbins={versions[psf_method]['nbins']}_npatch={versions[psf_method]['npatch']}_th.npy"

            # load the rho and tau stats in PSFErrorFit object
            psf_fitter.load_rho_stat(path_rho_stat)
            psf_fitter.load_tau_stat(path_tau_stat)
            psf_fitter.load_covariance(f"{path_cov_tau}", cov_type="tau")

            best_fit_params, uncertainty = psf_fitter.get_mcmc_from_samples(_samples)
            chi2 = psf_fitter.eval_chi_square(best_fit_params[1, :])
            n_bins = psf_fitter.rho_stat_handler.rho_stats["theta"].shape[0]
            n_dof = 3 * n_bins - len(best_fit_params)
            f.write(
                rf"{psf_method} {'w' if leak_corr else 'w/o'} leakage correction & ${best_fit_params[1, 0]:.3f}^{{+{uncertainty[0, 0]:.3f}}}_{{-{uncertainty[1, 0]:.3f}}}$ & ${best_fit_params[1, 1]:.3f}^{{+{uncertainty[0, 1]:.3f}}}_{{-{uncertainty[1, 1]:.3f}}}$ & ${best_fit_params[1, 2]:.3f}^{{+{uncertainty[0, 2]:.3f}}}_{{-{uncertainty[1, 2]:.3f}}}$ & ${chi2:.0f}/{n_dof:.0f}$ \\"
                + "\n"
            )

    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write(
        "\\caption{Best fit parameters and reduced chi-squared for the PSF error models of PSFEx and MCCD using the raw corrected ellipticities.}\n"
    )
    f.write("\\label{tab:psf_error_model_best_fit}\n")
    f.write("\\end{table*}\n")

# %%
