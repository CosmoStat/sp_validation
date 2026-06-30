# %%
import IPython

ipython = IPython.get_ipython()

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from astropy.io import fits
from getdist import MCSamples, plots
from shear_psf_leakage.rho_tau_stat import PSFErrorFit, RhoStat, TauStat

plt.style.use("./matplotlib_config/paper.mplstyle")

plt.rcParams["text.usetex"] = True

sns.set_palette("colorblind")

# Matplotlib inline if in jupyter
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

base_dir = "/home/guerrini/sp_validation/cosmo_val/output/rho_tau_stats/"

# %%
versions = ["SP_v1.4.6.3_A", "SP_v1.4.6.3_leak_corr_A"]
labels = ["Fiducial sample", "Fiducial sample w/ leakage corr."]
colors = [f"C{i}" for i in range(len(versions))]

tau_stats = []
cov_taus = []
samples = []

for ver in versions:
    tau_stats.append(fits.getdata(f"{base_dir}/tau_stats_{ver}.fits"))
    cov_taus.append(np.load(f"{base_dir}/cov_tau_{ver}_th.npy"))
    samples.append(np.load(f"{base_dir}/samples_{ver}.npy"))

num_bins = tau_stats[0]["theta"].shape[0]

# %%
# Create chains object for getdist
chains = []
for i, ver in enumerate(versions):
    chains.append(
        MCSamples(
            samples=samples[i],
            names=[r"\alpha", r"\beta", r"\eta"],
            labels=[r"\alpha", r"\beta", r"\eta"],
            label=labels[i],
        )
    )

# %%
g = plots.get_subplot_plotter()

markers = {
    r"\alpha": 0.0,
    r"\beta": 1.0,
    r"\eta": 1.0,
}
line_args = [{"color": colors[i]} for i in range(len(versions))]
g.triangle_plot(
    chains,
    filled=True,
    legend_labels=labels,
    legend_loc="upper right",
    line_args=line_args,
    contour_colors=colors,
    markers=markers,
)
g.export("./plots/psf_leakage_params.pdf")

# %%
# Inference of the xi_sys parameters
sep_units = "arcmin"
coord_units = "degrees"
theta_min = 1.0
theta_max = 250.0
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

psf_fitter = PSFErrorFit(rho_stats_handler, tau_stats_handler, base_dir)

# %%
markers = ["o", "h", "x", "s", "D", "^"]

plt.figure()

offset = 0.02

for idx, (ver, color, marker, sample) in enumerate(
    zip(versions, colors, markers, samples)
):
    print("Plotting verion:", ver)
    psf_fitter.load_rho_stat("rho_stats_" + ver + ".fits")
    quant = 0.84
    quantiles = [1 - quant, quant]
    xi_psf_sys_samples = np.array([]).reshape(0, num_bins)
    for i in range(len(sample)):
        xi_psf_sys = psf_fitter.compute_xi_psf_sys(sample[i])
        xi_psf_sys_samples = np.vstack((xi_psf_sys_samples, xi_psf_sys))

    xi_psf_sys_mean = np.mean(xi_psf_sys_samples, axis=0)
    xi_psf_sys_quant = np.quantile(xi_psf_sys_samples, quantiles, axis=0)
    theta = psf_fitter.rho_stat_handler.rho_stats["theta"]

    if os.path.exists(
        base_dir
        + f"/../{ver}_xi_minsep={theta_min}_maxsep={theta_max}_nbins={nbins}_npatch=1.txt"
    ):
        xip = np.loadtxt(
            base_dir
            + f"/../{ver}_xi_minsep={theta_min}_maxsep={theta_max}_nbins={nbins}_npatch=1.txt",
            usecols=3,
        )[:nbins]  ## This will need modifications as the pipeline evolved

    ratio_mean = xi_psf_sys_mean / xip
    ratio_quant = xi_psf_sys_quant / xip

    jittered_theta = theta * (1 + idx * offset)

    plt.errorbar(
        jittered_theta,
        ratio_mean,
        yerr=[ratio_mean - ratio_quant[0], ratio_quant[1] - ratio_mean],
        fmt=marker,
        color=color,
        label=labels[idx],
        capsize=5,
    )

threshold = 0.10
plt.fill_between(
    [theta_min, theta_max],
    -threshold,
    threshold,
    color="black",
    alpha=0.1,
    label=rf"${threshold * 100}\%$ threshold",
)
plt.plot([theta_min, theta_max], [threshold, threshold], color="black", ls="--")
plt.plot([theta_min, theta_max], [-threshold, -threshold], color="black", ls="--")

plt.xscale("log")
plt.xlabel(r"$\vartheta\ [\mathrm{arcmin}]$")
plt.ylabel(r"$[\xi_\mathrm{sys}/\xi_+](\vartheta)$")
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.legend(fontsize=12)
plt.savefig("./plots/xi_sys_over_xi_plus.pdf")

plt.show()

# %%
versions = ["SP_v1.4.5_leak_corr", "SP_v1.4.6_leak_corr", "SP_v1.4.8_leak_corr"]
labels = ["Including small objects", "Fiducial sample", "Additional masking"]
colors = ["C0", "C1", "C2"]

tau_stats = []
cov_taus = []
samples = []

for ver in versions:
    tau_stats.append(fits.getdata(f"{base_dir}/tau_stats_{ver}.fits"))
    cov_taus.append(np.load(f"{base_dir}/cov_tau_{ver}_th.npy"))
    samples.append(np.load(f"{base_dir}/samples_{ver}.npy"))

num_bins = tau_stats[0]["theta"].shape[0]

# %%
# Create chains object for getdist
chains = []
for i, ver in enumerate(versions):
    chains.append(
        MCSamples(
            samples=samples[i],
            names=[r"\alpha", r"\beta", r"\eta"],
            labels=[r"\alpha", r"\beta", r"\eta"],
            label=labels[i],
        )
    )

# %%
g = plots.get_subplot_plotter()

markers = {
    r"\alpha": 0.0,
    r"\beta": 1.0,
    r"\eta": 1.0,
}
line_args = [{"color": colors[i]} for i in range(len(versions))]
g.triangle_plot(
    chains,
    filled=True,
    legend_labels=labels,
    legend_loc="upper right",
    line_args=line_args,
    contour_colors=colors,
    markers=markers,
)
g.export("./plots/psf_leakage_params.pdf")

# %%
# Inference of the xi_sys parameters
sep_units = "arcmin"
coord_units = "degrees"
theta_min = 1.0
theta_max = 250.0
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

psf_fitter = PSFErrorFit(rho_stats_handler, tau_stats_handler, base_dir)

# %%
markers = ["o", "h", "x"]

plt.figure()

offset = 0.02

for idx, (ver, color, marker, sample) in enumerate(
    zip(versions, colors, markers, samples)
):
    print("Plotting verion:", ver)
    psf_fitter.load_rho_stat("rho_stats_" + ver + ".fits")
    quant = 0.84
    quantiles = [1 - quant, quant]
    xi_psf_sys_samples = np.array([]).reshape(0, num_bins)
    for i in range(len(sample)):
        xi_psf_sys = psf_fitter.compute_xi_psf_sys(sample[i])
        xi_psf_sys_samples = np.vstack((xi_psf_sys_samples, xi_psf_sys))

    xi_psf_sys_mean = np.mean(xi_psf_sys_samples, axis=0)
    xi_psf_sys_quant = np.quantile(xi_psf_sys_samples, quantiles, axis=0)
    theta = psf_fitter.rho_stat_handler.rho_stats["theta"]

    if os.path.exists(
        base_dir
        + f"/../{ver}_xi_minsep={theta_min}_maxsep={theta_max}_nbins={nbins}_npatch=1.txt"
    ):
        xip = np.loadtxt(
            base_dir
            + f"/../{ver}_xi_minsep={theta_min}_maxsep={theta_max}_nbins={nbins}_npatch=1.txt",
            usecols=3,
        )[:nbins]  ## This will need modifications as the pipeline evolved

    ratio_mean = xi_psf_sys_mean / xip
    ratio_quant = xi_psf_sys_quant / xip

    jittered_theta = theta * (1 + idx * offset)

    plt.errorbar(
        jittered_theta,
        ratio_mean,
        yerr=[ratio_mean - ratio_quant[0], ratio_quant[1] - ratio_mean],
        fmt=marker,
        color=color,
        label=labels[idx],
        capsize=5,
    )

threshold = 0.10
plt.fill_between(
    [theta_min, theta_max],
    -threshold,
    threshold,
    color="black",
    alpha=0.1,
    label=r"$5\%$ threshold",
)
plt.plot([theta_min, theta_max], [threshold, threshold], color="black", ls="--")
plt.plot([theta_min, theta_max], [-threshold, -threshold], color="black", ls="--")

plt.xscale("log")
plt.xlabel(r"$\vartheta\ [\mathrm{arcmin}]$")
plt.ylabel(r"$[\xi_\mathrm{sys}/\xi_+](\vartheta)$")
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.legend()
plt.savefig("./plots/xi_sys_over_xi_plus_leak_corr.pdf")

plt.show()

# %%
