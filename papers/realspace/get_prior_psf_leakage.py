# # Covariance matrix and PSF leakage
#
# This notebook plots the combined covariance matrix, and samples and plots the 2D marginalised posteriors of the PSF leakage parameters $\alpha$ and $\beta$.


import os

if not os.path.exists("./Plots"):
    os.makedirs("./Plots")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits
from getdist import MCSamples, plots
from shear_psf_leakage.rho_tau_stat import PSFErrorFit, RhoStat, TauStat

# Use paper style and seaborn with husl palette
plt.style.use("/home/guerrini/matplotlib_config/paper.mplstyle")
# Set default palette - will be updated per plot as needed
sns.set_palette("husl")

g = plots.get_subplot_plotter(width_inch=30)
g.settings.axes_fontsize = 30
g.settings.axes_labelsize = 30
g.settings.alpha_filled_add = 0.7
g.settings.legend_fontsize = 25

ver = "v1.4.6.3"
blind = "B"


data_path = f"/home/guerrini/sp_validation/cosmo_inference/data/SP_{ver}_config/"

path_cosmo_val = "/home/guerrini/sp_validation/cosmo_val/output/"

roots = [f"SP_{ver}_{blind}", f"SP_{ver}_leak_corr_{blind}"]

labels = [f"SP_{ver}_{blind}", f"SP_{ver}_leak_corr_{blind}"]


data_vectors = []

for root in roots:
    data_vectors.append(
        fits.open(data_path + f"SP_{ver}_{blind}/cosmosis_{root}_masked.fits")
    )


def cov_to_corr(cov):
    """Convert a covariance matrix to a correlation matrix."""
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    corr[cov == 0] = 0
    return corr


# Print the covariance matrix for each root
for i, root in enumerate(roots):
    print(f"Covariance matrix for {labels[i]}:")
    cov = data_vectors[i]["COVMAT"].data

    n_bins = cov.shape[0] // 4

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(cov_to_corr(cov), vmin=-1, vmax=1, cmap="seismic")
    ax.set_aspect("equal")
    ax.set_yticks(np.array([10, 30, 50, 70]))
    ax.set_yticklabels(
        [
            r"$\xi_+(\vartheta)$",
            r"$\xi_-(\vartheta)$",
            r"$\tau_0(\vartheta)$",
            r"$\tau_2(\vartheta)$",
        ]
    )
    ax.set_xticks(np.array([10, 30, 50, 70]))
    ax.set_xticklabels(
        [
            r"$\xi_+(\vartheta)$",
            r"$\xi_-(\vartheta)$",
            r"$\tau_0(\vartheta)$",
            r"$\tau_2(\vartheta)$",
        ],
        rotation=45,
    )
    fig.colorbar(im, ax=ax)

    plt.savefig(f"./Plots/cov_matrix_{root}.png", bbox_inches="tight", dpi=300)


# Create dummy rho and tau stat handler.

# Inference of the xi_sys parameters
sep_units = "arcmin"
coord_units = "degrees"
theta_min = 1.0
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


# Create a PSFErrorFit instance
psf_fitter = PSFErrorFit(
    rho_stats_handler,
    tau_stats_handler,
    path_cosmo_val + "rho_tau_stats/",
    use_eta=False,
)

g = plots.get_subplot_plotter(width_inch=30)

g.settings.axes_fontsize = 30
g.settings.axes_labelsize = 30
g.settings.alpha_filled_add = 0.7
g.settings.legend_fontsize = 40

chains = []

# Load rho-, tau-statistics, and cov_tau from the data_vector
for i, root in enumerate(roots):
    print("Sampling PSF parameters for ", labels[i])
    path_rho = f"rho_stats_{root}.fits"
    path_tau = f"tau_stats_{root}.fits"
    path_cov_rho = f"cov_rho_{root}.npy"
    path_cov_tau = f"cov_tau_{root}_th.npy"
    psf_fitter.load_rho_stat(path_rho)
    psf_fitter.load_tau_stat(path_tau)
    psf_fitter.load_covariance(path_cov_rho, cov_type="rho")
    psf_fitter.load_covariance(path_cov_tau, cov_type="tau")
    samples_lq, _, _ = psf_fitter.get_least_squares_params_samples(
        npatch=None, apply_debias=False
    )

    samples_gd = MCSamples(
        samples=samples_lq, names=[r"\alpha", r"\beta"], labels=[r"\alpha", r"\beta"]
    )

    chains.append(samples_gd)

g.triangle_plot(
    chains,
    filled=True,
    legend_labels=labels,
    legend_loc="upper right",
)

# plt.savefig(f"./Plots/psf_leakage_params.png", bbox_inches='tight', dpi=300)
