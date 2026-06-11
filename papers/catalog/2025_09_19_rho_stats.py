# %%
import IPython

ipython = IPython.get_ipython()

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use(
    "./matplotlib_config/paper.mplstyle"
)

plt.rcParams["text.usetex"] = True

sns.set_palette("colorblind")

#Matplotlib inline if in jupyter
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

base_dir = "/home/guerrini/sp_validation/cosmo_val/output/rho_tau_stats/"

# %%
version = "SP_v1.4.6_leak_corr"
label = r"Reserved stars \texttt{PSFex} model"
color = "C1"

rho_stats = fits.getdata(f"{base_dir}/rho_stats_{version}.fits")
cov_rho = np.load(f"{base_dir}/cov_rho_{version}.npy")
n_bins = rho_stats['theta'].shape[0]

version_xi = "SP_v1.4.6_no_leak_corr_A"
data_vector_path = f"/home/guerrini/sp_validation/cosmo_inference/data/{version_xi}/cosmosis_{version_xi}.fits"
data_vector = fits.open(data_vector_path)
xi_plus = data_vector['XI_PLUS'].data['VALUE']
cov_xi = data_vector['COVMAT'].data

snr = np.sqrt(xi_plus @ np.linalg.inv(cov_xi[:n_bins, :n_bins]) @ xi_plus)
print(f"SNR of xi+: {snr:.2f}")

a = 0.02

threshold_rho_134 = 0.5 * xi_plus / snr
threshold_rho_25 = 0.5 * xi_plus / snr / a

plot_requirements = True

# %%
e_psf = "e^\mathrm{PSF}"
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

for i in range(6):
    axs[i].errorbar(
        rho_stats['theta'],
        np.abs(rho_stats[f'rho_{i}_p']),
        yerr=np.sqrt(cov_rho[i*n_bins:(i+1)*n_bins, i*n_bins:(i+1)*n_bins].diagonal()),
        fmt='o',
        linestyle='-',
        label=label,
        capsize=2,
        color=color
    )

    if plot_requirements:
        if i in [1, 3, 4]:
            axs[i].fill_between(
                rho_stats['theta'],
                0,
                y2=threshold_rho_134,
                color='gray',
                alpha=0.3,
            )
        elif i in [2, 5]:
            axs[i].fill_between(
                rho_stats['theta'],
                0,
                y2=threshold_rho_25,
                color='gray',
                alpha=0.3,
            )

    axs[i].set_xscale('log')
    axs[i].set_yscale('log')
    axs[i].set_ylabel(rf"$|\rho_{i}(\vartheta)|$")
    if i in [3, 4, 5]:
        axs[i].set_xlabel(r"$\vartheta$ [arcmin]")
    axs[i].set_title(titles[i])
    axs[i].set_xlim(0.9, 300)

handles, labels = axs[4].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.05),   # centered under all axes
    ncol=2,
    frameon=False,
    fontsize=16
)


plt.tight_layout()

plt.savefig("./plots/rho_stats.pdf", bbox_inches='tight')
plt.show()
# %%
rho_stats['theta']
# %%
