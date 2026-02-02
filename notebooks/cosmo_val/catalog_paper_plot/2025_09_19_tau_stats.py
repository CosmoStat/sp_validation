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

base_dir = "/home/guerrini/sp_validation/notebooks/cosmo_val/output/rho_tau_stats/"
# %%
versions = ["SP_v1.4.6", "SP_v1.4.6_leak_corr"]
labels = ["Fiducial sample", "Fiducial sample w/ leakage corr."]
colors = [f"C{i}" for i in range(len(versions))]

tau_stats = []
cov_taus = []
samples = []

for ver in versions:
    tau_stats.append(fits.getdata(f"{base_dir}/tau_stats_{ver}.fits"))
    cov_taus.append(np.load(f"{base_dir}/cov_tau_{ver}_th.npy"))
    samples.append(np.load(f"{base_dir}/samples_{ver}.npy"))

num_bins = tau_stats[0]['theta'].shape[0]

# %%
e_obs = "e^\mathrm{obs}"
e_psf = "e^\mathrm{PSF}"
delta_e_psf = "\delta e^\mathrm{PSF}"
delta_T_psf = "\delta T^\mathrm{PSF}"

titles = [
    rf"$\langle {e_obs} {e_psf} \rangle$",
    rf"$\langle {e_obs} {delta_e_psf} \rangle$",
    rf"$\langle {e_obs} {delta_T_psf} \rangle$",
]

y_labels = [
    r"$\tau_0(\vartheta)$",
    r"$\vartheta \, \tau_1(\vartheta)$",
    r"$\vartheta \, \tau_2(\vartheta)$",
]

dict_index_tau = {
    0: '0',
    1: '2',
    2: '5',
}

fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True)

axs = axs.flatten()

for i in range(3):
    for j in range(len(versions)):
        y_plot = np.copy(tau_stats[j][f'tau_{dict_index_tau[i]}_p'])
        yerr_plot = np.copy(np.sqrt(cov_taus[j][i*num_bins:(i+1)*num_bins, i*num_bins:(i+1)*num_bins].diagonal()))
        if i>0:
            y_plot *= tau_stats[j]['theta']
            yerr_plot *= tau_stats[j]['theta']

        axs[i].errorbar(
            tau_stats[j]['theta'],
            y_plot,
            yerr=yerr_plot,
            fmt='o',
            linestyle='-',
            label=labels[j],
            color=colors[j],
            capsize=2,
        )
    axs[i].set_xlim(1.0, 250.0)
    axs[i].set_xscale('log')
    axs[i].set_xlabel(r"$\vartheta$ [arcmin]")
    axs[i].set_ylabel(y_labels[i])
    axs[i].axhline(0, color='black', linestyle='--', linewidth=1)

handles, labels = axs[1].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.05),   # centered under all axes
    ncol=2,
    frameon=False,
    fontsize=16
)

plt.tight_layout()

plt.savefig("./plots/tau_stats.pdf", bbox_inches='tight')
plt.show()
# %%
