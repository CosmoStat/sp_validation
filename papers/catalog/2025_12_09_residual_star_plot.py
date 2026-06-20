# %%
import IPython

ipython = IPython.get_ipython()

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

plt.style.use(
    "./matplotlib_config/paper.mplstyle"
)

plt.rcParams["text.usetex"] = True

sns.set_palette("husl", 18)

#Matplotlib inline if in jupyter
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

base_dir = "/home/guerrini/sp_validation/cosmo_val/output/rho_tau_stats/"

# %%
path_star = "/n17data/UNIONS/WL/v1.4.x/full_starcat-0000000.fits"

cat_star = fits.getdata(path_star)

# To be checked
e1_star = cat_star['E1_STAR_HSM']
e2_star = cat_star['E2_STAR_HSM']
T_star = cat_star['SIGMA_STAR_HSM']**2
e1_psf = cat_star['E1_PSF_HSM']
e2_psf = cat_star['E2_PSF_HSM']
T_psf = cat_star['SIGMA_PSF_HSM']**2
M_4_1_star = cat_star['M_4_STAR_1']
M_4_2_star = cat_star['M_4_STAR_2']
M_4_1_psf = cat_star['M_4_PSF_1']
M_4_2_psf = cat_star['M_4_PSF_2']

# %%
histtype = 'step'
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(3, 2, wspace=0.1, hspace=0.5)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharey=ax1)
ax3 = fig.add_subplot(gs[2, :])
ax4 = fig.add_subplot(gs[0, 1], sharey=ax1)
ax5 = fig.add_subplot(gs[1, 1], sharey=ax1)

res_e1 = e1_star - e1_psf
res_e2 = e2_star - e2_psf
res_T = (T_star - T_psf) / T_star
res_M4_1 = M_4_1_star - M_4_1_psf
res_M4_2 = M_4_2_star - M_4_2_psf

# Remove 0.01% of stars with M4 component larger than 1.
mask = (np.abs(res_M4_1) <= 1) & (np.abs(res_M4_2) <= 1)

ax1.hist(
    e1_star - e1_psf,
    bins=50,
    range=(-0.05, 0.05),
    density=True,
    alpha=0.7,
    histtype=histtype,
    label=r"$\delta e_1$",
)
ax1.text(
    0.05, 0.7,
    f"Mean: {np.mean(res_e1):.2e}\n"
    f"Std: {np.std(res_e1):.2e}",
    transform=ax1.transAxes,
    fontsize=18,
    color='k'
)

ax2.hist(
    e2_star - e2_psf,
    bins=50,
    range=(-0.05, 0.05),
    density=True,
    alpha=0.7,
    histtype=histtype,
    label=r"$\delta e_2$",
)

ax2.text(
    0.05, 0.7,
    f"Mean: {np.mean(res_e2):.2e}\n"
    f"Std: {np.std(res_e2):.2e}",
    transform=ax2.transAxes,
    fontsize=18,
    color='k'
)

ax3.hist(
    (T_star - T_psf) / T_psf,
    bins=50,
    range=(-0.1, 0.1),
    density=True,
    alpha=0.7,
    histtype=histtype,
    label=r"$\delta r_{\rm hlr} / r_{\rm hlr}$",
)
ax3.text(
    0.05, 0.7,
    f"Mean: {np.mean(res_T):.2e}\n"
    f"Std: {np.std(res_T):.2e}",
    transform=ax3.transAxes,
    fontsize=18,
    color='k'
)

ax4.hist(
    M_4_1_star - M_4_1_psf,
    bins=50,
    range=(-0.05, 0.05),
    density=True,
    alpha=0.7,
    histtype=histtype,
    label=r"$\delta M_{4,1}$",
)

ax4.text(
    0.05, 0.7,
    f"Mean: {np.mean(res_M4_1[mask]):.2e}\n"
    f"Std: {np.std(res_M4_1[mask]):.2e}",
    transform=ax4.transAxes,
    fontsize=18,
    color='k'
)

ax5.hist(
    M_4_2_star - M_4_2_psf,
    bins=50,
    range=(-0.05, 0.05),
    density=True,
    alpha=0.7,
    histtype=histtype,
    label=r"$\delta M_{4,2}$",
)

ax5.text(
    0.05, 0.7,
    f"Mean: {np.mean(res_M4_2[mask]):.2e}\n"
    f"Std: {np.std(res_M4_2[mask]):.2e}",
    transform=ax5.transAxes,
    fontsize=18,
    color='k'
)

ax1.set_ylabel("Density", fontsize=20)
ax1.minorticks_on()
ax2.set_ylabel("Density", fontsize=20)
ax2.minorticks_on()
ax3.set_ylabel("Density", fontsize=20)
ax3.minorticks_on()
ax4.minorticks_on()
ax5.minorticks_on()

ax1.tick_params(
    axis='both', which='both', labelsize=20
)
ax2.tick_params(
    axis='both', which='both', labelsize=20
)
ax3.tick_params(
    axis='both', which='both', labelsize=20
)
ax4.tick_params(
    axis='both', which='both', labelsize=20
)
ax5.tick_params(
    axis='both', which='both', labelsize=20
)

ax1.set_title(r"$\delta e_1$", fontsize=22)
ax2.set_title(r"$\delta e_2$", fontsize=22)
ax3.set_title(r"$\delta r_{\rm hlr} / r_{\rm hlr}$", fontsize=22)
ax4.set_title(r"$\delta M_{4,1}$", fontsize=22)
ax5.set_title(r"$\delta M_{4,2}$", fontsize=22)

plt.tight_layout()

plt.savefig('./plots/residual_star_properties.pdf')

plt.show()

# %%
