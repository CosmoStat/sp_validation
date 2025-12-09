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

base_dir = "/home/guerrini/sp_validation/notebooks/cosmo_val/output/rho_tau_stats/"

# %%
path_star = "/n17data/UNIONS/WL/v1.4.x/unions_shapepipe_psf_2024_v1.4.a.fits"

cat_star = fits.getdata(path_star)

# To be checked
e1_star = cat_star['E1_STAR_HSM']
e2_star = cat_star['E2_STAR_HSM']
T_star = cat_star['SIGMA_STAR_HSM']**2
e1_psf = cat_star['E1_PSF_HSM']
e2_psf = cat_star['E2_PSF_HSM']
T_psf = cat_star['SIGMA_PSF_HSM']**2

# %%
histtype = 'step'
fig = plt.figure(figsize=(10, 3))
gs = GridSpec(1, 3, wspace=0.1)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharey=ax1)
ax3 = fig.add_subplot(gs[2], sharey=ax1)

ax1.hist(
    e1_star - e1_psf,
    bins=50,
    range=(-0.05, 0.05),
    density=True,
    alpha=0.7,
    histtype=histtype,
    label=r"$\delta e_1$",
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

ax3.hist(
    (T_star - T_psf) / T_psf,
    bins=50,
    range=(-0.1, 0.1),
    density=True,
    alpha=0.7,
    histtype=histtype,
    label=r"$\delta T / T$",
)

ax1.set_ylabel("Density")
ax1.minorticks_on()
ax2.minorticks_on()
ax3.minorticks_on()

ax2.yaxis.set_visible(False)
ax3.yaxis.set_visible(False)

ax1.set_title(r"$\delta e_1$", fontsize=16)
ax2.set_title(r"$\delta e_2$", fontsize=16)
ax3.set_title(r"$\delta T / T$", fontsize=16)

plt.tight_layout()

plt.savefig('./plots/residual_star_properties.pdf')

plt.show()

# %%
