# # Covmat mask analysis
#
# This notebook creates the plots to look at the ratio of the covaraiance matrices when applying the mask or not


import os

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("/home/guerrini/matplotlib_config/paper.mplstyle")

plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18

plt.rcParams["text.usetex"] = True
sns.set_palette("husl")

cat_dir = "/n17data/UNIONS/WL/v1.4.x/"
catalog_ver = "v1.4.6.3"
blind = "B"

nside = 8192
npix = hp.nside2npix(nside)

data_dir = "/n23data1/n06data/lgoh/scratch/UNIONS/cosmo_inference/data/"
curr_dir = os.getcwd()


# PLOT 2D MAP OF COVMAT masked vs unmasked RATIOS
nbins = 20
ndata = nbins * 2
full_ratio = np.zeros((ndata, ndata))

cov = np.loadtxt(data_dir + f"/covs/cov_SP_{catalog_ver}_{blind}.txt")
cov_masked = np.loadtxt(data_dir + f"/covs/cov_masked_SP_{catalog_ver}_{blind}.txt")

for i in range(ndata):
    for j in range(ndata):
        full_ratio[i][j] = cov_masked[i][j] / cov[i][j]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
extent = (0, ndata, ndata, 0)

vmin, vmax = np.percentile(full_ratio, [1, 99])

im3 = ax.imshow(full_ratio, cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent)

cbar = fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

ax.text(int(ndata / 4), ndata + 5, r"$\xi_+$", fontsize=15)
ax.text(3 * int(ndata / 4), ndata + 5, r"$\xi_-$", fontsize=15)
ax.text(-8, int(ndata / 4), r"$\xi_+$", fontsize=15, rotation=90)
ax.text(-8, 3 * int(ndata / 4), r"$\xi_-$", fontsize=15, rotation=90)
ax.set_xticks([0, 10, 20, 30, 40])
ax.set_yticks([0, 10, 20, 30, 40])
ax.set_yticklabels(["1'", "125'", "250'", "125'", "250'"])
ax.set_xticklabels(["1'", "125'", "250'", "125'", "250'"])
plt.axvline(x=int(ndata / 2), color="white", linewidth=1.0)
plt.axhline(y=int(ndata / 2), color="white", linewidth=1.0)

plt.savefig(
    f"./../../results/covmat_masked_unmasked_ratio_{catalog_ver}_{blind}.pdf",
    bbox_inches="tight",
)


theta = np.linspace(1, 250, 20)
plt.axhline(y=1, color="k", ls="--")
plt.plot(theta, np.diag(cov_masked)[:20] / np.diag(cov)[:20], label=r"$\xi_+$")
plt.plot(theta, np.diag(cov_masked)[20:] / np.diag(cov)[20:], label=r"$\xi_-$")

plt.xlabel(r"$\theta$ (arcmin)")
plt.ylabel("Cov masked / Cov unmasked")
plt.legend(fontsize=20)
plt.savefig(
    "./../../results/covmat_masked_unmasked_ratio_diag.pdf", bbox_inches="tight"
)
