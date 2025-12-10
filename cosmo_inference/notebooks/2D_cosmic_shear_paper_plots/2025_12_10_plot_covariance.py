# %%
import IPython

ipython = IPython.get_ipython()

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from getdist import plots, MCSamples

plt.style.use(
    "./matplotlib_config/paper.mplstyle"
)

plt.rcParams["text.usetex"] = True  

sns.set_palette("husl")

#Matplotlib inline if in jupyter
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# %%
cosmosis_dv = fits.open(
    "/home/guerrini/sp_validation/cosmo_inference/data/SP_v1.4.6_leak_corr_A_10_80/cosmosis_SP_v1.4.6_leak_corr_A_10_80.fits"
)

cov_matrix = cosmosis_dv["COVMAT"].data
# %%
def cov_to_corr(cov):
    """Convert covariance matrix to correlation matrix."""
    diag = np.sqrt(np.diag(cov))
    corr = cov / np.outer(diag, diag)
    return corr

# %%
labels_cov = [
    r"$\xi_+(\theta)$",
    r"$\xi_-(\theta)$",
    r"$\tau_0(\theta)$",
    r"$\tau_2(\theta)$",
]
# %%
plt.figure()

plt.imshow(cov_to_corr(cov_matrix), vmin=-1, vmax=1, cmap="coolwarm")
plt.colorbar(label="Correlation coefficient")

plt.xticks(
    ticks=[10, 30, 50, 70],
    labels=labels_cov,
)
plt.yticks(
    ticks=[10, 30, 50, 70],
    labels=labels_cov,
)

plt.savefig("./plots/cov_matrix_SP_v1.4.6_leak_corr.pdf")
plt.show()
# %%
