# %%
import IPython

ipython = IPython.get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import os
from pathlib import Path
import h5py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.io import fits

plt.style.use("./matplotlib_config/paper.mplstyle")

plt.rcParams["text.usetex"] = True

sns.set_palette("husl")

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# %%
path_cat_v146 = (
    "/n17data/UNIONS/WL/v1.4.x/v1.4.6/unions_shapepipe_cut_struc_2024_v1.4.6.fits"
)
path_cat_comprehensive = (
    "/n17data/UNIONS/WL/v1.4.x/unions_shapepipe_comprehensive_struc_2024_v1.X.c.hdf5"
)

# %%
# Star with plot with the sample v1.4.6
# Load catalog
cat_v146 = fits.open(path_cat_v146)[1].data

# %%
# Plot histogram of the response on the weak lensing sample
plt_diagonal = True
plt.figure()

plt.hist(
    cat_v146["R_g11"],
    bins=50,
    histtype="step",
    range=(-2.5, 2.5),
    density=True,
    color="C0",
    label=r"$R_{11}$",
)
plt.hist(
    cat_v146["R_g22"],
    bins=50,
    histtype="step",
    range=(-2.5, 2.5),
    density=True,
    color="C1",
    label=r"$R_{22}$",
)

if plt_diagonal:
    plt.hist(
        cat_v146["R_g12"],
        bins=50,
        histtype="step",
        range=(-2.5, 2.5),
        density=True,
        color="C2",
        label=r"$R_{12}$",
    )
    plt.hist(
        cat_v146["R_g21"],
        bins=50,
        histtype="step",
        range=(-2.5, 2.5),
        density=True,
        color="C3",
        label=r"$R_{21}$",
    )

plt.axvline(0, color="k", linestyle="--", alpha=0.7)

plt.axvline(
    np.mean(cat_v146["R_g11"]),
    color="C0",
    linestyle=":",
    alpha=0.7,
    label=rf"$\langle R_{11} \rangle = {np.mean(cat_v146['R_g11']):.2f}$",
)
plt.axvline(
    np.mean(cat_v146["R_g22"]),
    color="C1",
    linestyle=":",
    alpha=0.7,
    label=rf"$\langle R_{22} \rangle = {np.mean(cat_v146['R_g22']):.2f}$",
)

plt.ylabel("Density")
plt.xlabel("Response")
plt.minorticks_on()
plt.legend(fontsize=12)
plt.savefig("./plots/histogram_response_v146.pdf")
plt.show()

# %%
cat_comprehensive = h5py.File(path_cat_comprehensive, "r")
# %%
cat_comprehensive["data"]["RA"]
# %%
