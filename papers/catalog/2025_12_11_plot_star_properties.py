# %%
import IPython

ipython = IPython.get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")


import matplotlib.pyplot as plt
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
path_cat_star = "/n17data/UNIONS/WL/v1.4.x//full_starcat-0000000.fits"

# %%
# Load catalog
cat_stars = fits.open(path_cat_star)[1].data

e1_star, e2_star = cat_stars["HSM_G1_STAR"], cat_stars["HSM_G2_STAR"]
e1_psf, e2_psf = cat_stars["HSM_G1_PSF"], cat_stars["HSM_G2_PSF"]
t_star, t_psf = cat_stars["HSM_T_STAR"], cat_stars["HSM_T_PSF"]
mag = cat_stars["MAG"]

# %%
# Plot brighter fatter effect
plt.figure()

plt.hist(mag, bins=50, histtype="step", density=True, color="C0")

plt.show()


# %%
