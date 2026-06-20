"""
Sky maps of shape parameters from a FITS catalog using Healpy + Skymapper.
Author: Fabian Hervas Peters and Sacha Guerrini
Date: May 30, 2025
"""

# %%
import IPython

ipython = IPython.get_ipython()

import astropy.units as u
import bornraytrace.lensing as lensing
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skymapper as skm
from astropy.coordinates import SkyCoord
from astropy.io import fits
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

plt.style.use("./matplotlib_config/paper.mplstyle")

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 12

sns.set_palette("husl", 18)
sns.color_palette("rocket", as_cmap=True)

# Matplotlib inline if in jupyter
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")


# %%
def weighted_mean_per_pixel(pix, weights, values, nside):
    """
    Compute the weighted mean of given values within each HEALPix pixel.

    Parameters
    ----------
    pix : ndarray
        HEALPix pixel indices corresponding to each source.
    weights : ndarray
        Weight values for each source.
    values : ndarray
        Quantity whose weighted mean is computed.
    nside : int
        HEALPix NSIDE parameter defining map resolution.

    Returns
    -------
    result : ndarray
        Weighted mean value per pixel, with zeros for empty pixels.
    """
    npix = hp.nside2npix(nside)
    w_sum = np.bincount(pix, weights=weights, minlength=npix)
    v_sum = np.bincount(pix, weights=weights * values, minlength=npix)
    valid = w_sum > 0
    result = np.zeros(npix)
    result[valid] = v_sum[valid] / w_sum[valid]
    return result


def plot_galactic_planes(sky_map):
    """
    Overlay galactic plane lines (b = 0°, ±10°) on a given sky map.

    Parameters
    ----------
    sky_map : skymapper.Map
        Skymapper map instance to draw the galactic plane lines on.
    """
    planes = {
        r"$b=0^\circ$": {"l": np.linspace(-20, 270, 1000), "b": 0, "style": "solid"},
        r"$b=+10^\circ$": {
            "l": np.linspace(-20, 270, 1000),
            "b": 10,
            "style": "dashed",
        },
        r"$b=-10^\circ$": {
            "l": np.linspace(-20, 270, 1000),
            "b": -10,
            "style": "dashed",
        },
    }

    for label, params in planes.items():
        gal = SkyCoord(l=params["l"] * u.deg, b=params["b"] * u.deg, frame="galactic")
        eq = gal.transform_to("icrs")
        sky_map.plot(
            eq.ra.deg,
            eq.dec.deg,
            color="gray",
            linestyle=params["style"],
            linewidth=1.2,
        )


def plot_healpix_map(
    pix_vals, nside, map_style, label, output_name, show=False, cmap=None, mask=None
):
    """
    Render and save a HEALPix map in an Albers equal-area projection.

    Parameters
    ----------
    pix_vals : ndarray
        Value per HEALPix pixel.
    nside : int
        HEALPix NSIDE parameter.
    map_style : str
        Map type identifier (e.g., 'e1', 'e2', 'N_eff').
    label : str
        Label for the colorbar and output.
    output_name : str
        Filename for the saved figure.
    """
    ra, dec = hp.pix2ang(nside, np.arange(len(pix_vals)), nest=False, lonlat=True)

    lon_0, lat_0, lat_1, lat_2 = 253, 40, 45, 49
    proj = skm.Albers(lon_0, lat_0, lat_1, lat_2)

    fig = plt.figure(figsize=(24, 12))
    ax = fig.add_subplot(111, aspect=3.0)
    ax.set_xlabel("Right Ascension [deg]", fontsize=20)
    ax.yaxis.set_ticks([])

    sky_map = skm.Map(proj, ax=ax)
    plot_galactic_planes(sky_map)

    _, vertices = skm.healpix.getCountAtLocations(
        ra, dec, nside=nside, per_area=True, return_vertices=True
    )

    if mask is None:
        mask = pix_vals == 0

    masked_vals = np.ma.array(pix_vals, mask=mask)
    if cmap is None:
        cmap = plt.cm.coolwarm.copy()
    cmap.set_under("lightgrey")

    vmin, vmax = np.percentile(masked_vals[~masked_vals.mask], [1, 99])

    mappable = sky_map.healpix(
        masked_vals, color_percentiles=[2, 98], cmap=cmap, vmin=vmin, vmax=vmax
    )

    divider = make_axes_locatable(ax)
    # 'size' can be a percent string or fraction, e.g. "5%" or 0.05
    cax = divider.append_axes("bottom", size="2%", pad=0.6)

    cbar = fig.colorbar(
        mappable, cax=cax, orientation="horizontal", fraction=0.05, pad=0.1
    )
    cbar.set_label(label, fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    # adjust height by changing cax position (if you want)
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.5])

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    formatter.set_scientific(True)
    cbar.ax.xaxis.set_major_formatter(formatter)
    cbar.ax.xaxis.get_offset_text().set_fontsize(14)

    sky_map.ylim(bottom=-0.2, top=1.38)
    sky_map.xlim(-1.29, 1.30)

    sky_map.fig.set_size_inches(12, 8)
    sky_map.grid(sep=15, lat_min=0, lat_max=360, lon_min=-179, lon_max=179)

    ax.text(
        0.39,
        1.09,
        "Declination [deg]",
        fontsize=18,
        transform=ax.transAxes,
        va="center",
    )

    for dec, positio in zip([30, 45, 60, 60, 45, 30], [12, 23, 36, 94, 107, 119]):
        ax.text(
            (positio - 7) / 120,
            1.02,
            f"+{dec}°",
            fontsize=12,
            transform=ax.transAxes,
            va="center",
        )

    ax.set_yticklabels([])
    ax.tick_params(axis="y", left=False, right=False)
    ax.tick_params(axis="x", top=False)

    plt.savefig(output_name)
    if show:
        plt.show()
    plt.close()


def plot_healpix_map_ax(
    pix_vals, nside, map_style, label, fig, ax, cmap=None, mask=None
):
    """
    Render and save a HEALPix map in an Albers equal-area projection.

    Parameters
    ----------
    pix_vals : ndarray
        Value per HEALPix pixel.
    nside : int
        HEALPix NSIDE parameter.
    map_style : str
        Map type identifier (e.g., 'e1', 'e2', 'N_eff').
    label : str
        Label for the colorbar and output.
    output_name : str
        Filename for the saved figure.
    """
    ra, dec = hp.pix2ang(nside, np.arange(len(pix_vals)), nest=False, lonlat=True)

    lon_0, lat_0, lat_1, lat_2 = 253, 40, 45, 49
    proj = skm.Albers(lon_0, lat_0, lat_1, lat_2)

    ax.set_xlabel("Right Ascension [deg]", fontsize=12)
    ax.yaxis.set_ticks([])

    sky_map = skm.Map(proj, ax=ax)
    plot_galactic_planes(sky_map)

    _, vertices = skm.healpix.getCountAtLocations(
        ra, dec, nside=nside, per_area=True, return_vertices=True
    )

    if mask is None:
        mask = pix_vals == 0

    masked_vals = np.ma.array(pix_vals, mask=mask)
    if cmap is None:
        cmap = plt.cm.coolwarm.copy()
    cmap.set_under("lightgrey")

    vmin, vmax = np.percentile(masked_vals[~masked_vals.mask], [1, 99])

    mappable = sky_map.healpix(
        masked_vals, color_percentiles=[2, 98], cmap=cmap, vmin=vmin, vmax=vmax
    )

    divider = make_axes_locatable(ax)
    # 'size' can be a percent string or fraction, e.g. "5%" or 0.05
    cax = divider.append_axes("bottom", size="5%", pad=0.5)

    cbar = fig.colorbar(
        mappable, cax=cax, orientation="horizontal", fraction=0.05, pad=0.1
    )
    cbar.set_label(label, fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    # adjust height by changing cax position (if you want)
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.7])

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    formatter.set_scientific(True)
    cbar.ax.xaxis.set_major_formatter(formatter)
    cbar.ax.xaxis.get_offset_text().set_fontsize(14)

    sky_map.ylim(bottom=-0.2, top=1.38)
    sky_map.xlim(-1.29, 1.30)

    sky_map.fig.set_size_inches(12, 8)
    sky_map.grid(sep=15, lat_min=0, lat_max=360, lon_min=-179, lon_max=179)

    ax.text(
        0.35,
        1.22,
        "Declination [deg]",
        fontsize=12,
        transform=ax.transAxes,
        va="center",
    )

    for dec, positio in zip([30, 45, 60, 60, 45, 30], [12, 23, 36, 94, 107, 119]):
        ax.text(
            (positio - 7) / 120,
            1.05,
            f"+{dec}°",
            fontsize=12,
            transform=ax.transAxes,
            va="center",
        )

    ax.set_yticklabels([])
    ax.tick_params(axis="y", left=False, right=False)
    ax.tick_params(axis="x", top=True)
    ax.tick_params(axis="both", which="major", labelsize=12, top=True)

    return ax


def get_kappa_kaiser_squire(gamma_1, gamma_2):
    gamma_map = gamma_1 + 1j * gamma_2
    nside = hp.npix2nside(len(gamma_map))
    kappa_map = lensing.shear2kappa(gamma_map, lmax=2 * nside)
    return -kappa_map.real, -kappa_map.imag


# %%
path_cat_gal = (
    "/n17data/UNIONS/WL/v1.4.x/v1.4.6.3/unions_shapepipe_cut_struc_2024_v1.4.6.3.fits"
)
path_cat_star = "/n17data/UNIONS/WL/v1.4.x/full_starcat-0000000.fits"

cat_gal = fits.getdata(path_cat_gal)
cat_star = fits.getdata(path_cat_star)

# %%
# Plot maps of e1, e2 and N_eff
nside = 1024

mask = cat_gal["Dec"] > 20  # thrwowing out cosmos
ra_gal, dec_gal = cat_gal["RA"][mask], cat_gal["DEC"][mask]
weights_gal = cat_gal["w_des"][mask]
e1_gal, e2_gal = cat_gal["e1"][mask], cat_gal["e2"][mask]

pix = hp.ang2pix(nside, ra_gal, dec_gal, lonlat=True)

pix_vals_e1 = weighted_mean_per_pixel(pix, weights_gal, e1_gal, nside)
pix_vals_e2 = weighted_mean_per_pixel(pix, weights_gal, e2_gal, nside)
kappa_E_map, kappa_B_map = get_kappa_kaiser_squire(pix_vals_e1, pix_vals_e2)
mask_footprint = pix_vals_e1 == 0

apply_smoothing = True
sigma_deg = 0.15

if apply_smoothing:
    sigma_rad = np.radians(sigma_deg)
    pix_vals_e1 = hp.smoothing(pix_vals_e1, sigma=sigma_rad)
    pix_vals_e2 = hp.smoothing(pix_vals_e2, sigma=sigma_rad)
    kappa_E_map = hp.smoothing(kappa_E_map, sigma=sigma_rad)
    kappa_B_map = hp.smoothing(kappa_B_map, sigma=sigma_rad)

output_path_e1 = "./plots/sky_map_e1_healpix.pdf"

plot_healpix_map(
    pix_vals_e1,
    nside,
    map_style="e1",
    label=r"$\gamma_1$",
    output_name=output_path_e1,
    show=True,
    mask=mask_footprint,
)

output_path_e2 = "./plots/sky_map_e2_healpix.pdf"

plot_healpix_map(
    pix_vals_e2,
    nside,
    map_style="e2",
    label=r"$\gamma_2$",
    output_name=output_path_e2,
    show=True,
    mask=mask_footprint,
)

output_path_kappaE = "./plots/sky_map_kappaE_healpix.pdf"

plot_healpix_map(
    kappa_E_map,
    nside,
    map_style="kappa_E",
    label=r"$\kappa_E$",
    output_name=output_path_kappaE,
    show=True,
    cmap=plt.cm.magma.copy(),
    mask=mask_footprint,
)

output_path_kappaB = "./plots/sky_map_kappaB_healpix.pdf"


plot_healpix_map(
    kappa_B_map,
    nside,
    map_style="kappa_B",
    label=r"$\kappa_B$",
    output_name=output_path_kappaB,
    show=True,
    cmap=plt.cm.magma.copy(),
    mask=mask_footprint,
)

# %%
plt.figure()

plt.hist(
    kappa_E_map[~mask_footprint],
    bins=1000,
    density=True,
    histtype="step",
    label=r"$\kappa_E$",
)

plt.show()

# %%
hp.mollview(
    kappa_E_map,
    title=r"$\kappa_E$ map",
    unit=r"$\kappa_E$",
    cmap="coolwarm",
    min=-0.02,
    max=0.02,
)
plt.show()

# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 9))

plot_healpix_map_ax(
    pix_vals_e1,
    nside,
    map_style="e1",
    label=r"$\gamma_1$",
    fig=fig,
    ax=ax1,
    mask=mask_footprint,
)

plot_healpix_map_ax(
    pix_vals_e2,
    nside,
    map_style="e2",
    label=r"$\gamma_2$",
    fig=fig,
    ax=ax2,
    mask=mask_footprint,
)

plot_healpix_map_ax(
    kappa_E_map,
    nside,
    map_style="kappa_E",
    label=r"$\kappa$",
    fig=fig,
    ax=ax3,
    cmap=plt.cm.magma.copy(),
    mask=mask_footprint,
)

plt.savefig("./plots/sky_maps_e1_e2_kappaE_healpix.pdf", dpi=600)
plt.show()


# %%
# Plot number density map

# Get effective number density map
nside = 256
pix = hp.ang2pix(nside, ra_gal, dec_gal, lonlat=True)

npix = hp.nside2npix(nside)
w_sum = np.bincount(pix, weights=weights_gal, minlength=npix)
w2_sum = np.bincount(pix, weights=weights_gal**2, minlength=npix)
valid = w_sum > 0

n_eff = np.zeros(npix)
n_eff[valid] = (w_sum[valid]) ** 2 / w2_sum[valid]
pix_area_arcmin2 = hp.nside2pixarea(nside, degrees=True) * (60**2)
print(pix_area_arcmin2)
pix_vals_n_eff = n_eff / pix_area_arcmin2

plot_healpix_map(
    pix_vals_n_eff,
    nside,
    map_style="N_eff",
    label=r"$n_\mathrm{eff}\ [\mathrm{arcmin}^{-2}]$",
    output_name="./plots/sky_map_n_eff_healpix.pdf",
    show=True,
    cmap=sns.cubehelix_palette(as_cmap=True),
    mask=~valid,
)

# %%
# Plot shape noise
unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)

shape_noise_map = np.zeros(npix)
shape_noise_map[unique_pix] = np.bincount(
    idx_rep,
    weights=weights_gal**2 * (e1_gal**2 + e2_gal**2),
)
mask_shape_noise = shape_noise_map == 0
shape_noise_map[~mask_shape_noise] = np.sqrt(
    1 / 2 * shape_noise_map[~mask_shape_noise] / w2_sum[~mask_shape_noise]
)

plot_healpix_map(
    shape_noise_map,
    nside,
    map_style="shape_noise",
    label=r"$\sigma_e$",
    output_name="./plots/sky_map_shape_noise_healpix.pdf",
    show=True,
    cmap=sns.color_palette("crest", as_cmap=True),
    mask=mask_shape_noise,
)
# %%
# GLASS mock maps
path_glass_mock = (
    "/n09data/guerrini/glass_mock/results/unions_glass_sim_00109_4096.fits"
)

cat_glass = fits.getdata(path_glass_mock)


nside = 1024

mask = cat_glass["Dec"] > 20  # thrwowing out cosmos
ra_glass, dec_glass = cat_glass["RA"][mask], cat_glass["DEC"][mask]
weights_glass = cat_glass["w"][mask]
e1_glass, e2_glass = cat_glass["e1"][mask], cat_glass["e2"][mask]

pix = hp.ang2pix(nside, ra_glass, dec_glass, lonlat=True)

pix_vals_e1 = weighted_mean_per_pixel(pix, weights_glass, e1_glass, nside)
pix_vals_e2 = weighted_mean_per_pixel(pix, weights_glass, e2_glass, nside)
kappa_E_map, kappa_B_map = get_kappa_kaiser_squire(pix_vals_e1, pix_vals_e2)
mask_footprint = pix_vals_e1 == 0

apply_smoothing = True
sigma_deg = 0.15

if apply_smoothing:
    sigma_rad = np.radians(sigma_deg)
    pix_vals_e1 = hp.smoothing(pix_vals_e1, sigma=sigma_rad)
    pix_vals_e2 = hp.smoothing(pix_vals_e2, sigma=sigma_rad)
    kappa_E_map = hp.smoothing(kappa_E_map, sigma=sigma_rad)
    kappa_B_map = hp.smoothing(kappa_B_map, sigma=sigma_rad)

output_path_e1 = "./plots/sky_map_e1_glass_healpix.pdf"

plot_healpix_map(
    pix_vals_e1,
    nside,
    map_style="e1",
    label=r"$\gamma_1$",
    output_name=output_path_e1,
    show=True,
    mask=mask_footprint,
)

output_path_e2 = "./plots/sky_map_e2_glass_healpix.pdf"

plot_healpix_map(
    pix_vals_e2,
    nside,
    map_style="e2",
    label=r"$\gamma_2$",
    output_name=output_path_e2,
    show=True,
    mask=mask_footprint,
)

output_path_kappaE = "./plots/sky_map_kappaE_glass_healpix.pdf"

plot_healpix_map(
    kappa_E_map,
    nside,
    map_style="kappa_E",
    label=r"$\kappa_E$",
    output_name=output_path_kappaE,
    show=True,
    cmap=plt.cm.magma.copy(),
    mask=mask_footprint,
)

output_path_kappaB = "./plots/sky_map_kappaB_glass_healpix.pdf"


plot_healpix_map(
    kappa_B_map,
    nside,
    map_style="kappa_B",
    label=r"$\kappa_B$",
    output_name=output_path_kappaB,
    show=True,
    cmap=plt.cm.magma.copy(),
    mask=mask_footprint,
)
# %%
# Maps of star/PSF information
ra_star, dec_star = cat_star["RA"], cat_star["DEC"]
e1_star, e2_star = cat_star["E1_STAR_HSM"], cat_star["E2_STAR_HSM"]
e1_psf, e2_psf = cat_star["E1_PSF_HSM"], cat_star["E2_PSF_HSM"]
t_star, t_psf = cat_star["SIGMA_STAR_HSM"] ** 2, cat_star["SIGMA_PSF_HSM"] ** 2
weights_star = np.ones_like(ra_star)

# %%
pix_star = hp.ang2pix(nside, ra_star, dec_star, lonlat=True)

# Map number density of stars
npix = hp.nside2npix(nside)
w_sum_star = np.bincount(pix_star, weights=weights_star, minlength=npix)
valid_star = w_sum_star > 0

# %%
pix_vals_n_eff = w_sum_star / pix_area_arcmin2

plot_healpix_map(
    pix_vals_n_eff,
    nside,
    map_style="N_eff_stars",
    label=r"$n_\mathrm{stars}\ [\mathrm{arcmin}^{-2}]$",
    output_name="./plots/sky_map_n_eff_stars_healpix.pdf",
    show=True,
    cmap=sns.cubehelix_palette(as_cmap=True),
    mask=~valid_star,
)

# %%
pix_vals_dela_e1 = weighted_mean_per_pixel(
    pix_star, weights_star, e1_star - e1_psf, nside
)
pix_vals_dela_e2 = weighted_mean_per_pixel(
    pix_star, weights_star, e2_star - e2_psf, nside
)

plot_healpix_map(
    pix_vals_dela_e1,
    nside,
    map_style="delta_e1_star_psf",
    label=r"$e_1^{\mathrm{star}} - e_1^{\mathrm{PSF}}$",
    output_name="./plots/sky_map_delta_e1_star_psf_healpix.pdf",
    show=True,
    mask=pix_vals_dela_e1 == 0,
)

plot_healpix_map(
    pix_vals_dela_e2,
    nside,
    map_style="delta_e2_star_psf",
    label=r"$e_2^{\mathrm{star}} - e_2^{\mathrm{PSF}}$",
    output_name="./plots/sky_map_delta_e2_star_psf_healpix.pdf",
    show=True,
    mask=pix_vals_dela_e2 == 0,
)
# %%
pix_vals_delta_t = weighted_mean_per_pixel(
    pix_star, weights_star, (t_star - t_psf) / t_star, nside
)

plot_healpix_map(
    pix_vals_delta_t,
    nside,
    map_style="delta_t_star_psf",
    label=r"$(T^{\mathrm{star}} - T^{\mathrm{PSF}}) / T^{\mathrm{star}}$",
    output_name="./plots/sky_map_delta_t_star_psf_healpix.pdf",
    show=True,
    mask=pix_vals_delta_t == 0,
)

# %%
