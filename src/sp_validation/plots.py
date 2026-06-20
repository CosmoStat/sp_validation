"""PLOTS.

:Name: plots.py

:Description: This script contains methods for plots.

:Author: Martin Kilbinger


"""

from collections import Counter

import healpy as hp
import healsparse as hsp
import matplotlib.pylab as plt
import matplotlib.scale as mscale
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
import numpy as np
import skyproj
from astropy import units as u
from astropy.coordinates import SkyCoord
from cs_util import plots
from lenspack.geometry.projections.gnom import radec2xy

from sp_validation.plot_style import *


class SquareRootScale(mscale.ScaleBase):
    """
    ScaleBase class for generating square root scale.

    Usage example: axis.set_yscale('squareroot')

    """

    name = "squareroot"

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self, axis, **kwargs)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(0.0, vmin), vmax

    class SquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a) ** 0.5

        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()

    class InvertedSquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            return np.array(a) ** 2

        def inverted(self):
            return SquareRootScale.SquareRootTransform()

    def get_transform(self):
        return self.SquareRootTransform()


mscale.register_scale(SquareRootScale)


def plot_spatial_density(
    ra, dec, title, x_label, y_label, cbar_label, out_path, n_grid=1000, verbose=False
):
    """Plot Spatial Density.

    Plot spatial density distribution of objects.

    Parameters
    ----------
    ra, dec : array of float
        coordinates
    title : string
        plot title
    x_label, y_label : string
        x-/y-axis label
    cbar_label : string
        color bar label
    out_path : string
        output file path
    n_grid : int, optional, default=1000
        number of hex grid points
    verbose : bool, optional, default=False
        verbose output if True
    """
    plots.figure(figsize=(30, 30))

    if max(ra) > 360:
        ra_plot = ra - 360
    else:
        ra_plot = ra
    plt.hexbin(ra_plot, dec, gridsize=n_grid)

    cbar = plt.colorbar()
    cbar.set_label(cbar_label, rotation=270, labelpad=40)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plots.savefig(out_path)


def get_ticks(loc, N, new_min, new_max):
    """Get ticks.

    Return formatted axis ticks for plots.

    Parameters
    ----------
    loc : array of floats
        original tick locations
    N : number of pixels (in origina coordinates)
    new_min : float
        new coordinate minimum
    new_max : float
        new coordinate maximum

    Returns
    -------
    loc_new : array of floats
        new tick locations
    labels_new : array of strings
        new tick labels
    """
    loc_new = []
    labels_new = []

    for i in range(1, len(loc) - 1):
        lab = loc[i] / N * (new_max - new_min) + new_min
        loc_new.append(loc[i])
        labels_new.append(f"{lab:.1f}")

    return loc_new, labels_new


def plot_map(
    m,
    ra,
    dec,
    min_x,
    max_x,
    min_y,
    max_y,
    Nx,
    Ny,
    title,
    out_path,
    vlim=None,
    grid=True,
    clusters=None,
    map_cut_coords=None,
    dpi=100,
    colorbar=True,
):
    """Plot Map.

    Plots 2D map.

    Parameters
    ----------
    m : 2D array of float
        map
    ra, dec : array of float
        coordinates, for axis ticks
    title : string
        plot title
    out_path : string
        output file path
    vlim : array(2) of float, optional, default=None
        limits of map values, if not given compute from map
    grid : bool, optional
        if `True` (default) plot grid lines
    clusters :
        dictionary of cluster information, optional, default=None
    """
    plots.figure(figsize=(10, 10))

    # plot image
    plt.imshow(m)

    # Transform axis labels to ra, dec
    ra_min, ra_max = ra.min(), ra.max()
    ra_mean = np.mean(ra)
    dec_min, dec_max = dec.min(), dec.max()
    dec_mean = np.mean(dec)

    # save image limits
    xlim = plt.xlim()
    ylim = plt.ylim()

    # Set colorbar
    if not vlim:
        vlim = plt.gci().get_clim()
    else:
        plt.gci().set_clim(vlim)
    if colorbar:
        plt.colorbar()

    loc, labels = plt.xticks()
    loc_ra, labels_ra = get_ticks(loc, Nx, ra_min, ra_max)
    plt.xticks(loc_ra, labels=labels_ra)

    loc, labels = plt.yticks()
    loc_dec, labels_dec = get_ticks(loc, Ny, dec_min, dec_max)
    plt.yticks(loc_dec, labels=labels_dec)

    mean_x = (min_x + max_x) / 2
    mean_y = (min_y + max_y) / 2

    # plot grid
    if grid:
        grid_lines_ra = []
        grid_lines_dec = []
        n_per_line = 200

        # create lines of constant ra and varying dec, and vice versa

        # extend beyond projected image limits, to avoid image edges
        # without grid lines
        d = 2
        gl_ra = np.linspace(ra_min - d, ra_max + d, num=n_per_line)
        gl_dec = np.linspace(dec_min - d, dec_max + d, num=n_per_line)
        ra_list = np.arange(np.floor(ra_min - d), np.ceil(ra_max + d))
        dec_list = np.arange(np.floor(dec_min - d), np.ceil(dec_max + d))
        for ra in ra_list:
            grid_lines_ra.append([ra] * n_per_line)
            grid_lines_dec.append(gl_dec)
        for dec in dec_list:
            grid_lines_dec.append([dec] * n_per_line)
            grid_lines_ra.append(gl_ra)

        for grid_line_ra, grid_line_dec in zip(grid_lines_ra, grid_lines_dec):
            x, y = radec2xy(ra_mean, dec_mean, grid_line_ra, grid_line_dec)
            xx = (x + mean_x - min_x) / (max_x - min_x) * Nx
            yy = (y + mean_y - min_y) / (max_y - min_y) * Ny
            plt.plot(xx, yy, "w:", linewidth=0.5)

    # cut out if required
    if map_cut_coords:
        x_cut, y_cut = radec2xy(
            ra_mean,
            dec_mean,
            [map_cut_coords[0], map_cut_coords[1]],
            [map_cut_coords[2], map_cut_coords[3]],
        )
        xx = (x_cut + mean_x - min_x) / (max_x - min_x) * Nx
        yy = (y_cut + mean_y - min_y) / (max_y - min_y) * Ny
        xlim = plt.xlim(xx)
        ylim = plt.ylim(yy)

    # mark cluster positions
    if clusters:
        x_cluster = (clusters["x"] + mean_x - min_x) / (max_x - min_x) * Nx
        y_cluster = (clusters["y"] + mean_y - min_y) / (max_y - min_y) * Ny
        dy = 0.02
        plt.plot(
            x_cluster,
            y_cluster,
            "ro",
            mfc="none",
            markeredgewidth=0.9,
            markersize=12,
        )

    # go back to image limits
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.xlabel("R.A. [deg]")
    plt.ylabel("Dec [deg]")

    plt.title(title)

    plots.savefig(out_path)

    return vlim


def plot_binned_one(
    ax,
    quantity,
    bin_edges_x,
    bin_edges_y,
    vmin=None,
    vmax=None,
    title=None,
    xlabel=None,
    ylabel=None,
):

    # Note: transpose R slice to match (y, x) shape required by pcolormesh
    pcm = ax.pcolormesh(
        bin_edges_x, bin_edges_y, quantity, vmin=vmin, vmax=vmax, shading="auto"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    plt.colorbar(pcm, ax=ax)


def plot_binned(
    quantities,
    key,
    bin_edges_x,
    bin_edges_y,
    title_base,
    vmin=None,
    vmax=None,
    xlabel=None,
    ylabel=None,
):

    len_shape = len(quantities[key].shape)

    fig_size = 2 * len_shape
    fig = plt.figure(figsize=(fig_size, fig_size))

    if len_shape == 2:
        ax = plt.subplot2grid((1, 1), (0, 0))
        plot_binned_one(
            ax,
            quantities[key].T,
            bin_edges_x,
            bin_edges_y,
            vmin=vmin,
            vmax=vmax,
            title=title_base,
            xlabel=xlabel,
            ylabel=ylabel,
        )

    elif len_shape == 4:
        for idx in (0, 1):
            for jdx in (0, 1):
                ax = plt.subplot2grid((2, 2), (idx, jdx))

                if vmin is not None and vmax is not None:
                    if idx == jdx:
                        my_vmin = vmin["diag"]
                        my_vmax = vmax["diag"]
                    else:
                        my_vmin = vmin["offdiag"]
                        my_vmax = vmax["offdiag"]
                else:
                    my_vmin = None
                    my_vmax = None

                plot_binned_one(
                    ax,
                    quantities[key][:, :, idx, jdx].T,
                    bin_edges_x,
                    bin_edges_y,
                    vmin=my_vmin,
                    vmax=my_vmax,
                    title=f"${title_base}_{{{idx + 1}{jdx + 1}}}$",
                    xlabel=xlabel,
                    ylabel=ylabel,
                )

    plt.tight_layout()
    plots.savefig(f"{key}_binned.png")


def hsp_map_logical_or(maps, verbose=False):
    """
    Hsp Map Logical Or.

    Logical AND of HealSparseMaps.

    """
    if verbose:
        print("Combine all maps...")

    # Ensure consistency in coverage and data type
    nside_coverage = maps[0].nside_coverage
    nside_sparse = maps[0].nside_sparse
    dtype = maps[0].dtype

    for m in maps:
        # MKDEBUG TODO: Change nside if possible
        if m.nside_coverage != nside_coverage:
            raise ValueError(
                f"Coverage nside={m.nside_coverage} does not match {nside_coverage}"
            )
        if m.dtype != dtype:
            raise ValueError(f"Data type {m.dtype} does not match {dtype}")

    # Create an empty HealSparse map
    map_comb = hsp.HealSparseMap.make_empty(nside_coverage, nside_sparse, dtype=dtype)
    for idx, m in enumerate(maps):
        map_comb |= m

        if verbose:
            valid_pixels = map_comb.valid_pixels
            n_tot = np.sum(valid_pixels)
            n_true = np.count_nonzero(valid_pixels)
            n_false = n_tot - n_true
            print(
                f"after map {idx}: frac_true={n_true / n_tot:g}, frac_false={n_false / n_tot:g}"
            )

    return map_comb


def plot_area_mask(ra, dec, zoom, mask=None):
    """Plot Area Mask.

    Create sky plot of objects.

    Parameters
    ----------
    ra : list
        R.A. coordinates
    dec : list
        Dec. coordinates
    zoom : TBD
    mask: TBD, optional

    """
    if mask is None:
        mask == np.ones_like(ra)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))
    axes[0].hexbin(ra[mask], dec[mask], gridsize=100)
    axes[1].hexbin(ra[mask & zoom], dec[mask & zoom], gridsize=200)
    for idx in (0, 1):
        axes[idx].set_xlabel("R.A. [deg]")
        axes[idx].set_ylabel("Dec [deg]")


def sky_plots(dat, masks, labels, zoom_ra, zoom_dec):
    """Sky Plots.

    Plot sky regions with different masks.

    Parameters
    ----------
    masks : list
        masks to be applied
    labels : dict
        labels for masks
    zoom_ra : list
        min and max R.A. for zoom-in plot
    zoom_dec : list
        min and max Dec. for zoom-in plot

    """
    ra = dat["RA"][:]
    dec = dat["Dec"][:]

    zoom_ra = (room_ra[0] < dat["RA"]) & (dat["RA"] < zoom_ra[1])
    zoom_dec = (zoom_dec[0] < dat["Dec"]) & (dat["Dec"] < zoom_dec[1])
    zoom = zoom_ra & zoom_dec

    # No mask
    plot_area_mask(ra, dec, zoom)

    # SExtractor and SP flags
    m_flags = masks[labels["FLAGS"]]._mask & masks[labels["IMAFLAGS_ISO"]]._mask
    plot_area_mask(ra, dec, zoom, mask=m_flags)

    # Overlap regions
    m_over = masks[labels["overlap"]]._mask & m_flags
    plot_area_mask(ra, dec, zoom, mask=m_over)

    # Coverage mask
    m_point = masks[labels["npoint3"]]._mask & m_over
    plot_area_mask(ra, dec, zoom, mask=m_point)

    # Maximask
    m_maxi = masks[labels["1024_Maximask"]]._mask & m_point
    plot_area_mask(ra, dec, zoom, mask=m_maxi)

    m_comb = mask_combined._mask
    plot_area_mask(ra, dec, zoom, mask=m_comb)

    m_man = m_maxi & masks[labels["8_Manual"]]._mask
    plot_area_mask(ra, dec, zoom, mask=m_man)

    m_halos = (
        m_maxi
        & masks[labels["1_Faint_star_halos"]]._mask
        & masks[labels["2_Bright_star_halos"]]._mask
    )
    plot_area_mask(ra, dec, zoom, mask=m_halos)
