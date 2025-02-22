"""PLOTS.

:Name: plots.py

:Description: This script contains methods for plots.

:Author: Martin Kilbinger


"""

import numpy as np
import matplotlib.pylab as plt

from lenspack.geometry.projections.gnom import radec2xy

from sp_validation import util
from sp_validation import io
from sp_validation import basic
from sp_validation.plot_style import *

from cs_util import plots


def plot_spatial_density(
    ra,
    dec,
    title,
    x_label,
    y_label,
    cbar_label,
    out_path,
    n_grid=1000,
    verbose=False
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
        labels_new.append(f'{lab:.1f}')

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
    colorbar=True
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
            plt.plot(xx, yy, 'w:', linewidth=0.5)

    # cut out if required
    if map_cut_coords:
        x_cut, y_cut = radec2xy(
            ra_mean,
            dec_mean,
            [map_cut_coords[0], map_cut_coords[1]],
            [map_cut_coords[2], map_cut_coords[3]]
        )
        xx = (x_cut + mean_x - min_x) / (max_x - min_x) * Nx
        yy = (y_cut + mean_y - min_y) / (max_y - min_y) * Ny
        xlim = plt.xlim(xx)
        ylim = plt.ylim(yy)

    # mark cluster positions
    if clusters:
        x_cluster = (clusters['x'] + mean_x - min_x) / (max_x - min_x) * Nx
        y_cluster = (clusters['y'] + mean_y - min_y) / (max_y - min_y) * Ny
        dy = 0.02
        plt.plot(
            x_cluster,
            y_cluster,
            'ro',
            mfc='none',
            markeredgewidth=0.9,
            markersize=12,
        )

    # go back to image limits
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.xlabel('R.A. [deg]')
    plt.ylabel('Dec [deg]')

    plt.title(title)

    plots.savefig(out_path)

    return vlim


def plot_map_stacked(kappa, title, radius, output_path, vlim=None):
    """Plot Map Stacked.

    Plot stacked convergence map.

    Parameters
    ----------
    kappa : image
        map values
    title : string
        plot title
    output_path : string
        figure output file path

    vlim : array(2) of float, optional, default=None
        map limits; min and max of kappa if not given

    Returns
    -------
    array(2) of float
        map limits

    """
    plots.figure(figsize=(10, 10))

    # plot image
    plt.imshow(kappa)

    # set colorbar
    if not vlim:
        vlim = plt.gci().get_clim()
    else:
        plt.gci().set_clim(vlim)
    plt.colorbar()

    npix = kappa.shape[0]

    # mark center
    plt.plot(npix / 2 - 1, npix / 2 - 1, '+')

    # axes ticks
    n_ticks = 4
    loc = np.arange(0, npix + npix / n_ticks, step=npix / n_ticks)
    lab = np.round(
        np.arange(
            -radius,
            radius + radius * 2 / n_ticks,
            step=radius * 2 / n_ticks,
        ),
        1
    )
    plt.xticks(loc, labels=lab)
    plt.yticks(loc, labels=lab)

    plt.xlabel(r'separation $R$ [Mpc]')
    plt.ylabel(r'separation $R$ [Mpc]')

    plt.title(title)

    plots.savefig(output_path)

    return vlim
