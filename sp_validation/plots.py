"""PLOTS.

:Name: plots.py

:Description: This script contains methods for plots.

:Author: Martin Kilbinger

:Date: 2021

:Package: sp_validation

"""

import numpy as np
import matplotlib.pylab as plt

from lenspack.geometry.projections.gnom import radec2xy

from sp_validation import util
from sp_validation import io
from sp_validation import basic
from sp_validation.plot_style import *


def figure(figsize=(30, 30)):
    """Figure.

    Create figure

    Parameters
    ----------
    figsize : tuple, optional
        figure size, default is (30, 30)

    """
    plt.figure(figsize=figsize, facecolor='none')


def savefig(fname):
    """Save Figure.

    Save figure to file.

    Parameters
    ----------
    fname : str
        output file name

    """
    plt.savefig(fname, facecolor='w', bbox_inches='tight')


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
    figure(figsize=(30, 30))

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

    savefig(out_path)


def plot_histograms(
    xs,
    labels,
    title,
    x_label,
    y_label,
    x_range,
    n_bin,
    out_path,
    weights=None,
    colors=None,
    linestyles=None,
    vline_x=None,
    vline_lab=None,
    density=True,
):
    """Plot Histograms.

    Plot one or more 1D distributions.

    Parameters
    ----------
    xs : array of float
        array of values, each of which to plot the distribution
    labels : array of string
        plot labels
    title : string
        plot title
    x_label, y_label : string
        x-/y-axis label
    n_bin : int
        number of histogram bins
    out_path : string
        output file path
    weights : array of float, optional, default=None
        weights
    colors : array of string, optional, default=None
        plot colors
    linestyles : array of string, optional, default=None
        line styles
    vline_x : array of float, optional, default=None
        x-values of vertical lines if not None
    vline_lab : array of string, optional, default=None
        labels of vertical lines if not None
    density : bool, optional, default=True
        (normalised) density histogram if True
    """
    if weights is None:
        weights = [np.ones_like(x) for x in xs]
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
    if linestyles is None:
        linestyles = ['-'] * len(labels)

    figure(figsize=(15, 10))

    # Histogramsh
    for x, w, label, color, linestyle in zip(
            xs, weights, labels, colors, linestyles
    ):
        plt.hist(x, n_bin, weights=w, range=x_range, histtype='step',
                 color=color, linestyle=linestyle,
                 linewidth=1, density=density, label=label)

    # Horizontal lines
    if vline_x:
        ylim = plt.ylim()
        for x, lab in zip(vline_x, vline_lab):
            plt.vlines(
                x=x,
                ymax=ylim[1],
                ymin=ylim[0],
                linestyles='--',
                colors='k'
            )
            plt.text(x * 1.5, ylim[1] * 0.95, lab)
        plt.ylim(ylim)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    savefig(out_path)


def plot_data_1d(
    x,
    y,
    yerr,
    title,
    xlabel,
    ylabel,
    out_path,
    xlog=False,
    ylog=False,
    labels=None,
    colors=None,
    linestyles=None,
    eb_linestyles=None,
    ylim=None,
):
    """Plot points errors.

    Plot one-dimensional data points with errorbars.

    Parameters
    ----------
    x, y, yerr : array of array of float
        data
    title, xlabel, ylabel : string
        title and labels
    out_path : string
        output file path
    xlog, ylog : bool, optional, default=False
        logscale on x, y if True
    labels : array of string, optional, default=None
        plot labels, no labels if None
    color : array of string, optional, default=None
        line colors, matplotlib default colors if None
    linestyle : array of string, optional, default=None
        linestyle indicators, '-' if None
    eb_linestyle : array of string, optional, default=None
        errorbar linestyle indicators, '-' if None
    ylim : array(float, 2), optional, default=None
        y-axis limits, automatic if None
    """
    if labels is None:
        labels = [''] * len(x)
        do_legend = False
    else:
        do_legend = True
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
    if linestyles is None:
        linestyles = ['-'] * len(x)
    if eb_linestyles is None:
        eb_linestyles = ['-'] * len(x)

    figure(figsize=(15, 10))

    for i in range(len(x)):
        if np.isnan(yerr[i]).all():
            eb = plt.plot(
                x[i],
                y[i],
                label=labels[i],
                color=colors[i],
                linestyle=linestyles[i],
            )
        else:
            eb = plt.errorbar(
                x[i],
                y[i],
                yerr=yerr[i],
                label=labels[i],
                color=colors[i],
                linestyle=linestyles[i],
                marker='o',
                markerfacecolor='none',
                capsize=4,
            )
            eb[-1][0].set_linestyle(eb_linestyles[i])

    plt.hlines(
        y=0,
        xmin=plt.xlim()[0],
        xmax=plt.xlim()[1],
        linestyles='dashed',
    )

    if xlog:
        plt.xscale('log')
        plt.xticks(
            [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
            labels=[
                '0.1',
                '0.2',
                '0.5',
                '1',
                '2',
                '5',
                '10',
                '20',
                '50',
                '100',
                '200',
                '500',
            ]
        )
    if ylog:
        plt.yscale('log')

    if ylim:
        plt.ylim(ylim)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if do_legend:
        plt.legend()

    savefig(out_path)


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
        # print(loc[i], lab)
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
    clusters=None,
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
    clusters :
        dictionary of cluster information, optional, default=None
    """
    figure(figsize=(10, 10))

    # plot image
    plt.imshow(m)

    # save image limits
    xlim = plt.xlim()
    ylim = plt.ylim()

    # Set colorbar
    if not vlim:
        vlim = plt.gci().get_clim()
    else:
        plt.gci().set_clim(vlim)
    plt.colorbar()

    # Transform axis labels to ra, dec
    ra_min, ra_max = ra.min(), ra.max()
    ra_mean = np.mean(ra)
    dec_min, dec_max = dec.min(), dec.max()
    dec_mean = np.mean(dec)

    loc, labels = plt.xticks()
    loc_ra, labels_ra = get_ticks(loc, Nx, ra_min, ra_max)
    plt.xticks(loc_ra, labels=labels_ra)

    loc, labels = plt.yticks()
    loc_dec, labels_dec = get_ticks(loc, Ny, dec_min, dec_max)
    plt.yticks(loc_dec, labels=labels_dec)

    # plot grid
    grid_lines_ra = []
    grid_lines_dec = []
    n_per_line = 200

    # create lines of constant ra and varying dec, and vice versa

    # extend beyond projected image limits, to avoid image edges without grid
    # lines
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

    mean_x = (min_x + max_x) / 2
    mean_y = (min_y + max_y) / 2

    for grid_line_ra, grid_line_dec in zip(grid_lines_ra, grid_lines_dec):
        x, y = radec2xy(ra_mean, dec_mean, grid_line_ra, grid_line_dec)
        xx = (x + mean_x - min_x) / (max_x - min_x) * Nx
        yy = (y + mean_y - min_y) / (max_y - min_y) * Ny
        plt.plot(xx, yy, 'w:', linewidth=0.5)

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

    savefig(out_path)

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
    vlim : array(2) of float
        map limits
    """
    figure(figsize=(10, 10))

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

    savefig(output_path)

    return vlim
