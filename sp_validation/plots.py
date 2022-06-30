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


def savefig(fname, dpi=100):
    """Save Figure.

    Save figure to file.

    Parameters
    ----------
    fname : str
        output file name

    """
    plt.savefig(fname, facecolor='w', bbox_inches='tight', dpi=dpi)


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
    log=False,
    labels=None,
    colors=None,
    linestyles=None,
    eb_linestyles=None,
    linewidths=None,
    xlim=None,
    ylim=None
):
    """Plot Data 1D.

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
    labels : list, optional, default=None
        plot labels, no labels if None
    color : list, optional, default=None
        line colors, matplotlib default colors if None
    linestyle : list, optional, default=None
        linestyle indicators, '-' if None
    linewidths : list
        line widths, default is `2`
    eb_linestyle : array of string, optional, default=None
        errorbar linestyle indicators, '-' if None
    xlim : array(float, 2), optional, default=None
        x-axis limits, automatic if None
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
    if linewidths is None:
        linewidths = [2] * len(x)

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

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    plt.hlines(
        y=0,
        xmin=plt.xlim()[0],
        xmax=plt.xlim()[1],
        linestyles='dashed'
    )

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
    figure(figsize=(10, 10))

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
        print('MKDEBUG 1', xx)
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

    savefig(out_path, dpi=dpi)
    
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


def set_labels(p_dp, order, mix):
    """Set Label.

    Set labels for plot of 2D fit

    Parameters
    ----------
    d_dp : dict
        values with uncertainties of fit parameters
    order : str
        linear ('lin') or quadratic ('quad') model
    mix : bool
        mixing of components if True

    Returns
    -------
    dict :
        label strings

    """
    # Affine parameters
    label = {
        'A': (
            f'$a_{{11}}={p_dp["a11"]: .2ugL}$'
            + '\n' + f'$c_1={p_dp["c1"]: .2ugL}$'
        ),
        'D': (
            f'$a_{{22}}={p_dp["a22"]: .2ugL}$'
            + '\n' + f'$c_2={p_dp["c2"]: .2ugL}$'
        )
    }
    if order == 'quad':
        # Add quadratic parameters
        label['A'] = f'$q_{{111}}={p_dp["q111"]: .2ugL}$' + '\n' + label['A']
        label['D'] = f'$q_{{222}}={p_dp["q222"]: .2ugL}$' + '\n' + label['D']
    if mix:
        # Add mixture parameters
        label['B'] = f'$a_{{12}}={p_dp["a12"]: .2ugL}$'
        label['C'] = f'$a_{{12}}={p_dp["a12"]: .2ugL}$'
        if order == 'quad':
            label['B'] = (
                f'$q_{{211}}={p_dp["q211"]: .2ugL}$'
                + '\n'
                + f'$q_{{212}}={p_dp["q212"]: .2ugL}$'
                + '\n'
                + label['B']
            )
            label['C'] = (
                f'$q_{{122}}={p_dp["q122"]: .2ugL}$'
                + '\n'
                + f'$q_{{112}}={p_dp["q112"]: .2ugL}$'
                + '\n'
                + label['C']
            )

    return label


def plot_corr_2d(
    x,
    y,
    weights,
    res,
    p_dp,
    n_bin,
    order,
    mix,
    xlabel_arr,
    ylabel_arr,
    y_ground_truth=None,
    title=None,
    colors=None,
    out_path=None,
):
    """Plot Corr 2D.

    Plot 2D correlation data and fits.

    Parameters
    ----------
    x : array(double)
        input x value
    y : array(m) of double
        input y arrays
    weights  : array of double, optional, default=None
        weights of x points
    res : class lmfit.MinimizerResult
        results of the minization
    n_bin : double, optional, default=30
        number of points onto which data are binned
    order : str
        order of fit
    mix : bool
        mixing of components if True
    xlabel_arr, ylabel_arr : list of str
        x-and y-axis labels
    y_ground_truth : 2D np.array, optional
        ground truth model values (y1, y2) for plotting, default is `None`
    title : string, optional, default=''
        plot title
    colors : array(m) of string, optional, default=None
        line colors
    out_path : str, optional, default=None
        output file path, if not given, plot is not saved to file

    """
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

    # Compute binned data for pretty plotting.
    x_bin, y_bin, err_bin = util.compute_bins_func_2d(
        x,
        y,
        n_bin,
        mix,
        weights=weights
    )

    # Initialise mosaic figure
    figure_mosaic = """
    AB
    CD
    """
    fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(15, 15))

    # Get best-fit model on 2D binned grid
    y_model_all = np.zeros(shape=(2, n_bin, n_bin))
    y_model_all[0], y_model_all[1] = util.func_bias_2d_full(
        res.params,
        x_bin[0],
        x_bin[1],
        order=order,
        mix=mix
    )
    # Compute means and standard deviations
    y_model_mean = np.zeros(shape=(2, n_bin))
    y_model_upper = np.zeros(shape=(2, n_bin))
    y_model_lower = np.zeros(shape=(2, n_bin))
    for comp, ax in zip((0, 1), (1, 0)):
        y_model_mean[comp] = y_model_all[comp].mean(axis=ax)
        std = y_model_all[comp].std(axis=ax)
        y_model_upper[comp] = y_model_mean[comp] + std
        y_model_lower[comp] = y_model_mean[comp] - std

    # Set up quantities to plot in each panel
    xb = {}
    yd = {}
    ym = {}
    ymu = {}
    yml = {}
    xgt = {}
    ygt = {}
    dy = {}
    col = {}
    xl = {}
    yl = {}

    # Set component for each panel.
    # x: 0 in A, B; 1 in C, D
    # y: 0 in A, C; 1 in B, D
    panel_comp_x = {}
    panel_comp_y = {}
    for p in 'A', 'B':
        panel_comp_x[p] = 0
    for p in 'C', 'D':
        panel_comp_x[p] = 1
    for p in 'A', 'C':
        panel_comp_y[p] = 0
    for p in 'B', 'D':
        panel_comp_y[p] = 1

    # Assign quantities to plot with corresponding components
    for p in axes:
        xb[p] = x_bin[panel_comp_x[p]]
        xl[p] = xlabel_arr[panel_comp_x[p]]

        ym[p] = y_model_mean[panel_comp_y[p]]
        ymu[p] = y_model_upper[panel_comp_y[p]]
        yml[p] = y_model_lower[panel_comp_y[p]]
        yl[p] = ylabel_arr[panel_comp_y[p]]
        yd[p] = y_bin[panel_comp_y[p]][panel_comp_x[p]]
        dy[p] = err_bin[panel_comp_y[p]][panel_comp_x[p]]
        col[p] = colors[panel_comp_y[p]]

        if y_ground_truth:
            xgt[p] = x[panel_comp_x[p]]
            ygt[p] = y_ground_truth[panel_comp_y[p]]

    # Set plot labels to parameter best-fit + std
    label = set_labels(p_dp, order, mix)

    # Loop over panels
    for p in axes:

        # No off-diagonal plots if no mixing
        if not mix and p in ['B', 'C']:
            continue

        # Plot best-fit mean and mean +/- std
        axes[p].plot(xb[p], ym[p], c=col[p], label=label[p])
        axes[p].fill_between(
            xb[p],
            ymu[p],
            yml[p],
            color=col[p],
            interpolate=True,
            alpha=0.3
        )

        # Plot ground-truth model if provided
        if y_ground_truth:
            axes[p].plot(xgt[p], ygt[p], '.', c='k', markersize=0.4)

        # Plot binned data with error bars
        axes[p].errorbar(xb[p], yd[p], yerr=dy[p], c=col[p], fmt='.')

        # Set labels
        axes[p].set_xlabel(xl[p])
        axes[p].set_ylabel(yl[p])
        axes[p].legend()

    # Finish figure
    fig.suptitle(title)
    plt.tight_layout()

    # Save figure
    if out_path:
        plt.savefig(f'{out_path}.png', bbox_inches='tight')


def plot_bar_spin(par, s_ground_truth, output_path=None):
    """Plot Bar Spin.

    Create bar plot of spin coefficients.

    Parameters
    ----------
    par : dict of ufloat
        parameter values and standard deviations
    s_ground_truth : dict, optional
        ground truth parameter, for plotting, default is `None`
    output_path : str, optional
        plot output file if not `None` (default)

    """
    # Shift of real and imaginary components
    dx = 0.4

    # Colors of rea and imaginary components
    colors = {'real': 'b', 'imaginary': 'g'}

    # Set data for bar plot
    x = []
    y = []
    dy = []
    col = []
    s = set()
    for key in par:

        z = key[0]
        spin = int(key[1:])
        s.add(spin)
        if z == 'x':
            x.append(spin - dx)
            col.append(colors['real'])
        else:
            x.append(spin + dx)
            col.append(colors['imaginary'])

        y.append(par[key].nominal_value)
        dy.append(par[key].std_dev)

    fig, ax = plt.subplots()

    bars = ax.bar(
        x,
        y,
        yerr=dy,
        align='center',
        alpha=0.5,
        ecolor='black',
        capsize=8,
        width=0.8,
        color=col,
    )
    xlim = ax.get_xlim()
    ax.plot(xlim, [0, 0], 'k-')
    ax.set_ylabel(r"$z_s = x_s + \mathrm{i} y_s$")
    xl = list(s)
    ax.set_xticks(xl)
    ax.set_xlabel('$s$')

    for comp in colors:
        if colors[comp] in col:
            ax.bar(x, y, width=0, color=colors[comp], label=comp)
    ax.legend()

    x = []
    y = []
    if s_ground_truth:
        for key in s_ground_truth:
            z = key[0]
            spin = int(key[1:])
            if z == 'x':
                x.append(spin - dx)
            else:
                x.append(spin + dx)
            y.append(s_ground_truth[key])
        ax.plot(x, y, 'ro', markerfacecolor='none')

    plt.tight_layout()

    # Save the figure
    if output_path:
        plt.savefig(output_path)
