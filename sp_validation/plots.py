"""
  
:Name: plots.py

:Description: This script contains methods for plots.

:Author: Martin Kilbinger

:Date: 2021

:Package: sp_validation

"""

import numpy as np
import matplotlib.pylab as plt

from sp_validation import util
from sp_validation import io
from sp_validation import basic
from sp_validation.plot_style import *


def plot_spatial_density(ra, dec, title, x_label, y_label, cbar_label, out_path, n_grid=1000, verbose=False):
    """Plot Spatial Density

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

    plt.figure(figsize=(30, 30))

    plt.hexbin(ra, dec, gridsize=n_grid)

    cbar = plt.colorbar()
    cbar.set_label(cbar_label, rotation=270, labelpad=40)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(out_path)
    plt.show()


def plot_histograms(xs, labels, title, x_label, y_label, x_range, n_bin, out_path,
                    weights=None,
                    colors=None, linestyles=None,
                    vline_x=None, vline_lab=None):
    """Plot Histograms

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
    """

    if weights is None:
        weights = [np.ones_like(x) for x in xs]
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
    if linestyles is None:
        linestyles = ['-'] * len(labels)

    plt.figure(figsize=(15,10))

    # Histogramsh
    for x, w, label, color, linestyle in zip(xs, weights, labels, colors, linestyles):
        plt.hist(x, n_bin, weights=w, range=x_range, histtype='step',
                 color=color, linestyle=linestyle,
                 linewidth=1, density=True, label=label)
 
    # Horizontal lines
    if vline_x:
        ylim = plt.ylim()
        for x, lab in zip(vline_x, vline_lab):
            plt.vlines(x=x, ymax=ylim[1], ymin=ylim[0], linestyles='--', colors='k')
            plt.text(x*1.5, ylim[1]*0.95, lab)
        plt.ylim(ylim)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(out_path)


def plot_data_1d(x, y, yerr, title, xlabel, ylabel, out_path, xlog=False, ylog=False, labels=None,
                 colors=None, linestyles=None, eb_linestyles=None):
    """plot points errors

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
    label : array of string, optional, default=None
        plot labels, no labels if None
    color : array of string, optional, default=None
        line colors, matplotlib default colors if None
    linestyle : array of string, optional, default=None
        linestyle indicators, '-' if None
    eb_linestyle : array of string, optional, default=None
        errorbar linestyle indicators, '-' if None
    """

    if labels is None:
        labels = [''] * len(x)
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
    if linestyles is None:
        linestyles = ['-'] * len(x)
    if eb_linestyles is None:
        eb_linestyles = ['-'] * len(x)

    plt.figure(figsize=(15,10))

    for i in range(len(x)):
        eb = plt.errorbar(x[i], y[i], yerr=yerr[i], label=labels[i], color=colors[i], linestyle=linestyles[i],
                          marker='o', markerfacecolor='none', capsize=4)
        eb[-1][0].set_linestyle(eb_linestyles[i])

    plt.hlines(y=0, xmin=plt.xlim()[0], xmax=plt.xlim()[1], linestyles='dashed')

    if xlog == True:
        plt.xscale('log')
        plt.xticks([2, 5, 10, 20, 50, 100, 200], labels=['2', '5', '10', '20', '50', '100', '200'])
    if ylog == True:
        plt.yscale('log')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.savefig(out_path)
