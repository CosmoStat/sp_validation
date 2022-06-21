"""
  
:Name: correlation.py

:Description: This script contains methods to deal with
auto- and cross-correlations.

:Author: Martin Kilbinger

:Date: 2021

:Package: sp_validation

"""

import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat

from tqdm import tqdm

from sp_validation.basic import bootstrap_weighted_average

from sp_validation.plot_style import *
from sp_validation.io import print_stats

import treecorr


def affine_corr(
    x,
    y,
    xlabel,
    ylabel,
    mlabel=None,
    weights=None,
    n_bin=30,
    out_path=None,
    title='',
    colors=None,
    stats_file=None,
    verbose=False,
    parallel=False,
    n_jobs=-1,
    seed=None,
    rng=None,
):
    """Affine Corr

    Computes and plots affine correlation of y(n) as function of x.

    Parameters
    -----------
    x: array(double)
        input x value
    y: array(m) of double
        input y arrays
    xlabel, ylabel : string
        x-and y-axis labels
    mlabel : string(m), optional, default=None
        label for slope in the plot legend
    weights : array of double, optional, default=None
        weights of x points
    n_bin : double, optional, default=30
        number of points onto which data are binned
    out_path : string, optional, default=None
        output file path, if not given, plot is not saved to file
    title : string, optional, default=''
        plot title
    colors : array(m) of string, optional, default=None
        line colors
    stats_file : filehandler, optional, default=None
        output file for statistics
    verbose : bool, optional, default=False
        verbose output if True
    parallel: bool
        If True, use parallel computing. [Default: False]
    n_jobs: int
        Number of jobs to run in parallel. [Default: -1]
    seed: int
        Seed to initialize the randoms. [Default: None]
    rng: numpy.random.RandomState
        Random generator. [Default: None]
    """

    def get_seed(rng):
        return rng.randint(low=0, high=2**30, size=1)

    def lin(x, a, b):
        return a * x + b

    # Init randoms
    if isinstance(rng, np.random.RandomState):
        master_rng = rng
    else:
        master_rng = np.random.RandomState(seed)

    if mlabel is None:
        mlabel = np.ones('m')

    if weights is None:
        weights = np.ones_like(y[0])

    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

    size_all = len(y[0])
    for j in range(1, len(y)):
        if len(y[j]) != size_all:
            raise IndexError
            (
                f'Size {len(y[j])} of input #{i} different from  size {size_all} of input #0'
            )
    size_bin = int(size_all / n_bin)
    diff_size = size_all - size_bin

    # Prepare arrays for binned data
    x_arg_sort = np.argsort(x)
    x_bin = []
    y_bin = []
    err_bin = []

    for j in range(len(y)):
        y_bin.append([])
        err_bin.append([])

    # Bin data
    for i in tqdm(range(n_bin), total=n_bin, disable=not verbose):
        if i < diff_size:
            bin_size_tmp = size_bin + 1
            starter = 0
        else:
            bin_size_tmp = size_bin
            starter = diff_size
        ind = x_arg_sort[starter + i * bin_size_tmp : starter + (i + 1) * bin_size_tmp]

        x_bin.append(np.mean(x[ind]))

        for j in range(len(y)):
            r_jk = bootstrap_weighted_average(
                y[j][ind],
                weights[ind],
                seed=get_seed(master_rng),
                remove_size=0.2,
                n_realization=50,
                parallel=parallel,
                n_job=-1,
            )
            y_bin[j].append(r_jk[0])
            err_bin[j].append(r_jk[1])

    x_bin = np.array(x_bin)
    for j in range(len(y)):
        y_bin[j] = np.array(y_bin[j])
        err_bin[j] = np.array(err_bin[j])

    # Fit affine functions, plot function and data
    plt.figure(figsize=(10, 6))
    for j in range(len(y)):
        res = curve_fit(lin, x, y[j], p0=[0.01, 0.01], sigma=1/np.sqrt(weights))
        m_dm = ufloat(res[0][0], np.sqrt(res[1][0,0]))

        label = '${}={:.2ugL}$'.format(mlabel[j], m_dm)
        plt.plot(x_bin, lin(x_bin, *res[0]), c=colors[j], label=label)
        plt.errorbar(x_bin, y_bin[j], yerr=err_bin[j], c=colors[j], fmt='.')

        if stats_file:
            msg = '{}: {}={:.2ugP}'.format(xlabel, mlabel[j], m_dm)
            print_stats(msg, stats_file, verbose=verbose)

    # Finalise plots
    plt_xmin, plt_xmax = plt.xlim()
    plt.xlim(plt_xmin, plt_xmax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.title(title)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, bbox_inches='tight')


def affine_corr_n(
    x_arr,
    y,
    xlabel_arr,
    ylabel,
    mlabel=None,
    weights=None,
    n_bin=30,
    out_path_arr=None,
    title='',
    colors=None,
    stats_file=None,
    verbose=False,
    seed=None,
    parallel=False,
    n_jobs=-1
):
    """Affine Corr N

    Compute n affine correlations of y(m) versus x_arr[n].

    Parameters
    -----------
    x_arr: array(n, double)
        input x value
    y: array(m) of double
        input y arrays
    xlabel, ylabel : string
        x-and y-axis labels
    mlabel(m) : string, optional, default=None
        label for slope in the plot legend
    weights : array of double, optional, default=None
        weights of x points
    n_bin : double, optional, default=30
        number of points onto which data are binned
    out_path_arr) : array(n) of string, optional, default=None
        output file path, if not given, plot is not saved to file
    title : string, optional, default=''
        plot title
    colors(m) : array of string, optional, default=None
        line colors
    stats_file : filehandler, optional, default=None
        output file for statistics
    verbose : bool, optional, default=False
        verbose output if True
    seed: int
        Seed to initialize the randoms. [Default: None]
    parallel: bool
        If True, use parallel computing. [Default: False]
    n_jobs: int
        Number of jobs to run in parallel. [Default: -1]
    """

    master_rng = np.random.RandomState(seed)
    seeds = master_rng.randint(low=0, high=2**30, size=len(x_arr))

    if out_path_arr is None:
        out_path_arr = [None]*len(x_arr)
    for x, xlabel, out_path, seed_tmp in zip(
        x_arr, xlabel_arr, out_path_arr, seeds
    ):
        affine_corr(
            x,
            y,
            xlabel,
            ylabel,
            mlabel=mlabel,
            weights=weights,
            n_bin=n_bin,
            out_path=out_path,
            title=title,
            colors=colors,
            stats_file=stats_file,
            verbose=verbose,
            seed=seed_tmp,
            parallel=parallel,
            n_jobs=n_jobs,
        )


def xi_star_gal_tc(ra_gal, dec_gal, e1_gal, e2_gal, w_gal, ra_star, dec_star, e1_star, e2_star, w_star=None,
    theta_min_amin=2, theta_max_amin=200, n_theta=20):
    """xi star gal tc

    Cross-correlation between galaxy and star ellipticities.
    """

    cat_gal = treecorr.Catalog(ra=ra_gal, dec=dec_gal, g1=e1_gal, g2=e2_gal,
                               w=w_gal, ra_units='degrees', dec_units='degrees')
    cat_star = treecorr.Catalog(ra=ra_star, dec=dec_star, g1=e1_star, g2=e2_star,
                                w=w_star, ra_units='degrees', dec_units='degrees')

    TreeCorrConfig = {
        'ra_units': 'degrees',
        'dec_units': 'degrees',
        'sep_units': 'arcminutes',
        'min_sep': theta_min_amin,
        'max_sep': theta_max_amin,
        'nbins': n_theta
    }

    ng = treecorr.GGCorrelation(TreeCorrConfig)

    ng.process(cat_gal, cat_star)

    return ng


def correlation_12_22(ra_1, dec_1, e1_1, e2_1, weights_1, ra_2, dec_2, e1_2, e2_2,
    theta_min_amin=2, theta_max_amin=200, n_theta=20):
    """Correlation 12 22

    Correlation functions between two samples 1 and 2. Compute xi_12 and xi_22.

    Parameters
    ----------
    ra_1, dec_1 : array of float
        coordinates of sample 1
    e1_1, e2_1 : array of float
        ellipticities of sample 1
    weights_1 : array of float
        weights of sample 1
    ra_2, dec_2 : array of float
        coordinates of sample 2
    e1_2, e2_2 : array of float
        ellipticities of sample 2
    theta_min_amin : float, optional
        minimum angular scale in arcmin, default is 2
    theta_max_amin : float, optional
        maximum angular scale in arcmin, default is 200
    n_theta : int, optional
        number of angular scales, default is 20

    Returns
    -------
    xi_12, xi_22 : correlations
        correlations 12, and 22
    """

    r_corr_12 = xi_star_gal_tc(
        ra_1,
        dec_1,
        e1_1,
        e2_1,
        weights_1,
        ra_2,
        dec_2,
        e1_2,
        e2_2,
        theta_min_amin=theta_min_amin,
        theta_max_amin=theta_max_amin,
        n_theta=n_theta
    )
    r_corr_22 = xi_star_gal_tc(
        ra_2,
        dec_2,
        e1_2,
        e2_2,
        np.ones_like(ra_2),
        ra_2,
        dec_2,
        e1_2,
        e2_2,
        theta_min_amin=theta_min_amin,
        theta_max_amin=theta_max_amin,
        n_theta=n_theta
    )

    return r_corr_12, r_corr_22


def alpha(r_corr_gp, r_corr_pp, e1_gal, e2_gal, weights_gal, e1_star, e2_star):
    """alpha

    Compute scale-dependent PSF leakage alpha.

    Parameters
    ----------
    r_corr_gp, r_corr_pp : correlations
        correlations galaxy-star, star-star
    e1_gal, e2_gal : array of float
        galaxy ellipticities
    weights_gal : array of float
        galaxy weights
    e1_star, e2_star : array of float
        galaxy ellipticities

    Returns
    -------
    alpha, sig_alpha : float
        mean and std of alpha
    """

    complex_gal = (
        np.average(e1_gal, weights=weights_gal)
        + np.average(e2_gal, weights=weights_gal)*1j
    )
    complex_psf = np.mean(e1_star) + np.mean(e2_star)*1j

    alpha_leak = (
        (r_corr_gp.xip - np.real(np.conj(complex_gal)*complex_psf))
        / (r_corr_pp.xip - np.abs(complex_psf)**2)
    )
    sig_alpha_leak = (
        np.abs(alpha_leak) * np.sqrt((np.sqrt(r_corr_gp.varxip) 
        / r_corr_gp.xip)**2
        + (np.sqrt(r_corr_pp.varxip) / r_corr_pp.xip)**2)
    )

    return alpha_leak, sig_alpha_leak
