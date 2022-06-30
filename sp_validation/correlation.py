"""CORRELATION.

:Name: correlation.py

:Description: This script contains methods to deal with
    auto- and cross-correlations.

:Author: Martin Kilbinger <martin.kilbinger@cea.fr>
         Axel Guinot

:Date: 2021

"""

import numpy as np
import matplotlib.pylab as plt
from lmfit import minimize, Parameters, fit_report
from uncertainties import ufloat

from sp_validation import util
from sp_validation.basic import jackknif_weighted_average
from sp_validation.plot_style import *
from sp_validation import plots
from sp_validation.io import print_stats

import treecorr


def func_bias_lin_1d(params, x_data):
    """Func Bias Lin 1D

    Function for linear 1D bias model.

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.array
        x-values of the data

    Returns
    -------
    numpy.array
        y-values of the model

    """
    m = params['m'].value
    c = params['c'].value

    y_model = m * x_data + c

    return y_model


def loss_bias_lin_1d(params, x_data, y_data, err):
    """Loss Bias Lin 1D

    Loss function for linear 1D model

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.array
        x-values of the data
    y_data : numpy.array
        y-values of the data
    err : numpy.array
        error values of the data

    Returns
    -------
    numpy.array
        residuals

    """
    y_model = func_bias_lin_1d(params, x_data)
    residuals = (y_model - y_data) / err
    return residuals


def func_bias_2d_full(params, x1, x2, order='lin', mix=False):
    """Func Bias 2D Full

    Function of 2D bias model evaluated on full 2D grid.

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x1 : list of float
        first component of x-values
    x2 : list of float
        second component of x-values
    order : str, optional
        order of fit, default is 'lin'
    mix : bool, optional
        mixing between components, default is `False`

    Returns
    -------
    2D np.array of float
        first component the 2D model y1(x1, x2) on the (x1, x2)-grid
    2D np.array of float
        second component the 2D model, y2(x1, x2) on the (x1, x2)-grid

    """

    len1 = len(x1)
    len2 = len(x2)

    # Initialise both components y1, y2 as 2D arrays
    y1 = np.zeros(shape=(len1, len2))
    y2 = np.zeros(shape=(len1, len2))

    # Create 2D mesh for input x1, x2 values
    v1, v2 = np.meshgrid(x1, x2, indexing='ij')

    # Compute both components y1, y2 over the meash
    y1, y2 = util.func_bias_2d(params, v1, v2, order=order, mix=mix) 

    return y1, y2


def loss_bias_2d(params, x_data, y_data, err, order, mix):
    """Loss Bias 2D

    Loss function for 2D model

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.array
        two-component x-values of the data
    y_data : numpy.array
        two-component y-values of the data
    err : numpy.array
        error values of the data, assumed the same for both components
    order : str
        order of fit
    mix : bool
        mixing of components if True

    Raises
    ------
    IndexError :
        if input arrays x1_data and x2_data have different lenght

    Returns
    -------
    numpy.array
        residuals

    """

    # Get x and y values of the input data
    x1_data = x_data[0]
    x2_data = x_data[1]
    y1_data = y_data[0]
    y2_data = y_data[1]

    if len(x1_data) != len(x2_data):
        raise IndexError('Length of both data components has to be equal')

    # Get model 1D y1 and y2 components
    y1_model, y2_model = util.func_bias_2d(
        params,
        x1_data,
        x2_data,
        order=order,
        mix=mix
    )

    # Compute residuals between data and model
    res1 = (y1_model - y1_data) / err
    res2 = (y2_model - y2_data) / err

    # Concatenate both components
    residuals = np.concatenate([res1, res2])

    return residuals


def print_fit_report(res, file=None):
    """Print Fit Report

    Print report of minimizing result.

    Parameters
    ----------
    res : class lmfit.MinimizerResult
        results of the minization
    file : filehandler, optional
        output to file; if `None` (default) output to `stdout`

    """
    # chi^2
    print(f'chi^2 = {res.chisqr}', file=file)

    # Reduced chi^2
    print(f'reduced chi^2 = {res.redchi}', file=file)

    # Akaike Information Criterium
    print(f'aic = {res.aic}', file=file)

    # Bayesian Information Criterium
    print(f'bic = {res.bic}', file=file)


def corr_2d(
    x,
    y,
    xlabel_arr,
    ylabel_arr,
    weights=None,
    order='lin',
    mix=False,
    n_bin=30,
    title='',
    colors=None,
    out_path=None,
    y_ground_truth=None,
    par_ground_truth=None,
    stats_file=None,
    verbose=False,
):
    """Corr 2D
    
    Compute and plot 2D linear and quadratic correlations of (y1, y2) as
    function of (x1, x2).
 
    Parameters
    -----------
    x : array(double)
        input x value

    y : array(m) of double
        input y arrays
    weights  : array of double, optional, default=None
        weights of x points
    order : str, optional
        order of fit, default is 'lin'
    mix : bool
        mixing of components if True
    xlabel_arr, ylabel_arr : list of str
        x-and y-axis labels
    n_bin : double, optional, default=30
        number of points onto which data are binned
    title : str, optional, default=''
        plot title
    colors : array(m) of str, optional, default=None
        line colors
    stats_file : filehandler, optional, default=None
        output file for statistics
    out_path : str, optional, default=None
        output file path, if not given, plot is not saved to file
    y_ground_truth : 2D np.array, optional
        ground truth model values (y1, y2) for plotting, default is `None`
    par_ground_truth : dict, optional
        ground truth parameter, for plotting, default is `None`
    verbose : bool, optional, default=False
        verbose output if True
    """
    
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

    if len(y) != 2 or len(x) != 2:
        raise IndexError(f'Input data needs to have two components')
    if any(len(y[0]) != c for c in {len(y[1]), len(x[0]), len(x[1])}):
        raise IndexError('Input data has inconsistent length')

    # Initialise parameters of model to fit
    params = Parameters()

    # Affine parameters
    for p_affine in ['a11', 'a22', 'c1', 'c2']:
        params.add(p_affine, value=0.0)

    if mix:
        # Linear mixing pararmeter
        params.add('a12', value=0.0)

    if order == 'quad':
        # Quadratic parameters
        for p_quad in ['q111', 'q222']:
            params.add(p_quad, value=0.0)

        if mix:
            # Quadratic mixing parameters 
            for p_quad_mix in ['q112', 'q122', 'q212', 'q211']:
                params.add(p_quad_mix, value=0.0)

    # Mininise loss function
    if weights is not None:
        err = 1 / np.sqrt(weights)
    else:
        err = np.ones_like(y[0])
    res = minimize(
        loss_bias_2d,
        params,
        args=(x, y, err, order, mix)
    )
    if stats_file:
        print_stats(
            f'2D fit order={order} mix={mix}:',
            stats_file,
            verbose=verbose
        )
        print_fit_report(res, file=stats_file)
    if verbose:
        print_fit_report(res)

    # Get best-fit parameter values and standard deviations
    p_dp = {}
    for p in res.params:
        p_dp[p] = ufloat(res.params[p].value, res.params[p].stderr)

    # Get spin coefficients
    s_ds = param_order2spin(p_dp, order, mix)

    # Output to stats file
    if stats_file:
        for p in res.params:
            print_stats(f'{p}={p_dp[p]:.3ugP}', stats_file, verbose=verbose)
        for spin in s_ds:
            print_stats(
                f'{spin}={s_ds[spin]:.3ugP}',
                stats_file,
                verbose=verbose
            )

    # Plots

    ## Spin compoments
    if out_path:
        out_path_spin = f'{out_path}_spin.png'
    else:
        out_path_spin = None

    if par_ground_truth:
        s_ground_truth = param_order2spin(par_ground_truth, order, mix)
    else:
        s_ground_truth = None
    plots.plot_bar_spin(
        s_ds,
        s_ground_truth=s_ground_truth,
        output_path=out_path_spin,
    )

    ## Curves
    plots.plot_corr_2d(
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
        y_ground_truth=y_ground_truth,
        title=title,
        colors=colors,
        out_path=out_path,
    )


def param_order2spin(p_dp, order, mix):

    s_ds = {}

    s_ds['x0'] = 0.5 * ( p_dp['a11'] + p_dp['a22'] )

    if order == 'quad' and mix:
        s_ds['x2'] = 0.5 * ( p_dp['q111'] + p_dp['q122'] )
        s_ds['y2'] = 0.5 * ( p_dp['q211'] - p_dp['q222'] )
        s_ds['x-2'] = 0.25 * ( p_dp['q111'] - p_dp['q122'] + p_dp['q212'] )
        s_ds['y-2'] = 0.25 * ( p_dp['q211'] - p_dp['q222'] - p_dp['q112'] )

    s_ds['x4'] = 0.5 * ( p_dp['a11'] - p_dp['a22'] )

    if mix:
        s_ds['y4'] = p_dp['a12']

    if order == 'quad' and mix:
        s_ds['x6'] = 0.25 * ( p_dp['q111'] - p_dp['q122'] - p_dp['q212'] )
        s_ds['y6'] = 0.25 * ( p_dp['q211'] - p_dp['q222'] + p_dp['q112'] )


    return s_ds


def affine_corr(
    x,
    y,
    xlabel,
    ylabel,
    mlabel=None,
    clabel=None,
    weights=None,
    n_bin=30,
    out_path=None,
    title='',
    colors=None,
    stats_file=None,
    verbose=False
):
    """Affine Corr.

    Computes and plots affine correlation of y(n) as function of x.

    Parameters
    ----------
    x: array(double)
        input x value
    y: array(m) of double
        input y arrays
    xlabel, ylabel : str
        x-and y-axis labels
    mlabel : str, optional, default=None
        label for slope in the plot legend
    clabel : str, optional, default=None
        label for offset in the plot legend
    weights : array of double, optional, default=None
        weights of x points
    n_bin : double, optional, default=30
        number of points onto which data are binned
    out_path : str, optional, default=None
        output file path, if not given, plot is not saved to file
    title : str, optional, default=''
        plot title
    colors : array(m) of str, optional, default=None
        line colors
    stats_file : filehandler, optional, default=None
        output file for statistics
    verbose : bool, optional, default=False
        verbose output if True
    """
    
    n_y = len(y)

    if mlabel is None:
        mlabel = np.full(n_y, 'm')
    if clabel is None:
        clabel = np.full(n_y, 'c')
        
    if weights is None:
        weights = np.ones_like(y[0])

    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

    size_all = len(y[0])
    for idx in range(1, n_y):
        if len(y[idx]) != size_all:
            raise IndexError
            (
                f'Size {len(y[idx])} of input #{idx} is different from size '
                + f'{size_all} of input #0'
            )
    size_bin = int(size_all / n_bin)
    diff_size = size_all - size_bin

    # Prepare arrays for binned data
    x_arg_sort = np.argsort(x)
    x_bin = []
    y_bin = []
    err_bin = []
    
    for idx in range(len(y)):
        y_bin.append([])
        err_bin.append([])

    # Bin data for plot
    for idx in range(n_bin):
        if idx < diff_size:
            bin_size_tmp = size_bin + 1
            starter = 0
        else:
            bin_size_tmp = size_bin
            starter = diff_size
        ind = x_arg_sort[
            starter + idx * bin_size_tmp : starter + (idx + 1) * bin_size_tmp
        ]

        x_bin.append(np.mean(x[ind]))

        for jdx in range(len(y)):
            r_jk = jackknif_weighted_average(
                y[jdx][ind],
                weights[ind],
                remove_size=0.2,
                n_realization=50
            )
            y_bin[jdx].append(r_jk[0])
            err_bin[jdx].append(r_jk[1])

    x_bin = np.array(x_bin)
    for jdx in range(len(y)):
        y_bin[jdx] = np.array(y_bin[jdx])
        err_bin[jdx] = np.array(err_bin[jdx])
 

    # Fit affine functions, plot function and data
    plt.figure(figsize=(10, 6))
    for jdx in range(len(y)):
        params = Parameters()
        params.add('m', value=0.01)
        params.add('c', value=0.01)
        res = minimize(
            loss_bias_lin_1d, params, args=(x, y[jdx], 1/np.sqrt(weights))
        )
        m_dm = ufloat(res.params['m'].value, res.params['m'].stderr)
        c_dc = ufloat(res.params['c'].value, res.params['c'].stderr)

        label = f'${mlabel[jdx]}={m_dm: .2ugL}, {clabel[jdx]}={c_dc: .2ugL}$'
        plt.plot(
            x_bin,
            func_bias_lin_1d(res.params, x_bin),
            c=colors[jdx],
            label=label
        )
        plt.errorbar(
            x_bin,
            y_bin[jdx],
            yerr=err_bin[jdx],
            c=colors[jdx],
            fmt='.'
        )

        if stats_file:
            msg = '{}: {}={:.2ugP}'.format(xlabel, mlabel[jdx], m_dm)
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
    clabel=None,
    weights=None,
    n_bin=30,
    out_path_arr=None,
    title='',
    colors=None,
    stats_file=None,
    verbose=False
):
    """Affine Corr N.

    Compute n affine correlations of y(m) versus x_arr[n].

    Parameters
    ----------
    x_arr: array(n, double)
        input x value
    y: array(m) of double
        input y arrays
    xlabel, ylabel : str
        x-and y-axis labels
    mlabel : str, optional, default=None
        label for slope in the plot legend
    clabel : str, optional, default=None
        label for offset in the plot legend
    weights : array of double, optional, default=None
        weights of x points
    n_bin : double, optional, default=30
        number of points onto which data are binned
    out_path_arr : array(n) of str, optional, default=None
        output file path, if not given, plot is not saved to file
    title : str, optional, default=''
        plot title
    colors(m) : array of str, optional, default=None
        line colors
    stats_file : filehandler, optional, default=None
        output file for statistics
    verbose : bool, optional, default=False
        verbose output if True
    """
    if out_path_arr is None:
        out_path_arr = [None] * len(y_arr)
    for x, xlabel, out_path in zip(x_arr, xlabel_arr, out_path_arr):
        affine_corr(
            x,
            y,
            xlabel,
            ylabel,
            mlabel=mlabel,
            clabel=clabel,
            weights=weights,
            n_bin=n_bin,
            out_path=out_path,
            title=title,
            colors=colors,
            stats_file=stats_file,
            verbose=verbose
        )


def xi_star_gal_tc(
    ra_gal,
    dec_gal,
    e1_gal,
    e2_gal,
    w_gal,
    ra_star,
    dec_star,
    e1_star,
    e2_star,
    w_star=None,
    theta_min_amin=2,
    theta_max_amin=200,
    n_theta=20,
):
    """Xi star gal tc.

    Cross-correlation between galaxy and star ellipticities.
    """

    unit = 'degrees'

    cat_gal = treecorr.Catalog(
        ra=ra_gal,
        dec=dec_gal,
        g1=e1_gal,
        g2=e2_gal,
        w=w_gal,
        ra_units=unit,
        dec_units=unit
    )
    cat_star = treecorr.Catalog(
        ra=ra_star,
        dec=dec_star,
        g1=e1_star,
        g2=e2_star,
        w=w_star,
        ra_units=unit,
        dec_units=unit
    )

    TreeCorrConfig = {
        'ra_units': unit,
        'dec_units': unit,
        'sep_units': 'arcminutes',
        'min_sep': theta_min_amin,
        'max_sep': theta_max_amin,
        'nbins': n_theta
    }
    ng = treecorr.GGCorrelation(TreeCorrConfig)

    ng.process(cat_gal, cat_star)

    return ng


def correlation_12_22(
    ra_1,
    dec_1,
    e1_1,
    e2_1,
    weights_1,
    ra_2,
    dec_2,
    e1_2,
    e2_2,
    theta_min_amin=2,
    theta_max_amin=200,
    n_theta=20,
):
    """Correlation 12 22.

    Shear correlation functions between two samples 1 and 2.
    Compute xi_12 and xi_22.

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
    """Alpha.

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
        + np.average(e2_gal, weights=weights_gal) * 1j
    )
    complex_psf = np.mean(e1_star) + np.mean(e2_star) * 1j

    alpha_leak = (
        (r_corr_gp.xip - np.real(np.conj(complex_gal) * complex_psf))
        / (r_corr_pp.xip - np.abs(complex_psf) ** 2)
    )
    sig_alpha_leak = (
        np.abs(alpha_leak)
        * np.sqrt(
            (np.sqrt(r_corr_gp.varxip) / r_corr_gp.xip) ** 2
            + (np.sqrt(r_corr_pp.varxip) / r_corr_pp.xip) ** 2
        )
    )

    return alpha_leak, sig_alpha_leak
