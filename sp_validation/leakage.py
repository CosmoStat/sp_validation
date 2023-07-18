"""LEAKAGE.

:Name: leakage.py

:Description: This package contains methods to deal with
    leakage.

:Authors: Martin Kilbinger <martin.kilbinger@cea.fr>
         Clara Bonini
         Axel Guinot

"""

import numpy as np
import matplotlib.pylab as plt
from lmfit import minimize, Parameters
from uncertainties import ufloat

from sp_validation.plot_style import *
from sp_validation import plots
from sp_validation.io import print_stats
from sp_validation import basic
from sp_validation import util
from sp_validation import io


def func_bias_quad_1D(params, x_data):
    """Func Bias Quad 1D.

    Function for quadratic 1D bias model.

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.ndarray
        x-values of the data

    Returns
    -------
    numpy.ndarray
        y-values of the model

    """
    q = params['q'].value
    m = params['m'].value
    c = params['c'].value

    y_model = q * x_data**2 + m * x_data + c

    return y_model


def loss_bias_quad_1d(params, x_data, y_data, err):
    """Loss Bias quad 1D.

    Loss function for Quadratic 1D model

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.ndarray
        x-values of the data
    y_data : numpy.ndarray
        y-values of the data
    err : numpy.ndarray
        error values of the data

    Returns
    -------
    numpy.ndarray
        residuals

    """
    y_model = func_bias_quad_1D(params, x_data)
    residuals = (y_model - y_data) / err
    return residuals


def quad_corr_quant(
    x,
    y,
    xlabel,
    ylabel,
    qlabel=None,
    mlabel=None,
    clabel=None,
    weights=None,
    n_bin=30,
    out_path=None,
    title='',
    colors=None,
    stats_file=None,
    verbose=False,
    seed=None,
    rng=None,
):
    """Quadratic Correlation Quantity.

    Computes and plots quadratic correlation of y(n) as function of x.

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
    seed: int
        Seed to initialize the randoms. [Default: None]
    rng: numpy.random.RandomState
        Random generator. [Default: None]

    Returns
    -------
    list
        1rst order coeff of each e_gal vs quantities for recap plot
    list
        2nd order coeff of each e_gal vs quantities for recap plot
    list
        names of the quantities associated to each slopes
    list
        errors of 1rst order coeff
    list
        errors of 2nd order coeff

    """
    # Init randoms
    if isinstance(rng, np.random.RandomState):
        master_rng = rng
    else:
        master_rng = np.random.RandomState(seed)

    n_y = len(y)

    if qlabel is None:
        qlabel = np.full(n_y, 'q')
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
            starter + idx * bin_size_tmp: starter + (idx + 1) * bin_size_tmp
        ]

        x_bin.append(np.mean(x[ind]))

        for j in range(len(y)):
            r_jk = basic.jackknif_weighted_average2(
                y[j][ind],
                weights[ind],
                remove_size=0.2,
                n_realization=50,
            )
            y_bin[j].append(r_jk[0])
            err_bin[j].append(r_jk[1])

    x_bin = np.array(x_bin)
    for jdx in range(len(y)):
        y_bin[jdx] = np.array(y_bin[jdx])
        err_bin[jdx] = np.array(err_bin[jdx])

    # Fit affine functions, plot function and data
    slope = []
    qslope = []
    ticks_names = []
    m_err = []
    q_err = []
    plt.figure(figsize=(10, 6))
    for jdx in range(len(y)):
        params = Parameters()
        params.add('q', value=0.01)
        params.add('m', value=0.01)
        params.add('c', value=0.01)

        # Optimize parameters
        res = minimize(
            loss_bias_quad_1d, params, args=(x, y[jdx], 1 / np.sqrt(weights))
        )

        qslope.append(res.params['q'].value)
        slope.append(res.params['m'].value)

        ticks_names.append(f"{xlabel}_e_{jdx+1}")
        q_dm = ufloat(res.params['q'].value, res.params['q'].stderr)
        m_dm = ufloat(res.params['m'].value, res.params['m'].stderr)
        c_dc = ufloat(res.params['c'].value, res.params['c'].stderr)

        q_err.append(res.params['q'].stderr)
        m_err.append(res.params['m'].stderr)

        label = (
            f"${qlabel[jdx]}={q_dm: .2ugL}, {mlabel[jdx]}={m_dm: .2ugL},"
            + f" {clabel[jdx]}={c_dc: .2ugL}$"
        )

        plt.plot(
            x_bin,
            func_bias_quad_1D(res.params, x_bin),
            c=colors[jdx],
            label=label,
        )

        plt.errorbar(
            x_bin,
            y_bin[jdx],
            yerr=err_bin[jdx],
            c=colors[jdx],
            fmt='.',
        )

        if stats_file:
            msg1 = '{}: {}={:.2ugP}'.format(xlabel, qlabel[jdx], q_dm)
            msg2 = '{}: {}={:.2ugP}'.format(xlabel, mlabel[jdx], m_dm)
            print_stats(msg1, stats_file, verbose=verbose)
            print_stats(msg2, stats_file, verbose=verbose)

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
    plt.close()

    return slope, qslope, ticks_names, m_err, q_err


def quad_corr_n_quant(
    x_arr,
    y,
    xlabel_arr,
    ylabel,
    qlabel=None,
    mlabel=None,
    clabel=None,
    weights=None,
    n_bin=30,
    out_path_arr=None,
    title='',
    colors=None,
    stats_file=None,
    verbose=False,
    seed=None,
):
    """Quadratic Correlation N Quantity.

    Compute n quadratic correlations of y(m) versus x_arr[n].

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
    seed: int
        Seed to initialize the randoms. [Default: None]

    """
    master_rng = np.random.RandomState(seed)
    seeds = master_rng.randint(low=0, high=2**30, size=len(x_arr))
    slopes = []
    qslopes = []
    ticks_label = []
    merr = []
    qerr = []

    if out_path_arr is None:
        out_path_arr = [None] * len(x_arr)
    for x, xlabel, out_path, seed_tmp in zip(
        x_arr, xlabel_arr, out_path_arr, seeds
    ):
        slope, qslope, ticks_names, m_err, q_err = quad_corr_quant(
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
            verbose=verbose,
            seed=seed_tmp,
        )

        for i in range(len(slope)):
            slopes.append(slope[i])
            qslopes.append(qslope[i])
            ticks_label.append(ticks_names[i])
            merr.append(m_err[i])
            qerr.append(q_err[i])

    ticks_positions = np.arange(1, len(slopes) + 1, 1)

    # Plot slopes
    plt.figure()
    plt.errorbar(
        ticks_positions,
        slopes,
        yerr=merr,
        color='peru',
        label='m',
        fmt='.',
    )

    plt.errorbar(
        ticks_positions,
        qslopes,
        yerr=qerr,
        color='crimson',
        label='q',
        fmt='.',
    )

    plt.xticks(
        ticks_positions,
        ticks_label,
        rotation=90,
        fontsize=10,
    )

    plt.yticks(fontsize=10)
    plt.axhline(
        y=0,
        color='black',
        linestyle='--',
    )
    plt.ylabel('q and m', fontsize=10)
    title = "(e1, e2) systematic tests (quadratic)"
    plt.title(title, fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path_arr[-1])
    plt.close()


def func_bias_lin_1d(params, x_data):
    """Func Bias Lin 1D.

    Function for linear 1D bias model.

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.ndarray
        x-values of the data

    Returns
    -------
    numpy.ndarray
        y-values of the model

    """
    m = params['m'].value
    c = params['c'].value

    y_model = m * x_data + c

    return y_model


def loss_bias_lin_1d(params, x_data, y_data, err):
    """Loss Bias Lin 1D.

    Loss function for linear 1D model

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.ndarray
        x-values of the data
    y_data : numpy.ndarray
        y-values of the data
    err : numpy.ndarray
        error values of the data

    Returns
    -------
    numpy.ndarray
        residuals

    """
    y_model = func_bias_lin_1d(params, x_data)
    residuals = (y_model - y_data) / err
    return residuals


def func_bias_2d_full(params, x1, x2, order='lin', mix=False):
    """Func Bias 2D Full.

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
    numpy.ndarray
        first component the 2D model y1(x1, x2) on the (x1, x2)-grid;
        2D array of float
    numpy.ndarray
        second component the 2D model, y2(x1, x2) on the (x1, x2)-grid;
        2D array of float

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
    """Loss Bias 2D.

    Loss function for 2D model

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.ndarray
        two-component x-values of the data
    y_data : numpy.ndarray
        two-component y-values of the data
    err : numpy.ndarray
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
    numpy.ndarray
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
    """Print Fit Report.

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
    """Corr 2D.

    Compute and plot 2D linear and quadratic correlations of (y1, y2) as
    function of (x1, x2).

    Parameters
    ----------
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
    y_ground_truth : numpy.ndarray, optional
        ground truth model values (y1, y2) for plotting (2D array),
        default is `None`;
    par_ground_truth : dict, optional
        ground truth parameter, for plotting, default is `None`
    verbose : bool, optional, default=False
        verbose output if True

    Returns
    -------
    lmfit.Parameters
        best-fit parameters

    """
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

    if len(y) != 2 or len(x) != 2:
        raise IndexError('Input data needs to have two components')
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

    # Spin compoments
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

    # Curves
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

    return res.params


def param_order2spin(p_dp, order, mix):
    """Param Order 2 Spin.

    Transform parameter from natural to spin coefficients.

    Parameters
    ----------
    p_dp : dict
        Parameter natural coefficients
    order : str
        expansion order, one of 'linear', 'quad'
    mix : bool
        ellipticity components are mixed if ``True``

    Returns
    -------
    dict
        Parameter spin coefficients

    """
    s_ds = {}

    s_ds['x0'] = 0.5 * (p_dp['a11'] + p_dp['a22'])

    if order == 'quad' and mix:
        s_ds['x2'] = 0.5 * (p_dp['q111'] + p_dp['q122'])
        s_ds['y2'] = 0.5 * (p_dp['q211'] - p_dp['q222'])
        s_ds['x-2'] = 0.25 * (p_dp['q111'] - p_dp['q122'] + p_dp['q212'])
        s_ds['y-2'] = 0.25 * (p_dp['q211'] - p_dp['q222'] - p_dp['q112'])

    s_ds['x4'] = 0.5 * (p_dp['a11'] - p_dp['a22'])

    if mix:
        s_ds['y4'] = p_dp['a12']

    if order == 'quad' and mix:
        s_ds['x6'] = 0.25 * (p_dp['q111'] - p_dp['q122'] - p_dp['q212'])
        s_ds['y6'] = 0.25 * (p_dp['q211'] - p_dp['q222'] + p_dp['q112'])

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
    verbose=False,
    seed=None,
    rng=None,
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
    seed: int
        Seed to initialize the randoms. [Default: None]
    rng: numpy.random.RandomState
        Random generator. [Default: None]

    Returns
    -------
    list
        slopes of the linear fits
    list
        errors of the slopes
    list
        labels of the linear fits

    """
    # Init randoms
    if isinstance(rng, np.random.RandomState):
        master_rng = rng
    else:
        master_rng = np.random.RandomState(seed)

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
            starter + idx * bin_size_tmp: starter + (idx + 1) * bin_size_tmp
        ]

        x_bin.append(np.mean(x[ind]))

        for j in range(len(y)):
            r_jk = basic.jackknif_weighted_average2(
                y[j][ind],
                weights[ind],
                remove_size=0.2,
                n_realization=50,
            )
            y_bin[j].append(r_jk[0])
            err_bin[j].append(r_jk[1])

    x_bin = np.array(x_bin)
    for jdx in range(len(y)):
        y_bin[jdx] = np.array(y_bin[jdx])
        err_bin[jdx] = np.array(err_bin[jdx])

    # Fit affine functions, plot function and data
    plt.figure(figsize=(10, 6))

    m_arr = []
    m_err_arr = []
    tick_name_arr = []

    for jdx in range(len(y)):
        params = Parameters()
        params.add('m', value=0.01)
        params.add('c', value=0.01)
        res = minimize(
            loss_bias_lin_1d, params, args=(x, y[jdx], 1 / np.sqrt(weights))
        )

        m_arr.append(res.params['m'].value)
        # MKDEBUG float required?
        m_err_arr.append(float(res.params['m'].stderr))
        tick_name_arr.append(f"{xlabel}_e{jdx+1}")

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

    plt.close()

    return m_arr, m_err_arr, tick_name_arr


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
    verbose=False,
    seed=None,
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
    seed: int
        Seed to initialize the randoms. [Default: None]

    """
    master_rng = np.random.RandomState(seed)
    seeds = master_rng.randint(low=0, high=2**30, size=len(x_arr))

    if out_path_arr is None:
        out_path_arr = [None] * len(x_arr)
    for x, xlabel, out_path, seed_tmp in zip(
        x_arr, xlabel_arr, out_path_arr, seeds
    ):
        m_arr, m_err_arr, tick_name_arr = affine_corr(
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
            verbose=verbose,
            seed=seed_tmp,
        )

    # Summary plot
    plt.figure()
    ticks_positions = np.arange(1, len(m_arr) + 1, 1)
    plt.errorbar(
        ticks_positions,
        m_arr,
        yerr=m_err_arr,
        color='peru',
        fmt='.'
    )
    plt.xticks(
        ticks_positions,
        tick_name_arr,
        rotation=90,
        fontsize=10,
    )
    plt.yticks(fontsize=10)
    plt.axhline(
        y=0,
        color='black',
        linestyle='--',
    )
    plt.ylabel('m')
    title = "(e1, e2) systematic tests"
    plt.title(title, fontsize=10)
    plt_xmin, plt_xmax = plt.xlim()
    plt.xlim(plt_xmin, plt_xmax)
    plt.tight_layout()
    plt.savefig(out_path_arr[-1])
    plt.close()


def corr_any_quant(dat, param, stats_file, label_quant=None, ratio=None):
    """Corr_any_quant.

    Compute and plot object-by-object ellipticity and any quantities relations.
    Plot also a recap plot of all slopes of the best fits of the e_gal vs
    other quantities.

    Parameters
    ----------
    dat : FITS.record
        input data
    param : class param
        parameters
    stats_file : file handler
        statistics output file

    """
    plot_dir_leakage = param.output_dir

    n_bin = 30

    colors = ['b', 'r']

    e1 = dat[param.e1_col]
    e2 = dat[param.e2_col]
    e = np.array([e1, e2])
    weights = dat['w']

    x_arr = []
    out_name_arr = []
    xlabel_arr = []

    if label_quant:
        xlabel_arr = label_quant
        for i in range(len(label_quant)):
            colname = label_quant[i]
            x_arr.append(dat[colname])
            out_name_arr.append(colname + '_vs_e_gal')

    if ratio:
        x_arr.append(dat[ratio[0]] / dat[ratio[1]])
        xlabel_arr.append(f"{ratio[0]}/{ratio[1]}")
        out_name_arr.append(f"{ratio[0]}_div_{ratio[1]}_vs_e_gal")

    ylabel = r'$e_{1,2}^{\rm gal}$'

    mlabel = ['m_1', 'm_2']
    clabel = ['c_1', 'c_2']

    print("Quadratic fit")
    out_path_arr = [f'{plot_dir_leakage}/{name}_quad' for name in out_name_arr]
    name = 'systematics_test_quad'
    out_path_arr.append(f'{plot_dir_leakage}/{name}')
    qlabel = ['q_1', 'q_2']
    quad_corr_n_quant(
        x_arr,
        e,
        xlabel_arr,
        ylabel,
        qlabel=qlabel,
        mlabel=mlabel,
        clabel=clabel,
        title="quadratic model",
        weights=weights,
        n_bin=n_bin,
        out_path_arr=out_path_arr,
        colors=colors,
        stats_file=stats_file,
        verbose=param.verbose
    )

    print("Linear fit")
    out_path_arr = [f'{plot_dir_leakage}/{name}_lin' for name in out_name_arr]
    name = 'systematics_test_lin'
    out_path_arr.append(f'{plot_dir_leakage}/{name}')
    affine_corr_n(
        x_arr,
        e,
        xlabel_arr,
        ylabel,
        mlabel=mlabel,
        clabel=clabel,
        title="linear model",
        weights=weights,
        n_bin=n_bin,
        out_path_arr=out_path_arr,
        colors=colors,
        stats_file=stats_file,
        verbose=param.verbose
    )
