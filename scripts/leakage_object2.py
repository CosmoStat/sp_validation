#!/usr/bin/env python3

import sys
import copy
import os
import numpy as np
from optparse import OptionParser
from astropy.io import ascii, fits
from lmfit import minimize, Parameters, fit_report
from cs_util import logging
from uncertainties import ufloat

from sp_validation.plot_style import *
from sp_validation import basic
from sp_validation import plots
from sp_validation import correlation
from sp_validation import io
from sp_validation.io import print_stats
from sp_validation.correlation import param_order2spin
###function from util.py

def func_bias_2d(params, x1_data, x2_data, order='lin', mix=False):
    """Func Bias 2D.

    Function of 2D bias model.

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x1_data : float or list of float
        first component of x-values of the data
    x2_data : float or list of float
        second component of x-values of the data
    order : str, optional
        order of fit, default is 'lin'
    mix : bool, optional
        mixing between components, default is `False`

    Returns
    -------
    list
        first component the 2D model, y1(x1, x2). Dimension
        is equal to x1_data and x2_data
    list
        second component the 2D model, y2(x1, x2). Dimension
        is equal to x1_data and x2_data

    """
    # Get affine parameters
    a11 = params['a11'].value
    a22 = params['a22'].value
    c1 = params['c1'].value
    c2 = params['c2'].value

    # Compute y-values for affine model
    y1_model = a11 * x1_data + c1
    y2_model = a22 * x2_data + c2

    if order == 'quad':
        # Add quadratic part
        q111 = params['q111'].value
        q222 = params['q222'].value
        y1_model += q111 * x1_data ** 2
        y2_model += q222 * x2_data ** 2

    if mix:
        # Add linear mixing part
        a12 = params['a12'].value
        y1_model += a12 * x2_data
        y2_model += a12 * x1_data

        if order == 'quad':
            # Add quadratic mixing part
            q112 = params['q112'].value
            q122 = params['q122'].value
            q212 = params['q212'].value
            q211 = params['q211'].value
            y1_model += q112 * x1_data * x2_data + q122 * x2_data ** 2
            y2_model += q212 * x1_data * x2_data + q211 * x1_data ** 2

    return y1_model, y2_model
    
def loss_bias_2d(params, x_data, y_data, err, order, mix):
    """Loss Bias 2D.

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
    y1_model, y2_model = func_bias_2d(
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
    y_ground_truth : 2D np.array, optional
        ground truth model values (y1, y2) for plotting, default is `None`
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

        
##############################################################################################    
class param:
    """Param Class.

    General class to store (default) variables.

    """

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def print(self, **kwds):
        """Print."""
        print(self.__dict__)

    def var_list(self, **kwds):
        """Get Variable List."""
        return vars(self)



def params_default():
    """Set default parameter values.

    Parameters
    ----------
    None

    Returns
    -------
    p_def: class param
       parameter values
    """

    p_def = param(
        output_dir='.',
        hdu_psf=1,
        e1_col='e1_uncal',
        e2_col='e2_uncal',
        e1_PSF_col='e1_PSF',
        e2_PSF_col='e2_PSF',
        size_PSF_col='fwhm_PSF',
    )

    return p_def


def parse_options(p_def):
    """Parse command line options.

    Parameters
    ----------
    p_def: class param
        parameter values

    Returns
    -------
    options: tuple
        Command line options
    args: string
        Command line string
    """

    usage  = "%prog [OPTIONS]"
    parser = OptionParser(usage=usage)

    parser.add_option(
        '-i',
        '--input_path_shear',
        dest='input_path_shear',
        type='string',
        help='input path of the extended shear catalogue'
    )
    parser.add_option(
        '-o',
        '--output_dir',
        dest='output_dir',
        default=p_def.output_dir,
        type='string',
        help=f'output_dir, default=\'{p_def.output_dir}\''
    )
    parser.add_option(
        '',
        '--e1_col',
        dest='e1_col',
        default=p_def.e1_col,
        type='string',
        help=f'e1 column name in galaxy catalogue, default=\'{p_def.e1_col}\''
    )
    parser.add_option(
        '',
        '--e2_col',
        dest='e2_col',
        default=p_def.e2_col,
        type='string',
        help=f'e2 column name in galaxy catalogue, default=\'{p_def.e2_col}\''
    )
    parser.add_option(
        '',
        '--e1_PSF_col',
        dest='e1_PSF_col',
        default=p_def.e1_PSF_col,
        type='string',
        help=f'PSF e1 column name, default=\'{p_def.e1_PSF_col}\''
    )
    parser.add_option(
        '',
        '--e2_PSF_col',
        dest='e2_PSF_col',
        default=p_def.e2_PSF_col,
        type='string',
        help=f'PSF e2 column name, default=\'{p_def.e2_PSF_col}\''
    )
    parser.add_option(
        '',
        '--size_PSF_col',
        dest='size_PSF_col',
        default=p_def.size_PSF_col,
        type='string',
        help=f'PSF size column name, default=\'{p_def.size_PSF_col}\''
    )
    parser.add_option(
        '-s',
        '--shapes',
        dest='sh',
        default=None,
        type='string',
        help=f'shape measurement method, default: read from parameter file'
    )
    parser.add_option(
        '-v',
        '--verbose',
        dest='verbose',
        action='store_true',
        help=f'verbose output'
    )
    parser.add_option(
        '-t',
        '--test',
        dest='test',
        action='store_true',
        help=f'test of 2D fit'
    )

    options, args = parser.parse_args()

    return options, args


def check_options(options):
    """Check command line options.

    Parameters
    ----------
    options: tuple
        Command line options

    Returns
    -------
    erg: bool
        Result of option check. False if invalid option value.
    """

    if not options.input_path_shear and not options.test:
        print(
            'Input path for shear catalogue (option \'-i\') '
            + 'required unless in test mode (option \'-t\')'
        )
        return False

    if options.e1_PSF_col == options.e2_PSF_col:
        print(
            'Column names for e1_PSF and e2_PSF are identical, '
            + 'this is surely a mistake'
        )
        return False

    return True


def update_param(p_def, options):
    """Return default parameter, updated and complemented according to options.
    
    Parameters
    ----------
    p_def:  class param
        parameter values
    optiosn: tuple
        command line options
    
    Returns
    -------
    param: class param
        updated paramter values
    """

    param = copy.copy(p_def)

    # Update keys in param according to options values
    for key in vars(param):
        if key in vars(options):
            setattr(param, key, getattr(options, key))

    # Add remaining keys from options to param
    for key in vars(options):
        if not key in vars(param):
            setattr(param, key, getattr(options, key))

    return param


def leakage_test(param, stats_file):
    """Leakage Test

    Test object-by-object leakage relations.

    """
    plot_dir_leakage = param.output_dir

    n_bin = 20

    colors = ['b', 'r']
    ylabel_arr = ['$y_1$', '$y_2$']
    mlabel = ['m_1', 'm_2']
    clabel = ['c_1', 'c_2']

    xlabel_arr = [
        r'$x_1$',
        r'$x_2$',
    ]

    # For testing
    np.random.seed(seed=6121975)
    
    xm = 1.0
    size = 2000
    sig_x = 0.5
    x_arr = [
        np.random.uniform(-xm, xm, size=size),
        np.random.uniform(-xm, xm, size=size)
    ]

    pars_gt = {
        'q111' : -0.9,
        'q222' : 0.3,
        'q112' : 1.8,
        'q122' : -1.3,
        'q212' : -2.0,
        'q211' : 0.25,
        'a11' :  -0.4,
        'a22' : 0.3,
        'a12' : 0.3,
        'c1' : 0.2,
        'c2' : -0.3,
    }

    # Ground truth parameter
    p_gt = Parameters()
    for par in pars_gt:
        p_gt.add(par, value=pars_gt[par])

    # Ground-truth data
    y1, y2 = func_bias_2d(
        p_gt,
        x_arr[0],
        x_arr[1],
        order='quad',
        mix=True
    )

    # Perturbation
    dy1 = np.random.normal(scale=sig_x, size=size)
    dy2 = np.random.normal(scale=sig_x, size=size)

    for order in ['lin', 'quad']:

        for mix in [False, True]:

            out_path = f'{plot_dir_leakage}/test_{order}_{mix}'
            corr_2d(
                x_arr,
                [y1 + dy1, y2 + dy2],
                xlabel_arr=xlabel_arr,
                ylabel_arr=ylabel_arr,
                order=order,
                mix=mix,
                title=f'test {order} {mix}',
                n_bin=n_bin,
                out_path=out_path,
                colors=colors,
                y_ground_truth=[y1, y2],
                par_ground_truth=p_gt,
                stats_file=stats_file,
                verbose=param.verbose,
        )

    print('Ground truth:')
    for par in p_gt:
        print(par, p_gt[par].value)

 
def leakage(dat, param, stats_file):
    """Leakage

    Compute and plot object-by-object PSF leakage relations.

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
    io.print_stats(f'{param.sh}:', stats_file, verbose=param.verbose)

    n_bin = 30

    colors = ['b', 'r']

    xlabel_arr = [
        r'$e_{1}^{\rm PSF}$',
        r'$e_{2}^{\rm PSF}$',
        r'$\mathrm{FWHM}^{\rm PSF}$ [arcsec]'
    ]
    
    e1 = dat[param.e1_col]
    e2 = dat[param.e2_col]
    e = np.array([e1, e2])
    weights = dat['w']

    x_arr = [
        dat[param.e1_PSF_col],
        dat[param.e2_PSF_col],
        dat[param.size_PSF_col]
    ]
    out_name_arr = [
        'PSF_e1_vs_e_gal',
        'PSF_e2_vs_e_gal',
        'PSF_size_vs_e_gal'
    ]

    ylabel_arr = [r'$e_1^{\rm gal}$', r'$e_2^{\rm gal}$']

    # Fit consistent spin-2 2D model
    mix = True
    for order in ['lin', 'quad']:
        out_path = (
            f'{plot_dir_leakage}/PSF_e_vs_e_gal_order-{order}_mix-{mix}'
        )
        par_best_fit = corr_2d(
            x_arr[:2],
            e,
            weights=weights,
            xlabel_arr=xlabel_arr[:2],
            ylabel_arr=ylabel_arr,
            order=order,
            mix=mix,
            title=f'{param.sh} {order} {mix}',
            n_bin=n_bin,
            out_path=out_path,
            colors=colors,
            stats_file=stats_file,
            verbose=param.verbose
        )
        fp_best_fit = open(f'{out_path}.json', 'w')
        par_best_fit.dump(fp_best_fit)

    # Fit separate 1D models, including size
    ylabel = r'$e_{1,2}^{\rm gal}$'
    mlabel = ['m_1', 'm_2']
    clabel = ['c_1', 'c_2']
    out_path_arr = [f'{plot_dir_leakage}/{name}' for name in out_name_arr]
    correlation.affine_corr_n(
        x_arr,
        e,
        xlabel_arr,
        ylabel,
        mlabel=mlabel,
        clabel=clabel,
        title=param.sh,
        weights=weights,
        n_bin=n_bin,
        out_path_arr=out_path_arr,
        colors=colors,
        stats_file=stats_file,
        verbose=param.verbose
    )


def main(argv=None):
    """Main

    Main program
    """

    # Set default parameters
    p_def = params_default()

    # Command line options
    options, args = parse_options(p_def)
    # Without option parsing, this would be: args = argv[1:]

    if check_options(options) is False:
        return 1

    param = update_param(p_def, options)

    # Save calling command

    logging.log_command(argv)

    os.mkdir(param.output_dir)
    stats_file = io.open_stats_file(param.output_dir, 'stats_file_leakage.txt')

    if param.test:
        leakage_test(param, stats_file)
        sys.exit(0)

    sys.path.append('.')
    import params as config
    if len(config.shapes) != 1:
        raise IndexError('number of shape measurement methods has to be one')
    if param.sh is None:
        param.sh = config.shapes[0]


    hdu_list = fits.open(param.input_path_shear)
    dat_shear = hdu_list[1].data

    # object-by-object alpha parameter
    leakage(dat_shear, param, stats_file)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
