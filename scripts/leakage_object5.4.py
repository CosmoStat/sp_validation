#!/usr/bin/env python3


import sys
import copy
import os
import numpy as np
from optparse import OptionParser
from astropy.io import fits
from lmfit import minimize, Parameters
from cs_util import logging
from uncertainties import ufloat


from sp_validation.plot_style import *
from sp_validation import basic
from sp_validation import plots
from sp_validation import correlation
from sp_validation import util
from sp_validation import io
from sp_validation.io import print_stats
from sp_validation.correlation import param_order2spin

##Edited library functions (from correlation.py)

###QUADRATIC FIT###############################################################

def func_bias_quad_1D(params,x_data):
    """Func Bias Quad 1D.

    Function for quadratic 1D bias model.

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
    q = params['q'].value
    m = params['m'].value
    c = params['c'].value
    
    y_model = q*x_data**2 + m * x_data + c
    return y_model


def loss_bias_quad_1d(params, x_data, y_data, err):
    """Loss Bias quad 1D.

    Loss function for Quadratic 1D model

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
    parallel=False,
    n_jobs=-1,
    seed=None,
    rng=None,
):
    """Quadratic Corr

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
    parallel: bool
        If True, use parallel computing. [Default: False]
    n_jobs: int
        Number of jobs to run in parallel. [Default: -1]
    seed: int
        Seed to initialize the randoms. [Default: None]
    rng: numpy.random.RandomState
        Random generator. [Default: None]
        
    Returns
    -------
    slope : array
        Array of the 1rst order coeff of each e_gal vs quantities for recap plot
    qslope : array
        Array of the 2nd order coeff of each e_gal vs quantities for recap plot
    ticks_name : array
        Names of the quantities associated to each slopes 
    m_err : array
        Error associate to each 1rst order coeff
    q_err : array
        Error associate to each 2nd order coeff
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
                #seed=get_seed(master_rng),
                remove_size=0.2,
                n_realization=50,
                #parallel=parallel,
                #n_job=-1,
            )
            y_bin[j].append(r_jk[0])
            err_bin[j].append(r_jk[1])

    x_bin = np.array(x_bin)
    for jdx in range(len(y)):
        y_bin[jdx] = np.array(y_bin[jdx])
        err_bin[jdx] = np.array(err_bin[jdx])

    # Fit affine functions, plot function and data
    slope = []
    qslope=[]
    ticks_names = []
    m_err =[]
    q_err=[]
    plt.figure(figsize=(10, 6))
    for jdx in range(len(y)):
        params = Parameters()
        params.add('q', value=0.01)
        params.add('m', value=0.01)
        params.add('c', value=0.01)
        res = minimize(
            loss_bias_quad_1d, params, args=(x, y[jdx], 1 / np.sqrt(weights))
        ) #optimize parameters 
        
        qslope.append(res.params['q'].value)
        slope.append(res.params['m'].value)
        
        ticks_names.append(xlabel+'_e'+str(jdx+1))
        q_dm = ufloat(res.params['q'].value, res.params['q'].stderr)
        m_dm = ufloat(res.params['m'].value, res.params['m'].stderr)
        c_dc = ufloat(res.params['c'].value, res.params['c'].stderr)
        
        q_err.append(res.params['q'].stderr)
        m_err.append(res.params['m'].stderr)
        
        label = f'${qlabel[jdx]}={q_dm: .2ugL}, {mlabel[jdx]}={m_dm: .2ugL}, {clabel[jdx]}={c_dc: .2ugL}$'

        plt.plot(
            x_bin,
            func_bias_quad_1D(res.params, x_bin),
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
    parallel=False,
    n_jobs=-1
):
    """Quad Corr N

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
    parallel: bool
        If True, use parallel computing. [Default: False]
    n_jobs: int
        Number of jobs to run in parallel. [Default: -1]
    """

    master_rng = np.random.RandomState(seed)
    seeds = master_rng.randint(low=0, high=2**30, size=len(x_arr))
    slopes = []
    qslopes = []
    ticks_label = []
    merr = []
    qerr= []

    if out_path_arr is None:
        out_path_arr = [None]*len(x_arr)
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
            parallel=parallel,
            n_jobs=n_jobs,
        )
        
        for i in range(len(slope)):
            slopes.append(slope[i])
            qslopes.append(qslope[i])
            ticks_label.append(ticks_names[i])
            merr.append(m_err[i])
            qerr.append(q_err[i])
        
        #slopes.append(slope[0])
        #slopes.append(slope[1])
        #ticks_label.append(ticks_names[0])
        #ticks_label.append(ticks_names[1])
      
    ticks_position = np.arange(1, len(slopes)+1,1)
    
    ##plot the slope by the 
    plt.figure()
    plt.errorbar(ticks_position, 
                 slopes, 
                 yerr=merr, 
                 color='peru',
                 label='m',
                 fmt='.'
    )
    
    plt.errorbar(ticks_position, 
                 qslopes, 
                 yerr=qerr, 
                 color='crimson',
                 label='q',
                 fmt='.'
    )
    
    plt.xticks(ticks_position, 
               ticks_label,
               rotation=90, 
               fontsize = 10
    )
    
    plt.yticks(fontsize = 10)
    plt.axhline(y=0, 
                color='black', 
                linestyle='--'
    )
    plt.ylabel('q and m',fontsize=10)
    title = "(e1, e2) systematic tests (quadratic)"
    plt.title(title, fontsize=10)
    plt.legend(fontsize=5)
    plt.tight_layout()
    plt.savefig(out_path_arr[-1])


###LINEAR FIT #################################################################


def affine_corr_quant(
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
    parallel=False,
    n_jobs=-1,
    seed=None,
    rng=None,
):
    """Affine Corr

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
    parallel: bool
        If True, use parallel computing. [Default: False]
    n_jobs: int
        Number of jobs to run in parallel. [Default: -1]
    seed: int
        Seed to initialize the randoms. [Default: None]
    rng: numpy.random.RandomState
        Random generator. [Default: None]
        
    Returns
    -------
    slope : array
        Array of the slopes of each e_gal vs quantities for recap plot
    ticks_name : array
        Names of the quantities associated to each slopes 
    m_err : array
        Error associate to each slopes
    
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
                #seed=get_seed(master_rng),
                remove_size=0.2,
                n_realization=50,
                #parallel=parallel,
                #n_job=-1,
            )
            y_bin[j].append(r_jk[0])
            err_bin[j].append(r_jk[1])

    x_bin = np.array(x_bin)
    for jdx in range(len(y)):
        y_bin[jdx] = np.array(y_bin[jdx])
        err_bin[jdx] = np.array(err_bin[jdx])

    # Fit affine functions, plot function and data
    slope = []
    ticks_names = []
    m_err =[]
    plt.figure(figsize=(10, 6))
    for jdx in range(len(y)):
        params = Parameters()
        params.add('m', value=0.01)
        params.add('c', value=0.01)
        res = minimize(
            correlation.loss_bias_lin_1d, params, args=(x, y[jdx], 1 / np.sqrt(weights))
        ) #optimize parameters 
        slope.append(res.params['m'].value)
        ticks_names.append(xlabel+'_e'+str(jdx+1))
        m_dm = ufloat(res.params['m'].value, res.params['m'].stderr)
        c_dc = ufloat(res.params['c'].value, res.params['c'].stderr)
        m_err.append(res.params['m'].stderr)
        label = f'${mlabel[jdx]}={m_dm: .2ugL}, {clabel[jdx]}={c_dc: .2ugL}$'

        plt.plot(
            x_bin,
            correlation.func_bias_lin_1d(res.params, x_bin),
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
        
    
    return slope,ticks_names, m_err
    

           
def affine_corr_n_quant(
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
    parallel=False,
    n_jobs=-1
):
    """Affine Corr N

    Compute n affine correlations of y(m) versus x_arr[n] and plot a 
    recap plot with the n slopes extracted from the linear fits for the n quantities.

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
    parallel: bool
        If True, use parallel computing. [Default: False]
    n_jobs: int
        Number of jobs to run in parallel. [Default: -1]
        
    """

    master_rng = np.random.RandomState(seed)
    seeds = master_rng.randint(low=0, high=2**30, size=len(x_arr))
    slopes = []
    ticks_label = []
    err = []

    if out_path_arr is None:
        out_path_arr = [None]*len(x_arr)
    for x, xlabel, out_path, seed_tmp in zip(
        x_arr, xlabel_arr, out_path_arr, seeds
    ):
        slope, ticks_names, m_err = affine_corr_quant(
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
            parallel=parallel,
            n_jobs=n_jobs,
        )
        
        for i in range(len(slope)):
            slopes.append(slope[i])
            ticks_label.append(ticks_names[i])
            err.append(float(abs(m_err[i])))
      
    ticks_position = np.arange(1, len(slopes)+1,1)
    
    ##plot the slope by the 
    plt.figure()
    plt.errorbar(ticks_position, 
                 slopes, 
                 yerr=err, 
                 color='peru',
                 fmt='.'
    )
    plt.xticks(ticks_position, 
               ticks_label,rotation=90, 
               fontsize = 10
    )
    plt.yticks(fontsize = 10)
    plt.axhline(y=0, 
                color='black', 
                linestyle='--'
    )
    plt.ylabel('m')
    title = "(e1, e2) systematic tests"
    plt.title(title, fontsize=10)
    plt_xmin, plt_xmax = plt.xlim()
    plt.xlim(plt_xmin, plt_xmax)
    plt.tight_layout()
    plt.savefig(out_path_arr[-1])

        
###############################################################################
###############################################################################
    
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
        RA_col = 'RA',
        Dec_col = 'Dec',
        mag_col = 'mag',
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
        '',
        '--RA',
        dest='RA',
        default=p_def.RA_col,
        type='string',
        help=f'RA column name, default=\'{p_def.RA_col}\''
    )
    
    parser.add_option(
        '',
        '--Dec',
        dest='Dec',
        default=p_def.Dec_col,
        type='string',
        help=f'PSF size column name, default=\'{p_def.Dec_col}\''
    )
    
    parser.add_option(
        '',
        '--mag',
        dest='mag',
        default=p_def.mag_col,
        type='string',
        help=f'magnitude column name, default=\'{p_def.mag_col}\''
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


    parser.add_option(
         '',
         '--header',
         dest='header',
         action='store_true',
         help=f'print the headers of the catalogue'
     )
     
    parser.add_option(
         '',
         '--linear',
         dest='linear',
         action='store_true',
         help=f'linear fit'
     )
    
    parser.add_option(
         '',
         '--quadratic',
         dest='quadratic',
         action='store_true',
         help=f'quadratic fit'
     )
    
    parser.add_option(
         '',
         '--ratio',
         dest='ratio',
         action='store_true',
         help=f'option for taking two columns and divided them'
     )
    
    parser.add_option(
         '',
         '--ratio_alone',
         dest='ratio_alone',
         default=None,
         action='store_true',
         help=f'option for running the code just for a ratio of two columns alone'
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
    y1, y2 = util.func_bias_2d(
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
            correlation.corr_2d(
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
        par_best_fit = correlation.corr_2d(
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

def corr_quant(dat, param, stats_file):
    """Corr_quant

    Compute and plot object-by-object ellipticity and other quantities relations.

    Parameters
    ----------
    dat : FITS.record
        input data
    param : class param
        parameters
    stats_file : file handler
        statistics output file

    """
    #premier test avec R.A.
    
    plot_dir_leakage = param.output_dir
    io.print_stats(f'{param.sh}:', stats_file, verbose=param.verbose)

    n_bin = 30

    colors = ['b', 'r']

    xlabel_arr = [
        r'$RA$',
        r'$Dec$',
        r'$mag$'
    ]
    
    e1 = dat[param.e1_col]
    e2 = dat[param.e2_col]
    e = np.array([e1, e2])
    weights = dat['w']

    x_arr = [
        dat[param.RA_col],
        dat[param.Dec_col],
        dat[param.mag_col]
    ]
    out_name_arr = [
        'RA_vs_e_gal',
        'Dec_vs_e_gal',
        'mag_vs_e_gal'
    ]

    ylabel_arr = [r'$e_1^{\rm gal}$', r'$e_2^{\rm gal}$']

    # Fit consistent spin-2 2D model
    mix = True
    for order in ['lin', 'quad']:
        out_path = (
            f'{plot_dir_leakage}/quantities_vs_e_gal_order-{order}_mix-{mix}'
        )
        par_best_fit = correlation.corr_2d(
            x_arr[:2],
            e,
            weights=weights,
            xlabel_arr=xlabel_arr[:2],
            ylabel_arr=ylabel_arr,
            order=order,
            mix=False,
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
    affine_corr_n_quant(
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

def corr_any_quant(dat, param, stats_file, label_quant=None,ratio=None):
    """Corr_any_quant

    Compute and plot object-by-object ellipticity and other quantities relations.
    Plot also a recap plot of all slopes of the best fits of the e_gal vs quantities

    Parameters
    ----------
    dat : FITS.record
        input data
    param : class param
        parameters
    stats_file : file handler
        statistics output file
        stats_file : file handler
            statistics output file

    """
    plot_dir_leakage = param.output_dir
    io.print_stats(f'{param.sh}:', stats_file, verbose=param.verbose)

    n_bin = 30

    colors = ['b', 'r']

    
    e1 = dat[param.e1_col]
    e2 = dat[param.e2_col]
    e = np.array([e1, e2])
    weights = dat['w']
    
    x_arr=[]
    out_name_arr = []
    xlabel_arr = []
    
    if label_quant:
        xlabel_arr = label_quant
        for i in range(len(label_quant)):
            colname = label_quant[i]
            x_arr.append(dat[colname])
            out_name_arr.append(colname+'_vs_e_gal')
    
    if param.ratio:
        x_arr.append(dat[ratio[0]]/dat[ratio[1]])
        xlabel_arr.append(str(ratio[0])+"/"+str(ratio[1]))
        out_name_arr.append(str(ratio[0])+"|"+str(ratio[1])+'_vs_e_gal')

    ylabel_arr = [r'$e_1^{\rm gal}$', r'$e_2^{\rm gal}$']
    ylabel = r'$e_{1,2}^{\rm gal}$'
    
    mlabel = ['m_1', 'm_2']
    clabel = ['c_1', 'c_2']
    out_path_arr = [f'{plot_dir_leakage}/{name}' for name in out_name_arr]
    name = 'systematics_test'
    out_path_arr.append(f'{plot_dir_leakage}/{name}')
    
    if param.quadratic:
        print("Quadratic Fit")
        qlabel = ['q_1','q_2']
        quad_corr_n_quant(
            x_arr,
            e,
            xlabel_arr,
            ylabel,
            qlabel=qlabel,
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

    else : 
        print("Linear Fit")
        affine_corr_n_quant(
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
    stats_file2 = io.open_stats_file(param.output_dir, 'stats_file_quant.txt') 

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
    
    if param.ratio:
        print("Data columns names :")
        print(dat_shear.dtype.names)
        print("Enter the two columns to be divided (x/y format) without whitespaces :")
        x_input = input('x: ')
        y_input = input('y: ')
        ratio = [x_input,y_input]
        
        if param.ratio_alone:
            label_quant = None
            corr_any_quant(dat_shear, param, stats_file2, label_quant, ratio)
            return 0
        
            
    #If the --header flag is in the command : print the header and ask the user to select the quantities in the terminal 
    if param.header:
        if param.ratio:
            print("Other quantities to select :")
        print("Data columns names :")
        print(dat_shear.dtype.names)
        change_header = input("Enter the list of the columns with seperate commas (,) and without whitespaces :  ")
        label_quant = [str(col) for col in change_header.split(',')]
        if param.ratio:
            print('columns selected :', ratio[0],'/',ratio[1], label_quant)
            corr_any_quant(dat_shear, param, stats_file2, label_quant,ratio)
            
        elif not param.ratio:
            print('columns selected :', label_quant)
            corr_any_quant(dat_shear, param, stats_file2, label_quant,ratio=None)
            
        return 0
            
    else : 
        # object-by-object alpha parameter
        #leakage(dat_shear, param, stats_file)
        
        #Same as the leakage() function but with other quantities (not with e_PSF)
        #corr_quant(dat_shear, param, stats_file2) 
        
        #Quantities array for the computation of the leakage:
        label_quant = [param.e1_PSF_col, param.e2_PSF_col, param.RA_col,param.Dec_col, param.mag_col, param.size_PSF_col]
        
        #Compute and plot the e_gal vs quantities (linear or quadratic fit and plot a global recap of the slopes)
        if param.ratio:
            print('columns selected :', ratio[0],'/',ratio[1], label_quant)
            corr_any_quant(dat_shear, param, stats_file2, label_quant,ratio) 
        else:
            print('columns selected :', label_quant)
            corr_any_quant(dat_shear, param, stats_file2, label_quant,ratio=None) 
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

