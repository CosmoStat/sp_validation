"""IO.

:Name: io.py

:Description: This script contains methods for input and output.

:Author: Martin Kilbinger


"""

import os

import numpy as np
from astropy.io import ascii, fits
from astropy.table import Table

from cs_util import cat


def make_out_dirs(output_dir, plot_dir, plot_subdirs, verbose=False):
    """Make output directories.

    Create output directories and subdirs

    Parameters
    ----------
    plot_dir: string
        directory name
    plot_subdirs: array of string
        subdirectory names
    verbose: bool, optional, default=False
        verbose output if True
    """
    for d in (output_dir, plot_dir):
        if not os.path.isdir(d):
            if verbose:
                print('Creating dir {}'.format(d))
            os.mkdir(d)
    for sd in plot_subdirs:
        dsd = '{}/{}'.format(plot_dir, sd)
        if not os.path.isdir(dsd):
            if verbose:
                print('Creating dir {}'.format(dsd))
            os.mkdir(dsd)


def open_stats_file(directory, file_name):
    """Open statistics file.

    Open output file for statistics

    Parameters
    ----------
    directory : string
        directory
    file_name : string
        file name
    """
    stats_file = open('{}/{}'.format(directory, file_name), 'w')

    return stats_file


def print_stats(msg, stats_file, verbose=False):
    """Print stats.

    Print message to stats file.

    Parameters
    ----------
    msg : string
        message
    stats_file : file handler
        statistics output file
    verbose : bool, optional, default=False
        print message to stdout if True
    """
    stats_file.write(msg)
    stats_file.write('\n')
    stats_file.flush()

    if verbose:
        print(msg)


def print_ratio(msg, numerator, denominator, stats_file, verbose=False):
    """Print Ratio.

    pretty-print ratio of two numbers

    msg : string
        message
    numerator : float
        ratio numerator
    denominator : float
        ratio denominator
    stats_file : file handler
        output staistic file
    verbose : bool, optional, default=False
        verbose output if True
    """
    if denominator != 0:
        ratio = numerator / denominator * 100
    else:
        ratio = 0

    print_stats(
        f'{msg} = {numerator}/{denominator}'
        + f' = {ratio:.1f}%',
        stats_file, verbose=verbose
    )


# MKDEBUG: This is not implemented in leakage, clean this up
def save_alpha(theta, alpha_leak, sig_alpha_leak, sh, output_dir):
    """Save Alpha.

    Save scale-dependent alpha

    Parameters
    ----------
    theta : list
        angular scales
    alpha_leak : list
        leakage alpha(theta)
    sig_alpha_leak : list
        standard deviation of alpha(theta)
    sh : str
        shape measurement method, e.g. 'ngmix'
    output_dir : str
        output directory

    """
    cols = [theta, alpha_leak, sig_alpha_leak]
    names = ['# theta[arcmin]', 'alpha', 'sig_alpha']
    fname = f'{output_dir}/alpha_leakage_{sh}.txt'
    cat.write_ascii_table_file(cols, names, fname)


def save_xi_sys(
    theta,
    xi_sys_p,
    xi_sys_m,
    xi_sys_std_p,
    xi_sys_std_m,
    xi_p_theo,
    xi_m_theo,
    sh,
    output_dir
):
    """Save Xi Sys.

    Save 'xi_sys' cross-correlation function.

    Parameters
    ----------
    theta : list
        angular scales
    xi_sys_p : list
        xi+ component of cross-correlation function
    xi_sys_m : list
        xi- component of cross-correlation function
    xi_sys_std_p : list
        xi+ component of cross-correlation standard deviation
    xi_sys_std_m : list
        xi- component of cross-correlation standard deviation
    xi_p_theo : list
        xi+ component of theoretical shear-shear correlation
    xi_m_theo : list
        xi- component of theoretical shear-shear correlation
    sh : str
        shape measurement method, e.g. 'ngmix'
    output_dir : str
        output directory

    """
    cols = [
        theta,
        xi_sys_p,
        xi_sys_m,
        xi_sys_std_p,
        xi_sys_std_m,
        xi_p_theo,
        xi_m_theo,
    ]
    names = [
        '# theta[arcmin]',
        'xi_+_sys',
        'xi_-_sys',
        'sigma(xi_+_sys)',
        'sigma(xi_-_sys)',
        'xi_+_theo',
        'xi_-_theo',
    ]
    fname = f'{output_dir}/xi_sys_{sh}.txt'
    cat.write_ascii_table_file(cols, names, fname)


def open_fits_or_npy(path, hdu_no=1):
    """Open FITS OR NPY.

    Open FITS or numpy binary file.

    Parameters
    ----------
    path : str
        path to input binary file
    hdu_no : int, optional
        HDU number, default is 1

    Returns
    -------
    FITS.rec or numpy.ndarray
        data

    """
    filename, file_extension = os.path.splitext(path)
    if file_extension in ['.fits', '.cat']:
        hdu_list = fits.open(path)
        data = hdu_list[hdu_no].data
    elif file_extension == '.npy':
        data = np.load(path)
    else:
        raise ValueError(f'Invalid file extension \'{file_extension}\'')

    return data
