"""IO.

:Name: io.py

:Description: This script contains methods for input and output.

:Author: Martin Kilbinger

:Date: 2021

:Package: sp_validation

"""

import os

import numpy as np
from astropy.io import ascii
from astropy.table import Table


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


def write_ascii_table_file(cols, names, fname):
    """Write Ascii Table File.

    Write ASCII file with table data

    Parameters
    ----------
    cols : list
        data columns
    names : list of str
        column names
    fname : str
        output file name

    """
    t = Table(cols, names=names)
    with open(fname, 'w') as fout:
        ascii.write(t, fout, delimiter='\t')


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
    names = ['# theta', 'alpha', 'sig_alpha']
    fname = f'{output_dir}/alpha_leakage_{sh}.txt'
    write_ascii_table_file(cols, names, fname)


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
    if file_extension == '.fits':
        hdu_list = fits.open(path)
        data = hdu_list[hdu_no]
    elif file_extension == '.npy':
        data = np.load(path)
    else:
        raise ValueError(f'Invalid file extension \'{file_extension}\'')

    return data
