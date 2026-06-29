"""IO.

:Name: io.py

:Description: This script contains methods for input and output.

:Author: Martin Kilbinger


"""

import os

import numpy as np


def make_out_dirs(output_dir, verbose=False):
    """Make output directory."""
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f'Created dir {output_dir}')


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


def write_binned_quantity(quantity, key, bin_edges, extra_key="quantity"):
    
    shape = quantity.shape
    len_shape = len(shape)
    nx, ny = shape[:2]
    
    filename = f"{key}_binned.npz"

    combined = {**bin_edges, extra_key: quantity}
    np.savez(filename, **combined)


def read_binned_quantity(filename):

    with np.load(filename) as data:
        return {key: data[key] for key in data.files}
