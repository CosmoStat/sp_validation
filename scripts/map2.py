#!/usr/bin/env python3

"""map2.py

Compute aperture-mass dispersion using ``treecorr``.
Requires input file with xi+ and xi- previously
computed with ``treecorr``.

:Author: Martin Kilbinger <martin.kilbinger@cea.fr>

"""

import sys

import numpy as np

from optparse import OptionParser
from astropy.io import fits
import matplotlib.pylab as plt

import treecorr

from cs_util import logging


def params_default():

    params = {
        'input_path': 'xip_xim.txt',
        'theta_min': 0.5,
        'theta_max': 200,
        'n_theta': 20,
        'n_Theta' : 10,
        'output_path' : './map2.txt',
    }

    short_options = {
        'input_path': '-i',
        'output_path': '-o',
    }

    types = {
        'theta_min': 'float',
        'theta_max': 'float',
        'n_theta': 'int',
        'n_Theta': 'int',
    }

    help_strings = {
        'input_path': 'input file containing xi+ and xi-, default={}',
        'theta_min': 'mininum angular scale [arcmin], default={}',
        'theta_max': 'maximum angular scale [arcmin], default={}',
        'n_theta': 'number of angular scales on input, default={}',
        'n_Theta': 'number of angular scales on output, default={}',
        'output_path': 'output path, default={}',
    }


    return params, short_options, types, help_strings


def parse_options(p_def, short_options, types, help_strings):
    """Parse command line options.

    Parameters
    ----------
    p_def : dict
        default parameter values
    help_strings : dict
        help strings for options

    Returns
    -------
    options: tuple
        Command line options
    """

    usage  = "%prog [OPTIONS]"
    parser = OptionParser(usage=usage)

    for key in p_def:
        if key in help_strings:

            if key in short_options:
                short = short_options[key]
            else:
                short = ''

            if key in types:
                typ = types[key]
            else:
                typ = 'string'

            parser.add_option(
                short,
                f'--{key}',
                dest=key,
                type=typ,
                default=p_def[key],
                help=help_strings[key].format(p_def[key]),
            )

    parser.add_option(
        '-v',
        '--verbose',
        dest='verbose',
        action='store_true',
        help=f'verbose output'
    )

    options, args = parser.parse_args()

    return options


def main(argv=None):

    params, short_options, types, help_strings  = params_default()

    options = parse_options(params, short_options, types, help_strings)

    # Update parameter values
    for key in vars(options):
        params[key] = getattr(options, key)

    # Save calling command
    logging.log_command(argv)

    # Initizlies correlation object
    gg = treecorr.GGCorrelation(
        min_sep=params['theta_min'],
        max_sep=params['theta_max'],
        bin_size=params['n_theta'],
    )

    # Open input catalogue
    if params['verbose']:
        print(f'Reading xi+ and xi- input file {params["input_path"]}...')
    gg.read(params['input_path'])

    # Set up angular smoothing scales on output
    R = np.geomspace(
        params['theta_min'] * 5,
        params['theta_max'] / 2,
        params['n_Theta'],
    )

    # Compute correlation
    if params['verbose']:
        print('Computing aperture mass dispersion...')
    gg.calculateMapSq(R, m2_uform='Schneider')

    # Write to file
    if params['verbose']:
        print(f'Writing output file {params["output_path"]}')
    gg.writeMapSq(params['output_path'], R=R, m2_uform='Schneider')

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
