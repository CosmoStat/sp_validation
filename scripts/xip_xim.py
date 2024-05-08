#!/usr/bin/env python3

"""xip_xim.py

Compute shear correlation functions using ``treecorr``.

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
        'input_path': 'shape_catalog_ngmix.fits',
        'key_ra': 'RA',
        'key_dec': 'DEC',
        'key_e1': 'e1',
        'key_e2': 'e2',
        'sign_e1': +1,
        'sign_e2': +1,
        'theta_min': 0.5,
        'theta_max': 200,
        'n_theta': 20,
        'output_path' : './xip_xim.txt',
    }

    short_options = {
        'input_path': '-i',
        'output_path': '-o',
    }

    types = {
        'sign_e1': 'int',
        'sign_e2': 'int',
    }

    help_strings = {
        'input_path': 'shear catalogue input path, default={}',
        'key_ra': 'column name for right ascension, default={}',
        'key_dec': 'column name for declination, default={}',
        'key_e1': 'column name for ellipticity component 1, default={}',
        'key_e2': 'column name for ellipticity component 2, default={}',
        'sign_e1': 'sign for ellipticity component 1, default={}',
        'sign_e2': 'sign for ellipticity component 2, default={}',
        'theta_min': 'mininum angular scale [arcmin], default={}',
        'theta_max': 'maximum angular scale [arcmin], default={}',
        'n_theta': 'number of angular scales, default={}',
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

    # Open input catalogue
    if params['verbose']:
        print(f'Reading catalogue {params["input_path"]}...')
    data = fits.getdata(params['input_path'])

    coord_units = 'degrees'
    if params['verbose']:
        print(                                                                  
            'Signs for ellipticity components ='
            + f' ({params["sign_e1"]:+d}, {params["sign_e2"]:+d})'
        )
    g1 = data[params['key_e1']] * params['sign_e1']
    g2 = data[params['key_e2']] * params['sign_e2']
    cat = treecorr.Catalog(
        ra=data[params['key_ra']],
        dec=data[params['key_dec']],
        g1=g1,
        g2=g2,
        w=data['w'],
        ra_units=coord_units,
        dec_units=coord_units,
    )

    # Set treecorr config info for correlation
    sep_units = 'arcmin'
    TreeCorrConfig = {
        'ra_units': coord_units,
        'dec_units': coord_units,
        'min_sep': params['theta_min'],
        'max_sep': params['theta_max'],
        'sep_units': sep_units,
        'nbins': params['n_theta'],
    }
    gg = treecorr.GGCorrelation(TreeCorrConfig)

    # Compute correlation
    if params['verbose']:
        print('Correlating...')
    gg.process(cat, cat)

    # Write to file
    if params['verbose']:
        print(f'Writing output file {params["output_path"]}')
    gg.write(params['output_path'])

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
