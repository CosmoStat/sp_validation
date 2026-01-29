#!/usr/bin/env python3

"""check_footprint.py

Check input catalogue against footprint mask.

:Author: Martin Kilbinger <martin.kilbinger@cea.fr>

:Date: 2022

"""

import sys
import os

import numpy as np
import healpy as hp

from astropy.io import fits
from astropy.table import Table

import matplotlib.pylab as plt

from optparse import OptionParser

from unions_wl import catalogue as wl_cat

from cs_util import logging
from cs_util import calc
from cs_util import plots
from cs_util import cat as cs_cat


def params_default():
    """PARAMS DEFAULT

    Return default parameter values and additional information
    about type and command line options.

    Returns
    -------
    list :
        parameter dict
        types if not default (``str``)
        help string dict for command line option
        short option letter dict

    """
    # Specify all parameter names and default values
    input_base = 'agn'
    params = {
        'input_cat': f'{input_base}.fits',
        'key_ra': 'ra',
        'key_dec': 'dec',
        'input_mask': 'mask.fits',
        'output_path': f'{input_base}_in_footprint.fits',
        'good' : 1,
        'plot': False,
    }

    # Parameters which are not the default, which is ``str``
    types = {
        'plot' : 'bool',
        'good' : 'int',
    }

    # Parameters which can be specified as command line option
    help_strings = {
        'input_cat': 'catalogue input path, default={}',
        'key_ra': 'right ascension column name, default={}',                    
        'key_dec': 'declination column name, default={}',
        'input_mask': 'mask input path, default={}',
        'output_path': 'output path of catalogue in footprint, default={}',
        'plot': 'create plot',
        'good': 'value of observed (good) pixels, default={}',
    }

    # Options which have one-letter shortcuts
    short_options = {
        'input_cat': '-i',
        'input_mask': '-m',
        'output_dir': '-o',
        'plot': '-p',
        'good': '-g',
    }

    params['input_base'] = input_base

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

            if typ == 'bool':                                                   
                parser.add_option(                                              
                    f'{short}',                                                 
                    f'--{key}',                                                 
                    dest=key,                                                   
                    default=False,                                              
                    action='store_true',                                        
                    help=help_strings[key].format(p_def[key]),
                )
            else:
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
        print(f'Reading catalogue {params["input_cat"]}...')
    dat = fits.getdata(params['input_cat'])
    ra = dat[params['key_ra']]
    dec = dat[params['key_dec']]

    # Read input mask
    mask, nest, nside = wl_cat.read_hp_mask(
        params['input_mask'],
        verbose=params['verbose']
    )

    # Get mask pixel numbers of coordinates
    ipix = hp.ang2pix(nside, ra, dec, nest=nest, lonlat=True)

    ## Get pixels in footprint, where mask is 1
    in_footprint = (mask[ipix] == params['good'])

    # Get numbers of pixels in footprint 
    ipix_in_footprint = ipix[in_footprint]

    # Get indices of coordinates in footprint
    idx_np = np.where(in_footprint)[0]

    if params['verbose']:
        n_in_footprint = len(idx_np)
        print(
            f'{n_in_footprint}/{len(ra)} ='
            + f' {n_in_footprint/len(ra):.2%} objects in footprint'
        )

    # Restrict data to footprint
    dat_in_footprint = dat[idx_np]

    # Write data in footprint to disk
    t = Table(dat_in_footprint)
    if params['verbose']:
        print(f'Writing objects in footprint to {params["output_path"]}')
    cols = []
    for col_ind,key in enumerate(t.keys()):
        cols.append(fits.Column(name=key, array=t[key], format=dat_in_footprint.columns[col_ind].format))
    cs_cat.write_fits_BinTable_file(cols, params["output_path"])

    # Create plots
    if params['plot']:
        out_path_plot = f'{params["input_base"]}.png'

        # Coordinates in footprint with mask
        point_size = 0.02
        if params['verbose']:
            print(f'Creating plot {out_path_plot}...')

        ra_center_deg = 151
        hp.mollview(mask, rot=(ra_center_deg, 0, 0))                      
        hp.projscatter(
            dat_in_footprint['RA'],
            dat_in_footprint['dec'],
            s=point_size,
            color='g',
            lonlat=True,
        )
        plt.savefig(out_path_plot)                                    
        plt.close() 

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
