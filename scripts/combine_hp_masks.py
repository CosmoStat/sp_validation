#!/usr/bin/env python

"""combine_hp_masks.py

Combine healpy tile masks into a joint footprint mask.

:Authors: Martin Kilbinger

"""

import sys
import re
from glob import glob
from optparse import OptionParser

import numpy as np

import matplotlib.pylab as plt

from astropy.io import fits
from astropy.table import Table
from astropy import wcs

import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)

import healpy as hp
from reproject import reproject_from_healpix, reproject_to_healpix

from tqdm import tqdm                                                           

from cs_util import logging


def params_default():
    """Params Default.

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
    params = {
        'input_tile_IDs': 'sp_output/found_ID_wshapes.txt',
        'input_dir_flags': 'output/run_sp_combined_flag/mask_runner/output',
        'input_dir_images': (
            'output/run_sp_combined_image/get_images_runner/output'
        ),
        'nside': 1024,
        'out_path': 'mask.fits',
        'out_path_plot': 'mask.png',
        'out_path2': None,
        'plot': '0',
        'verbose': False,
    }

    # Parameters which are not the default, which is ``str``
    types = {
        'nside': 'int',
        'plot': 'int',
        'verbose': 'bool',
    }

    # Parameters which can be specified as command line option
    help_strings = {
        'input_tile_IDs': 'input path of tile IDs, default={}',
        'input_dir_flags': (
            'input directory for pipeline flag files, default={}'
        ),
        'input_dir_images': (
            'input directory for images (headers with WCS, default={}'
        ),
        'nside': 'Output resolution parameter, default={}',
        'out_path': (
            'hp mask path; output if \'-p 0 or 1\';'
            + 'input if \'-p 2\'), can be list, default={}'
        ),
        'out_path2': 'hp mask path on output if \'-p 2',
        'out_path_plot': 'output path for plot',
        'plot': '0: no plot; 1: create plot; 2: plot only, read mask file',
        'verbose': 'verbose output if True',
    }

    # Options which have one-letter shortcuts
    short_options = {
        'input_tile_IDs': '-i',
        'nside': '-n',
        'out_path': '-o',
        'out_path_plot': '-O',
        'plot': '-p',
        'verbose': '-v',
    }

    return params, short_options, types, help_strings


def parse_options(p_def, short_options, types, help_strings):
    """Parse Options.
    
    Parse command line options.

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
                    f'{short}',
                    f'--{key}',
                    dest=key,
                    type=typ,
                    default=p_def[key],
                    help=help_strings[key].format(p_def[key]),
                )

    options, args = parser.parse_args()

    return options


def read_list(fname, col=None):
    """Read List.
       MKDEBUG from cfis.py

    Read list from ascii file.

    Parameters
    ----------
    fname : str
        ascii file name
    col : str, optional, default=None
        column name

    Returns
    -------
    list of str
        list of file names

    """
    if col is None:
        f = open(fname, 'r', encoding='latin1')
        file_list = [x.strip() for x in f.readlines()]
        f.close()
    else:
        import pandas as pd
        dat = pd.read_csv(fname, sep=r'\s+', dtype='string', header=None)
        if col not in dat:
            col = int(col)
        file_list = dat[col]

    file_list.sort()

    return file_list


def read_pixel_mask_files(ID_arr, params):
    """Read Pixel Mask Files.

    Read FITS files with pixel mask information.

    Parameters
    ----------
    ID_arr : list
        tile IDs
    params : dict
        options and parameter information

    Returns
    -------
    dict :
        healpix mask information per tile

    """
    if params['verbose']:
        print('Reading pixel mask files...')

    mask_1d = {}
    for ID in tqdm(ID_arr, disable=not params['verbose']):

        # Transform ID to file string format
        m = re.search(r'(\d{3}).{1}(\d{3})', ID)
        ID_file = f'{m[1]}-{m[2]}'

        # Read mask file
        path = f'{params["input_dir_flags"]}/pipeline_flag-{ID_file}.fits'
        hdu_mask = fits.open(path)
        mask = hdu_mask[0].data
        hdu_mask.close()

        # Get masked region indices (>= 1) and set to 0
        w_masked = (mask != 0)
        mask[w_masked] = 0

        # Get observed regions (0) and set to 1
        w_observed = (mask == 0)
        mask[w_observed] = 1

        # Read image header file
        img = open(f'{params["input_dir_images"]}/CFIS_image-{ID_file}.fits')
        header = fits.Header.fromtextfile(img)
        img.close()

        # Project to healpix.
        # Default ordering is RING (nested=False)
        mask_1d[ID], footprint = reproject_to_healpix(
            (mask, header),
            'c',
            nside=params['nside'],
        )

        # Get region outside footprint and set to 0
        w_outside = (footprint == 0)
        mask_1d[ID][w_outside] = 0

    return mask_1d


def combine_hp_masks(mask_1d, verbose=False):
    """Combine HelaPix Masks

    Combine healpix masks into Table

    Parameters
    ----------
    mask_1d : dict
        healpix masks per tile
    verbose : bool, optional
        verbose output if ``True``, default is ``False``

    Returns
    -------
    class Table :
        combined mask information

    """
    t = Table()

    # Add up all mask values
    idx = 0
    if verbose:
        print('Combining mask tile information...')
    for ID in tqdm(mask_1d, disable=not verbose):

        if idx == 0:
            t['flux'] = mask_1d[ID]
        else:
            t['flux'] += mask_1d[ID]

        idx += 1

    # Set all multiply added pixels back to 1
    w = t['flux'] > 1
    t['flux'][w] = 1

    # Exchange 0 <-> 1
    w_ok = (t['flux'] == 1)
    w_nok = (t['flux'] == 0)

    t['flux'][w_ok] = 0
    t['flux'][w_nok] = 1

    return t


def write_combined_hp_mask(t, nside, output_path):
    """Write Combined HealPix Mask.

    "Write healpix mask to FITS file.

    Parameters
    ----------
    t : class Table
        mask information
    nside : int
        healpix resolution parameter
    output_path : str
        output file path

    """
    # Set table header
    t.meta['ORDERING'] = 'RING'
    t.meta['COORDSYS'] = 'G'
    t.meta['NSIDE'] = nside
    t.meta['INDXSCHM'] = 'IMPLICIT'

    # Write to file
    t.write(output_path, overwrite=True, format='fits')


def main(argv=None):
    """Main.

    Main program.

    """
    params, short_options, types, help_strings = params_default()

    options = parse_options(params, short_options, types, help_strings)

    # Update parameter values
    for key in vars(options):
        params[key] = getattr(options, key)

    # Save calling command
    logging.log_command(argv)

    # Create mask
    if params['plot'] != 2:

        # For mask FITS file on output, make sure there are not
        # multiple output names

        # Get ID list
        ID_arr = read_list(params['input_tile_IDs'])

        # Read mask and header files, project to healpix
        mask_1d = read_pixel_mask_files(ID_arr, params)

        # Create output mask as Table
        t = combine_hp_masks(mask_1d, verbose=params['verbose'])

        write_combined_hp_mask(
            t,
            params['nside'],
            params["out_path"],
        )

    # Plot mask
    if params['plot'] > 0:

        if params['verbose']:
            print('Creating plot of combined mask(s)...')

        # Initialise empty mask, set all pixels to 1=unobserved
        mask_all = np.ones(shape=(hp.nside2npix(params['nside'])))

        # Combine input masks
        out_path_arr = glob(f'{params["out_path"]}')
        for out_path in out_path_arr:

            if params['verbose']:
                print(f'Reading file {out_path}...')

            # Read mask and multiply to cumulative mask.
            # Observed pixels (value=0) will be set to 0.
            mask_all *= hp.read_map(out_path)

        # Create and save plot

        # Correct plot for LF masks
        hp.mollview(mask_all, rot=(151, 0, 0))
        plt.savefig(params['out_path_plot'])
        plt.close()

        # Write combined map as FITS File
        if params['out_path2']:
            hp.write_map(params['out_path2'], mask_all, overwrite=True)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
