#!/usr/bin/env python3

import sys

import numpy as np
from astropy.io import ascii

from sp_validation.cat import *

from optparse import OptionParser


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
        survey = 'v1'
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
        '-s',
        '--survey',
        dest='survey',
        type='string',
        help='survey, one in \'v1\'|\'test\''
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



def get_R(fname_base, key_base=None):

    dat = ascii.read(f'{fname_base}.txt')
    if dat[-1]['patch'] != 'all':
        raise ValueError(
            f'Invalid file {fname}, last row does not correspond to patch=\'all\''
        )
    R = np.empty(shape=(2, 2))

    if not key_base:
        key_base = fname_base

    R[0, 0] = dat[-1][f'{key_base}_11']
    R[0, 1] = dat[-1][f'{key_base}_12']
    R[1, 0] = dat[-1][f'{key_base}_21']
    R[1, 1] = dat[-1][f'{key_base}_22']

    return R


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

    if param.survey == 'v1':
        n_patch = 7
        patches = [f'P{x}' for x in np.arange(n_patch) + 1]
    elif param.survey == 'test':
        patches = ['P7', 'W3', 'S4']

    sh = 'ngmix'

    survey = 'cfis_3500'
    pipeline = 'SP'
    version = '1.0'

    R = get_R('R', key_base='R_tot')
    R_shear = get_R('R_shear')
    R_select = get_R('R_select')

    print('R - R_shear + R_select = 0 ?')
    print(R - R_shear - R_select)

    fname = 'c.txt'
    dat = ascii.read(fname)
    if dat[-1]['patch'] != 'all':
        raise ValueError(
            f'Invalid file {fname}, last row does not correspond to patch=\'all\''
        )
    c = np.empty(2)
    c_err = np.empty(2)
    c[0] = dat[-1]['cw_1']
    c[1] = dat[-1]['cw_2']
    c_err[0] = dat[-1]['dmcw_1']
    c_err[1] = dat[-1]['dmcw_2']


    # Invert total response matrix
    Rm1 = np.linalg.inv(R)

    ra_all = np.array([])
    dec_all = np.array([])
    g1_corr_mc_all = np.array([])
    g2_corr_mc_all = np.array([])
    w_all = np.array([])
    mag_all = np.array([])
    snr_all = np.array([])

    for patch in patches:

        print(patch)

        input_path = f'{patch}/sp_output/shape_catalog_{sh}.fits'
        ra, dec, g1, g2, w, mag, _ = read_shape_catalog(input_path)

        ra_all = np.append(ra_all, ra)
        dec_all = np.append(dec_all, dec)
        w_all = np.append(w_all, w)
        mag_all = np.append(mag_all, mag)
        
        g = np.array([g1, g2])

        # Calibrate with global R and c
        c_corr = Rm1.dot(c)

        g_corr_mc = Rm1.dot(g)
        for comp in (0, 1):
            g_corr_mc[comp] = g_corr_mc[comp] - c_corr[comp]
        g1_corr_mc_all = np.append(g1_corr_mc_all, g_corr_mc[0])
        g2_corr_mc_all = np.append(g2_corr_mc_all, g_corr_mc[1])

    output_path = f'{survey}_{pipeline}_v{version}.fits'
    g_corr_mc_all = np.array([g1_corr_mc_all, g2_corr_mc_all])
    write_shape_catalog(output_path, ra_all, dec_all, g_corr_mc_all, w_all, mag_all, R, R_shear, R_select, c, c_err)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
