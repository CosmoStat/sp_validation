#!/usr/bin/env python3

import sys
import copy
import os
import numpy as np
from optparse import OptionParser
from astropy.io import fits
from lmfit import Parameters
from cs_util import logging


from sp_validation import leakage
from sp_validation import util
from sp_validation import io


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
        size_PSF_col='fwhm_PSF'
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
        '-v',
        '--verbose',
        dest='verbose',
        action='store_true',
        help='verbose output'
    )
    parser.add_option(
        '-t',
        '--test',
        dest='test',
        action='store_true',
        help='test of 2D fit'
    )
    parser.add_option(
         '',
         '--cols',
         dest='cols',
         type='string',
         help='list of column names for regression',
    )
    parser.add_option(
         '',
         '--cols_ratio',
         dest='cols_ratio',
         type='string',
         help='column names x_y for regression of their ratio (x/y)'
    )
    parser.add_option(
         '',
         '--PSF_Leakage',
         dest='PSF_Leakage',
         default=None,
         action='store_true',
         help='option for running the code for PSF Leakage'
    )
    parser.add_option(
         '',
         '--Obs_Leakage',
         dest='Obs_Leakage',
         default=None,
         action='store_true',
         help='option for running the code for Observational variables leakage'
     )

    options, args = parser.parse_args()

    return options, args


# MKDEBUG TODO: to cs_util. Also check shapepipe.
def my_string_split(string, num=-1, verbose=False, stop=False, sep=None):
    """My String Split.

    Split a *string* into a list of strings. Choose as separator
    the first in the list [space, underscore] that occurs in the string.
    (Thus, if both occur, use space.)

    Parameters
    ----------
    string : str
        Input string
    num : int
        Required length of output list of strings, -1 if no requirement.
    verbose : bool
        Verbose output
    stop : bool
        Stop programs with error if True, return None and continues otherwise
    sep : bool
        Separator, try ' ', '_', and '.' if None (default)

    Raises
    ------
    CfisError
        If number of elements in string and num are different, for stop=True
    ValueError
        If no separator found in string

    Returns
    -------
    list
        List of string on success, and None if failed

    """
    if string is None:
        return None

    if sep is None:
        has_space = string.find(' ')
        has_underscore = string.find('_')
        has_dot = string.find('.')

        if has_space != -1:
            my_sep = ' '
        elif has_underscore != -1:
            my_sep = '_'
        elif has_dot != -1:
            my_sep = '.'
        else:
            # no separator found, does string consist of only one element?
            if num == -1 or num == 1:
                my_sep = None
            else:
                raise Valueerror(
                    'No separator (\' \', \'_\', or \'.\') found in string'
                    + f' \'{string}\', cannot split'
                )
    else:
        if not string.find(sep):
            raise ValueError(
                f'No separator \'{sep}\' found in string \'{string}\', '
                + 'cannot split'
            )
        my_sep = sep

    res = string.split(my_sep)

    if num != -1 and num != len(res) and stop:
        raise CfisError(
            f'String \'{len(res)}\' has length {num}, required is {num}'
        )

    return res


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

    if (
        not options.PSF_Leakage
        and not options.Obs_Leakage
        and not options.test
    ):
        print(
            "One option out of --PSF_Leakage, --Obs_Leakage or -t is required"
        )
        return False

    if not options.Obs_Leakage and options.cols_ratio:
        print(f"Option 'cols_ratio' only valid for Obs_Leakage")

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

    param.cols = my_string_split(
        param.cols,
        verbose=param.verbose,
        stop=True,
    )
    param.cols_ratio = my_string_split(
        param.cols_ratio,
        num=2,
        verbose=param.verbose,
        stop=True,
    )

    return param


def leakage_test(param, stats_file):
    """Leakage Test

    Test object-by-object leakage relations.

    """
    plot_dir_leakage = param.output_dir

    n_bin = 20

    colors = ['b', 'r']
    ylabel_arr = ['$y_1$', '$y_2$']
    #mlabel = ['m_1', 'm_2']
    #clabel = ['c_1', 'c_2']

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
            leakage.corr_2d(
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


def PSF_leakage(dat, param, stats_file):
    """Leakage

    Compute and plot object-by-object PSF spin-consistent leakage relations.

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
        par_best_fit = leakage.corr_2d(
            x_arr[:2],
            e,
            weights=weights,
            xlabel_arr=xlabel_arr[:2],
            ylabel_arr=ylabel_arr,
            order=order,
            mix=mix,
            title=f'{order} {mix}',
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
    name = 'systematics_test'
    out_path_arr.append(f'{plot_dir_leakage}/{name}')
    leakage.affine_corr_n(
        x_arr,
        e,
        xlabel_arr,
        ylabel,
        mlabel=mlabel,
        clabel=clabel,
        title="",
        weights=weights,
        n_bin=n_bin,
        out_path_arr=out_path_arr,
        colors=colors,
        stats_file=stats_file,
        verbose=param.verbose
    )


def Obs_Leakage(dat_shear, param, stats_file):
    """Obs_Leakage

    Compute and plot object-by-object ellipticity and observational variables relations.
    Plot also a recap plot of all slopes of the best fits of the e_gal vs quantities

    Parameters
    ----------
    dat : FITS.record
        input data
    param : class param
        parameters
    stats_file : file handler
        statistics output file

    """
    # Get quantities to fix
    if not param.cols:
        # Get user input
        print("Data columns names :")
        print(dat_shear.dtype.names)
        change_header = input("Enter list of columns (comma-separated, no whitespaces: ")
        label_quant = [str(col) for col in change_header.split(',')]
    else:
        # Use command line argument
        label_quant = param.cols

    # Remove duplicates
    label_quant = list(set(label_quant))

    print('columns selected:', label_quant, end='')
    if param.cols_ratio:
        print(' ', param.cols_ratio[0],  '/', param.cols_ratio[1])
    leakage.corr_any_quant(dat_shear, param, stats_file, label_quant, ratio=param.cols_ratio)


def main(argv=None):
    """Main

    Main program
    """

    # Set default parameters
    p_def = params_default()

    # Command line options
    options, args = parse_options(p_def)

    if check_options(options) is False:
        return 1

    param = update_param(p_def, options)

    # Save calling command
    logging.log_command(argv)

    # Creation of the output directory
    if not os.path.exists(param.output_dir):
        os.mkdir(param.output_dir)

    # Creation of the statistics file handler
    stats_file = io.open_stats_file(param.output_dir, 'stats_file_leakage.txt')

    if param.test:
        # 2D spin-consistent test fit
        leakage_test(param, stats_file)
        return 0


    # Open Fits file of the input shear catalogue
    hdu_list = fits.open(param.input_path_shear)
    dat_shear = hdu_list[1].data

    if param.PSF_Leakage:

        # Object-by-object spin-consistent PSF leakage
        PSF_leakage(dat_shear, param, stats_file)

    if param.Obs_Leakage:
        # Object-by-object dependence of general variables
        Obs_Leakage(dat_shear, param, stats_file)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
