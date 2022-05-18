#!/usr/bin/env python3

import sys
import copy
import numpy as np
from optparse import OptionParser
from astropy.io import ascii

from shapepipe.utilities import cfis
from shapepipe.utilities import file_system

from sp_validation.cat import *
from sp_validation.plots import *
from sp_validation.util import transform_nan
from sp_validation.correlation import *
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
        size_PSF_col='fwhm_PSF',
        sh='ngmix',
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
        '-s',
        '--shapes',
        dest='sh',
        default=p_def.sh,
        type='string',
        help=f'shape measurement method, default=\'{p_def.sh}\''
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
        'm11' :  -0.4,
        'm22' : 0.3,
        'm12' : 0.3,
        'c1' : 0.2,
        'c2' : -0.3,
    }

    # Ground truth parameter
    p_gt = Parameters()
    for par in pars_gt:
        p_gt.add(par, value=pars_gt[par])

    # Ground-truth data
    y1, y2 = func_bias_2d(
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
            affine_corr_2d(
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
        affine_corr_2d(
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

    # Fit separate 1D models, including size
    ylabel = r'$e_{1,2}^{\rm gal}$'
    mlabel = ['m_1', 'm_2']
    clabel = ['c_1', 'c_2']
    out_path_arr = [f'{plot_dir_leakage}/{name}' for name in out_name_arr]
    affine_corr_n(
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
    cfis.log_command(argv)

    file_system.mkdir(param.output_dir)
    stats_file = io.open_stats_file(param.output_dir, 'stats_file_leakage.txt')

    if param.test:
        leakage_test(param, stats_file)
        sys.exit(0)

    sys.path.append('.')
    import params as config


    hdu_list = fits.open(param.input_path_shear)
    dat_shear = hdu_list[1].data

    # object-by-object alpha parameter
    leakage(dat_shear, param, stats_file)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
