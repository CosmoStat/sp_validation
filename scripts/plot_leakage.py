#!/usr/bin/env python3

import sys
import copy
import numpy as np
from glob import glob
from optparse import OptionParser
from astropy.io import ascii

from cs_util import logging

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
        '-o',
        '--output_dir',
        dest='output_dir',
        default=p_def.output_dir,
        type='string',
        help=f'output_dir, default=\'{p_def.output_dir}\''
    )
    parser.add_option(                                                          
        '-s',                                                                   
        '--shapes',                                                             
        dest='sh',                                                              
        default=None,                                                           
        type='string',                                                          
        help=f'shape measurement method, default: read from parameter file'     
    )                                                                      
    parser.add_option(
        '-v',
        '--verbose',
        dest='verbose',
        action='store_true',
        help=f'verbose output'
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


def plot_alpha_leakage(
        meanr,
        alpha_leak,
        sig_alpha_leak,
        shape_method,
        output_dir,
        xmin,
        xmax,
        ylim=None,
):
    """Plot Alpha Leakage.

    Plot scale-dependent leakage function alpha(theta).

    Parameters
    ----------
    meanr : list
        angular scales
    alpha_leak : list
        values of alpha
    sig_alpha_leak : list
        RMS of alpha
    output_dir : str
        output directory to save plot
    xmin : float
        smallest angular scale, interpreted in arcmin
    xmax : float
        largest angular scale, interpreted in arcmin
    ylim : list, optional
        y-axis plot limits, default is `Ǹone`

    """
    plot_dir_leakage = output_dir

    theta = meanr
    alpha_theta = alpha_leak
    yerr = sig_alpha_leak
    xlabel = r'$\theta$ [arcmin]'
    ylabel = r'$\alpha(\theta)$'
    title = shape_method
    out_path = f'{output_dir}/alpha_leakage_{shape_method}_all.png'

    linewidths = [1] * len(meanr)
    linewidths[0] = 3

    colors = ['grey', 'k', 'b', 'r', 'c', 'm', 'g', 'orange']
    labels = ['all', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']

    plot_data_1d(
        theta,
        alpha_theta,
        yerr,
        title,
        xlabel,
        ylabel,
        out_path,
        xlog=True,
        xlim=[xmin, xmax],
        ylim=ylim,
        linewidths=linewidths,
        colors=colors,
        labels=labels,
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

    # save calling command
    logging.log_command(argv)

    sys.path.append('.')
    import params as config
    if param.sh is None:
        param.sh = config.shapes[0]

    # files are command line arguments
    fnames = args

    if param.verbose:
        print('Input files: ', fnames)

    # read input files, append data
    theta = []
    alpha_leak = []
    sig_alpha_leak = []
    for idx, fn in enumerate(fnames):
        print('Loading ', fn)
        dat = ascii.read(fn)
        theta.append(dat['theta'])
        alpha_leak.append(dat['alpha'])
        sig = dat['sig_alpha']

        # if not first (reference) file: mark error bars to not plot
        if idx != 0:
            sig *= np.nan
        sig_alpha_leak.append(sig)

    # plot all alpha curves
    plot_alpha_leakage(
        theta,
        alpha_leak,
        sig_alpha_leak,
        param.sh,
        param.output_dir,
        config.theta_min_amin,
        config.theta_max_amin,
        config.leakage_alpha_ylim
    )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
