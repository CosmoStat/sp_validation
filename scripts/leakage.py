#!/usr/bin/env python3

import sys
import copy
import numpy as np
from optparse import OptionParser
from astropy.io import ascii

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
        '-i',
        '--input_path_shear',
        dest='input_path_shear',
        type='string',
        help='input path of the extended shear catalogue'
    )
    parser.add_option(
        '-I',
        '--input_path_PSF',
        dest='input_path_PSF',
        type='string',
        help='input path of the PSF catalogue'
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

    if not options.input_path_shear:
        print('Input path for shear catalogue (option \'-i\') required')
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


def leakage(dat, output_dir, stats_file, verbose=False):
    """Leakage

    Compute and plot object-by-object PSF leakage relations.

    Parameters
    ----------
    dat : FITS.record
        input data
    output_dir : str
        output directory for plots
    stats_file : file handler
        statistics output file
    verbose : bool, optional
        print message to stdout if True; default=False
    """

    sh = 'ngmix'

    plot_dir_leakage = output_dir
    io.print_stats(f'{sh}:', stats_file, verbose=verbose)

    n_bin = 30

    colors = ['b', 'r']
    ylabel = r'$e_{1,2}^{\rm gal}$'
    mlabel = ['m_1', 'm_2']

    xlabel_arr = [
        r'$e_{1}^{\rm PSF}$',
        r'$e_{2}^{\rm PSF}$',
        r'$\mathrm{FWHM}^{\rm PSF}$ [arcsec]'
    ]
    
    e1 = dat['e1_uncal']
    e2 = dat['e2_uncal']
    e = [e1, e2]
    weights = dat['w']

    x_arr = [
        dat['e1_PSF'],
        dat['e2_PSF'],
        dat['fwhm_PSF']
    ]
    out_name_arr = [
        'PSF_e1_vs_e_gal',
        'PSF_e2_vs_e_gal',
        'PSF_size_vs_e_gal'
    ]
    out_path_arr = [f'{plot_dir_leakage}/{name}' for name in out_name_arr]
    affine_corr_n(
        x_arr,
        e,
        xlabel_arr,
        ylabel,
        mlabel=mlabel,
        title=sh,
        weights=weights,
        n_bin=n_bin,
        out_path_arr=out_path_arr,
        colors=colors,
        stats_file=stats_file,
        verbose=True
    )


def compute_corr_gp_pp_alpha(
    dat_shear,
    dat_PSF,
    output_dir,
    stats_file,
    theta_min_amin,
    theta_max_amin,
    n_theta,
    verbose=False):
    """Leakage Scales

    Compute and plot scale-dependent PSF leakage functions.

    Parameters
    ----------
    dat_shear : FITS.record
        input shear data
    dat_PSF : FITS.record
        input PSF data
    output_dir : str
        output directory for plots
    stats_file : file handler
        statistics output file
    verbose : bool, optional
        print message to stdout if True; default=False

    Returns
    -------
    treecorr output
        galaxy-PSF correlation data
    treecorr output
        PSF-PSF correlation data    
    list of float
        values of alpha for a range of scales
    list of float
        values of the RMS of alpha for a range of scales
    """

    ra = dat_shear['RA']
    dec = dat_shear['Dec']
    e1_gal = dat_shear['e1']
    e2_gal = dat_shear['e2']
    weights = dat_shear['w']

    ra_star = dat_PSF['RA']
    dec_star = dat_PSF['Dec']
    e1_star = dat_PSF['e1']
    e2_star = dat_PSF['e2']

    # Correlation functions
    r_corr_gp, r_corr_pp = correlation_12_22(
        ra,
        dec,
        e1_gal,
        e2_gal, 
        weights,
        ra_star,
        dec_star,
        e1_star,
        e2_star,
        theta_min_amin=theta_min_amin,
        theta_max_amin=theta_max_amin,
        n_theta=n_theta
    )

    # Leakage
    alpha_leak, sig_alpha_leak = alpha(
        r_corr_gp,
        r_corr_pp,
        e1_gal,
        e2_gal,
        weights,
        e1_star,
        e2_star
    )

    return r_corr_gp, r_corr_pp, alpha_leak, sig_alpha_leak


def compute_alpha_mean(alpha_leak, sig_alpha_leak, stats_file, verbose=False):
    """Compute Alpha Mean

    Compute weighted mean of the leakage function alpha 

    Parameters
    ----------
    alpha_leak : list of float
        values of alpha for a range of scales
    sig_alpha_leak : list of float
        values of the RMS of alpha for a range of scales
    stats_file : file handler
        statistics output file
    verbose : bool, optional
        print message to stdout if True; default=False

    Returns
    -------
    float
        weighted mean of alpha
    """

    sh = 'ngmix'

    alpha_leak_mean = transform_nan(
        np.average(alpha_leak, weights=1/sig_alpha_leak**2)
    )
    print_stats(
        f'{sh}: Weighted average alpha = {alpha_leak_mean:.3g}',
        stats_file,
        verbose=verbose
    )

def plot_alpha_leakage(meanr, alpha_leak, sig_alpha_leak, output_dir, leakage_alpha_ylim):

    plot_dir_leakage = output_dir

    sh = 'ngmix'

    theta = [meanr]
    alpha_theta = [alpha_leak]
    yerr = [sig_alpha_leak]
    xlabel = r'$\theta$ [arcmin]'
    ylabel = r'$\alpha(\theta)$'
    title = sh
    out_path = f'{output_dir}/alpha_leakage{sh}.png'
    try:
        ylim  = leakage_alpha_ylim
    except:
        ylim = None

    plot_data_1d(
        theta,
        alpha_theta,
        yerr,
        title,
        xlabel,
        ylabel,
        out_path,
        xlog=True,
        ylim=ylim
    )


def compute_xi_sys(r_corr_gp, r_corr_pp):
    """Compute Xi Sys

    Compute galaxy - PSF systematics correlation function

    Parameters
    ----------
    r_corr_gp : treecorr output
        galaxy-PSF correlation data
    r_corr_pp : treecorr output
        PSF-PSF correlation data    

    Returns
    -------
    list of float
        xi_sys_+
    list of float
        xi_sys_-
    list of float
        RMS of xi_sys_+
    list of float
        RMS of xi_sys_-
    """

    C_sys_p = r_corr_gp.xip**2 / r_corr_pp.xip
    C_sys_m = r_corr_gp.xim**2 / r_corr_pp.xim

    C_sys_std_p = (
        np.abs(C_sys_p)
        * np.sqrt(
            (((2*r_corr_gp.xip**2 * np.sqrt(r_corr_gp.varxip)) \
              / r_corr_gp.xip)/r_corr_gp.xip**2)**2
            + (np.sqrt(r_corr_pp.varxip)/r_corr_pp.xip)**2
        )
    )
    
    C_sys_std_m = (
        np.abs(C_sys_m)
        * np.sqrt(
            (((2*r_corr_gp.xim**2 * np.sqrt(r_corr_gp.varxim)) \
              / r_corr_gp.xim)/r_corr_gp.xim**2)**2
            + (np.sqrt(r_corr_pp.varxim)/r_corr_pp.xim)**2
        )
    )

    return C_sys_p, C_sys_m, C_sys_std_p, C_sys_std_m


def plot_xi_sys(
    meanr,
    C_sys_p,
    C_sys_m,
    C_sys_std_p,
    C_sys_std_m,
    output_dir,
    stats_file,
    leakage_xi_sys_ylim,
    leakage_xi_sys_log_ylim,
    verbose=False
):
    """Plot Xi Sys

    Plot galaxy - PSF systematics correlation function

    Parameters
    ----------
    C_sys_p : list of float
        xi_sys_+
    C_sys_m : list of float
        xi_sys_-
    C_sys_std_p : list of float
        RMS of xi_sys_+
    C_sys_std_m : list of float
         RMS of xi_sys_-
    output_dir : str
        output directory for plots
    stats_file : file handler
        statistics output file
    verbose : bool, optional
        print message to stdout if True; default=False
    """

    sh = 'ngmix'

    labels = ['$\\xi^{\\rm sys}_+$', '$\\xi^{\\rm sys}_-$']

    title = 'Cross-correlation leakage'
    xlabel = '$\\theta$ [arcmin]'
    ylabel = 'Correlation function'

    theta = [meanr] * 2
    xi = [C_sys_p, C_sys_m]
    yerr = [C_sys_std_p, C_sys_std_m]
    
    comp_arr = [0, 1]
    symb_arr = ['+', '-']
    for comp, symb in zip(comp_arr, symb_arr):
        mean = np.mean(np.abs(xi[comp]))
        msg = f'{sh}: <|xi_sys_{symb}|> = {mean}'
        print_stats(msg, stats_file, verbose=verbose)

    try:
        ylim = leakage_xi_sys_ylim
    except:
        ylim = None
    out_path = f'{output_dir}/xi_sys_{sh}.pdf'
    
    plot_data_1d(
        theta,
        xi,
        yerr,
        title,
        xlabel,
        ylabel,
        out_path,
        xlog=True,
        ylim=ylim,
        labels=labels
    )

    try:
        ylim = leakage_xi_sys_log_ylim
    except:
        ylim = None
    out_path = f'{output_dir}/xi_sys_log_{sh}.pdf'
    plot_data_1d(
        theta,
        xi,
        yerr,
        title,
        xlabel,
        ylabel,
        out_path,
        xlog=True, 
        ylog=True,
        ylim=ylim,
        labels=labels
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

    sys.path.append('.')
    import params as config

    stats_file = io.open_stats_file(param.output_dir, 'stats_file_leakage.txt')

    hdu_list = fits.open(param.input_path_shear)
    dat_shear = hdu_list[1].data

    # object-by-object alpha parameter
    leakage(dat_shear, param.output_dir, stats_file, verbose=param.verbose)

    if param.input_path_PSF:
        hdu_list = fits.open(param.input_path_PSF)
        dat_PSF = hdu_list[1].data

        # scale-dependent alpha function
        r_corr_gp, r_corr_pp, alpha_leak, sig_alpha_leak = compute_corr_gp_pp_alpha(
            dat_shear,
            dat_PSF,
            param.output_dir,
            stats_file,
            config.theta_min_amin,
            config.theta_max_amin,
            config.n_theta,
            verbose=param.verbose
        )
        compute_alpha_mean(
            alpha_leak,
            sig_alpha_leak,
            stats_file,
            verbose=param.verbose
        )
        plot_alpha_leakage(
            r_corr_gp.meanr,
            alpha_leak,
            sig_alpha_leak,
            param.output_dir,
            config.leakage_alpha_ylim
        )

        # xi_sys
        C_sys_p, C_sys_m, C_sys_std_p, C_sys_std_m = compute_xi_sys(
            r_corr_gp,
            r_corr_pp
        )
        plot_xi_sys(
            r_corr_gp.meanr,
            C_sys_p,
            C_sys_m,
            C_sys_std_p,
            C_sys_std_m,
            param.output_dir,
            stats_file,
            config.leakage_xi_sys_ylim,
            config.leakage_xi_sys_log_ylim,
            verbose=param.verbose
        )
    else:
        if param.verbose:
            print('No PSF input catalogue given, skipping scale-dependent leakage')

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
