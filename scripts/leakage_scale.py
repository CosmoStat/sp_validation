#!/usr/bin/env python3

import sys
import copy
import os
import numpy as np
from optparse import OptionParser
from astropy.io import ascii, fits
import astropy.coordinates as coords
from astropy import units                                                       

from cs_util import logging
from cs_util import canfar

from sp_validation import cat
from cs_util import plots
from sp_validation import util
from sp_validation import correlation as corr
from sp_validation import io
from sp_validation import cosmology as cosmology


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
        e1_PSF_star_col='E1_PSF_HSM',
        e2_PSF_star_col='E2_PSF_HSM',
        ra_star_col='RA',
        dec_star_col='Dec',
        sh='sh',
        theta_min_amin=1,
        theta_max_amin=300,
        n_theta=20,
        leakage_alpha_ylim=[0.03, 0.15],
        leakage_xi_sys_ylim=[-4e5, 5e5],
        leakage_xi_sys_log_ylim=[2e-13, 5e-5],
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
        help='input path of the shear catalogue'
    )
    parser.add_option(
        '',
        '--hdu_psf',
        dest='hdu_psf',
        default=p_def.hdu_psf,
        type='int',
        help=f'HDU of PSF catalogue, default=\'{p_def.hdu_psf}\''
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
        '-I',
        '--input_path_PSF',
        dest='input_path_PSF',
        type='string',
        help='input path of the PSF catalogue'
    )
    parser.add_option(
        '',
        '--e1_PSF_star_col',
        dest='e1_PSF_star_col',
        default=p_def.e1_PSF_star_col,
        type='string',
        help='e1 PSF column name in star catalogue, '
            + f'default=\'{p_def.e1_PSF_star_col}\''
    )
    parser.add_option(
        '',
        '--ra_star_col',
        dest='ra_star_col',
        default=p_def.ra_star_col,
        type='string',
        help='right ascension column name in star catalogue, '
            + f'default=\'{p_def.ra_star_col}\''
    )
    parser.add_option(
        '',
        '--dec_star_col',
        dest='dec_star_col',
        default=p_def.dec_star_col,
        type='string',
        help='declination column name in star catalogue, '
            + f'default=\'{p_def.dec_star_col}\''
    )
    parser.add_option(
        '',
        '--e2_PSF_star_col',
        dest='e2_PSF_star_col',
        default=p_def.e2_PSF_star_col,
        type='string',
        help='e2 PSF column name in star catalogue, '
            + f'default=\'{p_def.e2_PSF_star_col}\''
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
        '-s',                                                                   
        '--shapes',                                                             
        dest='sh',                                                              
        default=None,                                                           
        type='string',                                                          
        help=f'shape measurement method, default: read from parameter file'     
    )
    parser.add_option(                                                          
        '-t',
        '--close_pair_tolerance',
        dest='close_pair_tolerance',                                                              
        default=None,
        type='string',                                                          
        help='tolerance angle for close objects in star catalogue'
    )
    parser.add_option(                                                          
        '-m',                                                                   
        '--close_pair_mode',                                                             
        dest='close_pair_mode',                                                              
        default=None,
        type='string',                                                          
        help='mode for close objects in star catalogue, one of'
            + '\'remove\', \'average\''
    )
    parser.add_option(                                                          
        '-p',
        '--patch',
        dest='patch',                                                              
        default=None,
        type='int',                                                          
        help='patch number'
    )
    parser.add_option(
        '',
        '--cut',
        dest='cut',
        type='string',
        help='list of criteria (white-space separated, do not use \'_\')'
            + ' to cut data, e.g. \'w>0_mask!=0\''
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
    if not options.input_path_PSF:
        print('Input path for PSF catalogue (option \'-I\') required')
        return False

    if options.e1_PSF_star_col == options.e2_PSF_star_col:
        print(
            'Column names for e1_PSF and e2_PSF are identical, '
            + 'this is surely a mistake'
        )
        return False

    if options.close_pair_tolerance and not options.close_pair_mode:
        print('No mode for close pairs given (option \'-m\')')
        return False

    if (
        options.close_pair_mode
        and options.close_pair_mode not in ['remove', 'average']
    ):
        print('Invalid mode (option \'-m\')')
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


def handle_close_objects(
    dat_PSF,
    ra_star_col,
    dec_star_col,
    tolerance,
    mode,
    stats_file=None,
    verbose=False
):
    """Handle Close Objects.

    Deal with close objects in PSF catalogue.

    Parameters
    ----------
    dat_PSF : FITS.record
        input PSF data
    ra_star_col : str
        column name for right ascension
    dec_star_col : str
        column name for declination
    tolerance : str
        smallest distance to define close object,
        numerical value and unit as string
    mode : str
        mode to handle close objects, 'remove' or 'average'
    stats_file : file handler
        statistics output file
    verbose : bool, optional
        print message to stdout if True; default=False

    Returns
    -------
    FITS.record
        processed PSF data

    """
    n_star = len(dat_PSF)

    tolerance_angle = coords.Angle(tolerance)

    io.print_stats(
        f'close object distance = {tolerance_angle}',
        stats_file,
        verbose=verbose
    )

    # Create SkyCoord object from star positions
    coordinates = coords.SkyCoord(
        ra=dat_PSF[ra_star_col],
        dec=dat_PSF[dec_star_col],
        unit='deg'
    )

    # Search PSF catalogue in itself around tolerance angle
    indices1, indices2, d2d, d3d = coordinates.search_around_sky(                              
        coordinates,
        tolerance_angle
    )

    # Count multiplicity of indices = number of matches of search
    count = np.bincount(indices1)
    dat_PSF_proc = {}

    # Copy unique objects (multiplicity of unity)
    for col in dat_PSF.dtype.names:
        dat_PSF_proc[col] = dat_PSF[col][count == 1]
    n_non_close = len(dat_PSF_proc[ra_star_col])
    io.print_stats(
        f'found {n_non_close}/{n_star} = {n_non_close / n_star:.1%} '
        + 'non-close objects',
        stats_file,
        verbose=verbose
    )


    # Deal with repeated objects (multiplicity > 1)
    multiples = (count != 1)
    if not multiples.any():

        # No multiples found -> no action
        io.print_stats('no close objects found', stats_file, verbose=verbose)

    else:

        # Get index list of multiple objects
        idx_mult = np.where(multiples)[0]
        if mode == 'average':

            # Initialise additional data vector
            dat_PSF_mult = {}
            for col in dat_PSF.dtype.names:
                dat_PSF_mult[col] = []

            done = np.array([])
            n_avg_rem = 0

            # Loop over repeated indices
            for idx in idx_mult:

                # If already used: ignore this index
                if idx in done:
                    continue

                # Get indices in data index list corresponding to
                # this multiple index
                w = np.where(indices1 == idx)[0]

                # Get indices in data
                ww = indices2[w]

                # Append mean to additional data vector
                for col in dat_PSF.dtype.names:
                    mean = np.mean(dat_PSF[col][ww])
                    dat_PSF_mult[col].append(mean)

                # Register indixes to avoid repetition
                done = np.append(done, ww)
                n_avg_rem += len(ww) - 1

            n_avg = len(dat_PSF_mult[ra_star_col])
            io.print_stats(
                f'adding {n_avg}/{n_star} = {n_avg / n_star:.1%} '
                + 'averaged objects',
                stats_file,
                verbose=verbose
            )

            for col in dat_PSF.dtype.names:
                dat_PSF_proc[col] = np.append(
                    dat_PSF_proc[col],
                    dat_PSF_mult[col]
                )
        elif mode == 'remove':
            n_rem = len(idx_mult)
            io.print_stats(
                f'removing {n_rem}/{n_star} = {n_rem / n_star:.1%} '
                + 'close objects',
                stats_file,
                verbose=verbose
            )

    # Test
    coordinates_proc = coords.SkyCoord(                                              
        ra=dat_PSF_proc[ra_star_col],
        dec=dat_PSF_proc[dec_star_col],
        unit='deg'
    )      
    idx, d2d, d3d = coords.match_coordinates_sky(
        coordinates_proc,
        coordinates_proc,
        nthneighbor=2
    )
    non_close = (d2d > tolerance_angle).all()
    io.print_stats(
        f'Check: all remaining distances > {tolerance_angle}? {non_close}',
        stats_file,
        verbose=verbose
    )
    if mode == 'average':
        io.print_stats(
            f'Check: n_non_close + n_avg + n_avg_rem = n_star? '
            + f'{n_non_close} + {n_avg} + {n_avg_rem} = '
            + f'{n_non_close + n_avg + n_avg_rem} ({n_star})',
            stats_file,
            verbose=verbose
        )
    elif mode == 'remove':
        io.print_stats(
            f'Check: n_non_close + n_rem = n_star? {n_non_close} '
            + f'+ {n_rem} = {n_non_close + n_rem} ({n_star})',
            stats_file,
            verbose=verbose
        )

    n_in = len(dat_PSF[ra_star_col])
    n_out = len(dat_PSF_proc[dec_star_col])
    
    if n_in == n_out:
        io.print_stats(
            f'keeping all {n_out} stars',
            stats_file,
            verbose=verbose
        )
    else:
        io.print_stats(
            f'keeping {n_out}/{n_in} = {n_out/n_in:.1%} stars',
            stats_file,
            verbose=verbose
        )

    return dat_PSF_proc


def compute_corr_gp_pp_alpha(
    dat_shear,
    dat_PSF,
    param,
    stats_file,
    theta_min_amin,
    theta_max_amin,
    n_theta,
    verbose=False
):
    """Compute Corr GP PP Alpha

    Compute and plot scale-dependent PSF leakage functions.

    Parameters
    ----------
    dat_shear : FITS.record
        input shear data
    dat_PSF : FITS.record
        input PSF data
    param : class param                                                         
        parameters                  
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
    e1_gal = dat_shear[param.e1_col]
    e2_gal = dat_shear[param.e2_col]
    weights = dat_shear['w']

    ra_star = dat_PSF[param.ra_star_col]
    dec_star = dat_PSF[param.dec_star_col]
    e1_star = dat_PSF[param.e1_PSF_star_col]
    e2_star = dat_PSF[param.e2_PSF_star_col]

    # Correlation functions
    r_corr_gp, r_corr_pp = corr.correlation_12_22(
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
    alpha_leak, sig_alpha_leak = corr.alpha(
        r_corr_gp,
        r_corr_pp,
        e1_gal,
        e2_gal,
        weights,
        e1_star,
        e2_star
    )

    return r_corr_gp, r_corr_pp, alpha_leak, sig_alpha_leak


def compute_alpha_mean(
        alpha_leak,
        sig_alpha_leak,
        shape_method,
        stats_file,
        verbose=False
):
    """Compute Alpha Mean

    Compute weighted mean of the leakage function alpha 

    Parameters
    ----------
    alpha_leak : list of float
        values of alpha for a range of scales
    sig_alpha_leak : list of float
        values of the RMS of alpha for a range of scales
    shape_method : str
        shape measurement method, e.g. 'ngmix'
    stats_file : file handler
        statistics output file
    verbose : bool, optional
        print message to stdout if True; default=False

    Returns
    -------
    float
        weighted mean of alpha
    """

    alpha_leak_mean = util.transform_nan(
        np.average(alpha_leak, weights=1/sig_alpha_leak**2)
    )
    io.print_stats(
        f'{shape_method}: Weighted average alpha = {alpha_leak_mean:.3g}',
        stats_file,
        verbose=verbose
    )


def plot_alpha_leakage(
        meanr,
        alpha_leak,
        sig_alpha_leak,
        shape_method,
        output_dir,
        xmin,
        xmax,
        leakage_alpha_ylim
):
    """Plot Alpha Leakage.

    Plot scale-dependent leakage function alpha(theta)

    """
    plot_dir_leakage = output_dir

    theta = [meanr]
    alpha_theta = [alpha_leak]
    yerr = [sig_alpha_leak]
    xlabel = r'$\theta$ [arcmin]'
    ylabel = r'$\alpha(\theta)$'
    title = shape_method
    out_path = f'{output_dir}/alpha_leakage_{shape_method}.png'
    try:
        ylim  = leakage_alpha_ylim
    except:
        ylim = None

    plots.plot_data_1d(
        theta,
        alpha_theta,
        yerr,
        title,
        xlabel,
        ylabel,
        out_path,
        xlog=True,
        xlim=[xmin, xmax],
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
    shape_method,
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
        msg = f'{shape_method}: <|xi_sys_{symb}|> = {mean}'
        io.print_stats(msg, stats_file, verbose=verbose)

    try:
        ylim = leakage_xi_sys_ylim
    except:
        ylim = None
    out_path = f'{output_dir}/xi_sys_{shape_method}.pdf'
    
    plots.plot_data_1d(
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
    out_path = f'{output_dir}/xi_sys_log_{shape_method}.pdf'
    plots.plot_data_1d(
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


def get_nz():

    nz_base = 'nz.CFHTLenSmatched.W3'                                               
    nz_ext = 'txt'                                                                  
    nz_date = '202012'                                                              
    nz_version = 'v1'                                                               
    data_dir = '.'

    nz_name = '{}_{}_{}.{}'.format(nz_base, nz_date, nz_version, nz_ext)            
    source = f'vos:cfis/cosmostat/cosmology/redshifts/{nz_name}'                    
    target = f'{data_dir}/{nz_name}'                                                
                                                                                
    canfar.download(source, target, verbose=True)                                   
    z, nz = np.loadtxt(target, unpack=True)

    return z, nz


def get_theo_xi_planck(theta):

    Om = 0.3153                                                                     
    sig8 = 0.8111                                                                   
    ns = 0.9649                                                                     
    Ob = 0.0493                                                                     
    h = 0.6736

    z, nz = get_nz()

    xi_p, xi_m = cosmology.get_theo_xi(                                 
        theta,
        z,                                                                  
        nz,                                                                 
        Omega_m=Om,                                                         
        h=h,                                                                
        Omega_b=Ob,                                                         
        sig8=sig8,                                                          
        ns=ns
    )

    return xi_p, xi_m


def plot_xi_sys_ratio(
        meanr,
        C_sys_p,
        C_sys_m,
        C_sys_std_p,
        C_sys_std_m,
        xi_p_planck,
        xi_m_planck,
        shape_method,
        output_dir,
        stats_file,
        verbose=False,
):

    labels = [
        '$\\xi^{\\rm sys}_+ / \\xi_+$',
        '$\\xi^{\\rm sys}_- / \\xi_-$',
    ]

    title = 'Cross-correlation leakage ratio'
    xlabel = '$\\theta$ [arcmin]'
    ylabel = 'Correlation function ratio'

    theta = [meanr] * 2
    xi = [C_sys_p / xi_p_planck, C_sys_m / xi_m_planck]
    yerr = [C_sys_std_p / xi_p_planck, C_sys_std_m / xi_m_planck]
    
    comp_arr = [0, 1]
    symb_arr = ['+', '-']
    for comp, symb in zip(comp_arr, symb_arr):
        mean = np.mean(np.abs(xi[comp]))
        msg = f'{shape_method}: <|xi_sys_{symb}| / xi_{symb}> = {mean}'
        io.print_stats(msg, stats_file, verbose=verbose)

    out_path = f'{output_dir}/xi_sys_{shape_method}_ratio.pdf'

    ylim = [0, 0.5]
    
    plots.plot_data_1d(
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
    logging.log_command(argv)

    sys.path.append('.')

    # MKDEBUG TODO: replace
    os.mkdir(param.output_dir)
    stats_file = io.open_stats_file(param.output_dir, 'stats_file_leakage.txt')

    # Read galaxy catalogue
    hdu_list = fits.open(param.input_path_shear)
    dat_shear = hdu_list[1].data
    n_shear = len(dat_shear)
    io.print_stats(
        f'{n_shear} galaxies found in shear catalogue',
        stats_file,
        verbose=param.verbose
    )

    # Apply cuts to galaxy catalogue if required
    dat_shear = cat.cut_data(dat_shear, param.cut, verbose=param.verbose)

    # Read star catalogue
    dat_PSF = io.open_fits_or_npy(param.input_path_PSF, hdu_no=param.hdu_psf)

    # Deal with close objects in PSF catalogue (= stars on same position
    # from different exposures)
    if param.close_pair_tolerance:
        dat_PSF = handle_close_objects(
            dat_PSF,
            param.ra_star_col,
            param.dec_star_col,
            param.close_pair_tolerance,
            param.close_pair_mode,
            stats_file=stats_file,
            verbose=param.verbose
        )

    # scale-dependent alpha function
    r_corr_gp, r_corr_pp, alpha_leak, sig_alpha_leak = compute_corr_gp_pp_alpha(
        dat_shear,
        dat_PSF,
        param,
        stats_file,
        param.theta_min_amin,
        param.theta_max_amin,
        param.n_theta,
        verbose=param.verbose
    )
    print(
        any((r_corr_gp.meanr - r_corr_pp.meanr) / r_corr_gp.meanr > 0.1)
    )

    io.save_alpha(
        r_corr_gp.meanr,
        alpha_leak,
        sig_alpha_leak,
        param.sh,
        param.output_dir
    )
    compute_alpha_mean(
        alpha_leak,
        sig_alpha_leak,
        param.sh,
        stats_file,
        verbose=param.verbose
    )
    plot_alpha_leakage(
        r_corr_gp.meanr,
        alpha_leak,
        sig_alpha_leak,
        param.sh,
        param.output_dir,
        param.theta_min_amin,
        param.theta_max_amin,
        param.leakage_alpha_ylim
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
        param.sh,
        param.output_dir,
        stats_file,
        param.leakage_xi_sys_ylim,
        param.leakage_xi_sys_log_ylim,
        verbose=param.verbose
    )

    xi_p_planck, xi_m_planck = get_theo_xi_planck(r_corr_gp.meanr)
    plot_xi_sys_ratio(
        r_corr_gp.meanr,
        C_sys_p,
        C_sys_m,
        C_sys_std_p,
        C_sys_std_m,
        xi_p_planck,
        xi_m_planck,
        param.sh,
        param.output_dir,
        stats_file,
        verbose=param.verbose,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
