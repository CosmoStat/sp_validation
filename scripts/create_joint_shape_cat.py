#!/usr/bin/env python3

import sys

import numpy as np
import copy
from astropy.io import ascii
from astropy.io import fits
from optparse import OptionParser

from cs_util import logging
from cs_util import cat
from cs_util import plots

from sp_validation.cat import *
from sp_validation.calibration import *



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
        survey = 'v1',
        star_cat_base_path='./star_cat',
    )

    return p_def


def parse_options(p_def):
    """Parse command line options.

    Parameters
    ----------
    p_def : class param
        parameter values

    Returns
    -------
    tuple
        Command line options
    str
        Command line str

    """
    usage  = "%prog [OPTIONS]"
    parser = OptionParser(usage=usage)

    parser.add_option(
        '-s',
        '--survey',
        dest='survey',
        type='string',
        help='survey, allowed is \'v1\'|\'test\'|\'v1_small\'|\'Pa+Pb+...\'',
    )
    parser.add_option(
        '-S',
        '--star_cat_base_path',
        dest='star_cat_base_path',
        default=p_def.star_cat_base_path,
        type='string',
        help=f'star catalogue base path, default is {p_def.star_cat_base_path}',
    )

    parser.add_option(
        '-v',
        '--verbose',
        dest='verbose',
        action='store_true',
        help='verbose output',
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
    bool
        Result of option check. False if invalid option value.

    """
    return True


def update_param(p_def, options):
    """Return default parameter, updated and complemented according to options.
    
    Parameters
    ----------
    p_def:  class param
        parameter values
    options: tuple
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
            f'Invalid file {fname_base}, last row does not correspond to patch=\'all\''
        )
    R = np.empty(shape=(2, 2))

    if not key_base:
        key_base = fname_base

    R[0, 0] = dat[-1][f'{key_base}_11']
    R[0, 1] = dat[-1][f'{key_base}_12']
    R[1, 0] = dat[-1][f'{key_base}_21']
    R[1, 1] = dat[-1][f'{key_base}_22']

    return R


def merge_catalogues(
    patches,
    base_path,
    input_sub_path,
    output_path,
    sh,
    R_select=None,
    return_mean_e=False,
    return_mean_R_shear=False,
    hdu_in=1,
    hist_base=None,
    verbose=False,
):
    """Merge Catalogues

    Merge catalogues from sub-patches into one FITS file.

    Parameters
    ----------
    patches : list of str
        list of patches/sub-directories
    base_path : str
        base path
    input_sub_path : str
        common part of input path (below patch)
    output_path : str
        output file path
    sh : str
        shape measurement method, for DES weights use 'ngmix'
    R_select : np.array(2, 2), optional
        selection matrix
    return_mean_e : bool, optional
        return mean ellipticity if `True`; default is `False`
    return_mean_R_shear : bool, optional
        return mean response matrix if `True`; default is `False`
    hdu_in : int, optional
        input data HD, default is `1` 
    verbose : bool, optional
        verbose output if `True`; default is `False`

    Returns
    -------
    dict
        additive bias
    dict
        response matrix (multiplicative bias component)

    """
    dat_all = {}
    for idx, patch in enumerate(patches):

        if verbose:
            print(' ', patch)

        input_path = f'{base_path}/{patch}/{input_sub_path}'
        try:
            dat = fits.getdata(input_path, hdu_in)
        except:
            print(f"No data found in file {input_path} at HDU #{hdu_in}")
            print(f"Trying at HDU #{hdu_in-1}")
            dat = fits.getdata(input_path, hdu_in - 1)

        if idx == 0:
            col_names = dat.dtype.names
            for name in col_names:
                if name != 'w':
                    dat_all[name] = []
                else:
                    dat_all[name+'_iv'] = []
            dat_all['patch'] = []
        for name in col_names:
            if name != 'w':
                dat_all[name] = np.append(dat_all[name], dat[name])
            else:
                dat_all[name+'_iv'] = np.append(dat_all[name+'_iv'], dat[name])

        # Add patch number
        dat_all['patch'] = np.append(dat_all['patch'], [idx + 1] * len(dat))

    col_names = col_names + ('patch',)

    # Compute column for the DES weights (Gatti et al. 2021)
    if sh == 'ngmix':
        if verbose:
            print(f"Compute DES weights for the combined catalogue.")
        name = 'w_des'
        num_bins = 20
        dat_all['w_des'] = get_w_des(dat_all, num_bins)
        col_names = col_names + ('w_des',)

    column_all = []
    for name in col_names:
        if name != 'patch':
            my_format = 'D'
        else:
            my_format = 'I'
        if name == 'w':
            name = 'w_iv'
        column = fits.Column(name=name, array=dat_all[name], format=my_format)
        column_all.append(column)

    # Compute bias parameters if required
    c = np.empty(shape=(2))
    if return_mean_e:
        for idx in (0, 1):
            if 'w_des' in col_names:
                name = 'w_des'
            else:
                name = 'w_iv'
            c[idx] = np.average(
                dat_all[f'e{idx+1}_uncal'],
                weights=dat_all[name]
        )

    R_shear = np.empty(shape=(2, 2))
    R_shear_all = []
    labels = []
    if return_mean_R_shear:
        for idx in (0, 1):
            for jdx in (0, 1):
                R_shear[idx][jdx] = np.mean(dat_all[f'R_g{idx+1}{jdx+1}'])

                R_shear_all.append(dat_all[f'R_g{idx+1}{jdx+1}'])
                labels.append(f'$R_{{{idx+1}{jdx+1}}}$')

    if hist_base is not None:
        # Plot histogram of R_shear elements
        title = 'Shear response'
        x_label = 'response matrix element'
        y_label = 'frequency'
        x_range = (-3, 3)
        n_bins = 500
        out_path = f'R_{hist_base}.pdf'
        colors = ['blue', 'red','blue', 'red']                                          
        linestyles = ['-', '-', ':', ':']

        plots.plot_histograms(
            R_shear_all,
            labels,
            title,
            x_label,
            y_label,
            x_range,
            n_bins,
            out_path,
            colors=colors,
            linestyles=linestyles,
        )

    if R_select is not None:
        R = R_shear + R_select
        if verbose:
            print('Performing calibration on the whole catalog.')
        # Compute the calibration on the whole catalog for the ellipticities
        # Find the index of the columns e1 and e2 of calibrated ellipticities
        index_e1 = next((i for i, col in enumerate(column_all) if col.name == 'e1'), None)
        index_e2 = next((i for i, col in enumerate(column_all) if col.name == 'e2'), None)

        # Perform the calibration
        g = np.array([dat_all['e1_uncal'], dat_all['e2_uncal']])
        Rm1 = np.linalg.inv(R)

        c_corr = Rm1.dot(c)
        g_corr = Rm1.dot(g)
        for comp in (0, 1):
            g_corr[comp] = g_corr[comp] - c_corr[comp]

        # Replace the columns
        dat_all['e1'] = g_corr[0]
        dat_all['e2'] = g_corr[1]
        column_all[index_e1] = fits.Column(name='e1', array=g_corr[0], format='D')
        column_all[index_e2] = fits.Column(name='e2', array=g_corr[1], format='D')

        # Compute the leakage coefficients for each object
        if sh == 'ngmix':
            if verbose:
                print('Computing leakage coefficients per object.')
            num_bins = 20
            weight_type = 'des' if 'w_des' in col_names else 'iv'
            alpha_1, alpha_2 = get_alpha_leakage_per_object(dat_all, num_bins, weight_type)
            # Add the columns to the fits file
            column_all.append(fits.Column(name='alpha_1', array=alpha_1, format='D'))
            column_all.append(fits.Column(name='alpha_2', array=alpha_2, format='D'))


        if verbose:
            print(f"Wrting file {output_path}")
        cat.write_fits_BinTable_file(column_all, output_path, R, R_shear, R_select, c)
    else:
        if verbose:
            print('No calibration performed on the whole catalog.')
        g_corr = None
        cat.write_fits_BinTable_file(column_all, output_path)

    return c, R_shear, g_corr


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

    logging.log_command(argv)

    if param.survey == 'v1':
        n_patch = 7
        patches = [f'P{x}' for x in np.arange(n_patch) + 1]
    elif param.survey == 'test':
        patches = ['P7', 'W3', 'S4']
    elif param.survey == 'v1_small':
        patches = ['W3', 'P7']
    else:
        patches = param.survey.split("+")
    
    sh = 'ngmix'

    survey = 'unions'
    pipeline = 'shapepipe'
    year = 2024
    version = '1.4.a'

    additive_bias = 'from_extended'
    shear_response = 'from_extended'


    #if additive_bias == 'from_extended':
        #return_mean_e = True
    #else:
        #return_mean_e = False
    return_mean_e = additive_bias == "from_extended"
    return_mean_R_shear = shear_response == 'from_extended'

    R_select = get_R('R_select')

    # Extended catalogue
    if param.verbose:
        print('Merging extended catalogue')
    base_path = "."
    input_sub_path = f'sp_output/shape_catalog_extended_{sh}.fits'
    output_path = f'{survey}_{pipeline}_extended_{year}_v{version}.fits'
    c_ext, R_shear_ext, g_corr = merge_catalogues(
        patches,
        base_path,
        input_sub_path,
        output_path,
        sh,
        R_select=R_select,
        verbose=param.verbose,
        return_mean_e=return_mean_e,
        return_mean_R_shear=return_mean_R_shear,
        hist_base=sh,
    )

    fname = 'c.txt'
    dat = ascii.read(fname)
    if dat[-1]['patch'] != 'all':
        raise ValueError(
            f'Invalid file {fname}, last row does not correspond to patch=\'all\''
        )
    c_err = np.empty(2)
    c_err[0] = dat[-1]['dmcw_1']
    c_err[1] = dat[-1]['dmcw_2']

    c = np.empty(2)
    if additive_bias == 'from_extended':
        print('Getting additive bias from extended catalog')
        c = c_ext
    else:
        print('Getting additive bias from combined run')
        c[0] = dat[-1]['cw_1']
        c[1] = dat[-1]['cw_2']

    if shear_response == 'from_extended':
        print('Getting shear response from extended catalog')
        R_shear = R_shear_ext
        R = R_shear + R_select
    else:
        R_shear = get_R('R_shear', key_base='R_shear')

        # Check of total R
        R = get_R('R', key_base='R_tot')
        print('R - R_shear + R_select = 0 ?')
        print(R - R_shear - R_select)

    # Invert total response matrix
    Rm1 = np.linalg.inv(R)

    # Base catalogue
    ra_all = np.array([])
    dec_all = np.array([])
    g1_corr_mc_all = np.array([]) if g_corr is None else g_corr[0] #The columns are already computed from extended catalogue
    g2_corr_mc_all = np.array([]) if g_corr is None else g_corr[1]
    w_all = np.array([])
    mag_all = np.array([])
    snr_all = np.array([])
    patch_all = np.array([])
    if param.verbose:
        print('Merging base catalogue')
    for idx, patch in enumerate(patches):

        if param.verbose:
            print(' ', patch)

        input_path = f'{patch}/sp_output/shape_catalog_{sh}.fits'
        ra, dec, g1, g2, w, mag, _ = read_shape_catalog(input_path, w_name="w_iv")

        ra_all = np.append(ra_all, ra)
        dec_all = np.append(dec_all, dec)
        w_all = np.append(w_all, w)
        mag_all = np.append(mag_all, mag)
        patch_all = np.append(patch_all, [idx + 1] * len(ra))

        #Only performs the calibration if the calibration is not done on the extended catalog
        if g_corr is None:
            g = np.array([g1, g2])

            # Calibrate with global R and c
            c_corr = Rm1.dot(c)

            g_corr_mc = Rm1.dot(g)
            for comp in (0, 1):
                g_corr_mc[comp] = g_corr_mc[comp] - c_corr[comp]
            g1_corr_mc_all = np.append(g1_corr_mc_all, g_corr_mc[0])
            g2_corr_mc_all = np.append(g2_corr_mc_all, g_corr_mc[1])

    if sh == 'ngmix':
        ext_cat = fits.getdata(output_path , 1)
        w_all = ext_cat['w_des']
        e1_psf = ext_cat['e1_PSF']
        e2_psf = ext_cat['e2_PSF']
        alpha_1 = ext_cat['alpha_1']
        alpha_2 = ext_cat['alpha_2']
        e1_noleakage = g1_corr_mc_all - alpha_1 * e1_psf
        e2_noleakage = g2_corr_mc_all - alpha_2 * e2_psf
    output_path = f'{survey}_{pipeline}_{year}_v{version}.fits'
    g_corr_mc_all = np.array([g1_corr_mc_all, g2_corr_mc_all])

    add_col_data = { 'patch' : patch_all, }
    add_col_format = { 'patch' : 'I' }

    if sh == 'ngmix':
        add_col_data['e1_noleakage'] = e1_noleakage
        add_col_format['e1_noleakage'] = 'D'
        add_col_data['e2_noleakage'] = e2_noleakage
        add_col_format['e2_noleakage'] = 'D'
    
    write_shape_catalog(
        output_path, 
        ra_all,
        dec_all,
        w_all,
        mag=mag_all,
        g=g_corr_mc_all,
        R=R,
        R_shear=R_shear, 
        R_select=R_select,
        c=c,
        c_err=c_err,
        w_type="des",
        add_cols=add_col_data,
        add_cols_format=add_col_format, 
    )

    # PSF catalogue with single-epoch shapes (HSM moments)
    if param.verbose:
        print('Merging PSF catalogues (with single-epoch moments shapes)')
    # Path of ShapePipe run
    base_path = param.star_cat_base_path
    input_sub_path = (
        "output/run_sp_Ms/merge_starcat_runner/output/full_starcat-0000000.fits"
    )

    output_path = f'{survey}_{pipeline}_psf_{year}_v{version}.fits'
    merge_catalogues(
        patches,
        base_path,
        input_sub_path,
        output_path,
        None,
        hdu_in=2,
        verbose=param.verbose,
    )

    # Star catalogue with multi-epoch shapes (ngmix); from matching with
    # galaxy sample
    if param.verbose:
        print('Merging star catalogues (matched with galaxy catalogue for shapes')
    base_path = "."
    input_sub_path = f'sp_output/psf_catalog_{sh}.fits'
    output_path = f'{survey}_{pipeline}_star_{year}_v{version}.fits'
    merge_catalogues(
        patches,
        base_path,
        input_sub_path,
        output_path,
        None,
        verbose=param.verbose,
    )



if __name__ == "__main__":
    sys.exit(main(sys.argv))
