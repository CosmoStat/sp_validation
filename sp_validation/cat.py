"""
  
:Name: cat.py

:Description: This script contains methods to deal with catalogues.

:Author: Martin Kilbinger

:Date: 2021

:Package: sp_validation

"""

import numpy as np

from astropy.io import fits

from sp_validation import util
from sp_validation import io
from sp_validation import basic


def print_some_quantities(dd, ell_col_name, ell_n_comp, stats_file, invalid=-10, verbose=False):
    """Print some quantities.

    Output some summary statistics from a catalogue.

    Parameters
    ----------
    dd : dict
        galaxy catalog
    ell_col_name : array of string
        ellipticity column name(s)
    ell_n_comp : int
        dimension (= number of components) of ellipticity column.
        Should be 1 or 2.
    stats_file : file handler
        summary statistics output file handler
    invalid : float, optional, default -10
        flag objects with ellipticty value = invalid
    verbose : bool, optional, default=False
        verbose output if True
    """

    # Print all column names
    if verbose:
        print('Column names:')
        print(dd.dtype.names)
        print('')

    n_tot = len(dd)
    msg = 'Total number of objects = {} = {}'.format(n_tot, util.millify(n_tot))
    io.print_stats(msg, stats_file, verbose=verbose)

    # Get ellipticity columns
    all_ell = []
    if ell_n_comp == 1:
        for i in (0, 1):
            all_ell.append(dd[ell_col_name[i]])
    elif ell_n_comp == 2:
        for i in (0, 1):
            all_ell.append(dd[ell_col_name][:,i])

    # Tolerance around invalid value
    EPS = 0.0001

    # Index list of valid objects
    ind_val = np.zeros(shape=(2, n_tot), dtype=np.bool)
    for i in (0, 1):
       ind_val[i] = np.abs(all_ell[i] - invalid) > EPS
    # Valid objects = those for which both ellipticity
    # are valid
    ind_v = ind_val[0] & ind_val[1]

    # Select valid objects
    for i in (0, 1):
        all_ell[i] = all_ell[i][ind_v]

    n_tot_val = len(np.where(ind_v)[0])
    msg = 'Total number of valid objects = {} = {}'.format(n_tot_val, util.millify(n_tot_val))
    io.print_stats(msg, stats_file, verbose=verbose)

    msg = 'Fraction of invalid objects = {}/{} = {:.3g}%\n' \
          ''.format(n_tot - n_tot_val, n_tot,
                    (n_tot - n_tot_val) / n_tot * 100)
    io.print_stats(msg, stats_file, verbose=verbose)

    # Mean ellipticity
    ell = np.zeros(2)
    for i in (0, 1):
        ell[i] = all_ell[i].mean()

    io.print_stats('Mean ellipticity of valid objects:', stats_file, verbose=verbose)
    for i in (0, 1):
        msg = '<e_{}> = {:.3g}'.format(i+1, ell[i])
        io.print_stats(msg, stats_file, verbose=verbose)


def check_matching(d1, d2, keys_1, keys_2, thresh, stats_file, name=None, verbose=False):
    """Check matching

    Check matching between two catalogues.

    Parameters
    ----------
    d1, d2 : dict
        catalogs
    key_1, key_2 : array(2) of string
        column keys for d1, d2, corresponding to x, y
    thres : float
        threshold for matching, in deg
    stats_file : file handler
        summary statistics output file handler
    verbose : bool, optional, default=False
        verbose output if True

    Returns
    -------
    ind : array of int
        index list of d2 of objects that were matched to d1
    mask_area_tiles : array of int
        index list of tiles in footprint
    """
    
    if name is not None:
        # Filter stars outside footprint for efficiency
        get_mask = getattr(basic, 'get_mask_footprint_{}'.format(name))
        mask_area_tiles = get_mask(d1[keys_1[0]], d1[keys_1[1]])
        if len(np.where(mask_area_tiles)[0]) == 0:
            raise ValueError('Error: no object found in field \'{}\''.format(name))
    else:
        mask_area_tiles = np.arange(len(d1))


    # Match stars from exposure (PSF) catalogue to total catalogue
    ind = basic.match_stars2(d2['XWIN_WORLD'], d2['YWIN_WORLD'], d1['RA'][mask_area_tiles],
                             d1['DEC'][mask_area_tiles], thresh=thresh)

    n_tot = len(d1[keys_1[0]][mask_area_tiles])
    msg = 'Number of matched stars from exposures to total catalogue = {}/{} = {:.1f}%' \
        ''.format(len(ind), n_tot, len(ind) / n_tot * 100)
    io.print_stats(msg, stats_file, verbose=verbose)

    # Remove stars matched to more than one target object
    ind = np.array(list(set(ind)))

    msg = 'Number of matched stars after removing multiple matches = {}/{} = {:.1f}%' \
        ''.format(len(ind), n_tot, len(ind) / n_tot * 100)
    io.print_stats(msg, stats_file, verbose=verbose)

    return ind, mask_area_tiles, n_tot


def check_invalid(dd, key, comp, val, stats_file, name=None, verbose=False):
    """Check invalid objects

    Check whether objects have invalid values.

    Parameters
    ----------
    dd : dict
        catalog
    key : array of string
        key names of columns to check
    comp : array of int
        components for above columns
    val : array of float
        values for above columns indicating invalid entries
    stats_file : file handler
        summary statistics output file handler
    name : array of string, optional, default=None
        for output message. If None, key strings are used
    verbose : bool, optional, default=False
        verbose output if True
    """

    if name is None:
        name = key

    n_all = len(dd)

    for i in range(len(key)):
    
        w = dd[key[i]][:,comp[i]] == val[i]
        n_inv_psf = len(np.where(w)[0])
        msg = 'Invalid {} found for {}/{} = {:.1g}% objects' \
              ''.format(name[i], n_inv_psf, n_all, n_inv_psf / n_all)
        io.print_stats(msg, stats_file, verbose=verbose)


def match_subsample(dd, ind, mask, pos_key, ell_key, n_ref, stats_file, verbose=False):
    """Match subsample

    Match subsample of catalogue.

    Parameters
    ----------
    dd : dict
        catalog
    ind : array of int
        index list of d2 of objects that were matched to d1
    mask : array of bool
        boolean mask
    pos_key : array(2) of string
        key names for position columns
    ell_key : string
        key name for ellipticity column
    n_ref : int
        reference number of objects
    stats_file : file handler
        summary statistics output file handler
    verbose : bool, optional, default=False
        verbose output if True

    Returns
    -------
    ra, dec : array of float
        positions
    g : array(2) of float
        ellipticities
    """
    
    msg = 'Number of stars matched to valid sample = {}/{} = {:.1f}%' \
          ''.format(len(dd[pos_key[0]][ind][mask]), n_ref,
                    len(dd[pos_key[0]][ind][mask]) / n_ref * 100)
    io.print_stats(msg, stats_file, verbose=verbose)

    ra = dd[pos_key[0]][ind][mask]
    dec = dd[pos_key[1]][ind][mask]
    g1 = dd[ell_key][:,0][ind][mask]
    g2 = dd[ell_key][:,1][ind][mask]
    g = np.array([g1, g2])
    
    return ra, dec, g


def match_spread_class(dd, ind, mask, stats_file, n_ref, verbose=False):
    """Match spread class

    Match
    """
    
    tot_star = n_ref
    tot_as_star = len(np.where(dd['SPREAD_CLASS'][ind][mask] == 0)[0])
    tot_as_gal = len(np.where(dd['SPREAD_CLASS'][ind][mask] == 1)[0])
    tot_as_other = len(np.where(dd['SPREAD_CLASS'][ind][mask] == 2)[0])

    msg = 'Number of stars selected as star (SPREAD_CLASS=0)   = {}/{} = {:.1f}%' \
      ''.format(tot_as_star, tot_star, tot_as_star/tot_star*100)
    io.print_stats(msg, stats_file, verbose=verbose)

    msg = 'Number of stars selected as galaxy (SPREAD_CLASS=1) = {}/{} = {:.1f}%' \
          ''.format(tot_as_gal, tot_star, tot_as_gal/tot_star*100)
    io.print_stats(msg, stats_file, verbose=verbose)
 
    msg = 'Number of stars selected as other (SPREAD_CLASS=2)  = {}/{} = {:.1f}%' \
          ''.format(tot_as_other,tot_star, tot_as_other/tot_star*100)
    io.print_stats(msg, stats_file, verbose=verbose)


def write_shape_catalog(output_path, ra, dec, g1, g2, w, mag, snr,
                        R, R_shear, R_select, c, c_err, alpha_leakage,
                        g1_uncal=None, g2_uncal=None, R_11=None,
                        R_22=None, R_12=None, R_21=None):
    """Write Shape Catalog

    Write catalogue with galaxy shapes = shear estimates.

    Parameters
    ----------
    output_path : string
        output file path
    ra, dec : arrays of float
        coordinates in deg
    g1, g2 : arrays of float
        calibrated reduced shear estimate components, corrected for multiplicative
        and additive bias, gi = R^-1 gi_uncal - ci
    w : array of float
        weights
    mag, snr : arrays of float
        magnitude, signal-to-noise ratio
    R : 2x2 matrix of float
        Mean full response matrix
    R_shear : 2x2 matrix of float
        Mean shear response matrix
    R_select : 2x2 matrix of float
        Global selection response matrix
    c : array(2) of float
        additive shear bias
    c_err : array(2) of float
        error of c
    alpha_leakage : float
        Mean scale-dependent PSF leakage
    """

    # Data HDU
    c_ra = fits.Column(name='ra', array=ra, format='D', unit='deg')
    c_dec = fits.Column(name='dec', array=dec, format='D', unit='deg')
    c_g1 = fits.Column(name='g1', array=g1, format='D')
    c_g2 = fits.Column(name='g2', array=g2, format='D')
    c_w = fits.Column(name='w', array=w, format='D')
    c_mag = fits.Column(name='mag', array=mag, format='D')
    c_snr = fits.Column(name='snr', array=snr, format='D')
    cols = [c_ra, c_dec, c_g1, c_g2, c_w, c_mag, c_snr]

    for x, name in zip([g1_uncal, g2_uncal, R_11, R_22, R_12, R_21],
                       ['g1_uncal', 'g2_uncal', 'R_11', 'R_22', 'R_12', 'R_21']):
        if x is not None:
            cols.append(fits.Column(name=name, array=x, format='D'))

    table_hdu = fits.BinTableHDU.from_columns(cols)

    table_hdu.header['TTYPE3'] = ('g1', 'Calibrated reduced shear estimate, 1st comp')
    table_hdu.header['TTYPE4'] = ('g2', 'Calibrated reduced shear estimate, 2nd comp')
    table_hdu.header['TTYPE5'] = ('w', 'Weight')
    table_hdu.header['TTYPE6'] = ('mag', 'Magnitude = MAG_AUTO (SExtractor)')
    table_hdu.header['TTYPE7'] = ('snr', 'Signal-to-noise ratio = flux/flux_std')
    
    ntype = 8
    for x in ([g1_uncal, g2_uncal], ['g1_uncal', 'g2_uncal']):
        if x is not None:
            table_hdu.header['TTYPE{}'.format(ntype)] = (name, 'uncalibrated reduced shear')
            ntype += 1
    for x, name in zip([R_11, R_22, R_12, R_21], ['R_11', 'R_22', 'R_12', 'R_21']):
        if x is not None:
            table_hdu.header['TTYPE{}'.format(ntype)] = (name, f'full response matrix {name}')
            ntype += 1

    # Primary HDU with information in header
    primary_header = fits.Header()

    primary_header['R'] = (r'<R>', r'Mean full response matrix <R> = <R_shear> + <R_select>')
    primary_header['R_11'] = (R[0,0], 'Full response matrix comp 1 1')
    primary_header['R_12'] = (R[0,1], 'Full response matrix comp 1 2')
    primary_header['R_21'] = (R[1,0], 'Full response matrix comp 2 1')
    primary_header['R_22'] = (R[1,1], 'Full response matrix comp 2 2')

    primary_header['R_g'] = (r'<R_g>', r'Mean shear response matrix <R_shear>')
    primary_header['R_g11'] = (R_shear[0,0], 'Mean shear response matrix comp 1 1')
    primary_header['R_g12'] = (R_shear[0,1], 'Mean shear response matrix comp 1 2')
    primary_header['R_g21'] = (R_shear[1,0], 'Mean shear response matrix comp 2 1')
    primary_header['R_g22'] = (R_shear[1,1], 'Mean shear response matrix comp 2 2')

    primary_header['R_S'] = (r'<R_S>', r'Global selection response matrix <R_select>')
    primary_header['R_S11'] = (R_select[0,0], 'Global selection response matrix comp 1 1')
    primary_header['R_S12'] = (R_select[0,1], 'Global selection response matrix comp 1 2')
    primary_header['R_S21'] = (R_select[1,0], 'Global selection response matrix comp 2 1')
    primary_header['R_S22'] = (R_select[1,1], 'Global selection response matrix comp 2 2')

    primary_header['c_1'] = (c[0], 'Additive bias 1st comp')
    primary_header['c1_err'] = (c_err[0], 'Standard deviation of c_1')
    primary_header['c2'] = (c[1], 'Additive bias 2nd comp')
    primary_header['c2_err'] = (c_err[1], 'Standard deviation of c_2')

    primary_header['w'] = ('Weight', r'1 / (2*sig_SN^2 + sig^2(g_1) + sig^2(g_2)')
    primary_header['sig_SN'] = (0.34, 'Shape noise RMS')

    primary_header['alpha'] = (alpha_leakage, 'Mean scale-dependent PSF leakage')

    primary_hdu = fits.PrimaryHDU(header=primary_header)

    # Final file
    hdu_list = fits.HDUList([primary_hdu, table_hdu])

    hdu_list.writeto(output_path, overwrite=True)
