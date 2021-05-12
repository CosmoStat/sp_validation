"""
  
:Name: cat.py

:Description: This script contains methods to deal with catalogues.

:Author: Martin Kilbinger

:Date: 2021

:Package: sp_validation

"""

import numpy as np

from sp_validation import util
from sp_validation import io
from sp_validation import basic


def print_some_quantities(dd, ell_col_name, ell_n_comp, stats_file, verbose=False):
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
    verbose : bool, optional, default=False
        verbose output if True
    """

    # Print all column names
    if verbose:
        print('Column names:')
        print(dd.dtype.names)
        print('')

    # Mean ellipticity looks reasonable, no `nan`?
    ell = np.zeros(2)
    if ell_n_comp == 1:
        for i in (0, 1):
            ell[i] = dd[ell_col_name[i]].mean()
    elif ell_n_comp == 2:
        for i in (0, 1):
            ell[i] = dd[ell_col_name][:,i].mean()

    n_tot = len(dd)
    msg = 'Total number of objects = {} = {}\n'.format(n_tot, util.millify(n_tot))
    io.print_stats(msg, stats_file, verbose=verbose)

    io.print_stats('Mean ellipticity:', stats_file, verbose=verbose)
    for i in (0, 1):
        msg = '<e_{}> = {:.3g}'.format(i, ell[i])
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
