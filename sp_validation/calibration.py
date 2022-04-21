"""
  
:Name: calibration.py

:Description: This script contains methods for shear calibration.

:Author: Martin Kilbinger

:Date: 2021

:Package: sp_validation

"""

import numpy as np

from astropy.io import fits

from sp_validation import util
from sp_validation import io
from sp_validation import basic
from sp_validation.survey import get_footprint


def get_calibrated_quantities(gal_metacal, shape_method='ngmix'):
    """Get Calibrated Quantities

    Return catalogue quantities for objects calibrated for multiplicative bias.

    Parameters
    ----------
    gal_metacal : dict
        galaxy metacalibration catalogue
    shape_method : string, optional, default='ngmix'
        shape measurement method, one in 'ngmix', 'galsim'
    verbose : optional, bool, default=False
        verbose output if True

    Returns
    -------
    g_corr : array(2, ngal) of float
        shear estimates calibrated for multiplicative bias
    g_uncorr : array(2, ngal) of float
        uncalibrated shear estimates
    w : array of float
        weights
    mask : array of bool
        mask to indicate valid objects in "no-shear" sample
    """

    # mask for 'no shear' images
    mask = gal_metacal.mask_dict['ns']

    # uncalibrated shear estimates
    g_uncorr = np.array([gal_metacal.ns['g1'][mask], gal_metacal.ns['g2'][mask]])

    # calibratied shear estimates: multiply with inverse response matrix
    g_corr = np.linalg.inv(gal_metacal.R).dot(g_uncorr)

    # weights
    w = gal_metacal.ns['w'][mask]

    return g_corr, g_uncorr, w, mask


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
