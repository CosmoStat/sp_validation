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
