"""CALIBRATION.

:Name: calibration.py

:Description: This script contains methods for shear calibration.

:Author: Martin Kilbinger

"""

import numpy as np

from astropy.io import fits

from sp_validation import util
from sp_validation import io
from sp_validation import basic
from sp_validation.survey import get_footprint


def get_calibrated_quantities(gal_metacal, shape_method='ngmix'):
    """Get Calibrated Quantities.

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
    g_uncorr = np.array([
        gal_metacal.ns['g1'][mask],
        gal_metacal.ns['g2'][mask]
    ])

    # calibratied shear estimates: multiply with inverse response matrix
    g_corr = np.linalg.inv(gal_metacal.R).dot(g_uncorr)

    # weights
    w = gal_metacal.ns['w'][mask]

    return g_corr, g_uncorr, w, mask
