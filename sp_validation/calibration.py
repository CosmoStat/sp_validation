"""CALIBRATION.

:Name: calibration.py

:Description: This script contains methods for shear calibration.

:Author: Martin Kilbinger

"""

import numpy as np
import pandas as pd

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

def get_w_des(cat_gal, num_bins):
    """
    Get DES weights. (Gatti et al. 2021)
    Return an array of DES weights obtained by binning in SNR and size and computing the ratio between
    the shear response and the shape noise.

    Parameters
    ----------
    cat_gal: dict
        A catalog of galaxies containing the response matrix and the uncalibrated ellipticities
    num_bins : int
        Number of bins to use for the binning of the SNR and size.

    Returns
    -------
    w : array of float
        DES weights
    """
    size_ratio = cat_gal['NGMIX_T_NOSHEAR'] / cat_gal['NGMIX_Tpsf_NOSHEAR']

    #Build a pandas dataframe to perform the binning and the computation of the weights
    df_gal = pd.DataFrame(
        np.array([
            np.array(cat_gal['e1_uncal'], dtype=np.float32),
            np.array(cat_gal['e2_uncal'], dtype=np.float32),
            np.array(cat_gal['R_g11'], dtype=np.float32),
            np.array(cat_gal['R_g12'], dtype=np.float32),
            np.array(cat_gal['R_g21'], dtype=np.float32),
            np.array(cat_gal['R_g22'], dtype=np.float32),
            np.array(cat_gal['snr'], dtype=np.float32),
            np.array(size_ratio, dtype=np.float32)
        ]).T, columns=['e1', 'e2', 'R_g11', 'R_g12', 'R_g21', 'R_g22', 'snr', 'size']
    )

    del size_ratio

    #Create logarithmic bins in size and SNR
    bin_edges_snr = np.logspace(
        np.log10(df_gal['snr'].min()), np.log10(300), num_bins + 1
    )

    df_gal['snr_log_bins'] = pd.cut(
        df_gal['snr'], bin_edges_snr, labels=False
    )
    df_gal.loc[np.isnan(df_gal['snr_log_bins']), 'snr_log_bins'] = num_bins - 1

    bin_edges_size = np.logspace(
        np.log10(df_gal['size'].min()), 
        np.log10(df_gal['size'].max()),
        num_bins + 1
    )
    df_gal['size_log_bins'] = pd.cut(df_gal['size'], bin_edges_size, labels=False)

    #Compute shape noise in each bin
    shape_noise = np.zeros((num_bins, num_bins))

    for i in range(num_bins):
        for j in range(num_bins):
            bin_mask = (df_gal['snr_log_bins'] == i) & (df_gal['size_log_bins'] == j)
            ngal = np.sum(bin_mask)
            shape_noise[i, j] = 0.5*(np.sum(df_gal[bin_mask]['e1']**2)/ngal+np.sum(df_gal[bin_mask]['e2']**2)/ngal)

    #Compute the response in each bin
    response = np.zeros((num_bins, num_bins))

    for i in range(num_bins):
        for j in range(num_bins):
            bin_mask = (df_gal['snr_log_bins'] == i) & (df_gal['size_log_bins'] == j)
            response[i, j] = 0.5*(np.average(df_gal[bin_mask]['R_g11'])+np.average(df_gal[bin_mask]['R_g22']))

    #Compute the weights
    for i in range(num_bins):
        for j in range(num_bins):
            bin_mask = (df_gal['size_log_bins'] == i) & (df_gal['snr_log_bins'] == j)
            df_gal.loc[bin_mask, 'w_des'] = response[i, j]**2/shape_noise[i, j]
        
    return np.array(df_gal['w_des'])