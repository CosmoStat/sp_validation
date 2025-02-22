"""CALIBRATION.

:Name: calibration.py

:Description: This script contains methods for shear calibration.

:Author: Martin Kilbinger

"""

import numpy as np
import pandas as pd

from astropy.io import fits
import statsmodels.api as sm

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
            np.array(cat_gal['e1_uncal'], dtype=np.float64),
            np.array(cat_gal['e2_uncal'], dtype=np.float64),
            np.array(cat_gal['R_g11'], dtype=np.float64),
            np.array(cat_gal['R_g12'], dtype=np.float64),
            np.array(cat_gal['R_g21'], dtype=np.float64),
            np.array(cat_gal['R_g22'], dtype=np.float64),
            np.array(cat_gal['snr'], dtype=np.float64),
            np.array(size_ratio, dtype=np.float64)
        ]).T, columns=['e1', 'e2', 'R_g11', 'R_g12', 'R_g21', 'R_g22', 'snr', 'size']
    )
    
    del size_ratio

    #Create logarithmic bins in size and SNR
    bin_edges_snr = np.logspace(
        np.log10(df_gal['snr'].min()), np.log10(300.), num_bins + 1
    )

    df_gal['snr_log_bins'] = pd.cut(
        df_gal['snr'], bin_edges_snr, labels=False
    )
    df_gal.loc[np.isnan(df_gal['snr_log_bins']), 'snr_log_bins'] = num_bins - 1

    bin_edges_size = np.logspace(
        np.log10(0.5), 
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

def get_alpha_leakage_per_object(cat_gal, num_bins, weight_type='des'):
    """
    Compute the leakage per object (Li et al. 2024)
    Return an array of leakage coefficients obtained by binning in SNR and size.

    Parameters
    ----------
    cat_gal : dict
        A catalog of galaxies containing galaxy ellipticity, PSF ellipticity, SNR and size of the galaxy and the PSF.
    num_bins : int
        Number of bins

    Returns
    -------
    alpha_1 : np.array
        Array containing the correction coefficient for the PSF leakage per object for the first component.
    alpha_2 : np.array
        Array containing the correction coefficient for the PSF leakage per object for the second component.
    """
    assert weight_type in ['des', 'iv'], "weight_type must be either 'des' or 'iv'"
    #Compute the size ratio
    size_ratio = cat_gal['NGMIX_Tpsf_NOSHEAR']/(cat_gal['NGMIX_T_NOSHEAR']+cat_gal['NGMIX_Tpsf_NOSHEAR'])

    df_gal = pd.DataFrame(np.array([
        np.array(cat_gal['e1'], dtype=np.float64),
        np.array(cat_gal['e2'], dtype=np.float64),
        np.array(cat_gal['e1_PSF'], dtype=np.float64),
        np.array(cat_gal['e2_PSF'], dtype=np.float64),
        np.array(cat_gal['snr'], dtype=np.float64),
        np.array(cat_gal[f'w_{weight_type}'], dtype=np.float64),
        np.array(size_ratio, dtype=np.float64)
    ]).T, columns=['e1', 'e2', 'e1_PSF', 'e2_PSF', 'snr', 'w', 'size_ratio'])

    del size_ratio

    n_bins_snr = num_bins
    n_bins_r = num_bins

    #Create logarithmic bins in size and SNR
    df_gal.loc[:, 'bin_R'] = pd.qcut(df_gal['size_ratio'], n_bins_r, labels=False, retbins=False)

    #initialize bin snr
    df_gal.loc[:, 'bin_snr'] = -999

    for ibin_r in range(n_bins_r):
        #select galaxies in the bin
        mask_binr = df_gal['bin_R'].values == ibin_r

        #bin in snr
        df_gal.loc[mask_binr, 'bin_snr'] = pd.qcut(df_gal.loc[mask_binr, 'snr'], n_bins_snr, labels=False, retbins=False)

    #group by bin
    df_gal_grouped = df_gal.groupby(['bin_R', 'bin_snr'])
    ngroups = df_gal_grouped.ngroups

    #Performing first round calibration
    alpha_df = pd.DataFrame(0.,
                            index = np.arange(ngroups),
                            columns = ['R', 'SNR',
                                       'alpha_1', 'alpha_2',
                                       'alpha_1_err', 'alpha_2_err'])
    
    i_group = 0
    for name, group in df_gal_grouped:

        #Save weighted average
        alpha_df.loc[i_group, 'R'] = np.average(group['size_ratio'], weights=group['w'])
        alpha_df.loc[i_group, 'SNR'] = np.average(group['snr'], weights=group['w'])

        #Fit linear model to compute alpha
        e1_out = np.array(group['e1'])
        e2_out = np.array(group['e2'])
        weight_out = np.array(group['w'])
        e1_PSF = np.array(group['e1_PSF'])
        e2_PSF = np.array(group['e2_PSF'])
        del group

        #Fit e1
        mod_wls = sm.WLS(e1_out, sm.add_constant(e1_PSF), weights=weight_out)
        try:
            res_wls = mod_wls.fit()
        except:
            print("Linear regression fit for PSF leakage failed")
            raise
        alpha_df.loc[i_group, 'alpha_1'] = res_wls.params[1]
        alpha_df.loc[i_group, 'alpha_1_err'] = np.sqrt(res_wls.cov_params()[1, 1])
        del res_wls, mod_wls

        #Fit e2
        mod_wls = sm.WLS(e2_out, sm.add_constant(e2_PSF), weights=weight_out)
        res_wls = mod_wls.fit()
        alpha_df.loc[i_group, 'alpha_2'] = res_wls.params[1]
        alpha_df.loc[i_group, 'alpha_2_err'] = np.sqrt(res_wls.cov_params()[1, 1])
        del weight_out, res_wls, mod_wls

        i_group += 1

        #Fit polynomial to remove general trend
    fitting_weight = 1./np.square(alpha_df['alpha_1_err'].values)
    A = np.vstack([fitting_weight*1,
    fitting_weight*np.power(alpha_df['SNR'].values, -2),
    fitting_weight*np.power(alpha_df['SNR'].values, -3),
    fitting_weight*alpha_df['R'].values,
    fitting_weight*alpha_df['R'].values*np.power(alpha_df['SNR'].values, -2)]).T
    poly1 = np.linalg.lstsq(A, fitting_weight*alpha_df['alpha_1'].values, rcond=None)[0]
    del fitting_weight, A

    fitting_weight = 1./np.square(alpha_df['alpha_2_err'].values)
    A = np.vstack([fitting_weight*1,
    fitting_weight*np.power(alpha_df['SNR'].values, -2),
    fitting_weight*np.power(alpha_df['SNR'].values, -3),
    fitting_weight*alpha_df['R'].values,
    fitting_weight*alpha_df['R'].values*np.power(alpha_df['SNR'].values, -2)]).T
    poly2 = np.linalg.lstsq(A, fitting_weight*alpha_df['alpha_2'].values, rcond=None)[0]
    del fitting_weight, A

    #Compute alpha to remove the general trend
    #e1
    alpha_1 = poly1[0] \
        + poly1[1] * np.power(df_gal['snr'], -2) \
        + poly1[2] * np.power(df_gal['snr'], -3) \
        + poly1[3] * df_gal['size_ratio'] \
        + poly1[4] * df_gal['size_ratio'] * np.power(df_gal['snr'], -2)

    df_gal.loc[:, 'e1_cor'] = df_gal['e1'].values - alpha_1 * df_gal['e1_PSF'].values
    del poly1

    #e2
    alpha_2 = poly2[0] \
        + poly2[1] * np.power(df_gal['snr'], -2) \
        + poly2[2] * np.power(df_gal['snr'], -3) \
        + poly2[3] * df_gal['size_ratio'] \
        + poly2[4] * df_gal['size_ratio'] * np.power(df_gal['snr'], -2)
    df_gal.loc[:, 'e2_cor'] = df_gal['e2'].values - alpha_2 * df_gal['e2_PSF'].values
    del poly2

    #Initialise second run of calibration
    df_gal.loc[:, 'alpha_1_cor'] = -999.
    df_gal.loc[:, 'alpha_2_cor'] = -999.
    alpha_df.loc[:, 'alpha_1_corr'] = -999.
    alpha_df.loc[:, 'alpha_2_corr'] = -999.
    alpha_df.loc[:, 'alpha_1_corr_err'] = -999.
    alpha_df.loc[:, 'alpha_2_corr_err'] = -999.

    #Second run of calibration
    for i_group in range(n_bins_r*n_bins_snr):
        #Get the mask
        mask_group = (df_gal['bin_R'].values == i_group//n_bins_snr) & (df_gal['bin_snr'].values == i_group%n_bins_snr)
        #Fit linear model to compute alpha
        e1_out = np.array(df_gal.loc[mask_group, 'e1_cor'])
        e2_out = np.array(df_gal.loc[mask_group, 'e2_cor'])
        weight_out = np.array(df_gal.loc[mask_group, 'w'])
        e1_PSF = np.array(df_gal.loc[mask_group, 'e1_PSF'])
        e2_PSF = np.array(df_gal.loc[mask_group, 'e2_PSF'])
        
        #Fit e1
        mod_wls = sm.WLS(e1_out, sm.add_constant(e1_PSF), weights=weight_out)
        res_wls = mod_wls.fit()
        alpha_df.loc[i_group, 'alpha_1_corr'] = res_wls.params[1]
        alpha_df.loc[i_group, 'alpha_1_corr_err'] = np.sqrt(res_wls.cov_params()[1, 1])
        del res_wls, mod_wls

        #Fit e2
        mod_wls = sm.WLS(e2_out, sm.add_constant(e2_PSF), weights=weight_out)
        res_wls = mod_wls.fit()
        alpha_df.loc[i_group, 'alpha_2_corr'] = res_wls.params[1]
        alpha_df.loc[i_group, 'alpha_2_corr_err'] = np.sqrt(res_wls.cov_params()[1, 1])
        del weight_out, res_wls, mod_wls

        df_gal.loc[mask_group, 'alpha_1_corr'] = alpha_df.loc[i_group, 'alpha_1_corr']
        df_gal.loc[mask_group, 'alpha_2_corr'] = alpha_df.loc[i_group, 'alpha_2_corr']

    alpha_1 += df_gal['alpha_1_corr'].values
    alpha_2 += df_gal['alpha_2_corr'].values

    return alpha_1, alpha_2

def get_calibrate_e_from_cat(path_cat_gal, weight_type='des', verbose=False):
    """
    Calibrates ellipticities from a galaxy catalog with a certain weight type.

    Parameters
    ----------
    path_cat_gal : str
        Path to the galaxy catalog
    weight_type : str, optional, default='des'
        Type of weight to use. Options are 'des' (DES weight) or 'iv' (inverse variance)
    verbose : bool, optional, default=False
        If True, print intermediate results

    Returns
    -------
    g_cal : np.array
        Calibrated ellipticities
    """
    assert (weight_type in ['iv', 'des']), "The weight_type is not correct. Options 'iv' (inverse variance) or 'des'."
    
    hdu_gal = fits.open(path_cat_gal)

    cat_gal = hdu_gal[1].data

    R_select = np.array([
        [hdu_gal[0].header['R_S11'], hdu_gal[0].header['R_S12']],
        [hdu_gal[0].header['R_S21'], hdu_gal[0].header['R_S22']]
    ])

    if verbose:
        print('R_select\n', R_select)

    g = np.array([cat_gal['e1_uncal'], cat_gal['e2_uncal']])

    c = np.average(g, axis=1, weights=cat_gal['w_'+weight_type])

    if verbose:
        print('Additive bias\n', c)

    R_shear = np.zeros((2, 2))
    for iidx in range(2):
        for jidx in range(2):
            R_shear[iidx, jidx] = np.mean(cat_gal[f'R_g{iidx+1}{jidx+1}'])

    if verbose:
        print('R_shear\n', R_shear)

    R = R_shear + R_select

    if verbose:
        print('R\n', R)
    
    c_corr = np.linalg.inv(R) @ c
    g_cal = np.linalg.inv(R) @ g

    for comp in (0, 1):
        g_cal[comp] -= c_corr[comp]

    return g_cal[0], g_cal[1]

def get_calibrate_no_leakage_e_from_cat(path_cat_gal, weight_type='des', verbose=False):
    """
    Calibrate ellipticities and removes leakage from a galaxy catalog with a certain weight type.

    Parameters
    ----------
    path_cat_gal : str
        Path to the galaxy catalog
    weight_type : str, optional, default='des'
        Type of weight to use. Options are 'des' (DES weight) or 'iv' (inverse variance)
    verbose : bool, optional, default=False
        If True, print intermediate results
    
    Returns
    -------
    e1_noleak : np.array
        Calibrated ellipticities without leakage for the first component
    e2_noleak : np.array
        Calibrated ellipticities without leakage for the second component
    """
    e1, e2 = get_calibrate_e_from_cat(path_cat_gal, weight_type, verbose)

    cat_gal = fits.getdata(path_cat_gal)
    e1_noleak = e1 - cat_gal['alpha_1']*cat_gal['e1_PSF']
    e2_noleak = e2 - cat_gal['alpha_2']*cat_gal['e2_PSF']

    return e1_noleak, e2_noleak