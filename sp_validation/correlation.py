"""
  
:Name: correlation.py

:Description: This script contains methods to deal with
auto- and cross-correlations.

:Author: Martin Kilbinger

:Date: 2021

:Package: sp_validation

"""

import numpy as np

import treecorr


def xi_star_gal_tc(ra_gal, dec_gal, e1_gal, e2_gal, w_gal, ra_star, dec_star, e1_star, e2_star, w_star=None):
    """xi star gal tc

    Cross-correlation between galaxy and star ellipticities.
    """

    cat_gal = treecorr.Catalog(ra=ra_gal, dec=dec_gal, g1=e1_gal, g2=e2_gal,
                               w=w_gal, ra_units='degrees', dec_units='degrees')
    cat_star = treecorr.Catalog(ra=ra_star, dec=dec_star, g1=e1_star, g2=e2_star,
                                w=w_star, ra_units='degrees', dec_units='degrees')

    TreeCorrConfig = {'ra_units': 'degrees', 'dec_units': 'degrees',
                      'max_sep': 200, 'min_sep': 2, 'sep_units': 'arcminutes',
                      'nbins': 20}

    ng = treecorr.GGCorrelation(TreeCorrConfig)

    ng.process(cat_gal, cat_star)

    return ng


def T_to_fwhm(T):
    """T to fwhm

    Transform from size T to FWHM.

    Parameters
    ----------
    T : (array of) float
        input size(s)

    Returns
    -------
    fwhm : (array of) float
        output fwhm(s)
    """

    return T / 1.17741 * 2.355


def sigma_to_fwhm(sigma, pixel_size=1):
    """sigma to fwhm

    Transform from size sigma to FWHM.

    Parameters
    ----------
    sigma : (array of) float
        input size(s)
    pixel_size : float, optional, default=1
        pixel size in arcsec, set to 1 if no scaling
        required

    Returns
    -------
    fwhm : (array of) float
        output fwhm(s)
    """

    return sigma * 2.355 * pixel_size


def correlation_12_22(ra_1, dec_1, e1_1, e2_1, weights_1, ra_2, dec_2, e1_2, e2_2):
    """Correlation 12 22

    Correlation functions between two samples 1 and 2. Compute xi_12 and xi_22.

    Parameters
    ----------
    ra_1, dec_1 : array of float
        coordinates of sample 1
    e1_1, e2_1 : array of float
        ellipticities of sample 1
    weights_1 : array of float
        weights of sample 1
    ra_2, dec_2 : array of float
        coordinates of sample 2
    e1_2, e2_2 : array of float
        ellipticities of sample 2

    Returns
    -------
    xi_12, xi_22 : correlations
        correlations 12, and 22
    """

    r_corr_12 = xi_star_gal_tc(ra_1, dec_1, e1_1, e2_1, weights_1,
                               ra_2, dec_2, e1_2, e2_2)
    r_corr_22 = xi_star_gal_tc(ra_2, dec_2, e1_2, e2_2, np.ones_like(ra_2),
                               ra_2, dec_2, e1_2, e2_2)

    return r_corr_12, r_corr_22


def alpha(r_corr_gp, r_corr_pp, e1_gal, e2_gal, weights_gal, e1_star, e2_star):
    """alpha

    Compute scale-dependent PSF leakage alpha.

    Parameters
    ----------
    r_corr_gp, r_corr_pp : correlations
        correlations galaxy-star, star-star
    e1_gal, e2_gal : array of float
        galaxy ellipticities
    weights_gal : array of float
        galaxy weights
    e1_star, e2_star : array of float
        galaxy ellipticities

    Returns
    -------
    alpha, sig_alpha : float
        mean and std of alpha
    """

    complex_gal = np.average(e1_gal, weights=weights_gal) + np.average(e2_gal, weights=weights_gal)*1j
    complex_psf = np.mean(e1_star) + np.mean(e2_star)*1j

    alpha_leak = (r_corr_gp.xip - np.real(np.conj(complex_gal)*complex_psf)) \
                       / (r_corr_pp.xip - np.abs(complex_psf)**2)
    sig_alpha_leak = np.abs(alpha_leak) \
                           * np.sqrt((np.sqrt(r_corr_gp.varxip) / r_corr_gp.xip)**2
                                     + (np.sqrt(r_corr_pp.varxip) / r_corr_pp.xip)**2)

    return alpha_leak, sig_alpha_leak
