"""
  
:Name: galaxy.py

:Description: This script contains methods to deal with
galaxy and star images.

:Author: Martin Kilbinger

:Date: 2021

:Package: sp_validation

"""


import numpy as np

from sp_validation import io


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


def classification_galaxy_base(dd):
    """Classification Galaxy Base
    
    Return mask corresponding to basic classification for galaxies.
    """
    
    # spread model class, add two times the uncertainty to be conservative
    sm_classif = dd['SPREAD_MODEL'] + 2 * dd['SPREADERR_MODEL']

    cut_common = \
        (sm_classif > 0.0035) \
        & (dd['SPREAD_MODEL'] > 0) \
        & (dd['SPREAD_MODEL'] < 0.03) \
        & (dd['MAG_AUTO'] < 26) \
        & (dd['MAG_AUTO'] > 20) \
        & (dd['FLAGS'] == 0) \
        & (dd['IMAFLAGS_ISO'] == 0) \
        & (dd['N_EPOCH'] > 0)

    return cut_common

def classification_galaxy_ngmix(dd, cut_common, stats_file, verbose=False):
    """Classification Galaxy Ngmx
    
    Return mask corresponding to ngmix classification of galaxies 
    """
    
    m_gal_ngmix = (
        cut_common
        & (dd['NGMIX_MCAL_FLAGS'] == 0)
        & (dd['NGMIX_ELL_PSFo_NOSHEAR'][:,0] != -10)
        & (dd['NGMIX_MOM_FAIL'] == 0)
        & (dd['NGMIX_N_EPOCH'] > 0)
    )

    n_gal_ngmix = len(np.where(m_gal_ngmix)[0])
    n_tot = len(dd)

    io.print_ratio(
        'ngmix: Objects selected as galaxies',
        n_gal_ngmix,
        n_tot,
        stats_file,
        verbose=verbose)
    
    return m_gal_ngmix


def classification_galaxy_galsim(dd, cut_common, stats_file, verbose=False):
    """Classification Galaxy Galsim
    
    Return mask corresponding to galsim classification of galaxies"""

    m_gal_galsim = (
        cut_common
        & (dd['GALSIM_PSF_ELL_ORIGINAL_PSF'][:,0] != -10)
    )

    n_gal_galsim = len(np.where(m_gal_galsim)[0])
    n_tot = len(dd)

    io.print_ratio(
        'galsim: Objects selected as galaxies',
        n_gal_galsim,
        n_tot,
        stats_file,
        verbose=verbose)

    return m_gal_galsim
