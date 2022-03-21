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

def classification_galaxy_overlap(dd):
    """Classification Galaxy Overlap

    Return mask corresponding to non-overlapping tile areas.

    """
    # Duplicate objects due to tile overlaps
    cut_overlap = (
        dd['FLAG_TILING'] == 1
    )

    return cut_overlap

def classification_galaxy_base(
    dd,
    cut_overlap,
    gal_mag_bright=20,
    gal_mag_faint=26,
    n_epoch_min=1,
    do_spread_model=True
    ):
    """Classification Galaxy Base
    
    Return mask corresponding to basic classification for galaxies.

    """
    if do_spread_model:
        # spread model class, add two times the uncertainty to be conservative
        sm_classif = dd['SPREAD_MODEL'] + 2 * dd['SPREADERR_MODEL']
        cut_sm = sm_classif > 0.0035

        cut_sm_all = (
            cut_sm
            & (dd['SPREAD_MODEL'] > 0)
            & (dd['SPREAD_MODEL'] < 0.03)
        )
    else:
        # Do not use spread model
        cut_sm_all = True

    cut_common = (
        cut_overlap
        & cut_sm_all
        & (dd['MAG_AUTO'] < gal_mag_faint)
        & (dd['MAG_AUTO'] > gal_mag_bright)
        & (dd['FLAGS'] == 0)
        & (dd['IMAFLAGS_ISO'] == 0)
        & (dd['N_EPOCH'] >= n_epoch_min)
    )

    return cut_common


def classification_galaxy_ngmix(dd, cut_common, stats_file=None, verbose=False):
    """Classification Galaxy Ngmx
    
    Return mask corresponding to ngmix classification of galaxies 
    """
    
    m_gal_ngmix = (
        cut_common
        & (dd['NGMIX_MCAL_FLAGS'] == 0)
        & (dd['NGMIX_ELL_PSFo_NOSHEAR'][:,0] != -10)
        & (dd['NGMIX_MOM_FAIL'] == 0)
    )

    n_gal_ngmix = len(np.where(m_gal_ngmix)[0])
    n_tot = len(dd)

    if stats_file:
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
