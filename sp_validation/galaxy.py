"""
  
:Name: galaxy.py

:Description: This script contains methods to deal with
galaxy and star images.

:Author: Martin Kilbinger

:Date: 2021

:Package: sp_validation

"""

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


