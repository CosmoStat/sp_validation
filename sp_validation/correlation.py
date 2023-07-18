"""CORRELATION.

:Name: correlation.py

:Description: This script contains methods to deal with
    auto- and cross-correlations.

:Author: Martin Kilbinger <martin.kilbinger@cea.fr>
         Axel Guinot

"""

import numpy as np


from sp_validation.plot_style import *
from sp_validation import util

import treecorr


def func_bias_lin_1d(params, x_data):
    """Func Bias Lin 1D.

    Function for linear 1D bias model.

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.array
        x-values of the data

    Returns
    -------
    numpy.array
        y-values of the model

    """
    m = params['m'].value
    c = params['c'].value

    y_model = m * x_data + c

    return y_model


def loss_bias_lin_1d(params, x_data, y_data, err):
    """Loss Bias Lin 1D.

    Loss function for linear 1D model

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.array
        x-values of the data
    y_data : numpy.array
        y-values of the data
    err : numpy.array
        error values of the data

    Returns
    -------
    numpy.array
        residuals

    """
    y_model = func_bias_lin_1d(params, x_data)
    residuals = (y_model - y_data) / err
    return residuals


def func_bias_2d_full(params, x1, x2, order='lin', mix=False):
    """Func Bias 2D Full.

    Function of 2D bias model evaluated on full 2D grid.

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x1 : list of float
        first component of x-values
    x2 : list of float
        second component of x-values
    order : str, optional
        order of fit, default is 'lin'
    mix : bool, optional
        mixing between components, default is `False`

    Returns
    -------
    2D np.array of float
        first component the 2D model y1(x1, x2) on the (x1, x2)-grid
    2D np.array of float
        second component the 2D model, y2(x1, x2) on the (x1, x2)-grid

    """
    len1 = len(x1)
    len2 = len(x2)

    # Initialise both components y1, y2 as 2D arrays
    y1 = np.zeros(shape=(len1, len2))
    y2 = np.zeros(shape=(len1, len2))

    # Create 2D mesh for input x1, x2 values
    v1, v2 = np.meshgrid(x1, x2, indexing='ij')

    # Compute both components y1, y2 over the meash
    y1, y2 = util.func_bias_2d(params, v1, v2, order=order, mix=mix)

    return y1, y2


def loss_bias_2d(params, x_data, y_data, err, order, mix):
    """Loss Bias 2D.

    Loss function for 2D model

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.array
        two-component x-values of the data
    y_data : numpy.array
        two-component y-values of the data
    err : numpy.array
        error values of the data, assumed the same for both components
    order : str
        order of fit
    mix : bool
        mixing of components if True

    Raises
    ------
    IndexError :
        if input arrays x1_data and x2_data have different lenght

    Returns
    -------
    numpy.array
        residuals

    """
    # Get x and y values of the input data
    x1_data = x_data[0]
    x2_data = x_data[1]
    y1_data = y_data[0]
    y2_data = y_data[1]

    if len(x1_data) != len(x2_data):
        raise IndexError('Length of both data components has to be equal')

    # Get model 1D y1 and y2 components
    y1_model, y2_model = util.func_bias_2d(
        params,
        x1_data,
        x2_data,
        order=order,
        mix=mix
    )

    # Compute residuals between data and model
    res1 = (y1_model - y1_data) / err
    res2 = (y2_model - y2_data) / err

    # Concatenate both components
    residuals = np.concatenate([res1, res2])

    return residuals


def print_fit_report(res, file=None):
    """Print Fit Report.

    Print report of minimizing result.

    Parameters
    ----------
    res : class lmfit.MinimizerResult
        results of the minization
    file : filehandler, optional
        output to file; if `None` (default) output to `stdout`

    """
    # chi^2
    print(f'chi^2 = {res.chisqr}', file=file)

    # Reduced chi^2
    print(f'reduced chi^2 = {res.redchi}', file=file)

    # Akaike Information Criterium
    print(f'aic = {res.aic}', file=file)

    # Bayesian Information Criterium
    print(f'bic = {res.bic}', file=file)


def param_order2spin(p_dp, order, mix):
    """Param Order 2 Spin.

    Transform parameter from natural to spin coefficients.

    Parameters
    ----------
    p_dp : dict
        Parameter natural coefficients
    order : str
        expansion order, one of 'linear', 'quad'
    mix : bool
        ellipticity components are mixed if ``True``

    Returns
    -------
    dict :
        Parameter spin coefficients

    """
    s_ds = {}

    s_ds['x0'] = 0.5 * (p_dp['a11'] + p_dp['a22'])

    if order == 'quad' and mix:
        s_ds['x2'] = 0.5 * (p_dp['q111'] + p_dp['q122'])
        s_ds['y2'] = 0.5 * (p_dp['q211'] - p_dp['q222'])
        s_ds['x-2'] = 0.25 * (p_dp['q111'] - p_dp['q122'] + p_dp['q212'])
        s_ds['y-2'] = 0.25 * (p_dp['q211'] - p_dp['q222'] - p_dp['q112'])

    s_ds['x4'] = 0.5 * (p_dp['a11'] - p_dp['a22'])

    if mix:
        s_ds['y4'] = p_dp['a12']

    if order == 'quad' and mix:
        s_ds['x6'] = 0.25 * (p_dp['q111'] - p_dp['q122'] - p_dp['q212'])
        s_ds['y6'] = 0.25 * (p_dp['q211'] - p_dp['q222'] + p_dp['q112'])

    return s_ds


def xi_star_gal_tc(
    ra_gal,
    dec_gal,
    e1_gal,
    e2_gal,
    w_gal,
    ra_star,
    dec_star,
    e1_star,
    e2_star,
    w_star=None,
    theta_min_amin=2,
    theta_max_amin=200,
    n_theta=20,
):
    """Xi star gal tc.

    Cross-correlation between galaxy and star ellipticities.

    """
    unit = 'degrees'

    cat_gal = treecorr.Catalog(
        ra=ra_gal,
        dec=dec_gal,
        g1=e1_gal,
        g2=e2_gal,
        w=w_gal,
        ra_units=unit,
        dec_units=unit
    )
    cat_star = treecorr.Catalog(
        ra=ra_star,
        dec=dec_star,
        g1=e1_star,
        g2=e2_star,
        w=w_star,
        ra_units=unit,
        dec_units=unit
    )

    TreeCorrConfig = {
        'ra_units': unit,
        'dec_units': unit,
        'sep_units': 'arcminutes',
        'min_sep': theta_min_amin,
        'max_sep': theta_max_amin,
        'nbins': n_theta
    }
    ng = treecorr.GGCorrelation(TreeCorrConfig)

    ng.process(cat_gal, cat_star)

    return ng


def correlation_12_22(
    ra_1,
    dec_1,
    e1_1,
    e2_1,
    weights_1,
    ra_2,
    dec_2,
    e1_2,
    e2_2,
    theta_min_amin=2,
    theta_max_amin=200,
    n_theta=20,
):
    """Correlation 12 22.

    Shear correlation functions between two samples 1 and 2.
    Compute xi_12 and xi_22.

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
    theta_min_amin : float, optional
        minimum angular scale in arcmin, default is 2
    theta_max_amin : float, optional
        maximum angular scale in arcmin, default is 200
    n_theta : int, optional
        number of angular scales, default is 20

    Returns
    -------
    xi_12, xi_22 : correlations
        correlations 12, and 22

    """
    r_corr_12 = xi_star_gal_tc(
        ra_1,
        dec_1,
        e1_1,
        e2_1,
        weights_1,
        ra_2,
        dec_2,
        e1_2,
        e2_2,
        theta_min_amin=theta_min_amin,
        theta_max_amin=theta_max_amin,
        n_theta=n_theta
    )
    r_corr_22 = xi_star_gal_tc(
        ra_2,
        dec_2,
        e1_2,
        e2_2,
        np.ones_like(ra_2),
        ra_2,
        dec_2,
        e1_2,
        e2_2,
        theta_min_amin=theta_min_amin,
        theta_max_amin=theta_max_amin,
        n_theta=n_theta
    )

    return r_corr_12, r_corr_22


def alpha(r_corr_gp, r_corr_pp, e1_gal, e2_gal, weights_gal, e1_star, e2_star):
    """Alpha.

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
    complex_gal = (
        np.average(e1_gal, weights=weights_gal)
        + np.average(e2_gal, weights=weights_gal) * 1j
    )
    complex_psf = np.mean(e1_star) + np.mean(e2_star) * 1j

    alpha_leak = (
        (r_corr_gp.xip - np.real(np.conj(complex_gal) * complex_psf))
        / (r_corr_pp.xip - np.abs(complex_psf) ** 2)
    )
    sig_alpha_leak = (
        np.abs(alpha_leak)
        * np.sqrt(
            (np.sqrt(r_corr_gp.varxip) / r_corr_gp.xip) ** 2
            + (np.sqrt(r_corr_pp.varxip) / r_corr_pp.xip) ** 2
        )
    )

    return alpha_leak, sig_alpha_leak
