"""COSMOLOGY.

:Name: cosmology.py

:Description: This file contains methods for science
              validation of a weak-lensing shape catalogue.
              Depends on a cosmological model.

:Author: Axel Guinot, Martin Kilbinger

"""

import numpy as np
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from tqdm import tqdm

from astropy import cosmology
from astropy.io import fits

from lenspack.geometry.projections.gnom import radec2xy

from cs_util import canfar

from sp_validation import basic
from sp_validation.survey import get_footprint


# For theoretical modelling of cluster lensing
try:
    import clmm
except Exception:
    print('Could not import clmm, continuing...')

try:
    # import clmm.modeling as cm
    from clmm import Cosmology
except Exception:
    print('Could not import clmm.Cosmology, continuing...')


# For correlation function calculations
import treecorr

# For theoretical modelling of the shear-shear correlation function
try:
    import pyccl as ccl
except Exception:
    print('Could not import pyccl')


# Convergence maps

def stack_mm3(
    ra,
    dec,
    e1,
    e2,
    w,
    cluster_ra,
    cluster_dec,
    cluster_z,
    radius=100,
    n_match=100000,
    tree=None,
):
    """Add docstring.

    ...

    """
    # Project data
    mean_dec = np.mean(dec)
    mean_ra = np.mean(ra)
    xx, yy = radec2xy(mean_ra, mean_dec, ra, dec)
    xx_clust, yy_clust = radec2xy(mean_ra, mean_dec, cluster_ra, cluster_dec)

    # From Z to comobile
    h = 0.7
    cosmo = cosmology.FlatLambdaCDM(H0=h * 100., Om0=0.3)
    deg_to_rad = np.pi / 180.

    if tree is None:
        tree = cKDTree(np.array([xx, yy]).T)

    k = 0
    for ra_c, dec_c, z_c in tqdm(
        zip(xx_clust, yy_clust, cluster_z),
        total=len(xx_clust),
    ):

        d_ang = cosmo.angular_diameter_distance(z_c).value   # Rad

        R_max_ang = radius / d_ang  # Rad         / deg_to_rad  # Deg

        res_match = tree.query(np.array([ra_c, dec_c]).T, k=n_match)

        ind_gal = res_match[1][np.where(res_match[0] < R_max_ang)]

        ra_centered = (xx[ind_gal] - ra_c) / R_max_ang
        dec_centered = (yy[ind_gal] - dec_c) / R_max_ang
        if k == 0:
            all_ra = ra_centered
            all_dec = dec_centered
            all_e1 = e1[ind_gal]
            all_e2 = e2[ind_gal]
            all_w = w[ind_gal]
        else:
            all_ra = np.concatenate((all_ra, ra_centered))
            all_dec = np.concatenate((all_dec, dec_centered))
            all_e1 = np.concatenate((all_e1, e1[ind_gal]))
            all_e2 = np.concatenate((all_e2, e2[ind_gal]))
            all_w = np.concatenate((all_w, w[ind_gal]))

        k += 1

    return all_ra, all_dec, all_e1, all_e2, all_w


def gamma_T_tc(ra_pos, dec_pos, ra_cat, dec_cat, e1_cat, e2_cat, w_cat=None):
    """Gamma T tc.

    Compute cross-correlation between positions (forground) and lensing
    (background) catalogue. Also called galaxy-galaxy lensing or population
    lensing.

    Parameters
    ----------
    ra_pos : array of float
        RA coordinates of foreground catalogue
    dec_pos : array of float
        DEC coordinates of foreground catalogue
    ra_cat : array of float
        RA coordinates of background catalogue
    dec_cat : array of float
        DEC coordinates of background catalogue
    e1_cat : array of float
        ellipticity component 1 of background catalogue
    e2_cat : array of float
        ellipticity component 2 of background catalogue
    w_cat : array of float, optional, default=None
        weight of background catalogue

    Returns
    -------
    meanr : array of float
        spatial bin centres
    meanlogr : array of float
        log of spatial bin centres
    xi : array of float
        tangential shear (E-mode)
    xi_im : array of float
        cross-component shear (B- or parity mode)
    rms : array of float
        R.M.S of both xi and xi_im
    """
    cat_pos = treecorr.Catalog(
        ra=ra_pos,
        dec=dec_pos,
        ra_units='degrees',
        dec_units='degrees',
    )
    cat_gal = treecorr.Catalog(
        ra=ra_cat,
        dec=dec_cat,
        g1=e1_cat,
        g2=e2_cat,
        w=w_cat,
        ra_units='degrees',
        dec_units='degrees',
    )

    TreeCorrConfig = {
        'ra_units': 'degrees',
        'dec_units': 'degrees',
        'max_sep': 60,
        'min_sep': 0.7,
        'sep_units': 'arcminutes',
        'nbins': 30,
    }

    ng = treecorr.NGCorrelation(TreeCorrConfig)

    ng.process(cat_pos, cat_gal)

    return ng.meanr, ng.meanlogr, ng.xi, ng.xi_im, np.sqrt(ng.varxi)


def xi_gal_gal_tc(
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
    """Add docstring.

    ...

    """
    cat_gal = treecorr.Catalog(
        ra=ra_gal,
        dec=dec_gal,
        g1=e1_gal,
        g2=e2_gal,
        w=w_gal,
        ra_units='degrees',
        dec_units='degrees',
    )
    cat_star = treecorr.Catalog(
        ra=ra_star,
        dec=dec_star,
        g1=e1_star,
        g2=e2_star,
        w=w_star,
        ra_units='degrees',
        dec_units='degrees',
    )

    TreeCorrConfig = {
        'ra_units': 'degrees',
        'dec_units': 'degrees',
        'sep_units': 'arcminutes',
        'min_sep': theta_min_amin,
        'max_sep': theta_max_amin,
        'nbins': n_theta
    }

    ng = treecorr.GGCorrelation(TreeCorrConfig)

    ng.process(cat_gal, cat_star)

    return ng


def get_theo_xi(
    theta,
    z,
    nz,
    Omega_m=0.295,
    h=0.672,
    Omega_b=0.0516,
    sig8=0.7745,
    ns=1.044,
):
    """Add docstring.

    ...

    """
    cosmo = ccl.Cosmology(
        Omega_c=Omega_m - Omega_b,
        Omega_b=Omega_b,
        h=h,
        sigma8=sig8,
        n_s=ns,
        transfer_function='eisenstein_hu',
    )

    # Create objects to represent tracers of the weak lensing signal with this
    # number density (with has_intrinsic_alignment=False)
    lens1 = ccl.WeakLensingTracer(cosmo, dndz=(z, nz))

    # Calculate the angular cross-spectrum of the two tracers as a function
    # of ell
    ell = np.logspace(0, np.log10(10000), 1000)
    cl = ccl.angular_cl(cosmo, lens1, lens1, ell)

    xip_fit = ccl.correlation(
        cosmo,
        ell,
        cl,
        theta / 60,
        type='GG+',
        method='Bessel',
    )
    xim_fit = ccl.correlation(
        cosmo,
        ell,
        cl,
        theta / 60,
        type='GG-',
        method='Bessel',
    )

    return xip_fit, xim_fit


def get_clusters(
    cluster_cat_name,
    vos_dir,
    output_dir,
    field_name,
    verbose=False,
):
    """Get Clusters.

    Return cluster information from file on VOspace

    Parameters
    ----------
    cluster_cat_name : string
        cluster catalogue file name
    vos_dir : string
        directory on VOspace
    field_name : string
        survey footprint name
    verbose : bool, optional, default=False
        verbose output if True

    Returns
    -------
    tuple
        cluster information (ra, dec, z, SZ-mass)
    """
    out_path = f'{output_dir}/{cluster_cat_name}'
    canfar.download(
        f'{vos_dir}/{cluster_cat_name}',
        out_path,
        verbose=verbose
    )

    cluster_cat = fits.getdata(out_path)
    m_good_cluster = (cluster_cat['MSZ'] != 0) & (cluster_cat['COSMO'] == True)

    m_cluster_foot = get_footprint(
        field_name,
        cluster_cat['RA'][m_good_cluster],
        cluster_cat['DEC'][m_good_cluster],
    )
    cluster_cut = {
        'ra': cluster_cat['RA'][m_good_cluster][m_cluster_foot],
        'dec': cluster_cat['DEC'][m_good_cluster][m_cluster_foot],
        'z': cluster_cat['REDSHIFT'][m_good_cluster][m_cluster_foot],
        'M': cluster_cat['MSZ'][m_good_cluster][m_cluster_foot] * 1e14,
    }

    return cluster_cut
