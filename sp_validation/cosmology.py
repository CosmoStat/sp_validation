"""COSMOLOGY.

:Name: cosmology.py

:Description: This file contains methods for science
              validation of a weak-lensing shape catalogue.
              Depends on a cosmological model.

:Author: Axel Guinot, Martin Kilbinger

:Date: 2021

:Package: sp_validation

"""

import numpy as np
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from tqdm import tqdm

from astropy import cosmology
from astropy.io import fits

#from lenspack.geometry.projections.gnom import radec2xy

from sp_validation import basic
from sp_validation import util
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


# For (obsolete?) convergence maps
from astropy.convolution import convolve_fft, Tophat2DKernel
from scipy import fftpack

# For correlation function calculations
import treecorr

# For theoretical modelling of the shear-shear correlation function
try:
    import pyccl as ccl
except Exception:
    print('Could not import pyccl')


# Cluster lensing

def gamma_T_tc_spatial(
    cluster_cat,
    ra_cat,
    dec_cat,
    e1_cat,
    e2_cat,
    w_cat=None,
):
    """Add docstring.

    ...

    """
    n_cluster = len(cluster_cat['ra'])

    h = 0.72
    cosmo = cosmology.FlatLambdaCDM(H0=h * 100., Om0=0.3)
    arcmin_to_rad = np.pi / 180. / 60.

    R_mpc = cluster_cat['R'] / h

    cat_gal = treecorr.Catalog(
        ra=ra_cat,
        dec=dec_cat,
        g1=e1_cat,
        g2=e2_cat,
        w=w_cat,
        ra_units='degrees',
        dec_units='degrees',
    )

    meanr = []
    meanlogr = []
    xi = []
    xi_im = []
    varxi = []
    for i in tqdm(range(n_cluster), total=n_cluster):

        d_ang = cosmo.angular_diameter_distance(cluster_cat['z'][i]).value

        R_v_ang = R_mpc[i] / d_ang / arcmin_to_rad

        cat_cluster = treecorr.Catalog(
            ra=[cluster_cat['ra'][i]],
            dec=[cluster_cat['dec'][i]],
            ra_units='degrees',
            dec_units='degrees',
        )

        TreeCorrConfig = {
            'ra_units': 'degrees',
            'dec_units': 'degrees',
            'max_sep': 3. * R_v_ang,
            'min_sep': R_v_ang / 8.,
            'sep_units': 'arcminutes',
            'nbins': 20,
        }

        ng = treecorr.NGCorrelation(TreeCorrConfig)

        ng.process(cat_cluster, cat_gal)

        meanr.append(ng.meanr * d_ang * arcmin_to_rad / R_mpc[i])
        meanlogr.append(ng.meanlogr)
        xi.append(ng.xi)
        xi_im.append(ng.xi_im)
        varxi.append(ng.varxi)

    meanr = np.array(meanr)
    meanlogr = np.array(meanlogr)
    xi = np.array(xi)
    xi_im = np.array(xi_im)
    varxi = np.array(varxi)

    return meanr, meanlogr, xi, xi_im, varxi


def gamma_T_tc_spatial_parallel(
    cluster_cat,
    ra_cat,
    dec_cat,
    e1_cat,
    e2_cat,
    w_cat=None,
):
    """Add docstring.

    ...

    """

    def job_func(clust_ra, clust_dec, clust_z, clust_R, cat_gal, cosmo):
        """Add docstring.

        ...

        """
        d_ang = cosmo.angular_diameter_distance(clust_z).value

        R_v_ang = clust_R / d_ang / arcmin_to_rad

        cat_cluster = treecorr.Catalog(
            ra=[clust_ra],
            dec=[clust_dec],
            ra_units='degrees',
            dec_units='degrees',
        )

        TreeCorrConfig = {
            'ra_units': 'degrees',
            'dec_units': 'degrees',
            'max_sep': 3. * R_v_ang,
            'min_sep': R_v_ang / 8.,
            'sep_units': 'arcminutes',
            'nbins': 20,
        }

        ng = treecorr.NGCorrelation(TreeCorrConfig)

        ng.process(cat_cluster, cat_gal)

        return np.array([
            ng.meanr * d_ang * arcmin_to_rad / clust_R,
            ng.meanlogr,
            ng.xi,
            ng.xi_im,
            ng.varxi
        ])

    n_cluster = len(cluster_cat['ra'])

    h = 0.72
    cosmo = cosmology.FlatLambdaCDM(H0=h * 100., Om0=0.3)
    arcmin_to_rad = np.pi / 180. / 60.

    R_mpc = cluster_cat['R'] / h

    cat_gal = treecorr.Catalog(
        ra=ra_cat,
        dec=dec_cat,
        g1=e1_cat,
        g2=e2_cat,
        w=w_cat,
        ra_units='degrees',
        dec_units='degrees',
    )

    meanr = []
    meanlogr = []
    xi = []
    xi_im = []
    varxi = []

    res = Parallel(n_jobs=5, backend='loky')(delayed(job_func)(
        cluster_cat['ra'][i],
        cluster_cat['dec'][i],
        cluster_cat['z'][i],
        R_mpc[i],
        cat_gal,
        cosmo
    ) for i in tqdm(range(n_cluster), total=n_cluster))

    return res


def get_theo_gamT(
    cluster_mass,
    cluster_z,
    z_source=0.65,
    H0=72.,
    Omega_m=0.3,
    Omega_b=0.045,
    delta_mdef=500,
    theta=None,
    z_distrib=None,
    meanbin_z=None,
    nz=None,
):
    """Add docstring.

    ...

    """

    def heaviside(a, trunc=0):
        """Add docstring.

        ...

        """
        out = np.zeros_like(a)
        ind = np.where(a > trunc)
        out[ind] = 1
        return out

    def get_z_norm(nz):
        """Add docstring.

        ...

        """
        nz, bins_z = np.histogram(
            nz,
            500,
            density=True,
            range=(0., np.max(nz)),
        )
        meanbin_z = np.mean((bins_z[1:], bins_z[:-1]), 0)
        int_nz = simps(nz, meanbin_z)

        return int_nz

    def get_z_norm2(meanbin_z, nz):
        """Add docstring.

        ...

        """
        int_nz = simps(nz, meanbin_z)

        return int_nz

    def get_norm_gamT(nz, z_min, z_norm=1):
        """Add docstring.

        ...

        """
        nz, bins_z = np.histogram(
            nz,
            500,
            density=True,
            range=(0, np.max(nz)),
        )
        meanbin_z = np.mean((bins_z[1:], bins_z[:-1]), 0)
        trunc_ind = np.where(meanbin_z > z_min)[0]
        int_nz = simps(nz[trunc_ind], meanbin_z[trunc_ind])
        z_mean = simps(
            nz[trunc_ind] * meanbin_z[trunc_ind],
            meanbin_z[trunc_ind]
        ) / int_nz

        return int_nz / z_norm, z_mean

    def get_norm_gamT2(meanbin_z, nz, z_min, z_norm=1):
        """Add docstring.

        ...

        """
        trunc_ind = np.where(meanbin_z > z_min)[0]
        int_nz = simps(nz[trunc_ind], meanbin_z[trunc_ind])
        z_mean = simps(
            nz[trunc_ind] * meanbin_z[trunc_ind],
            meanbin_z[trunc_ind]) / int_nz

        return int_nz / z_norm, z_mean

    def get_gt_model(
        m,
        concentration,
        cluster_z,
        z_distrib,
        cosmo,
        n_bins=500,
    ):
        """Add docstring.

        ...

        """
        nz, bins_z = np.histogram(
            z_distrib,
            n_bins,
            density=True,
            range=(0, np.max(z_distrib)),
        )
        meanbin_z = np.mean((bins_z[1:], bins_z[:-1]), 0)
        gt_models = []
        for i in range(n_bins):
            gt_tmp = clmm.predict_reduced_tangential_shear(
                R * H0 / 100.,
                m,
                concentration,
                cluster_z,
                meanbin_z[i],
                cosmo,
                delta_mdef=delta_mdef,
                halo_profile_model='nfw',
            )
            gt_models.append(gt_tmp)
        gt_models = np.array(gt_models)
        integrand = (
            gt_models * nz.reshape((n_bins, 1))
            * heaviside(meanbin_z, cluster_z).reshape((n_bins, 1))
        )

        final_gt = np.array([
            simps(integrand[:, i], meanbin_z)
            for i in range(integrand.shape[1])
        ])
        return final_gt

    def get_gt_model2(
        m,
        concentration,
        cluster_z,
        meanbin_z,
        nz,
        cosmo,
        n_bins=500,
    ):
        """Add docstring.

        ...

        """
        f = interp1d(meanbin_z, nz)
        zz = np.arange(
            meanbin_z.min(),
            meanbin_z.max(),
            (meanbin_z.max() - meanbin_z.min()) / n_bins,
        )
        gt_models = []
        for i in range(n_bins):
            gt_tmp = clmm.predict_reduced_tangential_shear(
                R * H0 / 100.,
                m,
                concentration,
                cluster_z,
                zz[i],
                cosmo,
                delta_mdef=delta_mdef,
                halo_profile_model='nfw',
            )
            gt_models.append(gt_tmp)
        gt_models = np.array(gt_models)
        integrand = (
            gt_models * f(zz).reshape((n_bins, 1))
            * heaviside(zz, cluster_z).reshape((n_bins, 1))
        )

        final_gt = np.array([
            simps(integrand[:, i], zz) for i in range(integrand.shape[1])
        ])
        return final_gt

    arcmin_to_rad = np.pi / 180. / 60.

    cluster_concentration = c_XRAY(cluster_z, cluster_mass, h=H0 / 100.)

    astropy_cosmo = cosmology.FlatLambdaCDM(H0=H0, Om0=Omega_m, Ob0=Omega_b)

    # cosmo_ccl = cm.cclify_astropy_cosmo(astropy_cosmo)
    cosmo = Cosmology(H0=100 * h, Omega_dm0=Om - Ob, Omega_b0=Ob, Omega_k0=0)

    d_ang = astropy_cosmo.angular_diameter_distance(cluster_z).value

    R = theta * d_ang * arcmin_to_rad

    if False:
        r3d = np.logspace(-2, 2, 100)
    else:
        r3d = R

    if False:
        factor = 1
    else:
        if z_distrib is not None:
            norm_nz = get_z_norm(z_distrib)
            factor, z_mean_src = get_norm_gamT(z_distrib, cluster_z, norm_nz)
        else:
            norm_nz = get_z_norm2(meanbin_z, nz)
            factor, z_mean_src = get_norm_gamT2(
                meanbin_z,
                nz,
                cluster_z,
                norm_nz,
            )

    if z_distrib is not None:
        gt = get_gt_model(
            cluster_mass,
            cluster_concentration,
            cluster_z,
            z_distrib,
            cosmo,
            n_bins=500,
        )
    else:
        gt = get_gt_model2(
            cluster_mass,
            cluster_concentration,
            cluster_z,
            meanbin_z,
            nz,
            cosmo,
            n_bins=500,
        )

    return gt


def get_theo_mass(
    cluster_z,
    theta,
    gt_profile,
    gt_sig,
    z_source,
    H0=72.,
    Omega_m=0.3,
    Omega_b=0.045,
    z_distrib=None,
    meanbin_z=None,
    nz=None,
    delta_mdef=500,
):
    """Add docstring.

    ...

    """

    def heaviside(a, trunc=0):
        """Add docstring.

        ...

        """
        out = np.zeros_like(a)
        ind = np.where(a > trunc)
        out[ind] = 1
        return out

    def get_z_norm(nz):
        """Add docstring.

        ...

        """
        nz, bins_z = np.histogram(
            nz,
            500,
            density=True,
            range=(0., np.max(nz)),
        )
        meanbin_z = np.mean((bins_z[1:], bins_z[:-1]), 0)
        int_nz = simps(nz, meanbin_z)

        return int_nz

    def get_z_norm2(meanbin_z, nz):
        """Add docstring.

        ...

        """
        int_nz = simps(nz, meanbin_z)

        return int_nz

    def get_norm_gamT(nz, z_min, z_norm=1):
        """Add docstring.

        ...

        """
        nz, bins_z = np.histogram(nz, 500, density=True, range=(0, np.max(nz)))
        meanbin_z = np.mean((bins_z[1:], bins_z[:-1]), 0)
        trunc_ind = np.where(meanbin_z > z_min)[0]
        int_nz = simps(nz[trunc_ind], meanbin_z[trunc_ind])
        z_mean = simps(
            nz[trunc_ind] * meanbin_z[trunc_ind], meanbin_z[trunc_ind]
        ) / int_nz

        return int_nz / z_norm, z_mean

    def get_norm_gamT2(meanbin_z, nz, z_min, z_norm=1):
        """Add docstring.

        ...

        """
        trunc_ind = np.where(meanbin_z > z_min)[0]
        int_nz = simps(nz[trunc_ind], meanbin_z[trunc_ind])
        z_mean = simps(
            nz[trunc_ind] * meanbin_z[trunc_ind], meanbin_z[trunc_ind]
        ) / int_nz

        return int_nz / z_norm, z_mean

    if z_distrib is not None:
        norm_nz = get_z_norm(z_distrib)
        factor, z_mean_src = get_norm_gamT(z_distrib, cluster_z, norm_nz)
    else:
        norm_nz = get_z_norm2(meanbin_z, nz)
        factor, z_mean_src = get_norm_gamT2(meanbin_z, nz, cluster_z, norm_nz)

    arcmin_to_rad = np.pi / 180. / 60.

    astropy_cosmo = cosmology.FlatLambdaCDM(H0=H0, Om0=Omega_m, Ob0=Omega_b)

    # cosmo_ccl = cm.cclify_astropy_cosmo(astropy_cosmo)
    cosmo = Cosmology(H0=100 * h, Omega_dm0=Om - Ob, Omega_b0=Ob, Omega_k0=0)

    def get_gt_model(
        m,
        concentration,
        cluster_z,
        z_distrib,
        cosmo,
        n_bins=500,
    ):
        """Add docstring.

        ...

        """
        nz, bins_z = np.histogram(
            z_distrib,
            n_bins,
            density=True,
            range=(0, np.max(z_distrib)),
        )
        meanbin_z = np.mean((bins_z[1:], bins_z[:-1]), 0)
        gt_models = []
        for i in range(n_bins):
            gt_tmp = clmm.predict_reduced_tangential_shear(
                R * H0 / 100.,
                m,
                concentration,
                cluster_z,
                meanbin_z[i],
                cosmo,
                delta_mdef=delta_mdef,
                halo_profile_model='nfw',
            )
            gt_models.append(gt_tmp)
        gt_models = np.array(gt_models)
        integrand = (
            gt_models * nz.reshape((n_bins, 1))
            * heaviside(meanbin_z, cluster_z).reshape((n_bins, 1))
        )

        final_gt = np.array([
            simps(integrand[:, i], meanbin_z)
            for i in range(integrand.shape[1])
        ])
        return final_gt

    def get_gt_model2(
        m,
        concentration,
        cluster_z,
        meanbin_z,
        nz,
        cosmo,
        n_bins=500,
    ):
        """Add docstring.

        ...

        """
        f = interp1d(meanbin_z, nz)
        zz = np.arange(
            meanbin_z.min(),
            meanbin_z.max(),
            (meanbin_z.max() - meanbin_z.min()) / n_bins,
        )
        gt_models = []
        for i in range(n_bins):
            gt_tmp = clmm.predict_reduced_tangential_shear(
                R * H0 / 100.,
                m,
                concentration,
                cluster_z,
                zz[i],
                cosmo,
                delta_mdef=delta_mdef,
                halo_profile_model='nfw',
            )
            gt_models.append(gt_tmp)
        gt_models = np.array(gt_models)
        integrand = (
            gt_models * f(zz).reshape((n_bins, 1))
            * heaviside(zz, cluster_z).reshape((n_bins, 1))
        )

        final_gt = np.array([
            simps(integrand[:, i], zz) for i in range(integrand.shape[1])
        ])
        return final_gt

    def nfw_to_shear_profile(logm, profile_info):
        [R, gt_profile, gt_sig, z_source] = profile_info
        m = 10. ** logm
        print(m / 1e14)
        concentration = c_XRAY(cluster_z, m, h=H0 / 100.)
        gt_model = get_gt_model(m, concentration, cluster_z, z_source, cosmo)
        return sum((gt_model - gt_profile) ** 2 / gt_sig ** 2)

    def nfw_to_shear_profile2(logm, profile_info):
        [R, gt_profile, gt_sig, meanbin_z, nz] = profile_info
        m = 10. ** logm
        print(m / 1e14)
        concentration = c_XRAY(cluster_z, m, h=H0 / 100.)
        gt_model = get_gt_model2(
            m,
            concentration,
            cluster_z,
            meanbin_z,
            nz,
            cosmo,
        )
        return sum((gt_model - gt_profile) ** 2 / gt_sig ** 2)

    def minimizer(func, x0, args, tolerence=1e-6):
        """Add docstring.

        ...

        """
        minimize_obj = minimize(func, x0, args=args, tol=tolerence)

        return minimize_obj.x[0], minimize_obj.fun, minimize_obj.hess_inv

    def get_best_m(func, args, n_guess=20, tolerence=1e-6, n_jobs=-1):
        """Add docstring.

        ...

        """
        logm_0s = np.random.uniform(13., 15., n_guess)

        res = Parallel(n_jobs=n_jobs, backend='loky')(delayed(minimizer)(
            func,
            x0,
            args,
            tolerence
        ) for x0 in tqdm(logm_0s, total=n_guess))

        res = list((zip(*res)))

        best_values = res[0]
        mini_values = res[1]

        arg_min = np.argmin(mini_values)

        cov_value = (
            np.max((1, np.abs(res[1][arg_min]))) * tolerence * res[2][arg_min]
        )

        return best_values[arg_min], cov_value

    logm_0 = np.random.uniform(13., 15., 1)[0]

    d_ang = astropy_cosmo.angular_diameter_distance(cluster_z).value

    R = theta * d_ang * arcmin_to_rad

    if z_distrib is not None:
        logm_est, cov = get_best_m(
            nfw_to_shear_profile,
            args=[R, gt_profile, gt_sig, z_distrib],
            tolerence=1e-3,
        )
        m_est = 10. ** logm_est
        final_model = get_gt_model(
            m_est,
            c_XRAY(cluster_z, m_est, h=H0 / 100.),
            cluster_z,
            z_distrib,
            cosmo,
            n_bins=100,
        )
    else:
        logm_est, cov = get_best_m(
            nfw_to_shear_profile2,
            args=[R, gt_profile, gt_sig, meanbin_z, nz],
            tolerence=1e-3,
        )
        m_est = 10. ** logm_est
        final_model = get_gt_model2(
            m_est,
            c_XRAY(cluster_z, m_est, h=H0 / 100.),
            cluster_z,
            meanbin_z,
            nz,
            cosmo,
            n_bins=100,
        )

    return m_est, final_model, cov


# Convergence maps

def stack_mm2(
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
    ra_corr = correct_ra2(ra, dec, mean_dec)
    ra_clust_corr = correct_ra2(cluster_ra, cluster_dec, mean_dec)

    # From Z to comobile
    h = 0.7
    cosmo = cosmology.FlatLambdaCDM(H0=h * 100., Om0=0.3)
    deg_to_rad = np.pi / 180.

    if tree is None:
        tree = cKDTree(np.array([ra_corr, dec]).T)

    k = 0
    for ra_c, dec_c, z_c in tqdm(
        zip(ra_clust_corr, cluster_dec, cluster_z),
        total=len(ra_clust_corr)
    ):

        d_ang = cosmo.angular_diameter_distance(z_c).value   # Rad

        R_max_ang = radius / d_ang / deg_to_rad  # Deg

        res_match = tree.query(np.array([ra_c, dec_c]).T, k=n_match, n_jobs=-1)

        ind_gal = res_match[1][np.where(res_match[0] < R_max_ang)]

        ra_centered = (ra_corr[ind_gal] - ra_c) / R_max_ang
        dec_centered = (dec[ind_gal] - dec_c) / R_max_ang
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

        res_match = tree.query(np.array([ra_c, dec_c]).T, k=n_match, n_jobs=-1)

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


def g2k_fft(g1, g2, dx):
    """Convert gamma.

    Convert gamma to kappa in Fourier space
    (Sect. 2.2 from Chang et al. 2016, MNRAS, 459, 3203)

    """
    g1_3d_ft = fftpack.fft2(g1)
    g2_3d_ft = fftpack.fft2(g2)
    FF1 = fftpack.fftfreq(len(g1_3d_ft))
    FF2 = fftpack.fftfreq(len(g1_3d_ft[0]))

    dk = 1.0 / dx * 2 * np.pi                     # max delta_k in 1/arcmin
    kx = np.dstack(np.meshgrid(FF2, FF1))[:, :, 0] * dk
    ky = np.dstack(np.meshgrid(FF2, FF1))[:, :, 1] * dk
    kx2 = kx**2
    ky2 = ky**2
    k2 = kx2 + ky2

    # kappa to everything
    k2[k2 == 0] = 1e-15
    g2kappaE_ft = g1_3d_ft / k2 * (kx2 - ky2) + g2_3d_ft / k2 * 2 * (kx * ky)
    g2kappaB_ft = (
        -1 * g1_3d_ft / k2 * 2 * (kx * ky) + g2_3d_ft / k2 * (kx2 - ky2)
    )

    return fftpack.ifft2(g2kappaE_ft).real, fftpack.ifft2(g2kappaB_ft).real


def make_mass_map(ra, dec, g1, g2, w, proj=True):
    """Add docstring.

    ...

    """
    params = {
        'nside': 2048,  # healpy resolution
        'smooth': 64.,  # smoothing scale to apply (if asked) to the maps
        'jk': 20,  # number of Jack-Knife realizations
        'pix': 0.8,  # angular pixel scale in arcmin
    }

    if proj:
        ra_proj = correct_ra2(ra, dec)  # Projected RA
    else:
        ra_proj = ra
    params['ra_max'] = np.max(ra_proj)
    params['ra_min'] = np.min(ra_proj)
    params['dec_max'] = np.max(dec)
    params['dec_min'] = np.min(dec)
    numbins_ra = max(
        1,
        round((params['ra_max'] - params['ra_min']) * 60. / params['pix'], 0)
    )  # number of bins in ra_l.
    numbins_dec = max(
        1,
        round((params['dec_max'] - params['dec_min']) * 60. / params['pix'], 0)
    )  # number of bins in dec_l.
    params['numbins_ra'] = numbins_ra
    params['numbins_dec'] = numbins_dec

    ellip1 = make_maps(g1, w, dec, ra_proj, numbins_dec, numbins_ra)
    ellip2 = make_maps(g2, w, dec, ra_proj, numbins_dec, numbins_ra)

    ellip1[(ellip1 < np.inf) is False] = 0.
    ellip2[(ellip2 < np.inf) is False] = 0.
    kappa_E, kappa_B = g2k_fft(ellip1, ellip2, 1)

    params['smooth_mode'] = 'tophat'

    smooth_s = get_ang_smoothing(params)
    if params['smooth_mode'] == 'boxcar':
        smooth_kernel = np.ones((smooth_s, smooth_s))
    elif params['smooth_mode'] == 'tophat':
        smooth_kernel = Tophat2DKernel(smooth_s)
    else:
        print("ERROR: Unknown smooth kernel")

    kappa_E_sm = convolve_fft(
        kappa_E,
        smooth_kernel,
        normalize_kernel=np.sum,
        nan_treatment='fill',
    )
    kappa_B_sm = convolve_fft(
        kappa_B,
        smooth_kernel,
        normalize_kernel=np.sum,
        nan_treatment='fill',
    )

    return kappa_E_sm[::-1, ::-1], kappa_B_sm[::-1, ::-1], params


def make_maps(map, w, dec, ra, ndec, nra):
    """Make maps.

    Make tagged maps, weighted by SNR

    """
    map_2d, edges = np.histogramdd(
        np.array([dec, ra]).T,
        bins=(ndec, nra),
        weights=map * w,
    )
    w_2d, edges = np.histogramdd(
        np.array([dec, ra]).T,
        bins=(ndec, nra),
        weights=w,
    )
    w_2d[w_2d == 0] = np.nan
    return map_2d / w_2d


def stack_mm(clust_ra, clust_dec, ra, dec, g1, g2, w, output_size=50):
    """Add docstring.

    ...

    """
    print(len(clust_ra))

    ke, kb, params = make_mass_map(ra, dec, g1, g2, w)

    mean_dec = np.mean(dec)
    step_ra = params['numbins_ra'] / (params['ra_max'] - params['ra_min'])
    step_dec = params['numbins_dec'] / (params['dec_max'] - params['dec_min'])
    ra_tmp = correct_ra2(clust_ra, clust_dec, mean_dec)
    ra_pix = params['numbins_ra'] - np.abs(ra_tmp - params['ra_min']) * step_ra
    dec_pix = (
        params['numbins_dec'] - np.abs(clust_dec - params['dec_min'])
        * step_dec
    )
    plt.figure(figsize=(30, 20))
    plt.imshow(ke)
    plt.plot(ra_pix, dec_pix, 'k+')

    mean_dec = np.mean(dec)
    step_ra = params['numbins_ra'] / (params['ra_max'] - params['ra_min'])
    step_dec = params['numbins_dec'] / (params['dec_max'] - params['dec_min'])
    stacked_img_e = np.zeros((output_size, output_size))
    stacked_img_b = np.zeros((output_size, output_size))
    output_size_2 = int(output_size / 2.)
    k = 0
    for ra_c, dec_c in zip(clust_ra, clust_dec):
        print('{}'.format(k), end='\r')
        ra_tmp = correct_ra2(ra_c, dec_c, mean_dec)
        ra_pix = int(
            params['numbins_ra'] - np.abs(ra_tmp - params['ra_min']) * step_ra
        )
        dec_pix = int(
            params['numbins_dec'] - np.abs(dec_c - params['dec_min'])
            * step_dec
        )
        if (
            (ra_pix - output_size_2 < 0)
            or (dec_pix - output_size_2 < 0)
            or (ra_pix + output_size_2 > ke.shape[0])
            or (dec_pix + output_size_2 > ke.shape[1])
        ):
            continue

        stacked_img_e += ke[
            ra_pix - output_size_2:ra_pix + output_size_2,
            dec_pix - output_size_2:dec_pix + output_size_2
        ]
        stacked_img_b += kb[
            ra_pix - output_size_2:ra_pix + output_size_2,
            dec_pix - output_size_2:dec_pix + output_size_2
        ]
        k += 1

    print(k)

    return stacked_img_e, stacked_img_b


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
        corr_type='L+',
        method='Bessel',
    )
    xim_fit = ccl.correlation(
        cosmo,
        ell,
        cl,
        theta / 60,
        corr_type='L-',
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
    util.download(
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
