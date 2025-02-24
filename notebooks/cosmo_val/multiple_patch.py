from shear_psf_leakage.rho_tau_stat import RhoStat, TauStat
from tqdm import tqdm
import numpy as np

sep_units = 'arcmin'
coord_units = 'degrees'
theta_min = 0.1
theta_max = 250
nbins = 20


TreeCorrConfig_xi = {
    'ra_units': coord_units,
    'dec_units': coord_units,
    'min_sep': theta_min,
    'max_sep': theta_max,
    'sep_units': sep_units,
    'nbins': nbins,
    'var_method':'jackknife',
}

paths = ['/n17data/sguerrini/multiple_covariance_LF/', '/n17data/sguerrini/multiple_covariance_LF_2/', '/n17data/sguerrini/multiple_covariance_LF_3/']


for path in paths:
    for i in tqdm(range(100)):

        rho_stats_handler = RhoStat(
            output=path,
            treecorr_config=TreeCorrConfig_xi,
            verbose=True
        )

        rho_stats_handler.build_cat_to_compute_rho(
            '/home/mkilbing/astro/data/CFIS/v1.0/SP_LFmask/P3/unions_shapepipe_psf_2022_v1.4_mtheli4k.fits',
            catalog_id='SP_v1.4-P3_LFmask_'+str(i),
            square_size=True,
            mask=True,
            hdu=1
        )
        only_p = lambda corrs: np.array([corr.xip for corr in corrs]+[corr.xim for corr in corrs]).flatten()
        rho_stats_handler.compute_rho_stats('SP_v1.4-P3_LFmask_'+str(i), 'rho_stats_SP_v1.4-P3_LFmask.fits',
                                            save_cov=True, func=only_p, var_method='jackknife')

        tau_stats_handler = TauStat(
        catalogs=rho_stats_handler.catalogs,
        output=path,
        treecorr_config=TreeCorrConfig_xi,
        verbose=True
        )

        tau_stats_handler.build_cat_to_compute_tau(
                '/home/mkilbing/astro/data/CFIS/v1.0/SP_LFmask/P3/unions_shapepipe_2022_v1.4_mtheli4k.fits',
                    cat_type='gal',
                    catalog_id='SP_v1.4-P3_LFmask_'+str(i),
                    square_size=True,
                    mask=True,
        )

        only_p = lambda corrs: np.array([corr.xip for corr in corrs]+[corr.xim for corr in corrs]).flatten()
        tau_stats_handler.compute_tau_stats(
            'SP_v1.4-P3_LFmask_'+str(i),
            'tau_stats_SP_v1.4-P3_LFmask.fits',
            save_cov=True,
            func=only_p,
            var_method='jackknife',
        )

        del rho_stats_handler
        del tau_stats_handler