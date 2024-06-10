import yaml
import time 
import numpy as np

from astropy.io import fits

from shear_psf_leakage.rho_tau_cov import CovTauTh

def get_params_rho_tau(cat, survey="other"):

    # Set parameters
    params = {}
    # TODO to yaml file
    params["ra_col"] = cat['psf']["ra_col"]
    params["dec_col"] = cat['psf']["dec_col"]
    params["e1_PSF_col"] = cat['psf']["e1_PSF_col"]
    params["e2_PSF_col"] = cat['psf']["e2_PSF_col"]
    params["e1_star_col"] = cat['psf']["e1_star_col"]
    params["e2_star_col"] = cat['psf']["e2_star_col"]
    params["PSF_size"] = cat['psf']["PSF_size"]
    params["star_size"] = cat['psf']["star_size"]
    params["square_size"] = cat['psf']["square_size"]
    params["R11"] = np.array([1])
    params["R22"] = np.array([1])
    if survey != 'DES':
        params["PSF_flag"] = cat['psf']["PSF_flag"]
        params["star_flag"] = cat['psf']["star_flag"]
    if survey == 'DES':
        params["R11"] = cat['shear']["R11"]
        params["R22"] = cat['shear']["R22"]   
    params["ra_units"] = "deg"
    params["dec_units"] = "deg"

    params["w_col"] = cat['shear']["w"]
    params["e1_col"] = cat['shear']["e1_col"]
    params["e2_col"] = cat['shear']["e2_col"]

    return params

if __name__ == "__main__":

    base_dir = '/home/guerrini/data/'

    versions = ['SP_v1.4-P3', 'SP_v1.4-P3_LFmask', 'SP_v1.4-P1+3', 'SP_v1.3_LFmask_8k', 'SP_axel_v0.0', 'DES']

    path_config = '/home/guerrini/sp_validation/notebooks/cosmo_val/cat_config.yaml'
    output_dir = '/home/guerrini/sp_validation/notebooks/cosmo_val/output/rho_tau_stats/'

    nbin_ang = 100
    nbin_rad = 200

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
    }

    with open(path_config, 'r') as file:
        cat = yaml.load(file.read(), Loader=yaml.FullLoader)

    for ver in versions:

        params = get_params_rho_tau(cat[ver], survey=ver)
        
        info = cat[ver]
        A = info['cov_th']['A']*60*60
        n_e = info['cov_th']['n_e']
        n_psf = info['cov_th']['n_psf']

        path_gal = base_dir+info['subdir']+'/'+info['shear']['path']
        path_psf = base_dir+info['subdir']+'/'+info['psf']['path']
        hdu_psf = info['psf']['hdu']

        print("Computing the covariance matrix for the version: ", ver)
        start_time = time.time()
        cov_tau_th = CovTauTh(
            path_gal=path_gal, path_psf=path_psf, hdu_psf=hdu_psf, treecorr_config=TreeCorrConfig_xi,
            A=A, n_e=n_e, n_psf=n_psf, params=params
        )
        print("--- Computation of the rho and tau statistics %s seconds ---" % (time.time() - start_time))

        start_time = time.time()

        cov = cov_tau_th.build_cov(nbin_ang=nbin_ang, nbin_rad=nbin_rad)

        print("--- Covariance computation %s seconds ---" % (time.time() - start_time))
        np.save(output_dir+'/cov_tau_'+ver+'_th.npy', cov)
        print("Saved covariance matrix of version: ", ver)