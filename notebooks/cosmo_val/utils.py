import numpy as np
import time
import yaml
import os

from shear_psf_leakage.rho_tau_cov import CovTauTh
from shear_psf_leakage.rho_tau_stat import RhoStat, TauStat

not_square_size = ['DES', 'SP_v1.3_LFmask_8k', 'SP_v1.3_LFmask_8k_SN8', 'SP_v1.3_LFmask_8k_F2']

def get_params_rho_tau(cat, survey="other"):

    # Set parameters
    params = {}
    # TODO to yaml file
    if survey == "DES":
        params["patch_number"] = 120
        print("DES, jackknife patch number = 120")
    elif survey == 'SP_axel_v0.0':
        params["patch_number"] = 120
        print("SP_Axel_v0.0, jackknife patch number =120")
    elif survey == 'SP_v1.4-P3' or survey == 'SP_v1.4-P3_LFmask':
        params["patch_number"] = 120
        print("SP_v1.4, jackknife patch number =120")
    else:
        params["patch_number"] = 150
    params["ra_col"] = cat['psf']["ra_col"]
    params["dec_col"] = cat['psf']["dec_col"]
    params["e1_PSF_col"] = cat['psf']["e1_PSF_col"]
    params["e2_PSF_col"] = cat['psf']["e2_PSF_col"]
    params["e1_star_col"] = cat['psf']["e1_star_col"]
    params["e2_star_col"] = cat['psf']["e2_star_col"]
    params["PSF_size"] = cat['psf']["PSF_size"]
    params["star_size"] = cat['psf']["star_size"]
    params["square_size"] = survey not in not_square_size
    if survey != 'DES':
        params["PSF_flag"] = cat['psf']["PSF_flag"]
        params["star_flag"] = cat['psf']["star_flag"]
    params["ra_units"] = "deg"
    params["dec_units"] = "deg"
   

    params["w_col"] = cat['shear']["w"]
    params["e1_col"] = cat['shear']["e1_col"]
    params["e2_col"] = cat['shear']["e2_col"]
    params["R11"] = cat['shear'].get("R11")
    params["R22"] = cat['shear'].get("R22")

    return params

def get_rho_tau_w_cov(config, version, treecorr_config, outdir, method, cov_rho=False):
    """
    Method to compute the covariance matrices of rho and tau-statistics of a given list of versions in cosmo_val.
    Also computes rho and tau-statistics.

    Parameters
    ----------
    versions : list
        List of versions to compute the covariance matrices for.
    method : str
        Method to compute the covariance matrices. Options are 'jk' or 'th'.
    """
    if method == 'th':
        nbin_ang, nbin_rad = 100, 200
        rho_stat_handler, tau_stat_handler = get_rho_tau(config, version, treecorr_config, outdir, cov_rho=cov_rho)
        get_theory_cov(config, version, treecorr_config, outdir, nbin_ang=nbin_ang, nbin_rad=nbin_rad)
        return rho_stat_handler, tau_stat_handler
    elif method == 'jk':
        return get_jackknife_cov(config, version, treecorr_config, outdir)
    elif method == 'sim':
        if os.path.exists(outdir+'/cov_tau_'+version+'_th.npy'):
            print(f"Covariance from simulation available at the following file: {outdir+'/cov_tau_'+version+'_th.npy'}")
            print(f"Computing rho and tau statistics for the version: {version}")
            return get_rho_tau(config, version, treecorr_config, outdir, cov_rho=cov_rho)
        else:
            raise ValueError("Covariance from simulation not available. Please compute it first.")
    else:
        raise ValueError("Method must be either 'jk' or 'th' or 'sim'.")

def get_rho_tau(config, version, treecorr_config, outdir, cov_rho=False):
    """
    Compute rho and tau statistics for a given version of the catalogue.

    Parameters
    ----------
    config: dict
        Configuration file.
    version : str
        Version of the catalogue to use.
    treecorr_config : dict
        Configuraion for treecorr.
    outdir : str
        Output directory.
    """

    params = get_params_rho_tau(config[version], survey=version)

    print("Compute Rho and Tau statistics for the version: ", version)
    start_time = time.time()

    out_base = f"rho_stats_{version}.fits"
    out_path = f"{outdir}/{out_base}"

    rho_stat_handler = RhoStat(
        output=outdir,
        treecorr_config=treecorr_config,
        verbose=True
    )

    if os.path.exists(out_path):
        print(f"Skipping rho statistics computation, file {out_path} already exists.")
        rho_stat_handler.load_rho_stats(out_base)
    else:

        rho_stat_handler.catalogs.set_params(params, outdir)

        mask = (version != 'DES')
        square_size = params["square_size"]

        # Build catalogues
        rho_stat_handler.build_cat_to_compute_rho(
            config[version]["psf"]["path"],
            catalog_id=version,
            square_size=square_size,
            mask=mask,
            hdu = config[version]["psf"]["hdu"] if config[version]["psf"]["hdu"] is not None else 1
        )

        if cov_rho: 
            if not os.path.exists(outdir+'/cov_rho_'+version+'.npy'):
                only_p = lambda corrs: np.array([corr.xip for corr in corrs]).flatten()
                rho_stat_handler.compute_rho_stats(version, out_base, save_cov=True, func=only_p, var_method='jackknife')
        else:
            rho_stat_handler.compute_rho_stats(version, out_base, var_method=None)


    out_base = f"tau_stats_{version}.fits"
    out_path = f"{outdir}/{out_base}"

    tau_stat_handler = TauStat(
        catalogs=rho_stat_handler.catalogs,
        output=outdir,
        treecorr_config=treecorr_config,
        verbose=True,
    )

    if os.path.exists(out_path):
        print(f"Skipping tau statistics computation, file {out_path} already exists.")
        tau_stat_handler.load_tau_stats(out_base)
    else:

        tau_stat_handler.catalogs.set_params(params, outdir)

        mask = (version != 'DES')

        square_size = params["square_size"]

        # Build the different catalogs if necessary
        if f"psf_{version}" not in tau_stat_handler.catalogs.catalogs_dict.keys():
            tau_stat_handler.build_cat_to_compute_tau(
                config[version]["psf"]["path"],
                cat_type='psf',
                catalog_id=version,
                square_size=square_size,
                mask=mask,
                hdu = config[version]["psf"]["hdu"] if config[version]["psf"]["hdu"] is not None else 1
            )

        # Build the catalog of galaxies. PSF was computed above
        tau_stat_handler.build_cat_to_compute_tau(
            config[version]["shear"]["path"],
            cat_type='gal',
            catalog_id=version,
            square_size=square_size,
            mask=mask,
        )


        # function to extract the tau_+
        tau_stat_handler.compute_tau_stats(
            version,
            out_base,
            var_method=None
        )

    print(f"Time to compute rho and tau statistics: {time.time() - start_time:.2f} s")
    return rho_stat_handler, tau_stat_handler


def get_theory_cov(config, version, treecorr_config, outdir, nbin_ang=100, nbin_rad=100):
    """
    Compute an analytical estimate of the covariance matrix of rho and tau-statistics.
    """

    params = get_params_rho_tau(config[version], survey=version)
    
    info = config[version]
    A = info['cov_th']['A']*60*60
    n_e = info['cov_th']['n_e']
    n_psf = info['cov_th']['n_psf']

    path_gal = info['shear']['path']
    path_psf = info['psf']['path']
    hdu_psf = info['psf']['hdu']
    
    print("Computing the covariance matrix for the version: ", version)
    start_time = time.time()

    if os.path.exists(outdir+'/cov_tau_'+version+'_th.npy'):
        print(f"Skipping covariance computation, file {outdir+'/cov_tau_'+version+'_th.npy'} already exists.")
        return
    
    cov_tau_th = CovTauTh(
        path_gal=path_gal, path_psf=path_psf, hdu_psf=hdu_psf, treecorr_config=treecorr_config,
        A=A, n_e=n_e, n_psf=n_psf, params=params
    )

    print("--- Computation of the rho and tau statistics for the covariance %s seconds ---" % (time.time() - start_time))

    cov = cov_tau_th.build_cov(nbin_ang=nbin_ang, nbin_rad=nbin_rad)
    print("--- Covariance computation %s seconds ---" % (time.time() - start_time))
    np.save(outdir+'/cov_tau_'+version+'_th.npy', cov)
    print("Saved covariance matrix of version: ", version)
    del cov_tau_th
    return

def get_jackknife_cov(config, version, treecorr_config, outdir, ncov=100):
    """
    Compute the covariance matrix of rho and tau-statistics using the jackknife method.
    Also compute rho and tau-statistics.
    """

    if os.path.exists(outdir+'/cov_tau_'+version+'_jk.npy'):
        print(f"Skipping covariance computation, file {outdir+'/cov_tau_'+version+'_jk.npy'} already exists.")
        rho_stat_handler = RhoStat(
            output=outdir,
            treecorr_config=treecorr_config,
            verbose=False)
        
        tau_stat_handler = TauStat(
            catalogs=rho_stat_handler.catalogs,
            output=outdir,
            treecorr_config=treecorr_config,
            verbose=True,
        )
        
        return rho_stat_handler, tau_stat_handler
    
    for i in range(ncov):

        if not os.path.exists(outdir+'/cov_tau_'+version+str(i)+'.npy'):

            params = get_params_rho_tau(config[version], survey=version)

            rho_stat_handler = RhoStat(
                output=outdir,
                treecorr_config=treecorr_config,
                verbose=False)
            
            out_base = f"rho_stats_{version}.fits"
            out_path = f"{outdir}/{out_base}"

            print(f"Computing rho-statistics of version {version} for jackknife patch {i+1}/{ncov}")

            rho_stat_handler.catalogs.set_params(params, outdir)

            mask = (version != 'DES')
            square_size = params["square_size"]

            # Build catalogues
            rho_stat_handler.build_cat_to_compute_rho(
                config[version]["psf"]["path"],
                catalog_id=version+str(i),
                square_size=square_size,
                mask=mask,
                hdu = config[version]["psf"]["hdu"]
            )

            # Compute and save rho stats
            only_p = lambda corrs: np.array([corr.xip for corr in corrs]).flatten()
            rho_stat_handler.compute_rho_stats(version+str(i), out_base, save_cov=True, func=only_p, var_method='jackknife')

            tau_stat_handler = TauStat(
                catalogs=rho_stat_handler.catalogs,
                output=outdir,
                treecorr_config=treecorr_config,
                verbose=True,
            )

            out_base = f"tau_stats_{version}.fits"
            out_path = f"{outdir}/{out_base}"

            tau_stat_handler.catalogs.set_params(params, outdir)

            mask = (version != 'DES')

            square_size = params["square_size"]

            # Build the different catalogs if necessary
            if f"psf_{version}{i}" not in tau_stat_handler.catalogs.catalogs_dict.keys():
                tau_stat_handler.build_cat_to_compute_tau(
                    config[version]["psf"]["path"],
                    cat_type='psf',
                    catalog_id=version,
                    square_size=square_size,
                    mask=mask,
                )

            # Build the catalog of galaxies. PSF was computed above
            tau_stat_handler.build_cat_to_compute_tau(
                config[version]["shear"]["path"],
                cat_type='gal',
                catalog_id=version+str(i),
                square_size=square_size,
                mask=mask,
            )


            # function to extract the tau_+
            only_p = lambda corrs: np.array([corr.xip for corr in corrs]).flatten()
            tau_stat_handler.compute_tau_stats(
                version+str(i),
                out_base,
                save_cov=True,
                func=only_p,
                var_method='jackknife',
            )

            if (i+1) != ncov:
                del rho_stat_handler, tau_stat_handler

    cov_tau_loc = np.zeros((60, 60))
    cov_rho_loc = np.zeros((120, 120))
    for i in range(ncov):
        cov_tau_loc += np.load(outdir+f'/cov_tau_{version}{i}.npy')
        cov_rho_loc += np.load(outdir+f'/cov_rho_{version}{i}.npy')
        os.remove(outdir+f'/cov_tau_{version}{i}.npy')
        os.remove(outdir+f'/cov_rho_{version}{i}.npy')
    
    cov_tau = cov_tau_loc/ncov
    cov_rho = cov_rho_loc/ncov

    np.save(outdir+'/cov_tau_'+version+'_jk.npy', cov_tau)
    np.save(outdir+'/cov_rho_'+version+'.npy', cov_rho)

    return rho_stat_handler, tau_stat_handler

def get_samples(psf_fitter, version, cov_type='jk', apply_debias=None, sampler='emcee'):
    """
    Samples (alpha, beta, eta) using the sampler 'emcee' or 'lsq'

    Parameters
    ----------
    psf_fitter : PSFFitter
        Instance of PSFFitter.
    version : str
        Version of the catalogue to use.
    cov_type : str
        Type of covariance matrix to use. Options are 'jk' or 'th' or 'sim'.
    apply_debias : int
        If not None, apply debiasing to the least square method.
    sampler : str
        Sampler to use. Options are 'emcee' or 'lsq'. (Default: 'emcee')
    """
    if sampler=='emcee':
        return get_samples_emcee(psf_fitter, version, cov_type=cov_type, apply_debias=apply_debias)
    elif sampler=='lsq':
        return get_samples_lsq(psf_fitter, version, cov_type=cov_type, apply_debias=apply_debias)
    else:
        raise ValueError("Sampler must be either 'emcee' or 'lsq'.")


def get_samples_emcee(psf_fitter, version, nwalkers=124, nsamples=10000, cov_type='jk', apply_debias=None):
    """
    Samples (alpha, beta, eta) using the covariance of the tau statistics and emcee.

    Parameters
    ----------
    psf_fitter : PSFFitter
        Instance of PSFFitter.
    version : str
        Version of the catalogue to use.
    nwalkers : int
        Number of walkers to use in the MCMC. (Default: 124)
    nsamples : int
        Number of samples to draw from the MCMC. (Default 10000)
    cov_type : str
        Type of covariance matrix to use. Options are 'jk' or 'th' or 'sim'. (Default: 'jk')
    """
    #Load rho and tau stats
    psf_fitter.load_rho_stat('rho_stats_'+ version + '.fits')
    psf_fitter.load_tau_stat('tau_stats_'+ version + '.fits')

    #Check if the path exists
    sample_file_path = psf_fitter.get_sample_file_path(version)
    if os.path.exists(sample_file_path):
        print(f"Skipping sample computation, file {sample_file_path} already exists.")
        flat_samples = psf_fitter.load_samples(version)
        mcmc_result, q = psf_fitter.get_mcmc_from_samples(version)
        print(mcmc_result)
    #Or run MCMC
    else:
        print("MCMC sampling")
        psf_fitter.load_covariance('cov_tau_' + version + '_' + cov_type + '.npy')

        npatch = apply_debias if (apply_debias is not None) else None
        flat_samples, mcmc_result, q = psf_fitter.run_chain(
            nwalkers=nwalkers, nsamples=nsamples,
            npatch=npatch,
            apply_debias=npatch is not None,
            savefig='mcmc_samples_'+version+'.png'
        )
        psf_fitter.save_samples(flat_samples, version)
    return flat_samples, mcmc_result, q

def get_samples_lsq(psf_fitter, version, apply_debias=None, cov_type='jk'):
    """
    Samples (alpha, beta, eta) using the covariance of the tau statistics and least square method.

    Parameters
    ----------
    psf_fitter : PSFFitter
        Instance of PSFFitter.
    version : str
        Version of the catalogue to use.
    apply_debias : int
        If not None, apply debiasing to the least square method. (Default: None)
    """
    #Load rho and tau stats
    psf_fitter.load_rho_stat('rho_stats_'+ version + '.fits')
    psf_fitter.load_tau_stat('tau_stats_'+ version + '.fits')

    #Check if the path exists
    sample_file_path = psf_fitter.get_sample_path(version)
    if os.path.exists(sample_file_path):
        print(f"Skipping sample computation, file {sample_file_path} already exists.")
        flat_samples = psf_fitter.load_samples(version)
        mcmc_result, q = psf_fitter.get_mcmc_from_samples(flat_samples)
        print(mcmc_result)
    #Or run MCMC
    else:
        print("Least square sampling")
        psf_fitter.load_covariance('cov_tau_' + version + '_'+cov_type+'.npy', cov_type='tau')
        psf_fitter.load_covariance('cov_rho_'+version+'.npy', cov_type='rho')
        npatch = apply_debias if (apply_debias is not None) else None
        flat_samples, mcmc_result, q = psf_fitter.get_least_squares_params_samples(
            npatch=npatch, apply_debias=(npatch is not None)
        )
        psf_fitter.save_samples(flat_samples, version)
    return flat_samples, mcmc_result, q
