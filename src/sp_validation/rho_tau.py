import gc
import os
import time
from pathlib import Path

import numpy as np
from shear_psf_leakage.rho_tau_cov import CovTauTh
from shear_psf_leakage.rho_tau_stat import RhoStat, TauStat

# SquareRootScale now lives in sp_validation.plots; re-exported here so that
# `from sp_validation.rho_tau import SquareRootScale` keeps working.
from sp_validation.plots import SquareRootScale  # noqa: F401


def _extract_xip(correlations):
    """Return flattened array of xip values from a list of correlations."""
    return np.array([corr.xip for corr in correlations]).flatten()


def get_params_rho_tau(cat):

    # Set parameters
    params = {}
    params["ra_PSF_col"] = cat["psf"]["ra_col"]
    params["dec_PSF_col"] = cat["psf"]["dec_col"]
    params["e1_PSF_col"] = cat["psf"]["e1_PSF_col"]
    params["e2_PSF_col"] = cat["psf"]["e2_PSF_col"]
    params["e1_star_col"] = cat["psf"]["e1_star_col"]
    params["e2_star_col"] = cat["psf"]["e2_star_col"]
    params["PSF_size"] = cat["psf"]["PSF_size"]
    params["star_size"] = cat["psf"]["star_size"]
    params["square_size"] = cat["psf"].get("square_size", False)
    params["PSF_flag"] = cat["psf"].get("PSF_flag")
    params["star_flag"] = cat["psf"].get("star_flag")
    params["ra_units"] = "deg"
    params["dec_units"] = "deg"
    params["patch_number"] = cat.get("patch_number", 100)  # Default patch number to 100

    params["ra_col"] = cat["shear"].get("ra_col", "RA")
    params["dec_col"] = cat["shear"].get("dec_col", "Dec")
    params["w_col"] = cat["shear"]["w_col"]
    params["e1_col"] = cat["shear"]["e1_col"]
    params["e2_col"] = cat["shear"]["e2_col"]
    params["tomo_bin_id_col"] = cat["shear"].get("tomo_bin_id_col")
    params["R11"] = cat["shear"].get("R11")
    params["R22"] = cat["shear"].get("R22")

    return params


def get_rho_tau_w_cov(
    config,
    version,
    treecorr_config,
    outdir,
    base,
    method,
    mask_star=None,
    mask_gal=None,
    cov_rho=False,
    ncov=100,
    compute_minus=True,
    **kwargs,
):
    """Compute rho/tau statistics and, if requested, their covariance."""
    if method == "th":
        nbin_ang, nbin_rad = kwargs.get("nbin_ang", 100), kwargs.get("nbin_rad", 200)
        compute_minus = kwargs.get("compute_minus", True)
        rho_stat_handler, tau_stat_handler = get_rho_tau(
            config,
            version,
            treecorr_config,
            outdir,
            base,
            cov_rho=cov_rho,
        )
        get_theory_cov(
            config,
            version,
            treecorr_config,
            outdir,
            base,
            nbin_ang=nbin_ang,
            nbin_rad=nbin_rad,
            compute_minus=compute_minus,
            mask_star=mask_star,
            mask_gal=mask_gal,
        )
        return rho_stat_handler, tau_stat_handler
    elif method == "jk":
        return get_jackknife_cov(config, version, treecorr_config, outdir, base)
    elif method == "sim":
        tau_cov_path = Path(outdir) / f"cov_tau_{base}_th.npy"

        if tau_cov_path.exists():
            print(f"Found existing covariance at {tau_cov_path}")
            print(f"Computing rho/tau statistics for {version}")
            return get_rho_tau(
                config,
                version,
                treecorr_config,
                outdir,
                base,
                cov_rho=cov_rho,
            )
        else:
            raise ValueError(
                "Covariance from simulation not available. Please compute it first."
            )
    else:
        raise ValueError("Method must be either 'jk' or 'th' or 'sim'.")


def get_rho_tau(
    config,
    version,
    treecorr_config,
    outdir,
    base,
    mask_star=None,
    mask_gal=None,
    cov_rho=False,
    force_run=False,
):
    """
    Compute rho and tau statistics for a given version of the catalogue.

    Parameters
    ----------
    config: dict
        Configuration file.
    version : str
        Version of the catalogue to use.
    treecorr_config : dict
        TreeCorr configuration (must include 'min_sep', 'max_sep', and 'nbins').
    outdir : str
        Output directory.
    base : str
        Base name for output files.
    mask_star : array-like, optional
        Boolean mask for stars. If None, no masking is applied.
    mask_gal : array-like, optional
        Boolean mask for galaxies. If None, no masking is applied.
    cov_rho : bool, optional
        If True, compute the covariance of rho statistics.
    force_run : bool, optional
        If True, force the computation even if output files already exist.
    """

    params = get_params_rho_tau(config[version])

    print("Compute Rho and Tau statistics for the version: ", version)
    start_time = time.time()

    outdir_path = Path(outdir)
    rho_path = outdir_path / f"rho_stats_{base}.fits"
    catalog_id = f"{base}_jk" if cov_rho else base
    cov_rho_path = outdir_path / f"cov_rho_{catalog_id}.npy" if cov_rho else None

    rho_stat_handler = RhoStat(
        output=outdir, treecorr_config=treecorr_config, verbose=True
    )

    rho_stats_exists = rho_path.exists()
    cov_exists = True if not cov_rho else cov_rho_path.exists()
    need_compute = (not rho_stats_exists) or (not cov_exists) or force_run

    if need_compute:
        rho_stat_handler.catalogs.set_params(params, outdir)

        square_size = params["square_size"]

        rho_stat_handler.build_cat_to_compute_rho(
            config[version]["psf"]["path"],
            catalog_id=catalog_id,
            square_size=square_size,
            mask=mask_star,
            hdu=(
                config[version]["psf"]["hdu"]
                if config[version]["psf"]["hdu"] is not None
                else 1
            ),
        )

        rho_stat_handler.compute_rho_stats(
            catalog_id,
            rho_path.name,
            save_cov=cov_rho,
            func=_extract_xip if cov_rho else None,
            var_method="jackknife" if cov_rho else None,
        )
        rho_stat_handler.load_rho_stats(rho_path.name)
    else:
        print(f"Skipping rho statistics computation, file {rho_path} already exists.")
        rho_stat_handler.load_rho_stats(rho_path.name)

    tau_path = outdir_path / f"tau_stats_{base}.fits"

    tau_stat_handler = TauStat(
        catalogs=rho_stat_handler.catalogs,
        output=outdir,
        treecorr_config=treecorr_config,
        verbose=True,
    )

    if tau_path.exists() and not force_run:
        print(f"Skipping tau statistics computation, file {tau_path} already exists.")
        tau_stat_handler.load_tau_stats(tau_path.name)
    else:
        tau_stat_handler.catalogs.set_params(params, outdir)

        square_size = params["square_size"]

        # Build the different catalogs if necessary
        if f"psf_{version}" not in tau_stat_handler.catalogs.catalogs_dict.keys():
            tau_stat_handler.build_cat_to_compute_tau(
                config[version]["psf"]["path"],
                cat_type="psf",
                catalog_id=version,
                square_size=square_size,
                mask=mask_star,
                hdu=(
                    config[version]["psf"]["hdu"]
                    if config[version]["psf"]["hdu"] is not None
                    else 1
                ),
            )

        # Build the catalog of galaxies. PSF was computed above
        tau_stat_handler.build_cat_to_compute_tau(
            config[version]["shear"]["path"],
            cat_type="gal",
            catalog_id=version,
            square_size=square_size,
            mask=mask_gal,
        )

        # function to extract the tau_+
        tau_stat_handler.compute_tau_stats(version, tau_path.name, var_method=None)

    print(f"Time to compute rho and tau statistics: {time.time() - start_time:.2f} s")
    return rho_stat_handler, tau_stat_handler


def get_theory_cov(
    config,
    version,
    treecorr_config,
    outdir,
    base,
    nbin_ang=100,
    nbin_rad=100,
    compute_minus=True,
    mask_star=None,  # TODO add the masking in the theory covariance
    mask_gal=None,
):
    """
    Compute an analytical estimate of the covariance matrix of rho and tau-statistics.
    """

    params = get_params_rho_tau(config[version])

    info = config[version]

    path_gal = info["shear"]["path"]
    path_psf = info["psf"]["path"]
    hdu_psf = info["psf"]["hdu"]

    target_cov = Path(outdir) / f"cov_tau_{base}_th.npy"

    if target_cov.exists():
        print(f"Skipping covariance computation, file {target_cov} already exists.")
        return

    print("Computing the covariance matrix for the version: ", version)
    start_time = time.time()

    cov_tau_th = CovTauTh(
        path_gal=path_gal,
        path_psf=path_psf,
        hdu_psf=hdu_psf,
        treecorr_config=treecorr_config,
        params=params,
        mask_star=mask_star,
        mask_gal=mask_gal,
    )

    elapsed = time.time() - start_time
    print(f"--- Rho/tau statistics for covariance computed in {elapsed:.2f}s ---")

    cov = cov_tau_th.build_cov(
        nbin_ang=nbin_ang, nbin_rad=nbin_rad, compute_minus=compute_minus
    )
    print(f"--- Covariance matrix assembled in {time.time() - start_time:.2f}s ---")
    target_cov.parent.mkdir(parents=True, exist_ok=True)
    np.save(target_cov, cov)
    print("Saved covariance matrix of version: ", version)
    del cov_tau_th
    gc.collect()
    return


def get_jackknife_cov(
    config,
    version,
    treecorr_config,
    outdir,
    base,
    npatch,
    ncov=100,
    mask_star=None,
    mask_gal=None,
    force_run=False,
):
    """
    Compute the covariance matrix of rho and tau-statistics using the jackknife method.
    Also compute rho and tau-statistics.
    """

    rho_filename = f"rho_stats_{base}.fits"
    tau_filename = f"tau_stats_{base}.fits"
    tau_cov_path = Path(outdir) / f"cov_tau_{base}_jk.npy"

    if tau_cov_path.exists() and not force_run:
        print(f"Skipping covariance computation, file {tau_cov_path} already exists.")
        rho_stat_handler = RhoStat(
            output=outdir, treecorr_config=treecorr_config, verbose=False
        )

        tau_stat_handler = TauStat(
            catalogs=rho_stat_handler.catalogs,
            output=outdir,
            treecorr_config=treecorr_config,
            verbose=True,
        )

        rho_path = Path(outdir) / rho_filename
        tau_path = Path(outdir) / tau_filename
        if rho_path.exists() and not force_run:
            rho_stat_handler.load_rho_stats(rho_path.name)
        if tau_path.exists() and not force_run:
            tau_stat_handler.load_tau_stats(tau_path.name)
        return rho_stat_handler, tau_stat_handler

    params = get_params_rho_tau(config[version], survey=version)

    rho_stat_handler = RhoStat(
        output=outdir, treecorr_config=treecorr_config, verbose=False
    )

    rho_stat_handler.catalogs.set_params(params, outdir)

    square_size = params["square_size"]

    tau_stat_handler = TauStat(
        catalogs=rho_stat_handler.catalogs,
        output=outdir,
        treecorr_config=treecorr_config,
        verbose=True,
    )

    tau_stat_handler.catalogs.set_params(params, outdir)

    for i in range(ncov):
        tau_chunk = outdir + f"/cov_tau_{version}{i}.npy"
        rho_chunk = outdir + f"/cov_rho_{version}{i}.npy"
        if not (os.path.exists(tau_chunk) and os.path.exists(rho_chunk)):
            print(f"Computing rho-statistics for {version} (patch {i + 1}/{ncov})")

            if f"psf_{version}{i}" not in rho_stat_handler.catalogs.catalogs_dict:
                # Build catalogues
                rho_stat_handler.build_cat_to_compute_rho(
                    config[version]["psf"]["path"],
                    catalog_id=version + str(i),
                    square_size=square_size,
                    mask=mask_star,
                    hdu=config[version]["psf"]["hdu"],
                )

                tau_stat_handler.catalogs.catalogs_dict = (
                    rho_stat_handler.catalogs.catalogs_dict
                )

                # Build the catalog of galaxies. PSF was computed above
                tau_stat_handler.build_cat_to_compute_tau(
                    config[version]["shear"]["path"],
                    cat_type="gal",
                    catalog_id=version + str(i),
                    square_size=square_size,
                    mask=mask_gal,
                )

            else:
                print(f"Computing the patch centers for patch {i + 1}/{ncov}")

                npatch = rho_stat_handler.catalogs._params["patch_number"]
                field = rho_stat_handler.catalogs.catalogs_dict[
                    f"psf_{version}{i}"
                ].getNField(max_top=int.bit_length(npatch) - 1, coords="spherical")
                patch, centers = field.run_kmeans(npatch)

                # Update the patch centers of the catalogs
                for key, cat in rho_stat_handler.catalogs.catalogs_dict.items():
                    cat._centers = centers
                    field = cat.getNField(
                        max_top=int.bit_length(npatch) - 1, coords="spherical"
                    )
                    cat._patch = field.kmeans_assign_patches(centers)

            # Compute and save rho stats
            rho_stat_handler.compute_rho_stats(
                version + str(i),
                rho_filename,
                save_cov=True,
                func=_extract_xip,
                var_method="jackknife",
            )

            # function to extract the tau_+
            tau_stat_handler.compute_tau_stats(
                version + str(i),
                tau_filename,
                save_cov=True,
                func=_extract_xip,
                var_method="jackknife",
            )

            # Update the keys in the dictionaries
            rho_dict = rho_stat_handler.catalogs.catalogs_dict
            tau_dict = tau_stat_handler.catalogs.catalogs_dict
            rho_dict[f"psf_{version}{i + 1}"] = rho_dict.pop(f"psf_{version}{i}")
            rho_dict[f"psf_error_{version}{i + 1}"] = rho_dict.pop(
                f"psf_error_{version}{i}"
            )
            rho_dict[f"psf_size_error_{version}{i + 1}"] = rho_dict.pop(
                f"psf_size_error_{version}{i}"
            )
            tau_dict[f"gal_{version}{i + 1}"] = tau_dict.pop(f"gal_{version}{i}")

    cov_tau_loc = np.zeros_like(np.load(outdir + f"/cov_tau_{version}0.npy"))
    cov_rho_loc = np.zeros_like(np.load(outdir + f"/cov_rho_{version}0.npy"))
    for i in range(ncov):
        cov_tau_loc += np.load(outdir + f"/cov_tau_{version}{i}.npy")
        cov_rho_loc += np.load(outdir + f"/cov_rho_{version}{i}.npy")
        os.remove(outdir + f"/cov_tau_{version}{i}.npy")
        os.remove(outdir + f"/cov_rho_{version}{i}.npy")

    cov_tau = cov_tau_loc / ncov
    cov_rho = cov_rho_loc / ncov

    tau_cov_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(tau_cov_path, cov_tau)

    cov_rho_path = Path(outdir) / f"cov_rho_{base}_jk.npy"
    cov_rho_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cov_rho_path, cov_rho)

    return rho_stat_handler, tau_stat_handler


def get_samples(
    psf_fitter,
    version,
    base,
    cov_type="jk",
    apply_debias=None,
    sampler="emcee",
):
    """Return (alpha, beta, eta) samples using ``emcee`` or least squares.

    Parameters
    ----------
    psf_fitter : PSFFitter
        PSF fitter instance that provides ``load_*`` helpers.
    version : str
        Catalog identifier whose rho/tau statistics are sampled.
    base : str
        Precomputed basename (e.g. ``SP_v1.4_minsep=…``) used for filenames.
    cov_type : str, optional
        Covariance label (``'jk'``, ``'th'``, or ``'sim'``). Defaults to ``'jk'``.
    apply_debias : int or None, optional
        Jackknife patch count used to debias samples. Disabled when ``None``.
    sampler : str, optional
        ``'emcee'`` for MCMC sampling, ``'lsq'`` for least squares
        (default ``'emcee'``).
    """
    if sampler == "emcee":
        return get_samples_emcee(
            psf_fitter,
            version,
            base,
            cov_type=cov_type,
            apply_debias=apply_debias,
        )
    elif sampler == "lsq":
        return get_samples_lsq(
            psf_fitter,
            version,
            base,
            cov_type=cov_type,
            apply_debias=apply_debias,
        )
    else:
        raise ValueError("Sampler must be either 'emcee' or 'lsq'.")


def get_samples_emcee(
    psf_fitter,
    version,
    base,
    nwalkers=124,
    nsamples=10000,
    cov_type="jk",
    apply_debias=None,
):
    """Draw (alpha, beta, eta) samples using ``emcee`` and the tau covariance.

    Parameters
    ----------
    psf_fitter : PSFFitter
        PSF fitter instance managing rho/tau statistics and covariances.
    version : str
        Catalog identifier whose rho/tau statistics are sampled.
    base : str
        Precomputed basename for locating statistics/covariance files.
    nwalkers : int, optional
        Number of walkers for the MCMC run (default ``124``).
    nsamples : int, optional
        Number of samples drawn per walker (default ``10000``).
    cov_type : str, optional
        Covariance label (``'jk'``/``'th'``/``'sim'``). Defaults to ``'jk'``.
    apply_debias : int or None, optional
        Jackknife patch count applied during debiasing. Disabled when ``None``.
    """
    # Load rho and tau stats
    psf_fitter.load_rho_stat(f"rho_stats_{base}.fits")
    psf_fitter.load_tau_stat(f"tau_stats_{base}.fits")

    # Check if the path exists (use base for cache key to account for different TreeCorr configs)
    sample_file_path = psf_fitter.get_sample_file_path(base)
    if os.path.exists(sample_file_path):
        print(f"Skipping sampling; {sample_file_path} exists.")
        flat_samples = psf_fitter.load_samples(base)
        mcmc_result, q = psf_fitter.get_mcmc_from_samples(base)
        print(mcmc_result)
    # Or run MCMC
    else:
        print("MCMC sampling")
        cov_filename = f"cov_tau_{base}_{cov_type}.npy"
        psf_fitter.load_covariance(cov_filename, cov_type="tau")

        debias_npatch = apply_debias if (apply_debias is not None) else None
        flat_samples, mcmc_result, q = psf_fitter.run_chain(
            nwalkers=nwalkers,
            nsamples=nsamples,
            npatch=debias_npatch,
            apply_debias=debias_npatch is not None,
            savefig="mcmc_samples_" + version + ".png",
        )
        psf_fitter.save_samples(flat_samples, base)
    return flat_samples, mcmc_result, q


def get_samples_lsq(
    psf_fitter,
    base,
    apply_debias=None,
    cov_type="jk",
):
    """Compute least-squares samples of (alpha, beta, eta) using the tau covariance.

    Parameters
    ----------
    psf_fitter : PSFFitter
        PSF fitter instance managing rho/tau statistics and covariances.
    base : str
        Precomputed basename for locating statistics/covariance files.
    apply_debias : int or None, optional
        Jackknife patch count applied during debiasing. Disabled when ``None``.
    cov_type : str, optional
        Covariance label (defaults to ``'jk'``).
    """
    # Load rho and tau stats
    psf_fitter.load_rho_stat(f"rho_stats_{base}.fits")
    psf_fitter.load_tau_stat(f"tau_stats_{base}.fits")

    # Check if the path exists (use base for cache key to account for different TreeCorr configs)
    sample_file_path = psf_fitter.get_sample_path(base)
    if os.path.exists(sample_file_path):
        print(f"Skipping sampling; {sample_file_path} exists.")
        flat_samples = psf_fitter.load_samples(base)
        mcmc_result, q = psf_fitter.get_mcmc_from_samples(flat_samples)
        print(mcmc_result)
    # Or run MCMC
    else:
        print("Least square sampling")
        tau_covariance = f"cov_tau_{base}_{cov_type}.npy"
        rho_covariance = f"cov_rho_{base}_jk.npy"
        psf_fitter.load_covariance(tau_covariance, cov_type="tau")
        psf_fitter.load_covariance(rho_covariance, cov_type="rho")
        debias_npatch = apply_debias if (apply_debias is not None) else None
        flat_samples, mcmc_result, q = psf_fitter.get_least_squares_params_samples(
            npatch=debias_npatch, apply_debias=(debias_npatch is not None)
        )
        psf_fitter.save_samples(flat_samples, base)
    return flat_samples, mcmc_result, q
