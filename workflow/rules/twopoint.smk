# Two-point data-vector rules: xi, rho/tau, and pseudo-Cl products.

# ---------------------------------------------------------------------------
# ξ± angular grids
# ---------------------------------------------------------------------------
# The table itself lives in common.py, where it can be built from a plain config
# dict and tested; these are the workflow's bindings to it.
XI_GRIDS = xi_grids(config, FIDUCIAL)


def xi_binning(grid):
    """The `minsep=..._maxsep=..._nbins=..._npatch=...` tag of a named grid."""
    return grid_binning(XI_GRIDS[grid])


rule xi:
    """TreeCorr ξ±(θ) for one version on one angular grid, as its sealed SACC part.

    One rule for every grid: outputs are named by their binning, so a request
    binds the wildcards and the grid label resolves from them. A `_tomo`
    suffix measures every tomographic bin pair into the one part.
    """
    input:
        catalog=get_shear_catalog,
    output:
        sacc=str(COSMO_VAL / "{version}_xi_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}{tomo_suffix}.sacc"),
    threads: 24
    params:
        ver="{version}",
        compute_tomography=lambda w: w.tomo_suffix == "_tomo",
        min_sep="{min_sep}",
        max_sep="{max_sep}",
        nbins="{nbins}",
        npatch="{npatch}",
        cat_config=CAT_CONFIG,
        output_dir=str(COSMO_VAL),
        grid=lambda w: grid_of(XI_GRIDS, w),
        b_target=config.get("cosmo_val", {}).get("b_target", 0.01),
        catalogue=lambda w: measurement_catalogue(w.version),
        blind=lambda w: blind_of(w.version),
    resources:
        # The fine integration grid needs more memory and wall time than the
        # ~20-bin reporting one; scale on nbins rather than splitting the rule.
        mem_mb=lambda w: 40000 if int(w.nbins) > 100 else 30000,
        disk_mb=20000,
        runtime=lambda w: 600 if int(w.nbins) > 100 else 360,
    script:
        "../scripts/run_2pcf.py"


def xi_onecov_covariance(w):
    """The OneCovariance matrix on a ξ± request's binning and tomography:
    Gaussian on the integration grid, non-Gaussian otherwise, as cosmo_val's
    covariance inputs are."""
    gaussian = "g" if grid_of(XI_GRIDS, w) == "integration" else "ng"
    return covariance_path(
        w.version, gaussian, w.min_sep, w.max_sep, w.nbins,
        DEFAULT_MASK_SUFFIX, w.tomo_suffix,
    )


rule xi_onecov:
    """A ξ± part with the OneCovariance matrix of its binning as its covariance.

    The matrix replaces the estimate the part was born with (jackknife or shot
    noise), reordered to the part's points; the result is a derivation of the
    part, so it keeps the part's blind.
    """
    input:
        xi=str(COSMO_VAL / "{version}_xi_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}{tomo_suffix}.sacc"),
        cov=xi_onecov_covariance,
    output:
        sacc=str(COSMO_VAL / "{version}_xi_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}{tomo_suffix}_onecov.sacc"),
    params:
        blind=lambda w: blind_of(w.version),
    resources:
        mem_mb=4000,
        runtime=10,
    script:
        "../scripts/xi_onecov_sacc.py"


rule rho_tau_stats:
    output:
        rho_stats=str(COSMO_VAL / "rho_tau_stats/rho_stats_{version}_tomo_bin_all_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        tau_stats=str(COSMO_VAL / "rho_tau_stats/tau_stats_{version}_tomo_bin_all_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        covariance=str(COSMO_VAL / ("rho_tau_stats/cov_tau_{version}_tomo_bin_all_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}_" + config.get("cosmo_val", {}).get("cov_estimate_method", "th") + ".npy")),
        rho_covariance=str(COSMO_VAL / "rho_tau_stats/cov_rho_{version}_tomo_bin_all_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}_jk.npy"),
        # Born-as-SACC ρ/τ part, written alongside the FITS.
        rho_tau=str(COSMO_VAL / "rho_tau_stats/rho_tau_{version}_tomo_bin_all_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.sacc"),
    threads: 48
    params:
        ver="{version}",
        cov_estimate_method=config.get("cosmo_val", {}).get("cov_estimate_method", "th"),
        min_sep="{min_sep}",
        max_sep="{max_sep}",
        nbins="{nbins}",
        npatch="{npatch}",
        cat_config=CAT_CONFIG,
        output_dir=str(COSMO_VAL),
        b_target=config.get("cosmo_val", {}).get("b_target", 0.01),
        catalogue=lambda w: measurement_catalogue(w.version, psf=True),
        blind=lambda w: blind_of(w.version),
    resources:
        mem_mb=64000,
        disk_mb=20000,
        runtime=360,
    script:
        "../scripts/run_rho_tau.py"


rule rho_tau_stats_tomo:
    """Measure τ per source bin, reusing the PSF-only ρ measurement."""
    input:
        rho=str(COSMO_VAL / "rho_tau_stats/rho_stats_{version}_tomo_bin_all_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
    output:
        rho_tau=str(COSMO_VAL / "rho_tau_stats/rho_tau_{version}_tomo_bin_all_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}_tomo.sacc"),
    threads: 48
    params:
        ver="{version}",
        cov_estimate_method=config.get("cosmo_val", {}).get("cov_estimate_method", "th"),
        min_sep="{min_sep}", max_sep="{max_sep}", nbins="{nbins}", npatch="{npatch}",
        cat_config=CAT_CONFIG, output_dir=str(COSMO_VAL),
        blind=lambda w: blind_of(w.version),
        compute_tomography=True,
    resources:
        mem_mb=64000, disk_mb=20000, runtime=360,
    script:
        "../scripts/run_rho_tau.py"


# Pseudo-Cl generation for harmonic-space data vectors and COSEBIS validation.

wildcard_constraints:
    binning="linear|logspace|powspace",


rule pseudo_cl:
    """Generate pseudo-Cl data vector (born as SACC) with configurable binning."""
    output:
        pseudo_cl=str(COSMO_VAL / "pseudo_cl_{version}_{binning}_nbins={nbins}{tomo_suffix}.sacc"),
    params:
        version="{version}",
        cat_config=CAT_CONFIG,
        nside=1024,
        npatch=1,
        cosmo_params=PLANCK18,
        binning="{binning}",
        nbins=lambda w: int(w.nbins),
        power=0.5,
        compute_tomography=lambda w: w.tomo_suffix == "_tomo",
        catalogue=lambda w: measurement_catalogue(w.version),
        blind=lambda w: blind_of(w.version),
    resources:
        mem_mb=32000,
        runtime=120,
    threads: 12
    script:
        "../scripts/generate_pseudo_cl.py"


rule pseudo_cl_cov:
    """Generate pseudo-Cl covariance with configurable binning."""
    output:
        pseudo_cl_cov=str(COSMO_VAL / "pseudo_cl_cov_{version}_{binning}_nbins={nbins}{tomo_suffix}.fits"),
    params:
        version="{version}",
        cat_config=CAT_CONFIG,
        nside=1024,
        npatch=1,
        cosmo_params=PLANCK18,
        binning="{binning}",
        nbins=lambda w: int(w.nbins),
        power=0.5,
        compute_tomography=lambda w: w.tomo_suffix == "_tomo",
        catalogue=lambda w: measurement_catalogue(w.version),
    resources:
        mem_mb=16000,
        runtime=180,
    threads: 12
    script:
        "../scripts/generate_pseudo_cl_cov.py"


PSEUDO_CL_VERSIONS = config["versions"]


rule pseudo_cl_all:
    """Generate pseudo-Cls for all versions."""
    input:
        expand(
            str(COSMO_VAL / "pseudo_cl_{version}_powspace_nbins=32.sacc"),
            version=PSEUDO_CL_VERSIONS,
        ),


rule pseudo_cl_cov_all:
    """Generate pseudo-Cl covariances for all versions."""
    input:
        expand(
            str(COSMO_VAL / "pseudo_cl_cov_{version}_powspace_nbins=32.fits"),
            version=PSEUDO_CL_VERSIONS,
        ),


rule pseudo_cl_fine_all:
    """Generate fine pseudo-Cls for COSEBIS."""
    input:
        expand(
            str(COSMO_VAL / "pseudo_cl_{version}_linear_nbins=2040.sacc"),
            version=config["versions"],
        ),
