# FIDUCIAL, COSMO_INFERENCE, COSMO_VAL, covariance_path defined in Snakefile
# NOTE: This subsystem is dormant — see fiber pseudo-cl-assets-in-inference-28589160
COSMO_INFERENCE_PROD = Path("/home/guerrini/sp_validation/cosmo_inference")  # dormant subsystem
PSEUDO_CL_DIR = COSMO_VAL.parent  # Same directory structure
GLASS_MOCK_VERSION = config["glass_mocks"].get("version", "v0")
GLASS_MOCK_SEED_RANGE = config["glass_mocks"]["seed_range"]

GLASS_MOCK_FITS_PATTERN = str(
    COSMO_INFERENCE_PROD
    / f"data/glass_mocks/{GLASS_MOCK_VERSION}/glass_mock_{{mock_id}}"
    / f"cosmosis_glass_mock_{GLASS_MOCK_VERSION}_{{mock_id}}.fits"
)
GLASS_MOCK_CONFIG_PATTERN = str(
    COSMO_INFERENCE_PROD
    / f"cosmosis_config/cosmosis_pipeline_glass_mocks_{GLASS_MOCK_VERSION}_glass_mock_{{mock_id}}.ini"
)

def pseudo_cl_assets(version):
    """Return pseudo-Cl and covariance paths for the requested catalog version."""
    cl_path = PSEUDO_CL_DIR / f"pseudo_cl_{version}.fits"
    cov_path = PSEUDO_CL_DIR / f"pseudo_cl_cov_g_ng_iNKA_{version}.fits"
    return str(cl_path), str(cov_path)

rule inference_prep:
    input:
        # Processed covariance matrix - use centralized covariance_path()
        cov_matrix=lambda w: covariance_path(w.version, w.blind, min_sep=w.min_sep, max_sep=w.max_sep, nbins=w.nbins),
        # Xi FITS files
        xi_plus=str(COSMO_VAL / "xi_plus_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        xi_minus=str(COSMO_VAL / "xi_minus_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        # n(z) file (using new location with base version mapping)
        nz_file=lambda w: build_redshift_path(w.version, w.blind),
        # rho/tau stats
        rho_stats=str(COSMO_VAL / "rho_tau_stats/rho_stats_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        tau_stats=str(COSMO_VAL / "rho_tau_stats/tau_stats_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        # tau covariance (tracked as dependency)
        tau_cov=str(COSMO_VAL / "rho_tau_stats/cov_tau_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}_th.npy"),
        pseudo_cl=lambda w: pseudo_cl_assets(w.version)[0],
        pseudo_cl_cov=lambda w: pseudo_cl_assets(w.version)[1],
    output:
        fits_file=str(
            COSMO_INFERENCE_PROD
            / "data/{version}_{blind}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}/cosmosis_{version}_{blind}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"
        ),
        config_file=str(
            COSMO_INFERENCE_PROD
            / "cosmosis_config/cosmosis_pipeline_{version}_{blind}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.ini"
        )
    params:
        cosmosis_root="{version}_{blind}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}",
        data_dir="/n09data/guerrini/output_chains/{version}_{blind}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}",
        output_root=str(COSMO_INFERENCE_PROD),
    threads: 1
    resources:
        mem_mb=8000,
        runtime=10,
    shell:
        """
        cd /n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference

        # Run inference preparation step with cosmosis_fitting.py
        python scripts/cosmosis_fitting.py \
            --cosmosis-root {params.cosmosis_root} \
            --nz-file {input.nz_file} \
            --data-dir {params.data_dir} \
            --output-root {params.output_root} \
            --xi {input.xi_plus} {input.xi_minus} \
            --cov-xi {input.cov_matrix} \
            --use-rho-tau \
            --rho-stats {input.rho_stats} \
            --tau-stats {input.tau_stats} \
            --cov-tau {input.tau_cov} \
            --cl-file {input.pseudo_cl} \
            --cov-cl {input.pseudo_cl_cov}
        """


rule inference_fiducial:
    input:
        # Use the same output patterns as inference_prep with FIDUCIAL params
        rules.inference_prep.output.fits_file.format(
            version=FIDUCIAL["version"], blind=FIDUCIAL["blind"],
            min_sep=FIDUCIAL["min_sep"], max_sep=FIDUCIAL["max_sep"],
            nbins=FIDUCIAL["nbins"], npatch=FIDUCIAL["npatch"]
        ),
        rules.inference_prep.output.config_file.format(
            version=FIDUCIAL["version"], blind=FIDUCIAL["blind"],
            min_sep=FIDUCIAL["min_sep"], max_sep=FIDUCIAL["max_sep"],
            nbins=FIDUCIAL["nbins"], npatch=FIDUCIAL["npatch"]
        )


rule inference_glass_mocks:
    input:
        expand(GLASS_MOCK_FITS_PATTERN, mock_id=[f"{i:05d}" for i in range(GLASS_MOCK_SEED_RANGE[0], GLASS_MOCK_SEED_RANGE[1] + 1)])


# fiducial_binning_suffix() defined in Snakefile


rule inference_prep_glass_mock:
    input:
        xi="/n09data/guerrini/glass_mock_v1.4.6/results/xi_glass_mock_{mock_id}_4096_nbins=20.fits",
        # Use centralized covariance_path() with fiducial mock version
        cov_matrix=covariance_path(FIDUCIAL["mock_version"], "A"),
        # n(z) file
        nz_file=build_redshift_path(FIDUCIAL["mock_version"], "A"),
        # Rho/tau stats: rho from real data, tau sampled
        rho_stats=str(COSMO_VAL / f"rho_tau_stats/rho_stats_{FIDUCIAL['mock_version']}{fiducial_binning_suffix()}.fits"),
        tau_stats="results/glass_mock_rhotau_samples/{mock_id}/tau_stats_sampled.fits",
        # Tau covariance (real data)
        tau_cov=str(COSMO_VAL / f"rho_tau_stats/cov_tau_{FIDUCIAL['mock_version']}{fiducial_binning_suffix()}_th.npy"),
        # C_ell data for dual config generation
        cl_file="/n09data/guerrini/glass_mock_v1.4.6/results/cl_glass_mock_{mock_id}_4096.npy",
        cl_cov=pseudo_cl_assets(FIDUCIAL["mock_version"])[1],
    output:
        fits_file=GLASS_MOCK_FITS_PATTERN,
        config_file=GLASS_MOCK_CONFIG_PATTERN,
    params:
        cosmosis_root=f"glass_mock_{GLASS_MOCK_VERSION}_{{mock_id}}",
        data_dir=f"/n09data/guerrini/glass_mock_chains/glass_mock_{GLASS_MOCK_VERSION}_{{mock_id}}",
        output_root=str(COSMO_INFERENCE_PROD),
        output_basename=f"glass_mocks/{GLASS_MOCK_VERSION}/glass_mock_{{mock_id}}",
    threads: 1
    resources:
        mem_mb=8000,
        runtime=10,
    shell:
        """
        cd /n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference

        mkdir -p {params.data_dir}

        python scripts/cosmosis_fitting.py --mock \
            --cosmosis-root {params.cosmosis_root} \
            --nz-file {input.nz_file} \
            --data-dir {params.data_dir} \
            --output-root {params.output_root} \
            --output-basename {params.output_basename} \
            --xi {input.xi} \
            --cov-xi {input.cov_matrix} \
            --use-rho-tau \
            --rho-stats {input.rho_stats} \
            --tau-stats {input.tau_stats} \
            --cov-tau {input.tau_cov} \
            --cl-file {input.cl_file} \
            --cov-cl {input.cl_cov}
        """

localrules:
    inference_prep,
    inference_fiducial,
    inference_glass_mocks,
