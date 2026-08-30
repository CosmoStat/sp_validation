# Imports from Snakefile: FIDUCIAL, COSMO_INFERENCE, COSMO_VAL, covariance_path, build_redshift_path, fiducial_binning_suffix
# NOTE: dormant subsystem, not run end-to-end. Reviving it needs the FITS
# content reconciled: cosmosis_fitting.py reads ELL/EE/BB + COVAR_FULL, while
# the producers write PSEUDO_CELL/ELL + COVAR_BB_BB.

# Output root for CosmoSIS data products + configs. COSMO_INFERENCE (common.py)
# already resolves to THIS repo's cosmo_inference dir, so the products land
# beside the code that builds them rather than in a contributor's home.
COSMO_INFERENCE_PROD = COSMO_INFERENCE
# Working directory for the cosmosis_fitting.py invocation — the same repo dir.
COSMO_INFERENCE_RUNDIR = str(COSMO_INFERENCE)

# External chain/mock locations are deployment-specific, so they live in config.
INFERENCE = config["inference"]
GLASS_MOCK_DATA_DIR = INFERENCE["glass_mock_data_dir"]    # precomputed mock xi/Cl products
GLASS_MOCK_CHAINS_DIR = INFERENCE["glass_mock_chains_dir"]  # mock chain output root

PSEUDO_CL_DIR = COSMO_VAL  # producer (twopoint.smk) writes pseudo_cl* here
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

PSEUDO_CL_TAG = pseudo_cl_tag(config)


def pseudo_cl_assets(version):
    """Pseudo-Cl and covariance paths for a catalog version.

    The producer (twopoint.smk) writes wildcard-tagged names; the consumer
    reconstructs them from the fiducial harmonic-binning config.
    """
    cl_path = PSEUDO_CL_DIR / f"pseudo_cl_{version}_{PSEUDO_CL_TAG}.fits"
    cov_path = PSEUDO_CL_DIR / f"pseudo_cl_cov_{version}_{PSEUDO_CL_TAG}.fits"
    return str(cl_path), str(cov_path)

# DORMANT: the SACC migration removed the data products rule inference_prep
# (and its inference_fiducial target) named, so its DAG no longer resolves; the
# rules are dropped rather than left red. Rewiring inference onto the assembled
# {version}.sacc is tracked separately.


rule inference_glass_mocks:
    input:
        expand(GLASS_MOCK_FITS_PATTERN, mock_id=[f"{i:05d}" for i in range(GLASS_MOCK_SEED_RANGE[0], GLASS_MOCK_SEED_RANGE[1] + 1)])


rule inference_prep_glass_mock:
    input:
        xi=f"{GLASS_MOCK_DATA_DIR}/xi_glass_mock_{{mock_id}}_4096_nbins=20.fits",
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
        cl_file=f"{GLASS_MOCK_DATA_DIR}/cl_glass_mock_{{mock_id}}_4096.npy",
        cl_cov=pseudo_cl_assets(FIDUCIAL["mock_version"])[1],
    output:
        fits_file=GLASS_MOCK_FITS_PATTERN,
        config_file=GLASS_MOCK_CONFIG_PATTERN,
    params:
        cosmosis_root=f"glass_mock_{GLASS_MOCK_VERSION}_{{mock_id}}",
        data_dir=f"{GLASS_MOCK_CHAINS_DIR}/glass_mock_{GLASS_MOCK_VERSION}_{{mock_id}}",
        output_root=str(COSMO_INFERENCE_PROD),
        output_basename=f"glass_mocks/{GLASS_MOCK_VERSION}/glass_mock_{{mock_id}}",
    threads: 1
    resources:
        mem_mb=8000,
        runtime=10,
    shell:
        """
        cd {COSMO_INFERENCE_RUNDIR}

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
    inference_prep_glass_mock,
    inference_glass_mocks,
