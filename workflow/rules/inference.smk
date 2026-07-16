# Imports from Snakefile: FIDUCIAL, COSMO_INFERENCE, COSMO_VAL, covariance_path, build_redshift_path, fiducial_binning_suffix
# NOTE: dormant subsystem. The file-name plumbing (config-driven paths + the
# producer-tagged pseudo-Cl names) is fixed and the DAG is valid, but it has not
# been run end-to-end. Reviving it still needs the FITS-CONTENT plumbing
# reconciled: cosmosis_fitting.py reads ELL/EE/BB + COVAR_FULL, while the
# producers write PSEUDO_CELL/ELL + COVAR_BB_BB.

# Output root for CosmoSIS data products + configs. COSMO_INFERENCE (common.py)
# already resolves to THIS repo's cosmo_inference dir, so the products land
# beside the code that builds them rather than in a contributor's home.
COSMO_INFERENCE_PROD = COSMO_INFERENCE
# Working directory for the cosmosis_fitting.py invocation — the same repo dir.
COSMO_INFERENCE_RUNDIR = str(COSMO_INFERENCE)

# External chain/mock locations are deployment-specific, so they live in config.
INFERENCE = config["inference"]
CHAINS_DIR = INFERENCE["chains_dir"]            # CosmoSIS chain output root (real data)
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

# Fiducial harmonic-binning tag the pseudo-Cl producer (twopoint.smk) stamps
# into the filename. These are NOT inference_prep wildcards, so the consumer
# reads them from config to reconstruct the exact name the producer emits
# (canonical: blind=A, powspace, nbins=32 — see twopoint.smk pseudo_cl_all).
HARMONIC_FIDUCIAL = config["harmonic"]["fiducial"]
PSEUDO_CL_TAG = (
    f"blind={HARMONIC_FIDUCIAL['blind']}"
    f"_{HARMONIC_FIDUCIAL['binning']}"
    f"_nbins={HARMONIC_FIDUCIAL['nbins']}"
)


def pseudo_cl_assets(version):
    """Return pseudo-Cl and covariance paths for the requested catalog version.

    The producer (twopoint.smk rules pseudo_cl / pseudo_cl_cov) writes
    wildcard-tagged names; the consumer reconstructs them from the fiducial
    harmonic-binning config so the requested path matches byte-for-byte.
    """
    cl_path = PSEUDO_CL_DIR / f"pseudo_cl_{version}_{PSEUDO_CL_TAG}.fits"
    cov_path = PSEUDO_CL_DIR / f"pseudo_cl_cov_{version}_{PSEUDO_CL_TAG}.fits"
    return str(cl_path), str(cov_path)

# ---------------------------------------------------------------------------
# DORMANT — pre-SACC cosmosis assembly. Migration to native SACC deferred to
# PR 7 (native-SACC inference consumption); do NOT deep-migrate here.
#
# The SACC migration (PR 4) removed the data products several of these inputs
# name, so this rule's DAG no longer resolves and is NOT reachable from the
# cosmo_val suite (cosmo_val_all never requests it). Stale inputs:
#   - xi_plus / xi_minus FITS: the `xi` rule now emits the coarse ξ± SACC part
#     ({version}_xi_coarse_...sacc), not per-sign FITS.
#   - pseudo_cl / pseudo_cl_cov via pseudo_cl_assets(): the `pseudo_cl` rule now
#     writes .sacc (pseudo_cl_assets still requests .fits).
# PR 7 rewires this to consume the assembled {version}.sacc (built by
# cosmo_val.smk's assemble_sacc rule) directly, retiring cosmosis_fitting.py's
# per-product FITS assembly. Until then the inference target is knowingly red.
# ---------------------------------------------------------------------------
rule inference_prep:
    input:
        # Processed covariance matrix - use centralized covariance_path()
        cov_matrix=lambda w: covariance_path(w.version, w.blind, min_sep=w.min_sep, max_sep=w.max_sep, nbins=w.nbins),
        # Xi FITS files — PRE-SACC (no longer produced; see dormant note above)
        xi_plus=str(COSMO_VAL / "xi_plus_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        xi_minus=str(COSMO_VAL / "xi_minus_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        # n(z) file (using new location with base version mapping)
        nz_file=lambda w: build_redshift_path(w.version, w.blind),
        # rho/tau stats
        rho_stats=str(COSMO_VAL / "rho_tau_stats/rho_stats_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        tau_stats=str(COSMO_VAL / "rho_tau_stats/tau_stats_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        # tau covariance (tracked as dependency)
        tau_cov=str(COSMO_VAL / "rho_tau_stats/cov_tau_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}_th.npy"),
        # pseudo_cl / pseudo_cl_cov — PRE-SACC (.fits path; producer now writes .sacc)
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
        data_dir=f"{CHAINS_DIR}/{{version}}_{{blind}}_minsep={{min_sep}}_maxsep={{max_sep}}_nbins={{nbins}}_npatch={{npatch}}",
        output_root=str(COSMO_INFERENCE_PROD),
    threads: 1
    resources:
        mem_mb=8000,
        runtime=10,
    shell:
        """
        cd {COSMO_INFERENCE_RUNDIR}

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
    inference_prep,
    inference_prep_glass_mock,
    inference_fiducial,
    inference_glass_mocks,
