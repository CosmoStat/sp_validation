# Imports from common (via `from common import *`): FIDUCIAL, COSMO_INFERENCE,
# COSMO_VAL, REPO_ROOT, covariance_path, redshift_path, pseudo_cl_tag.
# cv_analysis_sacc arrives from cosmo_val.smk (resolved
# lazily at DAG time, since that file is included after this one).
#
# Two paths live here:
#   * Real-data inference_prep — live: consumes the assembled {version}.sacc and
#     emits the converter 2pt-FITS + both engine inis (2pt_like, sacc_like).
#   * glass-mock rules — still cosmosis_fitting.py-based (their SACC migration is
#     out of scope); the pseudo-Cl file-name plumbing they depend on stays below.


# Output root for CosmoSIS data products + configs. COSMO_INFERENCE (common.py)
# already resolves to THIS repo's cosmo_inference dir, so the products land
# beside the code that builds them rather than in a contributor's home.
COSMO_INFERENCE_PROD = COSMO_INFERENCE
# Working directory for the (glass-mock) cosmosis_fitting.py invocation.
COSMO_INFERENCE_RUNDIR = str(COSMO_INFERENCE)

# External chain/mock locations are deployment-specific, so they live in config.
INFERENCE = config["inference"]
CHAINS_DIR = INFERENCE["chains_dir"]  # CosmoSIS chain output root (real data)
GLASS_MOCK_CHAINS_DIR = INFERENCE["glass_mock_chains_dir"]  # mock chain output root

PSEUDO_CL_DIR = COSMO_VAL  # producer (twopoint.smk) writes pseudo_cl* here
GLASS_MOCK_VERSION = config["glass_mocks"].get("version", "v0")
GLASS_MOCK_SEED_RANGE = config["glass_mocks"]["seed_range"]

GLASS_MOCK_FITS_PATTERN = str(
    COSMO_INFERENCE_PROD
    / f"data/glass_mocks/{GLASS_MOCK_SUITE}/{GLASS_MOCK_VERSION}/glass_mock_{{mock_id}}"
    / f"cosmosis_glass_mock_{GLASS_MOCK_VERSION}_{{mock_id}}.fits"
)
GLASS_MOCK_CONFIG_PATTERN = str(
    COSMO_INFERENCE_PROD
    / f"cosmosis_config/output/{GLASS_MOCK_SUITE}/cosmosis_pipeline_glass_mocks_{GLASS_MOCK_VERSION}_glass_mock_{{mock_id}}.ini"
)

# Fiducial harmonic-binning tag the pseudo-Cl producer (twopoint.smk) stamps
# into the filename. These are NOT inference_prep wildcards, so the consumer
# reads them from config to reconstruct the exact name the producer emits.
PSEUDO_CL_TAG = pseudo_cl_tag(config)


def pseudo_cl_assets(version):
    """Pseudo-Cl and covariance paths for a catalog version.

    The producer (twopoint.smk) writes wildcard-tagged names; the consumer
    reconstructs them from the fiducial harmonic-binning config.
    """
    cl_path = PSEUDO_CL_DIR / f"pseudo_cl_{version}_{PSEUDO_CL_TAG}.fits"
    cov_path = PSEUDO_CL_DIR / f"pseudo_cl_cov_{version}_{PSEUDO_CL_TAG}.fits"
    return str(cl_path), str(cov_path)

# ---------------------------------------------------------------------------
# Real-data inference prep (native SACC). Consumes the assembled
# analysis {version}.sacc (cosmo_val.smk's assemble_sacc rule) and emits the two
# file-prep products the A_ia (IA-only, ξ±) fiducial pipeline needs:
#   (a) the converter 2pt-FITS (sacc_to_twopoint_fits) + a generated 2pt_like ini
#       — the validating/legacy path (retiring cosmosis_fitting.py's assembly),
#   (b) a generated sacc_like ini pointing at the SACC directly — the native path
#       validated bit-for-bit against (a) (test_sacc_like.py).
# The converter is A_ia-scoped: no rho/tau sidecars, so it emits a pure-ξ FITS
# (it ignores the SACC's extra data types). This is file-prep only — the actual
# CosmoSIS sampling still runs via pipeline.sh against these products.
#
# The glass-mock rules below stay cosmosis_fitting.py-based; their SACC migration
# is tracked separately.
# ---------------------------------------------------------------------------
# Generated per-version configs land in the (env-overridable) output root.
INFERENCE_CONFIG_OUT = COSMO_INFERENCE_PROD / "cosmosis_config" / "output"
# Templates belong to the launched checkout, not the product output root.
INFERENCE_TEMPLATE_DIR = (
    REPO_ROOT / "cosmo_inference" / "cosmosis_config" / "templates"
)


def _csl_dir():
    """The CSL checkout that fills COSMOSIS_DIR / sacc_like csl_dir in the inis.

    Read lazily (at DAG time, inside inference_prep's params) rather than at
    module parse time: inference.smk is included by every paper workflow, but
    only papers that run inference (cosmo_val) carry inference.csl_dir. A missing
    key still fails loudly — just when the real-data inference is actually built,
    not when an unrelated (bmodes) workflow merely parses this file.
    """
    return INFERENCE["csl_dir"]


rule inference_prep:
    input:
        # The terminal assembled analysis SACC (cosmo_val.smk assemble_sacc). Bound
        # lazily through its helper so the filename tracks that rule, not a literal.
        sacc=lambda w: cv_analysis_sacc(w.version),
        # The two pipeline ini templates are static repo files, but binding them as
        # inputs (not params) puts them in the DAG, so editing a template
        # regenerates the configs rather than leaving stale output on disk.
        template_2pt=str(INFERENCE_TEMPLATE_DIR / "cosmosis_pipeline_A_ia.ini"),
        template_sacc=str(INFERENCE_TEMPLATE_DIR / "cosmosis_pipeline_A_ia_sacc.ini"),
    output:
        fits_file=str(COSMO_INFERENCE_PROD / "data/{version}/cosmosis_{version}.fits"),
        config_file_2pt=str(
            INFERENCE_CONFIG_OUT / "cosmosis_pipeline_{version}_A_ia.ini"
        ),
        config_file_sacc=str(
            INFERENCE_CONFIG_OUT / "cosmosis_pipeline_{version}_A_ia_sacc.ini"
        ),
    params:
        # SCRATCH = the per-version chain output root the generated inis point at.
        scratch=lambda w: f"{CHAINS_DIR}/{w.version}",
        cosmosis_dir=lambda w: _csl_dir(),
        blind=lambda w: blind_of(w.version),
    threads: 1
    resources:
        mem_mb=8000,
        runtime=10,
    # script jobs run in the workflow's container; run blocks execute on the host.
    script:
        "../scripts/generate_inference_config.py"


rule inference_fiducial:
    input:
        # The fiducial version's prep products (both engine inis + the FITS).
        rules.inference_prep.output.fits_file.format(version=FIDUCIAL["version"]),
        rules.inference_prep.output.config_file_2pt.format(version=FIDUCIAL["version"]),
        rules.inference_prep.output.config_file_sacc.format(version=FIDUCIAL["version"]),


rule inference_glass_mocks:
    input:
        expand(GLASS_MOCK_FITS_PATTERN, mock_id=[f"{i:05d}" for i in range(GLASS_MOCK_SEED_RANGE[0], GLASS_MOCK_SEED_RANGE[1] + 1)])


rule inference_prep_glass_mock:
    input:
        xi=f"{GLASS_MOCK_DIR}/xi_glass_mock_{{mock_id}}_4096_nbins=20.fits",
        # Use centralized covariance_path() with fiducial mock version
        cov_matrix=covariance_path(FIDUCIAL["mock_version"]),
        # n(z) file
        nz_file=redshift_path(FIDUCIAL["mock_version"]),
        # Rho/tau stats: rho from real data, tau sampled
        rho_stats=cv_rho_stats(FIDUCIAL["mock_version"]),
        tau_stats="results/glass_mock_rhotau_samples/{mock_id}/tau_stats_sampled.fits",
        # Tau covariance (real data)
        tau_cov=cv_cov_tau(FIDUCIAL["mock_version"]),
        # C_ell data for dual config generation
        cl_file=f"{GLASS_MOCK_DIR}/cl_glass_mock_{{mock_id}}_4096.npy",
        cl_cov=pseudo_cl_assets(FIDUCIAL["mock_version"])[1],
    output:
        fits_file=GLASS_MOCK_FITS_PATTERN,
        config_file=GLASS_MOCK_CONFIG_PATTERN,
    params:
        cosmosis_root=f"glass_mock_{GLASS_MOCK_VERSION}_{{mock_id}}",
        data_dir=f"{GLASS_MOCK_CHAINS_DIR}/{GLASS_MOCK_SUITE}/glass_mock_{GLASS_MOCK_VERSION}_{{mock_id}}",
        output_root=str(COSMO_INFERENCE_PROD),
        output_basename=f"glass_mocks/{GLASS_MOCK_SUITE}/{GLASS_MOCK_VERSION}/glass_mock_{{mock_id}}",
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
