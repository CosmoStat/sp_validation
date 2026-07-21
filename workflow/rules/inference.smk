# Imports from common (via `from common import *`): FIDUCIAL, COSMO_INFERENCE,
# COSMO_VAL, WORKFLOW_SCRIPTS, covariance_path, build_redshift_path,
# fiducial_binning_suffix. cv_analysis_sacc arrives from cosmo_val.smk (resolved
# lazily at DAG time, since that file is included after this one).
#
# Two paths live here:
#   * Real-data inference_prep — LIVE (PR 7): consumes the assembled {version}.sacc
#     and emits the converter 2pt-FITS + both engine inis (2pt_like, sacc_like).
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
# Real-data inference prep — LIVE (native SACC, PR 7). Consumes the assembled
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
# is out of scope for PR 7.
# ---------------------------------------------------------------------------
# Generated per-version configs land in the (env-overridable) output root.
INFERENCE_CONFIG_OUT = COSMO_INFERENCE_PROD / "cosmosis_config"
# The ini TEMPLATES are source files: anchor them on the running checkout (repo
# root = the workflow dir's parent, via WORKFLOW_SCRIPTS), NOT on the output root
# — so a template edit in this checkout drives the DAG even when COSMO_INFERENCE
# points elsewhere. In a normal (non-worktree) run the two roots coincide.
INFERENCE_TEMPLATE_DIR = (
    Path(os.path.dirname(WORKFLOW_SCRIPTS)).parent / "cosmo_inference" / "cosmosis_config"
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
    threads: 1
    resources:
        mem_mb=8000,
        runtime=10,
    run:
        import os
        import sys

        from sp_validation import sacc_io
        from sp_validation.sacc_io import sacc_to_twopoint_fits

        os.makedirs(os.path.dirname(output.fits_file), exist_ok=True)

        # (a) converter 2pt-FITS — pure ξ (A_ia scope; no rho/tau sidecars).
        sacc_to_twopoint_fits(sacc_io.load(input.sacc), output.fits_file, n_bins=1)

        # (b) + (c) the two generated pipeline inis, from the template inputs.
        # WORKFLOW_SCRIPTS (common.py) is the absolute generic-workflow scripts dir.
        sys.path.insert(0, WORKFLOW_SCRIPTS)
        from generate_inference_config import (
            _substitutions,
            generate_inference_config,
        )

        generate_inference_config(
            input.template_2pt,
            output.config_file_2pt,
            _substitutions(
                scratch=params.scratch,
                cosmosis_dir=params.cosmosis_dir,
                fits_file=output.fits_file,
            ),
        )
        generate_inference_config(
            input.template_sacc,
            output.config_file_sacc,
            _substitutions(
                scratch=params.scratch,
                cosmosis_dir=params.cosmosis_dir,
                sacc_file=input.sacc,
            ),
        )


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
