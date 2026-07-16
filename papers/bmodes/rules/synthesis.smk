# workflow/rules/synthesis.smk
"""
Synthesis — paper specs and aggregate targets.
Synthesis rules aggregate claims into papers and generate outputs for publication.
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Variables from included files: CONFIG_DIR, TAPESTRY_DIR (Snakefile)

# Claim rules that produce evidence.json — single source of truth for all_tapestry
# Each entry is a rule name; we access rules.X.output to get all outputs
CLAIM_RULES = [
    "cosebis_version_comparison",
    "cosebis_data_vector",
    "pure_eb_data_vector",
    "pure_eb_version_comparison",
    "pure_eb_covariance",
    "cl_data_vector",
    "cl_version_comparison",
    "config_space_pte_matrices",
    "harmonic_space_pte_matrices",
    "bb_covariance_blind_independence",
    "cosebis_filter_overlay",
]

# Wildcard claim rules expanded over their parameter values
_HARMONIC_COSEBIS_ANGULAR_RANGES = ["full", "fiducial"]


def _claim_outputs():
    """Get all outputs from claim rules."""
    return {name: getattr(rules, name).output for name in CLAIM_RULES}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Paper Macros
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

localrules: xi_cosmology_paper, paper_macros, bmodes_paper_spec, all_tapestry, unblinding_ceremony

rule xi_cosmology_paper:
    """Spec for B-mode reporting in configuration-space paper (Goh et al.).

    Depends on COSEBIS version comparison, pure E/B data vector, and covariance consistency.
    Reports fiducial version, n=6 COSEBIS, joint pure-mode PTEs at both full and fiducial scales.
    Also generates evidence.json for dashboard dependency tracking.
    """
    input:
        spec=f"{CONFIG_DIR}/xi_cosmology_paper.md",
        cosebis_evidence=rules.cosebis_version_comparison.output.evidence,
        pure_eb_evidence=rules.pure_eb_data_vector.output.evidence,
        bb_blind_evidence=rules.bb_covariance_blind_independence.output.evidence,
    output:
        macros="docs/unions_release/unions_2d_shear_xi/claims_macros.tex",
        evidence=f"{TAPESTRY_DIR}/xi_cosmology_paper/evidence.json",
    params:
        tapestry_dir=TAPESTRY_DIR,
    script:
        "../scripts/generate_paper_macros.py"


rule paper_macros:
    """Generate LaTeX macros and tables for B-modes paper (Daley et al.)."""
    input:
        cosebis_evidence=rules.cosebis_version_comparison.output.evidence,
        pure_eb_evidence=rules.pure_eb_data_vector.output.evidence,
        pure_eb_covariance=rules.pure_eb_covariance.output.evidence,
        # PTE composite evidence for table generation
        config_space_pte=rules.config_space_pte_matrices.output.evidence,
        harmonic_space_pte=rules.harmonic_space_pte_matrices.output.evidence,
    output:
        bmodes="docs/unions_release/unions_bmodes/claims_macros.tex",
        pte_table_results="docs/unions_release/unions_bmodes/pte_table_results.tex",
        pte_table_appendix="docs/unions_release/unions_bmodes/pte_table_appendix.tex",
    params:
        tapestry_dir=TAPESTRY_DIR,
    script:
        "../scripts/generate_paper_macros.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Paper Spec
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rule bmodes_paper_spec:
    """Generate evidence.json for bmodes_paper spec.

    Dependencies include both evidence files and figure outputs to ensure
    all stale plots are regenerated.
    """
    input:
        spec=f"{CONFIG_DIR}/bmodes_paper.md",
        # Upstream evidence (using rules.X.output for single source of truth)
        pure_eb_covariance=rules.pure_eb_covariance.output.evidence,
        pure_eb_data_vector=rules.pure_eb_data_vector.output.evidence,
        cosebis_data_vector=rules.cosebis_data_vector.output.evidence,
        cosebis_version_comparison=rules.cosebis_version_comparison.output.evidence,
        cl_data_vector=rules.cl_data_vector.output.evidence,
        cl_version_comparison=rules.cl_version_comparison.output.evidence,
        config_space_pte=rules.config_space_pte_matrices.output.evidence,
        harmonic_space_pte=rules.harmonic_space_pte_matrices.output.evidence,
        # Paper figure dependencies (ensures version comparison plots regenerate)
        pure_eb_version_comparison=rules.pure_eb_version_comparison.output.evidence,
        cosebis_bmode_stacked=rules.cosebis_version_comparison.output.paper_stacked,
        # Consistency checks
        bb_covariance_blind=rules.bb_covariance_blind_independence.output.evidence,
    output:
        evidence=f"{TAPESTRY_DIR}/bmodes_paper/evidence.json",
    run:
        import json
        from datetime import datetime
        from pathlib import Path

        evidence = {
            "id": "bmodes_paper",
            "generated": datetime.now().isoformat(),
            "evidence": {"type": "synthesis"},
            "output": {},
        }
        with open(output.evidence, "w") as f:
            json.dump(evidence, f, indent=2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Aggregate Targets
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_CEREMONY_BLIND = config.get("ceremony_blind", "A")
_CEREMONY_CHAIN_ROOT = "/n09data/guerrini/output_chains"
_CEREMONY_COSMOSIS_DIR = "/home/guerrini/sp_validation/cosmo_inference/data"

rule unblinding_ceremony:
    """Generate unblinding ceremony figure sequence.

    Via snakemake:
        snakemake unblinding_ceremony --config ceremony_blind=B --nolock
    Standalone:
        app python workflow/scripts/unblinding_ceremony.py B
    """
    input:
        xi_data=f"{_CEREMONY_COSMOSIS_DIR}/{FIDUCIAL_VERSION}_{_CEREMONY_BLIND}/cosmosis_{FIDUCIAL_VERSION}_{_CEREMONY_BLIND}.fits",
        pure_eb=f"results/paper_plots/intermediate/{FIDUCIAL_VERSION}_{_CEREMONY_BLIND}_pure_eb_semianalytic.npz",
        pseudo_cl=_pseudo_cl_path(FIDUCIAL_VERSION, blind=_CEREMONY_BLIND),
        pseudo_cl_cov=_pseudo_cl_cov_path(FIDUCIAL_VERSION, blind=_CEREMONY_BLIND),
        cosmosis_cell_fits=f"{_CEREMONY_COSMOSIS_DIR}/{FIDUCIAL_VERSION}_{_CEREMONY_BLIND}_fid/cosmosis_{FIDUCIAL_VERSION}_{_CEREMONY_BLIND}_fid_cell.fits",
        bestfit_dir=f"{_CEREMONY_CHAIN_ROOT}/best_fit/{FIDUCIAL_VERSION}_{_CEREMONY_BLIND}_10_80/shear_xi_plus/theta.txt",
    output:
        evidence=f"{TAPESTRY_DIR}/unblinding_ceremony/evidence.json",
    params:
        blind=_CEREMONY_BLIND,
        chain_version=FIDUCIAL_VERSION.replace("SP_", "").replace("_leak_corr", ""),
        chain_prefix=FIDUCIAL_VERSION,
        chain_root_dir=_CEREMONY_CHAIN_ROOT,
        results_dir="results/unblinding",
        bestfit_root_fid_cell=f"{FIDUCIAL_VERSION}_{_CEREMONY_BLIND}_fid_cell",
        bestfit_root_halofit_cell=f"{FIDUCIAL_VERSION}_{_CEREMONY_BLIND}_halofit_cell",
        bestfit_root_config=f"{FIDUCIAL_VERSION}_{_CEREMONY_BLIND}_10_80",
    shell:
        "python workflow/scripts/unblinding_ceremony.py {params.blind} --chain-version {params.chain_version}"


rule all_tapestry:
    """Aggregate target for all claim evidence and paper outputs."""
    input:
        bmodes_paper=rules.bmodes_paper_spec.output,
        xi_cosmology_paper=rules.xi_cosmology_paper.output,
        paper_macros=rules.paper_macros.output,
        harmonic_cosebis=expand(
            f"{TAPESTRY_DIR}/harmonic_config_cosebis_comparison_{{angular_range}}/evidence.json",
            angular_range=_HARMONIC_COSEBIS_ANGULAR_RANGES,
        ),
        **_claim_outputs(),
