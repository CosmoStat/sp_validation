"""
Paper outputs: LaTeX macros and PTE tables built from the figure rules'
evidence.json summaries, and the aggregate `paper` target.
"""

# Figure rules whose outputs the paper target builds
FIGURE_RULES = [
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

_HARMONIC_COSEBIS_ANGULAR_RANGES = ["full", "fiducial"]


localrules: xi_cosmology_paper_macros, paper_macros, paper


rule xi_cosmology_paper_macros:
    """B-mode macros for the configuration-space cosmology paper (Goh et al.):
    joint pure-mode PTEs and scale cuts at full and fiducial scales."""
    input:
        pure_eb_evidence=rules.pure_eb_data_vector.output.evidence,
    output:
        macros="docs/unions_release/unions_2d_shear_xi/claims_macros.tex",
    params:
        tapestry_dir=TAPESTRY_DIR,
    script:
        "../scripts/generate_paper_macros.py"


rule paper_macros:
    """LaTeX macros and PTE tables for the B-modes paper (Daley et al.)."""
    input:
        pure_eb_evidence=rules.pure_eb_data_vector.output.evidence,
        pure_eb_covariance=rules.pure_eb_covariance.output.evidence,
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


rule paper:
    """Every paper figure, macro file and PTE table."""
    input:
        rules.xi_cosmology_paper_macros.output,
        rules.paper_macros.output,
        expand(
            rules.harmonic_config_cosebis_comparison.output,
            angular_range=_HARMONIC_COSEBIS_ANGULAR_RANGES,
        ),
        *(getattr(rules, name).output for name in FIGURE_RULES),
