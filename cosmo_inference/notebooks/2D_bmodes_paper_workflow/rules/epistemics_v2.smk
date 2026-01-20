# workflow/rules/epistemics_v2.smk
"""
Spec-driven epistemic infrastructure.
Specs are markdown files in workflow/config/ — human-readable, trigger rebuilds when changed.
Results go to results/claims/.
"""

import os

# Directories
CONFIG_DIR = "workflow/config"
CLAIMS_DIR = "results/claims"
COSMO_VAL_OUTPUT = "/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/output"
# Sasha is source of truth for pseudo_cl files
SASHA_COSMO_VAL_OUTPUT = "/home/guerrini/sp_validation/notebooks/cosmo_val/output"
PAPER_FIGURES_DIR = "docs/unions_release/unions_bmodes/Figures"
SKILL_PATH = os.path.expanduser("~/.claude/skills/conducting-research/templates")


def _covariance_path(version, min_sep, max_sep, nbins, blind=None, gaussian="g"):
    """Construct covariance file path (epistemics_v2 version)."""
    if blind is None:
        blind = config["fiducial"]["blind"]
    base_name = f"covariance_{version}_{blind}_{gaussian}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_masked"
    return (
        "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/data/covariance/"
        f"{base_name}/{base_name}_processed.txt"
    )


def _reporting_cov_path(version, blind):
    """Path to reporting-scale covariance (non-Gaussian, masked)."""
    min_sep = config["fiducial"]["min_sep"]
    max_sep = config["fiducial"]["max_sep"]
    nbins = config["fiducial"]["nbins"]
    return _covariance_path(version, min_sep, max_sep, nbins, blind=blind, gaussian="ng")


BLINDS = ["A", "B", "C"]

# Note: cl_pte_matrix removed, harmonic_space_pte_matrices handles all versions


def _xi_integration_path(version):
    """Path to fine-binned 2PCF integration file."""
    return (
        f"{COSMO_VAL_OUTPUT}/{version}_xi_minsep={config['fiducial']['min_sep_int']}"
        f"_maxsep={config['fiducial']['max_sep_int']}"
        f"_nbins={config['fiducial']['nbins_int']}"
        f"_npatch={config['fiducial']['npatch']}.txt"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Claims
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rule cosebis_version_comparison:
    """B-mode visualization: COSEBIS B-modes across catalog versions.

    Plotting only - statistical PTEs are in cosebis_pte_matrix.
    """
    input:
        specs=[
            f"{CONFIG_DIR}/cosebis_version_comparison.md",
            f"{CONFIG_DIR}/cosebis.md",
            f"{CONFIG_DIR}/1d_plots.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        xi_integration=[_xi_integration_path(ver) for ver in config["versions"]],
        # Uses blind A for visualization (B-modes identical across blinds)
        # TODO: this is not true, fix..
        cov_integration=[
            _covariance_path(ver, config["fiducial"]["min_sep_int"], config["fiducial"]["max_sep_int"], config["fiducial"]["nbins_int"], blind="A")
            for ver in config["versions"]
        ],
    params:
        cov_base_dir="/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/data/covariance",
    output:
        evidence=f"{CLAIMS_DIR}/cosebis_version_comparison/evidence.json",
        figure_stacked=f"{CLAIMS_DIR}/cosebis_version_comparison/figure_stacked.png",
        paper_stacked=f"{PAPER_FIGURES_DIR}/cosebis_bmode_stacked.png",
    script:
        "../scripts/cosebis_version_comparison.py"


rule cosebis_data_vector:
    """B-mode data vector: COSEBIS B-modes for fiducial version (v1.4.6).

    Single-panel figure combining fiducial and full angular ranges.
    Paper figure for main text. PTEs are in cosebis_pte_matrix.
    """
    input:
        specs=[
            f"{CONFIG_DIR}/cosebis_data_vector.md",
            f"{CONFIG_DIR}/cosebis.md",
            f"{CONFIG_DIR}/1d_plots.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        xi_integration=_xi_integration_path(config["fiducial"]["version"]),
        cov_integration=_covariance_path(
            config["fiducial"]["version"],
            config["fiducial"]["min_sep_int"],
            config["fiducial"]["max_sep_int"],
            config["fiducial"]["nbins_int"],
            blind="A",
        ),
    params:
        cov_base_dir="/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/data/covariance",
    output:
        evidence=f"{CLAIMS_DIR}/cosebis_data_vector/evidence.json",
        figure=f"{CLAIMS_DIR}/cosebis_data_vector/figure.png",
        paper_figure=f"{PAPER_FIGURES_DIR}/cosebis_data_vector.png",
    script:
        "../scripts/cosebis_data_vector.py"


rule pure_eb_data_vector:
    """B-mode null test: Pure E/B data vector at fiducial scale cuts.

    Multi-blind: Computes PTEs for each blind (A, B, C) using per-blind
    MC-propagated E/B covariances from the Gaussian integration covariances.
    """
    input:
        specs=[
            f"{CONFIG_DIR}/pure_eb_data_vector.md",
            f"{CONFIG_DIR}/pure_eb.md",
            f"{CONFIG_DIR}/1d_plots.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        # Per-blind MC-propagated E/B covariances
        pure_eb_A=f"results/paper_plots/intermediate/{config['fiducial']['version']}_A_pure_eb_semianalytic.npz",
        pure_eb_B=f"results/paper_plots/intermediate/{config['fiducial']['version']}_B_pure_eb_semianalytic.npz",
        pure_eb_C=f"results/paper_plots/intermediate/{config['fiducial']['version']}_C_pure_eb_semianalytic.npz",
        # Per-blind reporting covariances (for total xi error bars)
        cov_a=_reporting_cov_path(config["fiducial"]["version"], "A"),
        cov_b=_reporting_cov_path(config["fiducial"]["version"], "B"),
        cov_c=_reporting_cov_path(config["fiducial"]["version"], "C"),
    output:
        evidence=f"{CLAIMS_DIR}/pure_eb_data_vector/evidence.json",
        paper_figure=f"{PAPER_FIGURES_DIR}/pure_eb_data_vector.png",
    script:
        "../scripts/pure_eb_data_vector.py"


rule pure_eb_version_comparison:
    """B-mode visualization: Pure E/B across catalog versions.

    Plotting only - statistical PTEs are in pure_eb_pte_matrix and config_space_pte_matrices.
    Uses E-mode errors from pure_eb covariance as proxy for total xi (E dominates).
    """
    input:
        specs=[
            f"{CONFIG_DIR}/pure_eb_version_comparison.md",
            f"{CONFIG_DIR}/pure_eb.md",
            f"{CONFIG_DIR}/1d_plots.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        pure_eb_data=[
            f"results/paper_plots/intermediate/{ver}_pure_eb_semianalytic.npz"
            for ver in config["versions"]
        ],
    output:
        evidence=f"{CLAIMS_DIR}/pure_eb_version_comparison/evidence.json",
        figure=f"{CLAIMS_DIR}/pure_eb_version_comparison/figure.png",
        paper_figure=f"{PAPER_FIGURES_DIR}/pure_eb_versions.png",
    script:
        "../scripts/pure_eb_version_comparison.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Harmonic-Space Claims (Cl)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rule cl_data_vector:
    """Harmonic-space B-mode power spectra at fiducial ell range."""
    input:
        specs=[
            f"{CONFIG_DIR}/cl_fiducial.md",
            f"{CONFIG_DIR}/cl.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        pseudo_cl=f"{SASHA_COSMO_VAL_OUTPUT}/pseudo_cl_{config['fiducial']['version']}.fits",
        pseudo_cl_cov=f"{SASHA_COSMO_VAL_OUTPUT}/pseudo_cl_cov_{config['fiducial']['version']}.fits",
    output:
        evidence=f"{CLAIMS_DIR}/cl_fiducial/evidence.json",
        figure=f"{CLAIMS_DIR}/cl_fiducial/figure.png",
        paper_figure=f"{PAPER_FIGURES_DIR}/cl_fiducial.png",
    script:
        "../scripts/cl_fiducial.py"


rule cl_version_comparison:
    """C_ell^BB version comparison across catalog versions."""
    input:
        specs=[
            f"{CONFIG_DIR}/cl_version_comparison.md",
            f"{CONFIG_DIR}/cl.md",
            f"{CONFIG_DIR}/cl_fiducial.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        cl_fiducial_evidence=rules.cl_data_vector.output.evidence,
        pseudo_cl=[
            f"{SASHA_COSMO_VAL_OUTPUT}/pseudo_cl_{ver}.fits"
            for ver in config["versions"]
        ],
        pseudo_cl_cov=[
            f"{SASHA_COSMO_VAL_OUTPUT}/pseudo_cl_cov_{ver}.fits"
            for ver in config["versions"]
        ],
    output:
        evidence=f"{CLAIMS_DIR}/cl_version_comparison/evidence.json",
        figure=f"{CLAIMS_DIR}/cl_version_comparison/figure.png",
        paper_figure=f"{PAPER_FIGURES_DIR}/cl_versions.png",
    script:
        "../scripts/cl_version_comparison.py"


# cl_pte_matrix removed — superseded by harmonic_space_pte_matrices which handles all versions

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pure E/B PTE Matrix
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rule pure_eb_pte_matrix:
    """PTE heatmap: Pure E/B B-modes across scale cut combinations."""
    input:
        specs=[
            f"{CONFIG_DIR}/pure_eb_pte_matrix.md",
            f"{CONFIG_DIR}/pure_eb_data_vector.md",
            f"{CONFIG_DIR}/2d_plots.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        pure_eb_evidence=rules.pure_eb_data_vector.output.evidence,
        # Pre-computed PTE matrices (from calculate_pure_eb_ptes.py)
        pte_data=f"results/paper_plots/intermediate/{config['fiducial']['version']}_pure_eb_ptes.npz",
    output:
        evidence=f"{CLAIMS_DIR}/pure_eb_pte_matrix/evidence.json",
        figure_xip=f"{CLAIMS_DIR}/pure_eb_pte_matrix/figure_xip.png",
        figure_xim=f"{CLAIMS_DIR}/pure_eb_pte_matrix/figure_xim.png",
        paper_figure_xip=f"{PAPER_FIGURES_DIR}/pure_eb_pte_xip.png",
        paper_figure_xim=f"{PAPER_FIGURES_DIR}/pure_eb_pte_xim.png",
    script:
        "../scripts/pure_eb_pte_matrix.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COSEBIS PTE Matrix (scatter-gather pattern)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _pte_scale_cut_pairs():
    """Return list of (i_min, i_max) index pairs for PTE matrix.

    Note: (9, 10) excluded due to numerical instability in cosmo_numba
    polynomial root finding with nmodes=20.
    """
    return [(i, j) for i in range(20) for j in range(i + 1, 21) if (i, j) != (9, 10)]


def _cov_integration_path(version, blind):
    """Covariance path for integration bins (Gaussian, for COSEBIS PTE)."""
    min_sep_int = config["fiducial"]["min_sep_int"]
    max_sep_int = config["fiducial"]["max_sep_int"]
    nbins_int = config["fiducial"]["nbins_int"]
    return _covariance_path(version, min_sep_int, max_sep_int, nbins_int, blind=blind, gaussian="g")


rule compute_cosebis_pte:
    """Scatter: Compute COSEBIS B-mode PTE for a single (version, blind, i_min, i_max) tuple."""
    input:
        xi_integration=lambda w: _xi_integration_path(w.version),
        cov_integration=lambda w: _cov_integration_path(w.version, w.blind),
    output:
        pte_json=f"{CLAIMS_DIR}/cosebis_pte_matrix/pte_values/{{version}}/{{blind}}/pte_{{i_min}}_{{i_max}}.json",
    params:
        nmodes=config["fiducial"]["nmodes"],
    wildcard_constraints:
        i_min=r"\d{3}",
        i_max=r"\d{3}",
        blind=r"[ABC]",
    threads: 1
    script:
        "../scripts/compute_cosebis_pte_single.py"


rule cosebis_pte_matrix:
    """Gather: Assemble PTE values into matrices, generate figures and evidence."""
    input:
        specs=[
            f"{CONFIG_DIR}/cosebis_pte_matrix.md",
            f"{CONFIG_DIR}/cosebis_version_comparison.md",
            f"{CONFIG_DIR}/2d_plots.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        pte_files=[
            f"{CLAIMS_DIR}/cosebis_pte_matrix/pte_values/{ver}/{blind}/pte_{i:03d}_{j:03d}.json"
            for ver in config["versions"]
            for blind in BLINDS
            for i, j in _pte_scale_cut_pairs()
        ],
    output:
        evidence=f"{CLAIMS_DIR}/cosebis_pte_matrix/evidence.json",
        figures=expand(
            f"{CLAIMS_DIR}/cosebis_pte_matrix/figure_{{version}}_{{blind}}_n{{nmodes}}.png",
            version=config["versions"],
            blind=BLINDS,
            nmodes=[6, 20],
        ),
        paper_figures=expand(
            f"{PAPER_FIGURES_DIR}/cosebis_pte_matrices_{{version}}_{{blind}}_n{{nmodes}}.png",
            version=config["versions"],
            blind=BLINDS,
            nmodes=[6, 20],
        ),
    script:
        "../scripts/cosebis_pte_matrix.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pure E/B Covariance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rule pure_eb_covariance:
    """Covariance structure: 6-block pure E/B covariance correlation matrix.

    Validates covariance structure for B-mode tests by showing:
    - E and B blocks are well-conditioned (~10^5)
    - Ambiguous blocks are ill-conditioned (~10^15, expected)
    - Correlation structure across 6 blocks (E+/E-/B+/B-/amb+/amb-)

    Uses blind A covariance for visualization (structure is similar across blinds).
    """
    input:
        specs=[
            f"{CONFIG_DIR}/pure_eb_covariance.md",
            f"{CONFIG_DIR}/pure_eb.md",
            f"{CONFIG_DIR}/covariance.md",
            f"{CONFIG_DIR}/2d_plots.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        pure_eb_data=f"results/paper_plots/intermediate/{config['fiducial']['version']}_A_pure_eb_semianalytic.npz",
    output:
        evidence=f"{CLAIMS_DIR}/pure_eb_covariance/evidence.json",
        figure=f"{CLAIMS_DIR}/pure_eb_covariance/figure.png",
        paper_figure=f"{PAPER_FIGURES_DIR}/eb_covariance.png",
    script:
        "../scripts/pure_eb_covariance.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PTE Matrix Composites (Results + Appendix)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rule config_space_pte_matrices:
    """Configuration-space PTE composites for B-modes paper.

    Main text: 1x3 composite for fiducial version (xi+^B, xi-^B, COSEBIS B_n)
    Appendix: 3x3 composite for all versions (3 rows x 3 statistics)
    """
    input:
        specs=[
            f"{CONFIG_DIR}/config_space_pte_matrices.md",
            f"{CONFIG_DIR}/pure_eb.md",
            f"{CONFIG_DIR}/cosebis.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        pure_eb_pte=[
            f"results/paper_plots/intermediate/{ver}_pure_eb_ptes.npz"
            for ver in config["versions"]
        ],
        cosebis_pte_files=[
            f"{CLAIMS_DIR}/cosebis_pte_matrix/pte_values/{ver}/A/pte_{i:03d}_{j:03d}.json"
            for ver in config["versions"]
            for i, j in _pte_scale_cut_pairs()
        ],
    output:
        evidence=f"{CLAIMS_DIR}/config_space_pte_matrices/evidence.json",
        figure_fiducial=f"{CLAIMS_DIR}/config_space_pte_matrices/figure_fiducial.png",
        figure_appendix=f"{CLAIMS_DIR}/config_space_pte_matrices/figure_appendix.png",
        paper_figure_fiducial=f"{PAPER_FIGURES_DIR}/config_space_pte_fiducial.png",
        paper_figure_appendix=f"{PAPER_FIGURES_DIR}/config_space_pte_composite_appendix.png",
    script:
        "../scripts/config_space_pte_matrices.py"


rule harmonic_space_pte_matrices:
    """Harmonic-space PTE figures for all versions.

    Results: Single-panel Cl^BB PTE matrix for fiducial v1.4.6
    Appendix: 2-panel composite (v1.4.5 | v1.4.8)
    """
    input:
        specs=[
            f"{CONFIG_DIR}/harmonic_space_pte_matrices.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        pseudo_cl=[
            f"{SASHA_COSMO_VAL_OUTPUT}/pseudo_cl_{ver}.fits"
            for ver in config["versions"]
        ],
        pseudo_cl_cov=[
            f"{SASHA_COSMO_VAL_OUTPUT}/pseudo_cl_cov_{ver}.fits"
            for ver in config["versions"]
        ],
    output:
        evidence=f"{CLAIMS_DIR}/harmonic_space_pte_matrices/evidence.json",
        figure_fiducial=f"{CLAIMS_DIR}/harmonic_space_pte_matrices/figure_fiducial.png",
        figure_appendix=f"{CLAIMS_DIR}/harmonic_space_pte_matrices/figure_appendix.png",
        paper_figure_fiducial=f"{PAPER_FIGURES_DIR}/cl_pte_heatmap.png",
        paper_figure_appendix=f"{PAPER_FIGURES_DIR}/cl_pte_composite_appendix.png",
    script:
        "../scripts/harmonic_space_pte_matrices.py"


rule harmonic_config_cosebis_comparison:
    """Cross-validate COSEBIS from harmonic-space C_ℓ vs configuration-space ξ±.

    Tests consistency between two independent paths to COSEBIS:
    - Harmonic: pseudo-C_ℓ → T_n(ℓ) integration → E_n, B_n
    - Config: ξ± → W_n(θ) integration → E_n, B_n

    Agreement validates both pseudo-C_ℓ estimation and COSEBIS machinery.
    """
    input:
        spec=f"{CONFIG_DIR}/harmonic_config_cosebis_comparison.md",
        config=f"{CONFIG_DIR}/config.yaml",
        # Finely-binned pseudo-Cls (nellbins=1000, linear Δℓ=16) for accurate COSEBIS conversion
        pseudo_cls=[
            f"{COSMO_VAL_OUTPUT}/pseudo_cl_{ver}_nellbins=1000.fits"
            for ver in config["versions"]
        ],
        pseudo_cls_cov=[
            f"{COSMO_VAL_OUTPUT}/pseudo_cl_cov_{ver}_nellbins=1000.fits"
            for ver in config["versions"]
        ],
        cov_integration=[
            _covariance_path(
                ver,
                config["fiducial"]["min_sep_int"],
                config["fiducial"]["max_sep_int"],
                config["fiducial"]["nbins_int"],
                blind="A",
            )
            for ver in config["versions"]
        ],
    output:
        comparison=f"{CLAIMS_DIR}/harmonic_config_cosebis_comparison/figure.png",
        stats=f"{CLAIMS_DIR}/harmonic_config_cosebis_comparison/stats.txt",
    params:
        versions=config["versions"],
        nmodes_long=config["cosebis"]["nmodes"],
        nmodes_short=config["cosebis"]["mode_subsets"][0],
        theta_min=config["cosebis"]["theta_min"],
        theta_max=config["cosebis"]["theta_max"],
        min_sep_int=config["fiducial"]["min_sep_int"],
        max_sep_int=config["fiducial"]["max_sep_int"],
        nbins_int=config["fiducial"]["nbins_int"],
        npatch=config["fiducial"]["npatch"],
    script:
        "../scripts/compare_harmonic_config_cosebis.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Paper Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

localrules: xi_cosmology_paper, paper_macros, cl_data_vector, cl_version_comparison, pure_eb_pte_matrix, pure_eb_covariance, pure_eb_version_comparison, cosebis_pte_matrix, config_space_pte_matrices, harmonic_space_pte_matrices, harmonic_config_cosebis_comparison

rule xi_cosmology_paper:
    """Spec for B-mode reporting in configuration-space paper (Goh et al.).

    Depends on the two B-mode claims plus covariance consistency, produces macros for Paper II.
    Reports v1.4.6, n=6 COSEBIS, joint pure-mode PTEs at both full and fiducial scales.
    Also generates evidence.json for dashboard dependency tracking.
    """
    input:
        spec=f"{CONFIG_DIR}/xi_cosmology_paper.md",
        cosebis_evidence=rules.cosebis_version_comparison.output.evidence,
        pure_eb_evidence=rules.pure_eb_data_vector.output.evidence,
        covariance_evidence=rules.covariance_blind_consistency.output.evidence,
    output:
        macros="docs/unions_release/unions_2d_shear_xi/claims_macros.tex",
        evidence=f"{CLAIMS_DIR}/xi_cosmology_paper/evidence.json",
    params:
        claims_dir=CLAIMS_DIR,
    script:
        "../scripts/generate_paper_macros.py"


rule paper_macros:
    """Generate LaTeX macros and tables for B-modes paper (Daley et al.)."""
    input:
        cosebis_evidence=rules.cosebis_version_comparison.output.evidence,
        pure_eb_evidence=rules.pure_eb_data_vector.output.evidence,
        # PTE composite evidence for table generation
        config_space_pte=rules.config_space_pte_matrices.output.evidence,
        harmonic_space_pte=rules.harmonic_space_pte_matrices.output.evidence,
    output:
        bmodes="docs/unions_release/unions_bmodes/claims_macros.tex",
        pte_table_results="docs/unions_release/unions_bmodes/pte_table_results.tex",
        pte_table_appendix="docs/unions_release/unions_bmodes/pte_table_appendix.tex",
    params:
        claims_dir=CLAIMS_DIR,
    script:
        "../scripts/generate_paper_macros.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Method Specs (foundational, no dependencies)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METHOD_SPECS = ["cosebis", "pure_eb", "covariance", "cl", "1d_plots", "2d_plots"]


rule method_spec_evidence:
    """Generate evidence.json for method specs (no dependencies)."""
    input:
        spec=f"{CONFIG_DIR}/{{method_spec}}.md",
    output:
        evidence=f"{CLAIMS_DIR}/{{method_spec}}/evidence.json",
    wildcard_constraints:
        method_spec="|".join(METHOD_SPECS),
    script:
        "/home/cdaley/.claude/skills/conducting-research/templates/scripts/generate_spec_evidence.py"


rule all_method_specs:
    """Build all method spec evidence files."""
    input:
        expand(f"{CLAIMS_DIR}/{{spec}}/evidence.json", spec=METHOD_SPECS),


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Paper Specs (depend on claims)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rule bmodes_paper_spec:
    """Generate evidence.json for bmodes_paper spec.

    Dependencies include both evidence files and figure outputs to ensure
    dashboard regenerates all stale plots.
    """
    input:
        spec=f"{CONFIG_DIR}/bmodes_paper.md",
        # Upstream evidence (using rules.X.output for single source of truth)
        pure_eb_covariance=rules.pure_eb_covariance.output.evidence,
        pure_eb_data_vector=rules.pure_eb_data_vector.output.evidence,
        cosebis_version_comparison=rules.cosebis_version_comparison.output.evidence,
        cl_fiducial=rules.cl_data_vector.output.evidence,
        config_space_pte=rules.config_space_pte_matrices.output.evidence,
        harmonic_space_pte=rules.harmonic_space_pte_matrices.output.evidence,
        # Figure dependencies from bmodes.smk (ensures dashboard triggers regeneration)
        plot_eb_xis=expand(
            "/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/output/{version}_eb_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_minsepint={min_sep_int}_maxsepint={max_sep_int}_nbinsint={nbins_int}_npatch={npatch}_varmethod=semi-analytic_xis.png",
            version=config["versions"],
            min_sep=config["fiducial"]["min_sep"],
            max_sep=config["fiducial"]["max_sep"],
            nbins=config["fiducial"]["nbins"],
            min_sep_int=config["fiducial"]["min_sep_int"],
            max_sep_int=config["fiducial"]["max_sep_int"],
            nbins_int=config["fiducial"]["nbins_int"],
            npatch=config["fiducial"]["npatch"],
        ),
        plot_cosebis_modes=expand(
            "/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/output/{version}_cosebis_minsep=" + str(config['fiducial']['min_sep_int']) + "_maxsep=" + str(config['fiducial']['max_sep_int']) + "_nbins=" + str(config['fiducial']['nbins_int']) + "_npatch={npatch}_varmethod=analytic_nmodes=" + str(config['fiducial']['nmodes']) + "_scalecut=" + str(config['fiducial']['fiducial_min_scale']) + "-" + str(config['fiducial']['fiducial_max_scale']) + "_cosebis.png",
            version=config["versions"],
            npatch=config["fiducial"]["npatch"],
        ),
    output:
        evidence=f"{CLAIMS_DIR}/bmodes_paper/evidence.json",
    script:
        "/home/cdaley/.claude/skills/conducting-research/templates/scripts/generate_spec_evidence.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dashboard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

localrules: claims_dashboard, serve_claims, paper_macros, method_spec_evidence, all_method_specs, bmodes_paper_spec, spec_dependencies, all_claims


rule all_claims:
    """Aggregate target for all claim evidence (used by spec_dependencies)."""
    input:
        method_specs=expand(f"{CLAIMS_DIR}/{{spec}}/evidence.json", spec=METHOD_SPECS),
        bmodes_paper=rules.bmodes_paper_spec.output,
        xi_cosmology_paper=rules.xi_cosmology_paper.output,
        cosebis_version_comparison=rules.cosebis_version_comparison.output,
        cosebis_data_vector=rules.cosebis_data_vector.output,
        pure_eb_data_vector=rules.pure_eb_data_vector.output,
        pure_eb_version_comparison=rules.pure_eb_version_comparison.output,
        pure_eb_covariance=rules.pure_eb_covariance.output,
        cl_fiducial=rules.cl_data_vector.output,
        cl_version_comparison=rules.cl_version_comparison.output,
        config_space_pte=rules.config_space_pte_matrices.output,
        harmonic_space_pte=rules.harmonic_space_pte_matrices.output,


rule spec_dependencies:
    """Extract spec dependency graph from snakemake DAG.

    Queries snakemake's own detailed-summary to derive which specs depend
    on which, based on actual input file declarations. Single source of truth.
    """
    output:
        deps=f"{CLAIMS_DIR}/deps.json",
    run:
        import subprocess, json
        from pathlib import Path

        r = subprocess.run(
            ["snakemake", "--forceall", "--detailed-summary", "all_claims"],
            capture_output=True, text=True
        )

        specs = {}
        for line in r.stdout.split("\n"):
            p = line.split("\t")
            if len(p) < 7 or "evidence.json" not in p[0]:
                continue
            sid = Path(p[0]).parent.name
            deps = []
            for x in p[4].split(","):
                x = x.strip()
                if x.endswith(".md") and Path(x).stem != sid:
                    deps.append(Path(x).stem)
                elif x.endswith("evidence.json") and Path(x).parent.name != sid:
                    deps.append(Path(x).parent.name)
            specs[sid] = {"deps": sorted(set(deps)), "date": p[1], "status": p[6]}

        with open(output.deps, "w") as f:
            json.dump(specs, f, indent=2)


rule claims_dashboard:
    """Render claims dashboard with specs and evidence."""
    input:
        config=f"{CONFIG_DIR}/config.yaml",
        deps=f"{CLAIMS_DIR}/deps.json",
        # Method specs (foundational, no dependencies)
        method_specs=expand(f"{CLAIMS_DIR}/{{spec}}/evidence.json", spec=METHOD_SPECS),
        # Paper specs
        bmodes_paper=rules.bmodes_paper_spec.output,
        xi_cosmology_paper=rules.xi_cosmology_paper.output,
        # All claim rules that produce evidence
        cosebis_version_comparison=rules.cosebis_version_comparison.output,
        cosebis_data_vector=rules.cosebis_data_vector.output,
        pure_eb_data_vector=rules.pure_eb_data_vector.output,
        pure_eb_version_comparison=rules.pure_eb_version_comparison.output,
        pure_eb_covariance=rules.pure_eb_covariance.output,
        cl_fiducial=rules.cl_data_vector.output,
        cl_version_comparison=rules.cl_version_comparison.output,
        # PTE matrix composites (supersede individual pure_eb_pte_matrix and cosebis_pte_matrix)
        config_space_pte=rules.config_space_pte_matrices.output,
        harmonic_space_pte=rules.harmonic_space_pte_matrices.output,
        paper_macros=rules.paper_macros.output,
    output:
        html=f"{CLAIMS_DIR}/index.html",
    params:
        project_name="UNIONS B-modes",
        tagline="Spec-driven validation",
        config_dir=CONFIG_DIR,
        claims_dir=CLAIMS_DIR,
        skill_path=SKILL_PATH,
    shell:
        """
        python {params.skill_path}/scripts/generate_claims_dashboard.py \
            {output.html} \
            --project-name "{params.project_name}" \
            --tagline "{params.tagline}" \
            --specs-dir {params.config_dir} \
            --claims-dir {params.claims_dir} \
            --config-file {input.config} \
            --deps-file {input.deps}
        """


rule serve_claims:
    """Serve the claims dashboard."""
    input:
        html=f"{CLAIMS_DIR}/index.html",
    params:
        skill_path=SKILL_PATH,
        config_dir=CONFIG_DIR,
        claims_dir=CLAIMS_DIR,
        port_start=8000,
    run:
        import subprocess, sys, socket

        def find_open_port(start_port, max_attempts=100):
            for port in range(start_port, start_port + max_attempts):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(('', port))
                        return port
                    except OSError:
                        continue
            raise RuntimeError(f"No open port found in range {start_port}-{start_port + max_attempts}")

        port = find_open_port(params.port_start)
        print(f"Starting dashboard on port {port}")

        script_path = os.path.join(params.skill_path, "scripts", "claims_server.py")
        workflow_root = os.path.abspath(os.path.join(workflow.basedir, ".."))
        config_dir_abs = os.path.join(workflow_root, params.config_dir)
        claims_dir_abs = os.path.join(workflow_root, params.claims_dir)
        print(f"Config: {config_dir_abs}")
        print(f"Claims: {claims_dir_abs}")
        subprocess.run([sys.executable, script_path,
                        "--claims-dir", claims_dir_abs,
                        "--specs-dir", config_dir_abs,
                        "--port", str(port)], check=True)
