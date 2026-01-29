# workflow/rules/claims.smk
"""
Claims — testable assertions that produce evidence.
Claims depend on methods (for technique definitions) and compute outputs (for data).
"""

import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONFIG_DIR = "workflow/config"
CLAIMS_DIR = "results/claims"
COSMO_VAL_OUTPUT = "/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/output"
PAPER_FIGURES_DIR = "docs/unions_release/unions_bmodes/Figures"

BLINDS = ["A", "B", "C"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Path Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _covariance_path(version, min_sep, max_sep, nbins, blind=None, gaussian="g"):
    """Construct covariance file path.

    Covariance depends on survey properties (area, n_eff, sigma_e, cosmology, mask),
    which are identical for corrected and uncorrected ellipticities. So we always
    use the _leak_corr version's covariance file.
    """
    if blind is None:
        blind = config["fiducial"]["blind"]
    # Normalize to _leak_corr for covariance lookup (same survey properties)
    cov_version = version if "_leak_corr" in version else f"{version}_leak_corr"
    base_name = f"covariance_{cov_version}_{blind}_{gaussian}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_masked"
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


def _xi_reporting_path(version):
    """Path to reporting-scale 2PCF file."""
    return (
        f"{COSMO_VAL_OUTPUT}/{version}_xi_minsep={config['fiducial']['min_sep']}"
        f"_maxsep={config['fiducial']['max_sep']}"
        f"_nbins={config['fiducial']['nbins']}"
        f"_npatch={config['fiducial']['npatch']}.txt"
    )


def _xi_integration_path(version):
    """Path to fine-binned 2PCF integration file."""
    return (
        f"{COSMO_VAL_OUTPUT}/{version}_xi_minsep={config['fiducial']['min_sep_int']}"
        f"_maxsep={config['fiducial']['max_sep_int']}"
        f"_nbins={config['fiducial']['nbins_int']}"
        f"_npatch={config['fiducial']['npatch']}.txt"
    )


def _cov_integration_path(version, blind):
    """Covariance path for integration bins (Gaussian, for COSEBIS PTE)."""
    min_sep_int = config["fiducial"]["min_sep_int"]
    max_sep_int = config["fiducial"]["max_sep_int"]
    nbins_int = config["fiducial"]["nbins_int"]
    return _covariance_path(version, min_sep_int, max_sep_int, nbins_int, blind=blind, gaussian="g")


def _pte_scale_cut_pairs():
    """Return list of (i_min, i_max) index pairs for PTE matrix.

    Note: (9, 10) excluded due to numerical instability in cosmo_numba
    polynomial root finding with nmodes=20.
    """
    return [(i, j) for i in range(20) for j in range(i + 1, 21) if (i, j) != (9, 10)]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COSEBIS Claims
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rule cosebis_version_comparison:
    """B-mode visualization: COSEBIS B-modes across catalog versions.

    Main figure: leak_corr versions only (v1.4.5, v1.4.6, v1.4.8) — catalog evolution
    Second figure: v1.4.6 leak_corr vs uncorrected — correction impact comparison

    Plotting only - statistical PTEs are in config_space_pte_matrices.
    """
    input:
        specs=[
            f"{CONFIG_DIR}/cosebis_version_comparison.md",
            f"{CONFIG_DIR}/cosebis.md",
            f"{CONFIG_DIR}/1d_plots.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        xi_integration=[_xi_integration_path(ver) for ver in config["versions"]],
        cov_integration=[
            _covariance_path(ver, config["fiducial"]["min_sep_int"], config["fiducial"]["max_sep_int"], config["fiducial"]["nbins_int"], blind=config["fiducial"]["blind"])
            for ver in config["versions"]
        ],
    params:
        cov_base_dir="/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/data/covariance",
    output:
        evidence=f"{CLAIMS_DIR}/cosebis_version_comparison/evidence.json",
        figure_stacked=f"{CLAIMS_DIR}/cosebis_version_comparison/figure_stacked.png",
        figure_correction=f"{CLAIMS_DIR}/cosebis_version_comparison/figure_correction.png",
        paper_stacked=f"{PAPER_FIGURES_DIR}/cosebis_bmode_stacked.png",
        paper_correction=f"{PAPER_FIGURES_DIR}/cosebis_correction_comparison.png",
    script:
        "../scripts/cosebis_version_comparison.py"


rule cosebis_data_vector:
    """B-mode data vector: COSEBIS B-modes for fiducial version (v1.4.6).

    Two figures:
    - Main: leak_corr fiducial with fiducial and full angular ranges
    - Appendix: uncorrected version for correction comparison
    """
    input:
        specs=[
            f"{CONFIG_DIR}/cosebis_data_vector.md",
            f"{CONFIG_DIR}/cosebis.md",
            f"{CONFIG_DIR}/1d_plots.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        xi_integration=_xi_integration_path(config["fiducial"]["version"]),
        xi_integration_uncorr=_xi_integration_path("SP_v1.4.6"),
        cov_integration=_covariance_path(
            config["fiducial"]["version"],
            config["fiducial"]["min_sep_int"],
            config["fiducial"]["max_sep_int"],
            config["fiducial"]["nbins_int"],
            blind=config["fiducial"]["blind"],
        ),
        cov_integration_uncorr=_covariance_path(
            "SP_v1.4.6",
            config["fiducial"]["min_sep_int"],
            config["fiducial"]["max_sep_int"],
            config["fiducial"]["nbins_int"],
            blind=config["fiducial"]["blind"],
        ),
    params:
        cov_base_dir="/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/data/covariance",
    output:
        evidence=f"{CLAIMS_DIR}/cosebis_data_vector/evidence.json",
        figure=f"{CLAIMS_DIR}/cosebis_data_vector/figure.png",
        figure_uncorrected=f"{CLAIMS_DIR}/cosebis_data_vector/figure_uncorrected.png",
        paper_figure=f"{PAPER_FIGURES_DIR}/cosebis_data_vector.png",
        paper_figure_uncorrected=f"{PAPER_FIGURES_DIR}/cosebis_data_vector_uncorrected.png",
    script:
        "../scripts/cosebis_data_vector.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pure E/B Claims
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Number of parallel chunks for MC covariance estimation
N_PURE_EB_CHUNKS = 40


rule precompute_pure_eb_chunk:
    """Compute a chunk of MC samples for pure E/B covariance (scatter).

    Unified rule handles both base case (empty blind_suffix) and per-blind case (_A/_B/_C).
    The blind param extracts the blind letter or defaults to "A" for base case.
    """
    wildcard_constraints:
        version=r"[^_]+_v[\d.]+(_leak_corr)?",  # e.g. SP_v1.4.6 or SP_v1.4.6_leak_corr
        blind_suffix=r"|_[ABC]",  # empty OR _A/_B/_C
    input:
        cov_integration=lambda w: _cov_integration_path(w.version, w.blind_suffix[1] if w.blind_suffix else "A"),
        xi_reporting=lambda w: _xi_reporting_path(w.version),
        xi_integration=lambda w: _xi_integration_path(w.version),
    output:
        "results/paper_plots/intermediate/chunks/{version}{blind_suffix}_pure_eb_chunk_{chunk_id}.npz",
    params:
        version="{version}",
        blind=lambda w: w.blind_suffix[1] if w.blind_suffix else "A",
        chunk_id="{chunk_id}",
        n_chunks=N_PURE_EB_CHUNKS,
        min_sep=config["fiducial"]["min_sep"],
        max_sep=config["fiducial"]["max_sep"],
        nbins=config["fiducial"]["nbins"],
        min_sep_int=config["fiducial"]["min_sep_int"],
        max_sep_int=config["fiducial"]["max_sep_int"],
        nbins_int=config["fiducial"]["nbins_int"],
        npatch=config["fiducial"]["npatch"],
        n_samples=config["covariance"]["n_samples"],
    resources:
        mem_mb=8000,
    script:
        "../scripts/precompute_pure_eb_chunk.py"


rule precompute_pure_eb:
    """Gather MC sample chunks and compute final pure E/B covariance.

    Unified rule handles both base case (empty blind_suffix) and per-blind case (_A/_B/_C).
    For per-blind (B, C), loads data vectors from base file.
    For base or A, computes data vectors fresh.
    """
    wildcard_constraints:
        version=r"[^_]+_v[\d.]+(_leak_corr)?",  # e.g. SP_v1.4.6 or SP_v1.4.6_leak_corr
        blind_suffix=r"|_[ABC]",  # empty OR _A/_B/_C
    input:
        chunks=lambda w: expand(
            f"results/paper_plots/intermediate/chunks/{{version}}{w.blind_suffix}_pure_eb_chunk_{{chunk_id}}.npz",
            version=w.version,
            chunk_id=range(N_PURE_EB_CHUNKS),
        ),
        xi_reporting=lambda w: _xi_reporting_path(w.version),
        xi_integration=lambda w: _xi_integration_path(w.version),
        # For explicit blinds B/C, we need the base file to get data vectors
        base_pure_eb=lambda w: (
            f"results/paper_plots/intermediate/{w.version}_pure_eb_semianalytic.npz"
            if w.blind_suffix in ["_B", "_C"] else []
        ),
    output:
        "results/paper_plots/intermediate/{version}{blind_suffix}_pure_eb_semianalytic.npz",
    params:
        version="{version}",
        blind=lambda w: w.blind_suffix[1] if w.blind_suffix else "A",
        min_sep=config["fiducial"]["min_sep"],
        max_sep=config["fiducial"]["max_sep"],
        nbins=config["fiducial"]["nbins"],
        min_sep_int=config["fiducial"]["min_sep_int"],
        max_sep_int=config["fiducial"]["max_sep_int"],
        nbins_int=config["fiducial"]["nbins_int"],
        npatch=config["fiducial"]["npatch"],
    resources:
        mem_mb=8000,
        runtime=5,
    script:
        "../scripts/gather_pure_eb_chunks.py"


rule pure_eb_data_vector:
    """B-mode null test: Pure E/B data vector at fiducial scale cuts.

    Uses blinds specified in config (default: A).
    MC-propagated E/B covariances from the Gaussian integration covariances.
    """
    input:
        specs=[
            f"{CONFIG_DIR}/pure_eb_data_vector.md",
            f"{CONFIG_DIR}/pure_eb.md",
            f"{CONFIG_DIR}/1d_plots.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        # Base pure_eb file (data vectors are identical across blinds)
        pure_eb=f"results/paper_plots/intermediate/{config['fiducial']['version']}_pure_eb_semianalytic.npz",
        # Per-blind covariances
        cov=[
            _reporting_cov_path(config["fiducial"]["version"], blind)
            for blind in config["blinds"]
        ],
    output:
        evidence=f"{CLAIMS_DIR}/pure_eb_data_vector/evidence.json",
        paper_figure=f"{PAPER_FIGURES_DIR}/pure_eb_data_vector.png",
    script:
        "../scripts/pure_eb_data_vector.py"


rule pure_eb_version_comparison:
    """B-mode visualization: Pure E/B across catalog versions.

    Main figure: leak_corr versions only (v1.4.5, v1.4.6, v1.4.8) — catalog evolution
    Second figure: v1.4.6 leak_corr vs uncorrected — correction impact comparison

    Plotting only - statistical PTEs are in config_space_pte_matrices.
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
        figure_correction=f"{CLAIMS_DIR}/pure_eb_version_comparison/figure_correction.png",
        paper_figure=f"{PAPER_FIGURES_DIR}/pure_eb_versions.png",
        paper_figure_correction=f"{PAPER_FIGURES_DIR}/pure_eb_correction_comparison.png",
    script:
        "../scripts/pure_eb_version_comparison.py"


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
        pure_eb_data=f"results/paper_plots/intermediate/{config['fiducial']['version']}_pure_eb_semianalytic.npz",
    output:
        evidence=f"{CLAIMS_DIR}/pure_eb_covariance/evidence.json",
        figure=f"{CLAIMS_DIR}/pure_eb_covariance/figure.png",
        paper_figure=f"{PAPER_FIGURES_DIR}/eb_covariance.png",
    script:
        "../scripts/pure_eb_covariance.py"


rule calculate_pure_eb_ptes:
    """Calculate PTE matrices for Pure E/B mode scale cut robustness.

    Per-blind: Uses blind-specific integration covariance for PTE calculation.
    The pure_eb_data vectors are identical across blinds; only covariance differs.
    """
    input:
        pure_eb_data="results/paper_plots/intermediate/{version}_pure_eb_semianalytic.npz",
        cov_integration=lambda w: _cov_integration_path(w.version, w.blind),
    output:
        "results/paper_plots/intermediate/{version}_{blind}_pure_eb_ptes.npz",
    wildcard_constraints:
        blind=r"[ABC]",
    params:
        version="{version}",
        npatch=config["fiducial"]["npatch"],
        n_samples=config["covariance"]["n_samples"],
    resources:
        mem_mb=16000,
        runtime=30,
    script:
        "../scripts/calculate_pure_eb_ptes.py"


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
        pseudo_cl=f"{COSMO_VAL_OUTPUT}/pseudo_cl_{config['fiducial']['version']}_blind={config['fiducial']['blind']}_powspace_nbins=32.fits",
        pseudo_cl_cov=f"{COSMO_VAL_OUTPUT}/pseudo_cl_cov_{config['fiducial']['version']}_blind={config['fiducial']['blind']}_powspace_nbins=32.fits",
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
            f"{COSMO_VAL_OUTPUT}/pseudo_cl_{ver}_blind={config['fiducial']['blind']}_powspace_nbins=32.fits"
            for ver in config["versions"]
        ],
        pseudo_cl_cov=[
            f"{COSMO_VAL_OUTPUT}/pseudo_cl_cov_{ver}_blind={config['fiducial']['blind']}_powspace_nbins=32.fits"
            for ver in config["versions"]
        ],
    output:
        evidence=f"{CLAIMS_DIR}/cl_version_comparison/evidence.json",
        figure=f"{CLAIMS_DIR}/cl_version_comparison/figure.png",
        paper_figure=f"{PAPER_FIGURES_DIR}/cl_versions.png",
    script:
        "../scripts/cl_version_comparison.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COSEBIS PTE Matrix (scatter-gather pattern)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
    resources:
        mem_mb=8000,
        runtime=30,
    script:
        "../scripts/compute_cosebis_pte_single.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PTE Matrix Composites (Results + Appendix)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rule config_space_pte_matrices:
    """Configuration-space PTE composites for B-modes paper.

    Main text: 1x3 composite for fiducial version (xi+^B, xi-^B, COSEBIS B_n)
    Appendix: 3x3 composite for all versions (3 rows x 3 statistics)

    Uses blind A covariance as fiducial. BB covariances are theoretically blind-independent;
    see cosmology_for_covariance.md wiki for investigation details.
    """
    input:
        specs=[
            f"{CONFIG_DIR}/config_space_pte_matrices.md",
            f"{CONFIG_DIR}/pure_eb.md",
            f"{CONFIG_DIR}/cosebis.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        # Claim dependencies
        pure_eb_data_vector=f"{CLAIMS_DIR}/pure_eb_data_vector/evidence.json",
        cosebis_data_vector=f"{CLAIMS_DIR}/cosebis_data_vector/evidence.json",
        # Data inputs (BB covariances are blind-independent)
        pure_eb_pte=[
            f"results/paper_plots/intermediate/{ver}_{config['fiducial']['blind']}_pure_eb_ptes.npz"
            for ver in config["versions"]
        ],
        cosebis_pte_files=[
            f"{CLAIMS_DIR}/cosebis_pte_matrix/pte_values/{ver}/{config['fiducial']['blind']}/pte_{i:03d}_{j:03d}.json"
            for ver in config["versions"]
            for i, j in _pte_scale_cut_pairs()
        ],
    output:
        evidence=f"{CLAIMS_DIR}/config_space_pte_matrices/evidence.json",
        figure_fiducial=f"{CLAIMS_DIR}/config_space_pte_matrices/figure_fiducial.png",
        figure_appendix=f"{CLAIMS_DIR}/config_space_pte_matrices/figure_appendix.png",
        figure_appendix_uncorrected=f"{CLAIMS_DIR}/config_space_pte_matrices/figure_appendix_uncorrected.png",
        paper_figure_fiducial=f"{PAPER_FIGURES_DIR}/config_space_pte_fiducial.png",
        paper_figure_appendix=f"{PAPER_FIGURES_DIR}/config_space_pte_composite_appendix.png",
        paper_figure_appendix_uncorrected=f"{PAPER_FIGURES_DIR}/config_space_pte_composite_appendix_uncorrected.png",
    script:
        "../scripts/config_space_pte_matrices.py"


rule harmonic_space_pte_matrices:
    """Harmonic-space PTE figures for all versions.

    Results: Single-panel Cl^BB PTE matrix for fiducial v1.4.6
    Appendix: 3-panel composites for leak_corr and uncorrected versions

    Uses fiducial blind covariance from config.
    """
    input:
        specs=[
            f"{CONFIG_DIR}/harmonic_space_pte_matrices.md",
            f"{CONFIG_DIR}/cl.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        pseudo_cl=[
            f"{COSMO_VAL_OUTPUT}/pseudo_cl_{ver}_blind={config['fiducial']['blind']}_powspace_nbins=32.fits"
            for ver in config["versions"]
        ],
        pseudo_cl_cov=[
            f"{COSMO_VAL_OUTPUT}/pseudo_cl_cov_{ver}_blind={config['fiducial']['blind']}_powspace_nbins=32.fits"
            for ver in config["versions"]
        ],
    output:
        evidence=f"{CLAIMS_DIR}/harmonic_space_pte_matrices/evidence.json",
        figure_fiducial=f"{CLAIMS_DIR}/harmonic_space_pte_matrices/figure_fiducial.png",
        figure_appendix=f"{CLAIMS_DIR}/harmonic_space_pte_matrices/figure_appendix.png",
        figure_appendix_uncorrected=f"{CLAIMS_DIR}/harmonic_space_pte_matrices/figure_appendix_uncorrected.png",
        paper_figure_fiducial=f"{PAPER_FIGURES_DIR}/cl_pte_heatmap.png",
        paper_figure_appendix=f"{PAPER_FIGURES_DIR}/cl_pte_composite_appendix.png",
        paper_figure_appendix_uncorrected=f"{PAPER_FIGURES_DIR}/cl_pte_composite_appendix_uncorrected.png",
    script:
        "../scripts/harmonic_space_pte_matrices.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BB Covariance Blind Independence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rule bb_covariance_blind_independence:
    """Test BB covariance blind-independence vs EE variation.

    BB covariances should be stable across blinds (null signal → no sample variance).
    EE covariances should vary (~10%) due to sample variance from cosmological signal.

    Covers all three analysis spaces: Pure E/B, COSEBIS, and harmonic (pseudo-Cl).
    """
    input:
        specs=[
            f"{CONFIG_DIR}/bb_covariance_blind_independence.md",
            f"{CONFIG_DIR}/covariance.md",
            f"{CONFIG_DIR}/pure_eb.md",
            f"{CONFIG_DIR}/cosebis.md",
            f"{CONFIG_DIR}/cl.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        # Per-blind MC-propagated pure E/B covariances
        # Base file (no suffix) uses blind=A, so reuse it for A
        pure_eb_A=f"results/paper_plots/intermediate/{config['fiducial']['version']}_pure_eb_semianalytic.npz",
        pure_eb_B=f"results/paper_plots/intermediate/{config['fiducial']['version']}_B_pure_eb_semianalytic.npz",
        pure_eb_C=f"results/paper_plots/intermediate/{config['fiducial']['version']}_C_pure_eb_semianalytic.npz",
        # COSEBIS: xi integration file (shared) + per-blind config-space covariances
        xi_integration=_xi_integration_path(config["fiducial"]["version"]),
        cov_integration_A=_cov_integration_path(config["fiducial"]["version"], "A"),
        cov_integration_B=_cov_integration_path(config["fiducial"]["version"], "B"),
        cov_integration_C=_cov_integration_path(config["fiducial"]["version"], "C"),
        # Per-blind harmonic covariances
        harmonic_A=f"{COSMO_VAL_OUTPUT}/pseudo_cl_cov_{config['fiducial']['version']}_blind=A_nellbins=32.fits",
        harmonic_B=f"{COSMO_VAL_OUTPUT}/pseudo_cl_cov_{config['fiducial']['version']}_blind=B_nellbins=32.fits",
        harmonic_C=f"{COSMO_VAL_OUTPUT}/pseudo_cl_cov_{config['fiducial']['version']}_blind=C_nellbins=32.fits",
    params:
        nmodes=config["fiducial"]["nmodes"],
        theta_min=config["cosebis"]["theta_min"],
        theta_max=config["cosebis"]["theta_max"],
    output:
        evidence=f"{CLAIMS_DIR}/bb_covariance_blind_independence/evidence.json",
        figure=f"{CLAIMS_DIR}/bb_covariance_blind_independence/figure.png",
    script:
        "../scripts/bb_covariance_blind_independence.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Local Rules Declaration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

localrules: cl_data_vector, cl_version_comparison, pure_eb_covariance, pure_eb_version_comparison, config_space_pte_matrices, harmonic_space_pte_matrices, bb_covariance_blind_independence
