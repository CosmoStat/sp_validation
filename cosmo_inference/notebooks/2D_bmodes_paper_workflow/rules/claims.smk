# workflow/rules/claims.smk
"""
Claims — testable assertions that produce evidence.
Claims depend on methods (for technique definitions) and compute outputs (for data).
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CONFIG_DIR, CLAIMS_DIR, PAPER_FIGURES_DIR, BLINDS, FIDUCIAL defined in Snakefile
# COSMO_VAL, COSMO_INFERENCE, covariance_path() defined in Snakefile
COSMO_VAL_OUTPUT = str(COSMO_VAL)  # String version for f-string interpolation

# Fiducial binning parameters — used by multiple pure E/B rules
# Avoids repeating FIDUCIAL[key] in each rule's params block
FIDUCIAL_BINNING = {
    "min_sep": FIDUCIAL["min_sep"],
    "max_sep": FIDUCIAL["max_sep"],
    "nbins": FIDUCIAL["nbins"],
    "min_sep_int": FIDUCIAL["min_sep_int"],
    "max_sep_int": FIDUCIAL["max_sep_int"],
    "nbins_int": FIDUCIAL["nbins_int"],
    "npatch": FIDUCIAL["npatch"],
}

# VERSION_LABELS from config for passing to plotting scripts
VERSION_LABELS = config["plotting"].get("version_labels", {})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Path Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _covariance_path(version, min_sep, max_sep, nbins, blind=None, gaussian="g"):
    """Construct covariance file path using centralized covariance_path() from Snakefile.

    TODO(generate-v1-4-10-1-covariance-55144852): v1.4.10.1 uses v1.4.6 covariance
    as workaround until proper covariance is generated. Same footprint justifies this.
    """
    if blind is None:
        blind = FIDUCIAL["blind"]
    return covariance_path(version, blind, gaussian=gaussian, min_sep=min_sep, max_sep=max_sep, nbins=nbins)


def _reporting_cov_path(version, blind):
    """Path to reporting-scale covariance (non-Gaussian, masked)."""
    return covariance_path(version, blind, gaussian="ng")


def _xi_reporting_path(version):
    """Path to reporting-scale 2PCF file."""
    return (
        f"{COSMO_VAL_OUTPUT}/{version}_xi_minsep={FIDUCIAL['min_sep']}"
        f"_maxsep={FIDUCIAL['max_sep']}_nbins={FIDUCIAL['nbins']}_npatch={FIDUCIAL['npatch']}.txt"
    )


def _xi_integration_path(version):
    """Path to fine-binned 2PCF integration file."""
    return (
        f"{COSMO_VAL_OUTPUT}/{version}_xi_minsep={FIDUCIAL['min_sep_int']}"
        f"_maxsep={FIDUCIAL['max_sep_int']}_nbins={FIDUCIAL['nbins_int']}_npatch={FIDUCIAL['npatch']}.txt"
    )


def _cov_integration_path(version, blind):
    """Covariance path for integration bins (Gaussian, for COSEBIS PTE)."""
    return covariance_path(
        version, blind, gaussian="g",
        min_sep=FIDUCIAL["min_sep_int"], max_sep=FIDUCIAL["max_sep_int"], nbins=FIDUCIAL["nbins_int"]
    )


def _pte_scale_cut_pairs():
    """Return list of (i_min, i_max) index pairs for PTE matrix.

    Note: (9, 10) excluded due to numerical instability in cosmo_numba
    polynomial root finding with nmodes=20.
    """
    return [(i, j) for i in range(20) for j in range(i + 1, 21) if (i, j) != (9, 10)]


# Pre-compute PTE scale cut pairs (called multiple times in rule inputs)
PTE_SCALE_CUT_PAIRS = _pte_scale_cut_pairs()


def _pseudo_cl_path(version, blind="A", nbins=32):
    """Return pseudo-Cl path for a catalog version.

    All leak-corrected versions use consistent local naming with blind and binning.
    """
    return f"{COSMO_VAL_OUTPUT}/pseudo_cl_{version}_blind={blind}_powspace_nbins={nbins}.fits"


def _pseudo_cl_cov_path(version, blind="A", nbins=32):
    """Return pseudo-Cl covariance path for a catalog version."""
    return f"{COSMO_VAL_OUTPUT}/pseudo_cl_cov_{version}_blind={blind}_nellbins={nbins}.fits"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COSEBIS Claims
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
        cov_integration=[_cov_integration_path(ver, "A") for ver in config["versions"]],
    params:
        cov_base_dir=str(COSMO_INFERENCE / "data/covariance"),
        version_labels=VERSION_LABELS,
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
        xi_integration=_xi_integration_path(FIDUCIAL["version"]),
        cov_integration=_cov_integration_path(FIDUCIAL["version"], "A"),
    params:
        cov_base_dir=str(COSMO_INFERENCE / "data/covariance"),
    output:
        evidence=f"{CLAIMS_DIR}/cosebis_data_vector/evidence.json",
        figure=f"{CLAIMS_DIR}/cosebis_data_vector/figure.png",
        paper_figure=f"{PAPER_FIGURES_DIR}/cosebis_data_vector.png",
    script:
        "../scripts/cosebis_data_vector.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pure E/B Claims
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Number of parallel chunks for MC covariance estimation
N_PURE_EB_CHUNKS = config["pure_eb"]["n_chunks"]


rule precompute_pure_eb_chunk:
    """Compute a chunk of MC samples for pure E/B covariance (scatter)."""
    input:
        cov_integration=lambda w: _cov_integration_path(w.version, "A"),
        xi_reporting=lambda w: _xi_reporting_path(w.version),
        xi_integration=lambda w: _xi_integration_path(w.version),
    output:
        "results/paper_plots/intermediate/chunks/{version}_pure_eb_chunk_{chunk_id}.npz",
    params:
        version="{version}",
        chunk_id="{chunk_id}",
        n_chunks=N_PURE_EB_CHUNKS,
        n_samples=config["covariance"]["n_samples"],
        **FIDUCIAL_BINNING,
    resources:
        mem_mb=8000,
    script:
        "../scripts/precompute_pure_eb_chunk.py"


rule precompute_pure_eb:
    """Gather MC sample chunks and compute final pure E/B covariance."""
    wildcard_constraints:
        version=r"[^_]+_v[\d.]+_leak_corr",  # e.g. SP_v1.4.6_leak_corr (no blind suffix)
    input:
        chunks=expand(
            "results/paper_plots/intermediate/chunks/{{version}}_pure_eb_chunk_{chunk_id}.npz",
            chunk_id=range(N_PURE_EB_CHUNKS),
        ),
        xi_reporting=lambda w: _xi_reporting_path(w.version),
        xi_integration=lambda w: _xi_integration_path(w.version),
    output:
        "results/paper_plots/intermediate/{version}_pure_eb_semianalytic.npz",
    params:
        version="{version}",
        **FIDUCIAL_BINNING,
    resources:
        mem_mb=8000,
        runtime=5,
    script:
        "../scripts/gather_pure_eb_chunks.py"


rule precompute_pure_eb_blind:
    """Precompute pure E/B decomposition with per-blind covariance."""
    input:
        cov_integration=lambda w: _cov_integration_path(config["fiducial"]["version"], w.blind),
    output:
        f"results/paper_plots/intermediate/{config['fiducial']['version']}_{{blind}}_pure_eb_semianalytic.npz",
    params:
        version=config["fiducial"]["version"],
        n_samples=config["covariance"]["n_samples"],
        **FIDUCIAL_BINNING,
    resources:
        mem_mb=32000,
        runtime=60,
    threads: 16  # Reduced from 48 to avoid libgomp thread creation failures
    script:
        "../scripts/precompute_pure_eb_covariance.py"


rule pure_eb_data_vector:
    """B-mode null test: Pure E/B data vector at fiducial scale cuts.

    Uses fiducial blind only (config.fiducial.blind) for PTE calculation.
    """
    input:
        specs=[
            f"{CONFIG_DIR}/pure_eb_data_vector.md",
            f"{CONFIG_DIR}/pure_eb.md",
            f"{CONFIG_DIR}/1d_plots.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        pure_eb=f"results/paper_plots/intermediate/{config['fiducial']['version']}_{config['fiducial']['blind']}_pure_eb_semianalytic.npz",
        cov=_reporting_cov_path(config["fiducial"]["version"], config["fiducial"]["blind"]),
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
    params:
        version_labels=VERSION_LABELS,
    output:
        evidence=f"{CLAIMS_DIR}/pure_eb_version_comparison/evidence.json",
        figure=f"{CLAIMS_DIR}/pure_eb_version_comparison/figure.png",
        paper_figure=f"{PAPER_FIGURES_DIR}/pure_eb_versions.png",
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
        pure_eb_data=f"results/paper_plots/intermediate/{FIDUCIAL['version']}_A_pure_eb_semianalytic.npz",
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
# Harmonic-Space Claims (Cl)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rule cl_data_vector:
    """Harmonic-space B-mode power spectra at fiducial ell range."""
    input:
        specs=[
            f"{CONFIG_DIR}/cl_data_vector.md",
            f"{CONFIG_DIR}/cl.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        pseudo_cl=_pseudo_cl_path(FIDUCIAL['version']),
        pseudo_cl_cov=_pseudo_cl_cov_path(FIDUCIAL['version']),
    output:
        evidence=f"{CLAIMS_DIR}/cl_data_vector/evidence.json",
        figure=f"{CLAIMS_DIR}/cl_data_vector/figure.png",
        paper_figure=f"{PAPER_FIGURES_DIR}/cl_data_vector.png",
    script:
        "../scripts/cl_data_vector.py"


rule cl_version_comparison:
    """C_ell^BB version comparison across catalog versions."""
    input:
        specs=[
            f"{CONFIG_DIR}/cl_version_comparison.md",
            f"{CONFIG_DIR}/cl.md",
            f"{CONFIG_DIR}/cl_data_vector.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        cl_data_vector_evidence=rules.cl_data_vector.output.evidence,
        pseudo_cl=[_pseudo_cl_path(ver) for ver in config["versions"]],
        pseudo_cl_cov=[_pseudo_cl_cov_path(ver) for ver in config["versions"]],
    params:
        version_labels=VERSION_LABELS,
        ell_min_cut=config["cl"]["fiducial_ell_min"],
        ell_max_cut=config["cl"]["fiducial_ell_max"],
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
        nmodes=FIDUCIAL["nmodes"],
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
            for i, j in PTE_SCALE_CUT_PAIRS
        ],
    params:
        blinds=BLINDS,
    output:
        evidence=f"{CLAIMS_DIR}/cosebis_pte_matrix/evidence.json",
        figures=expand(
            f"{CLAIMS_DIR}/cosebis_pte_matrix/figure_{{version}}_{{blind}}_n{{nmodes}}.png",
            version=config["versions"],
            blind=BLINDS,
            nmodes=config["cosebis"]["mode_subsets"],
        ),
        paper_figures=expand(
            f"{PAPER_FIGURES_DIR}/cosebis_pte_matrices_{{version}}_{{blind}}_n{{nmodes}}.png",
            version=config["versions"],
            blind=BLINDS,
            nmodes=config["cosebis"]["mode_subsets"],
        ),
    script:
        "../scripts/cosebis_pte_matrix.py"


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
        # Claim dependencies
        pure_eb_data_vector=f"{CLAIMS_DIR}/pure_eb_data_vector/evidence.json",
        cosebis_data_vector=f"{CLAIMS_DIR}/cosebis_data_vector/evidence.json",
        # Data inputs (fiducial blind only)
        pure_eb_pte=[
            f"results/paper_plots/intermediate/{ver}_{config['fiducial']['blind']}_pure_eb_ptes.npz"
            for ver in config["versions"]
        ],
        cosebis_pte_files=[
            f"{CLAIMS_DIR}/cosebis_pte_matrix/pte_values/{ver}/{config['fiducial']['blind']}/pte_{i:03d}_{j:03d}.json"
            for ver in config["versions"]
            for i, j in PTE_SCALE_CUT_PAIRS
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

    Results: Single-panel Cl^BB PTE matrix for fiducial version
    Appendix: N-panel composite for all versions from config.versions

    Uses fiducial blind covariance (blind independence validated in bb_covariance_blind_independence).
    """
    input:
        specs=[
            f"{CONFIG_DIR}/harmonic_space_pte_matrices.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        pseudo_cl=[_pseudo_cl_path(ver) for ver in config["versions"]],
        pseudo_cl_cov=[
            _pseudo_cl_cov_path(ver, blind=config["fiducial"]["blind"])
            for ver in config["versions"]
        ],
    params:
        version_labels=VERSION_LABELS,
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
    Uses blind=A fine-binned pseudo-Cl with Planck18 cosmology covariance.
    """
    input:
        spec=f"{CONFIG_DIR}/harmonic_config_cosebis_comparison.md",
        config=f"{CONFIG_DIR}/config.yaml",
        # Fine-binned pseudo-Cl for accurate C_ℓ → COSEBIS conversion
        pseudo_cls=f"{COSMO_VAL_OUTPUT}/pseudo_cl_{config['fiducial']['version']}_blind=A_ellstep=1.fits",
        pseudo_cls_cov=f"{COSMO_VAL_OUTPUT}/pseudo_cl_cov_{config['fiducial']['version']}_blind=A_ellstep=1.fits",
        cov_integration=_covariance_path(
            config["fiducial"]["version"],
            config["fiducial"]["min_sep_int"],
            config["fiducial"]["max_sep_int"],
            config["fiducial"]["nbins_int"],
            blind="A",
        ),
    output:
        comparison=f"{CLAIMS_DIR}/harmonic_config_cosebis_comparison/figure.png",
        stats=f"{CLAIMS_DIR}/harmonic_config_cosebis_comparison/stats.txt",
    params:
        version=config["fiducial"]["version"],
        nmodes_long=config["cosebis"]["nmodes"],
        nmodes_short=config["cosebis"]["mode_subsets"][0],
        theta_min=config["cosebis"]["theta_min"],
        theta_max=config["cosebis"]["theta_max"],
        min_sep_int=config["fiducial"]["min_sep_int"],
        max_sep_int=config["fiducial"]["max_sep_int"],
        nbins_int=config["fiducial"]["nbins_int"],
        npatch=config["fiducial"]["npatch"],
        # ℓ cuts for harmonic-space (avoid noisy low-ℓ modes in fine binning)
        ell_min=config["cl"]["ell_min"],
        ell_max=config["cl"]["ell_max"],
    script:
        "../scripts/compare_harmonic_config_cosebis.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Local Rules Declaration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

localrules: cl_data_vector, cl_version_comparison, pure_eb_pte_matrix, pure_eb_covariance, pure_eb_version_comparison, cosebis_pte_matrix, cosebis_version_comparison, cosebis_data_vector, config_space_pte_matrices, harmonic_space_pte_matrices, harmonic_config_cosebis_comparison
