# workflow/rules/claims.smk
"""
Claims — testable assertions that produce evidence.
Claims depend on methods (for technique definitions) and compute outputs (for data).
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CONFIG_DIR, CLAIMS_DIR, PAPER_FIGURES_DIR, BLINDS, FIDUCIAL, PLANCK18 defined in Snakefile
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

# Shorthand for fiducial and mock version strings (used in path construction)
FIDUCIAL_VERSION = FIDUCIAL["version"]
MOCK_VERSION = f"{FIDUCIAL['mock_version']}_leak_corr"

# Filter versions for different analysis types
# Pure E/B and PTEs only apply to leak-corrected versions
VERSIONS_LEAK_CORR = [v for v in config["versions"] if "_leak_corr" in v and "_ecut" not in v]

# All versions needed for per-version data vector plots (both leak-corrected and uncorrected)
VERSIONS_ALL_FOR_PLOTS = VERSIONS_LEAK_CORR + [v.replace("_leak_corr", "") for v in VERSIONS_LEAK_CORR]


def _extract_version_number(version_string):
    """Extract short version number from full version string for filenames."""
    import re
    match = re.search(r'(v[\d.]+)', version_string)
    return match.group(1) if match else version_string


def _per_version_figure_outputs(claim_dir):
    """Generate output dict for 9 per-version figures.

    Returns dict mapping output keys to paths for all 9 figures:
    - figure.png (paper figure)
    - figure_v{X.Y.Z}.png for each leak-corrected version
    - figure_v{X.Y.Z}_uncorrected.png for each uncorrected version
    """
    outputs = {"figure": f"{claim_dir}/figure.png"}
    for ver_lc in sorted(VERSION_LABELS.keys(), key=lambda v: -len(v)):
        ver_num = _extract_version_number(ver_lc)
        outputs[f"figure_{ver_num.replace('.', '_')}"] = f"{claim_dir}/figure_{ver_num}.png"
        outputs[f"figure_{ver_num.replace('.', '_')}_uncorrected"] = f"{claim_dir}/figure_{ver_num}_uncorrected.png"
    return outputs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Path Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
    return f"{COSMO_VAL_OUTPUT}/pseudo_cl_cov_{version}_blind={blind}_powspace_nbins={nbins}.fits"


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
        # COSEBIs only for leak-corrected versions
        xi_integration=[_xi_integration_path(ver) for ver in VERSIONS_LEAK_CORR],
        cov_integration=[_cov_integration_path(ver, "A") for ver in VERSIONS_LEAK_CORR],
    params:
        version_labels=VERSION_LABELS,
        versions=VERSIONS_LEAK_CORR,
    output:
        evidence=f"{CLAIMS_DIR}/cosebis_version_comparison/evidence.json",
        figure_stacked=f"{CLAIMS_DIR}/cosebis_version_comparison/figure_stacked.png",
        paper_stacked=f"{PAPER_FIGURES_DIR}/cosebis_bmode_stacked.pdf",
    script:
        "../scripts/cosebis_version_comparison.py"


rule cosebis_data_vector:
    """B-mode data vector: COSEBIS B-modes for all catalog versions.

    Single-panel figure combining fiducial and full angular ranges.
    Paper figure for main text. PTEs are in cosebis_pte_matrix.

    Produces 9 figures:
    - figure.png: fiducial version, leak-corrected, no title (paper)
    - figure_v{X.Y.Z}.png: each version, leak-corrected, with title
    - figure_v{X.Y.Z}_uncorrected.png: each version, uncorrected, with title
    """
    input:
        specs=[
            f"{CONFIG_DIR}/cosebis_data_vector.md",
            f"{CONFIG_DIR}/cosebis.md",
            f"{CONFIG_DIR}/1d_plots.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        # Per-version inputs: xi_{version} and cov_{version} for all versions
        **{f"xi_{ver}": _xi_integration_path(ver) for ver in VERSIONS_ALL_FOR_PLOTS},
        **{f"cov_{ver}": _cov_integration_path(ver, "A") for ver in VERSIONS_ALL_FOR_PLOTS},
    params:
        cov_base_dir=str(COSMO_INFERENCE / "data/covariance"),
    output:
        evidence=f"{CLAIMS_DIR}/cosebis_data_vector/evidence.json",
        paper_figure=f"{PAPER_FIGURES_DIR}/cosebis_data_vector.pdf",
        **_per_version_figure_outputs(f"{CLAIMS_DIR}/cosebis_data_vector"),
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
        cov_integration=lambda w: _cov_integration_path(w.version, w.blind),
        xi_reporting=lambda w: _xi_reporting_path(w.version),
        xi_integration=lambda w: _xi_integration_path(w.version),
    output:
        "results/paper_plots/intermediate/chunks/{version}_{blind}_pure_eb_chunk_{chunk_id}.npz",
    params:
        version="{version}",
        blind="{blind}",
        chunk_id="{chunk_id}",
        n_chunks=N_PURE_EB_CHUNKS,
        n_samples=config["covariance"]["n_samples"],
        cosmo_params=PLANCK18,
        **FIDUCIAL_BINNING,
    resources:
        mem_mb=8000,
    script:
        "../scripts/precompute_pure_eb_chunk.py"


rule precompute_pure_eb:
    """Gather MC sample chunks and compute final pure E/B covariance."""
    wildcard_constraints:
        version=r"[^_]+_v[\d.]+(_ecut\d+)?(_leak_corr)?",  # e.g. SP_v1.4.6, SP_v1.4.6_ecut07_leak_corr
        blind=r"[ABC]",
    input:
        chunks=expand(
            "results/paper_plots/intermediate/chunks/{{version}}_{{blind}}_pure_eb_chunk_{chunk_id}.npz",
            chunk_id=range(N_PURE_EB_CHUNKS),
        ),
        xi_reporting=lambda w: _xi_reporting_path(w.version),
        xi_integration=lambda w: _xi_integration_path(w.version),
    output:
        "results/paper_plots/intermediate/{version}_{blind}_pure_eb_semianalytic.npz",
    params:
        version="{version}",
        **FIDUCIAL_BINNING,
    resources:
        mem_mb=8000,
        runtime=5,
    script:
        "../scripts/gather_pure_eb_chunks.py"


rule pure_eb_data_vector:
    """B-mode null test: Pure E/B data vector at fiducial scale cuts.

    Uses fiducial blind only (FIDUCIAL["blind"]) for PTE calculation.

    Produces 9 figures:
    - figure.png: fiducial version, leak-corrected, no title (paper)
    - figure_v{X.Y.Z}.png: each version, leak-corrected, with title
    - figure_v{X.Y.Z}_uncorrected.png: each version, uncorrected, with title
    """
    input:
        specs=[
            f"{CONFIG_DIR}/pure_eb_data_vector.md",
            f"{CONFIG_DIR}/pure_eb.md",
            f"{CONFIG_DIR}/1d_plots.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        # Per-version inputs: pure_eb_{version} and cov_{version} for all versions
        **{f"pure_eb_{ver}": f"results/paper_plots/intermediate/{ver}_{FIDUCIAL['blind']}_pure_eb_semianalytic.npz"
           for ver in VERSIONS_ALL_FOR_PLOTS},
        **{f"cov_{ver}": _reporting_cov_path(ver, FIDUCIAL["blind"])
           for ver in VERSIONS_ALL_FOR_PLOTS},
    output:
        evidence=f"{CLAIMS_DIR}/pure_eb_data_vector/evidence.json",
        paper_figure=f"{PAPER_FIGURES_DIR}/pure_eb_data_vector.pdf",
        **_per_version_figure_outputs(f"{CLAIMS_DIR}/pure_eb_data_vector"),
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
        # Pure E/B only for leak-corrected versions
        pure_eb_data=[
            f"results/paper_plots/intermediate/{ver}_A_pure_eb_semianalytic.npz"
            for ver in VERSIONS_LEAK_CORR
        ],
    params:
        version_labels=VERSION_LABELS,
        versions=VERSIONS_LEAK_CORR,
    output:
        evidence=f"{CLAIMS_DIR}/pure_eb_version_comparison/evidence.json",
        figure=f"{CLAIMS_DIR}/pure_eb_version_comparison/figure.png",
        paper_figure=f"{PAPER_FIGURES_DIR}/pure_eb_versions.pdf",
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
        pure_eb_data=f"results/paper_plots/intermediate/{FIDUCIAL_VERSION}_A_pure_eb_semianalytic.npz",
    output:
        evidence=f"{CLAIMS_DIR}/pure_eb_covariance/evidence.json",
        figure=f"{CLAIMS_DIR}/pure_eb_covariance/figure.png",
        paper_figure=f"{PAPER_FIGURES_DIR}/eb_covariance.pdf",
    script:
        "../scripts/pure_eb_covariance.py"


rule calculate_pure_eb_ptes:
    """Calculate PTE matrices for Pure E/B mode scale cut robustness.

    Per-blind: Uses blind-specific integration covariance for PTE calculation.
    The pure_eb_data vectors are identical across blinds; only covariance differs.

    In practice, BB covariance is blind-independent (validated by
    bb_covariance_blind_independence), so downstream consumers (config_space_pte_matrices)
    only request blind A. The per-blind wildcard is retained for the blind independence test.
    """
    input:
        pure_eb_data="results/paper_plots/intermediate/{version}_A_pure_eb_semianalytic.npz",
        cov_integration=lambda w: _cov_integration_path(w.version, w.blind),
    output:
        "results/paper_plots/intermediate/{version}_{blind}_pure_eb_ptes.npz",
    wildcard_constraints:
        blind=r"[ABC]",
    params:
        version="{version}",
        npatch=FIDUCIAL["npatch"],
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
    """Harmonic-space B-mode power spectra for all catalog versions.

    Produces 9 figures:
    - figure.png: fiducial version, leak-corrected, no title (paper)
    - figure_v{X.Y.Z}.png: each version, leak-corrected, with title
    - figure_v{X.Y.Z}_uncorrected.png: each version, uncorrected, with title
    """
    input:
        specs=[
            f"{CONFIG_DIR}/cl_data_vector.md",
            f"{CONFIG_DIR}/cl.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        # Per-version inputs: pseudo_cl_{version} and pseudo_cl_cov_{version} for all versions
        **{f"pseudo_cl_{ver}": _pseudo_cl_path(ver) for ver in VERSIONS_ALL_FOR_PLOTS},
        **{f"pseudo_cl_cov_{ver}": _pseudo_cl_cov_path(ver) for ver in VERSIONS_ALL_FOR_PLOTS},
    params:
        ell_min_cut=config["cl"]["fiducial_ell_min"],
        ell_max_cut=config["cl"]["fiducial_ell_max"],
    output:
        evidence=f"{CLAIMS_DIR}/cl_data_vector/evidence.json",
        paper_figure=f"{PAPER_FIGURES_DIR}/cl_data_vector.pdf",
        **_per_version_figure_outputs(f"{CLAIMS_DIR}/cl_data_vector"),
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
        # Cl version comparison only for leak-corrected versions
        pseudo_cl=[_pseudo_cl_path(ver) for ver in VERSIONS_LEAK_CORR],
        pseudo_cl_cov=[_pseudo_cl_cov_path(ver) for ver in VERSIONS_LEAK_CORR],
    params:
        version_labels=VERSION_LABELS,
        versions=VERSIONS_LEAK_CORR,
        ell_min_cut=config["cl"]["fiducial_ell_min"],
        ell_max_cut=config["cl"]["fiducial_ell_max"],
    output:
        evidence=f"{CLAIMS_DIR}/cl_version_comparison/evidence.json",
        figure=f"{CLAIMS_DIR}/cl_version_comparison/figure.png",
        paper_figure=f"{PAPER_FIGURES_DIR}/cl_versions.pdf",
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
            f"{CONFIG_DIR}/2d_plots.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        # Claim dependencies
        pure_eb_data_vector=f"{CLAIMS_DIR}/pure_eb_data_vector/evidence.json",
        cosebis_data_vector=f"{CLAIMS_DIR}/cosebis_data_vector/evidence.json",
        # Data inputs (fiducial blind only)
        # Pure E/B and COSEBIs PTEs only for leak-corrected versions
        pure_eb_pte=[
            f"results/paper_plots/intermediate/{ver}_{FIDUCIAL['blind']}_pure_eb_ptes.npz"
            for ver in VERSIONS_LEAK_CORR
        ],
        cosebis_pte_files=[
            f"{CLAIMS_DIR}/cosebis_pte_matrix/pte_values/{ver}/{FIDUCIAL['blind']}/pte_{i:03d}_{j:03d}.json"
            for ver in VERSIONS_LEAK_CORR
            for i, j in PTE_SCALE_CUT_PAIRS
        ],
    output:
        evidence=f"{CLAIMS_DIR}/config_space_pte_matrices/evidence.json",
        figure_fiducial=f"{CLAIMS_DIR}/config_space_pte_matrices/figure_fiducial.png",
        figure_appendix=f"{CLAIMS_DIR}/config_space_pte_matrices/figure_appendix.png",
        paper_figure_fiducial=f"{PAPER_FIGURES_DIR}/config_space_pte_fiducial.pdf",
        paper_figure_appendix=f"{PAPER_FIGURES_DIR}/config_space_pte_composite_appendix.pdf",
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
            f"{CONFIG_DIR}/cl.md",
            f"{CONFIG_DIR}/2d_plots.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        # Harmonic PTE matrices only for leak-corrected versions
        pseudo_cl=[_pseudo_cl_path(ver) for ver in VERSIONS_LEAK_CORR],
        pseudo_cl_cov=[
            _pseudo_cl_cov_path(ver, blind=FIDUCIAL["blind"])
            for ver in VERSIONS_LEAK_CORR
        ],
    params:
        version_labels=VERSION_LABELS,
    output:
        evidence=f"{CLAIMS_DIR}/harmonic_space_pte_matrices/evidence.json",
        figure_fiducial=f"{CLAIMS_DIR}/harmonic_space_pte_matrices/figure_fiducial.png",
        figure_appendix=f"{CLAIMS_DIR}/harmonic_space_pte_matrices/figure_appendix.png",
        paper_figure_fiducial=f"{PAPER_FIGURES_DIR}/cl_pte_heatmap.pdf",
        paper_figure_appendix=f"{PAPER_FIGURES_DIR}/cl_pte_composite_appendix.pdf",
    script:
        "../scripts/harmonic_space_pte_matrices.py"


rule bb_covariance_blind_independence:
    """Test BB covariance blind-independence vs EE variation.

    BB covariances should be stable across blinds (null signal → no sample variance).
    EE covariances should vary (~10%) due to sample variance from cosmological signal.

    Covers all three analysis spaces: Pure E/B, COSEBIS, and harmonic (pseudo-Cl).

    Uses mock_version (v1.4.6) for per-blind covariances — blind independence is a
    property of survey geometry, not catalog version. Avoids generating B/C covariances
    for the fiducial version.
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
        # Per-blind MC-propagated pure E/B covariances (using mock_version for all blinds)
        **{f"pure_eb_{b}": f"results/paper_plots/intermediate/{MOCK_VERSION}_{b}_pure_eb_semianalytic.npz"
           for b in BLINDS},
        # COSEBIS: xi integration file (shared) + per-blind config-space covariances
        xi_integration=_xi_integration_path(MOCK_VERSION),
        **{f"cov_integration_{b}": _cov_integration_path(MOCK_VERSION, b) for b in BLINDS},
        # Per-blind harmonic covariances
        **{f"harmonic_{b}": _pseudo_cl_cov_path(MOCK_VERSION, b) for b in BLINDS},
        # Pseudo-Cl data vector (blind A only) for ell bin centers
        pseudo_cl=_pseudo_cl_path(MOCK_VERSION, "A"),
    params:
        nmodes=FIDUCIAL["nmodes"],
        theta_min=config["cosebis"]["theta_min"],
        theta_max=config["cosebis"]["theta_max"],
    output:
        evidence=f"{CLAIMS_DIR}/bb_covariance_blind_independence/evidence.json",
        figure=f"{CLAIMS_DIR}/bb_covariance_blind_independence/figure.png",
    script:
        "../scripts/bb_covariance_blind_independence.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Harmonic vs Configuration-Space COSEBIS Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_COSEBIS_NBINS = config["cl"].get("cosebis_nbins", 32)

_COSEBIS_ANGULAR_RANGES = {
    "full": (float(config["cosebis"]["theta_min"]), float(config["cosebis"]["theta_max"])),
    "fiducial": (float(FIDUCIAL["fiducial_min_scale"]), float(FIDUCIAL["fiducial_max_scale"])),
}

rule harmonic_config_cosebis_comparison:
    """Cross-validate COSEBIS from harmonic (pseudo-Cl) and config (xi_pm) paths.

    Parameterized by {angular_range} (full or fiducial scale cuts).
    Produces 9 per-version figures (1 paper + 4 corrected + 4 uncorrected)
    plus a version comparison figure. Uses cl.cosebis_nbins (default 96) for
    finer bandpower resolution that recovers COSEBIS modes 1-7.
    """
    wildcard_constraints:
        angular_range="full|fiducial",
    input:
        specs=[
            f"{CONFIG_DIR}/harmonic_config_cosebis_comparison.md",
            f"{CONFIG_DIR}/cosebis.md",
            f"{CONFIG_DIR}/cl.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        **{f"pseudo_cl_{ver}": _pseudo_cl_path(ver, nbins=_COSEBIS_NBINS) for ver in VERSIONS_ALL_FOR_PLOTS},
        **{f"pseudo_cl_cov_{ver}": _pseudo_cl_cov_path(ver, nbins=_COSEBIS_NBINS) for ver in VERSIONS_ALL_FOR_PLOTS},
        **{f"xi_{ver}": _xi_integration_path(ver) for ver in VERSIONS_ALL_FOR_PLOTS},
        **{f"cov_{ver}": _cov_integration_path(ver, FIDUCIAL["blind"]) for ver in VERSIONS_ALL_FOR_PLOTS},
    params:
        scale_cut=lambda wildcards: _COSEBIS_ANGULAR_RANGES[wildcards.angular_range],
    output:
        evidence=f"{CLAIMS_DIR}/harmonic_config_cosebis_comparison_{{angular_range}}/evidence.json",
        figure_versions=f"{CLAIMS_DIR}/harmonic_config_cosebis_comparison_{{angular_range}}/figure_versions.png",
        paper_figure=f"{PAPER_FIGURES_DIR}/harmonic_config_cosebis_{{angular_range}}.pdf",
        **_per_version_figure_outputs(f"{CLAIMS_DIR}/harmonic_config_cosebis_comparison_{{angular_range}}"),
    script:
        "../scripts/harmonic_config_cosebis_comparison.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COSEBIS Filter Overlay
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rule cosebis_filter_overlay:
    """W_n(ell) filter functions overlaid on 32-bin BB bandpower data.

    Builds intuition about the C_ell -> COSEBIS transform.
    Shows why coarse bandpowers underresolve higher COSEBIS modes.
    """
    input:
        specs=[
            f"{CONFIG_DIR}/cosebis_filter_overlay.md",
            f"{CONFIG_DIR}/cosebis.md",
            f"{CONFIG_DIR}/cl.md",
        ],
        config=f"{CONFIG_DIR}/config.yaml",
        pseudo_cl=_pseudo_cl_path(FIDUCIAL_VERSION),
        pseudo_cl_cov=_pseudo_cl_cov_path(FIDUCIAL_VERSION),
    output:
        evidence=f"{CLAIMS_DIR}/cosebis_filter_overlay/evidence.json",
        figure=f"{CLAIMS_DIR}/cosebis_filter_overlay/figure.png",
    script:
        "../scripts/plot_cosebis_filter_overlay.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Local Rules Declaration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

localrules: cl_data_vector, cl_version_comparison, pure_eb_covariance, pure_eb_data_vector, pure_eb_version_comparison, cosebis_version_comparison, cosebis_data_vector, config_space_pte_matrices, harmonic_space_pte_matrices, bb_covariance_blind_independence, harmonic_config_cosebis_comparison, cosebis_filter_overlay
