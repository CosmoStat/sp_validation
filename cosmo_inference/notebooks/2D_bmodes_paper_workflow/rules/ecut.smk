# workflow/rules/ecut.smk
"""
Ellipticity cut investigation: filter catalogs by |e| < threshold,
recompute survey properties, and run through the full B-mode pipeline.
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ECUT_CONFIG = config.get("ecut", {})
ECUT_VERSIONS = ECUT_CONFIG.get("versions_for_comparison", [])
ECUT_VERSION_LABELS = ECUT_CONFIG.get("version_labels", {})
ECUT_FIDUCIAL = ECUT_CONFIG.get("fiducial_for_comparison", config["fiducial"]["version"])

# Parent version mapping: ecut version → parent version for catalog source
# e.g., SP_v1.4.6_ecut07 → SP_v1.4.6
ECUT_PARENT_VERSIONS = {
    ver: ver.split("_ecut")[0]
    for ver in config.get("versions", [])
    if "_ecut" in ver and "_leak_corr" not in ver
}


def _ecut_parent_catalog(wildcards):
    """Resolve parent catalog path for an ecut version."""
    parent = ECUT_PARENT_VERSIONS.get(wildcards.version)
    if parent is None:
        raise ValueError(f"No parent version found for {wildcards.version}")
    cat_config = config[parent]
    shear_path = cat_config["shear"]["path"]
    if shear_path.startswith("/"):
        return shear_path
    return str(Path(cat_config.get("subdir", "")) / shear_path)


def _ecut_parent_area(version):
    """Get parent version's area for an ecut version."""
    base = version.replace("_leak_corr", "")
    parent = ECUT_PARENT_VERSIONS.get(base, base)
    return config[parent]["cov_th"]["A"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Catalog filtering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rule filter_catalog_ellipticity:
    """Filter a shear catalog by |e_leak_corrected| < threshold."""
    input:
        catalog=_ecut_parent_catalog,
    params:
        e_max=lambda w: float(w.version.split("ecut")[1]) / 10,  # ecut07 → 0.7
        e1_col="e1_leak_corrected",
        e2_col="e2_leak_corrected",
        w_col="w_des",
        parent_area_deg2=lambda w: _ecut_parent_area(w.version),
    output:
        catalog="results/ecut/{version}.fits",
        survey_props="results/ecut/{version}_survey_props.json",
    wildcard_constraints:
        version=r"SP_v[\d.]+_ecut\d+",
    resources:
        mem_mb=16000,
    script:
        "../scripts/filter_catalog_ellipticity.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Version comparison figures (parameterized by {comparison} wildcard)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Comparison sets: maps {comparison} wildcard → (versions, labels, fiducial)
COMPARISON_SETS = {
    "paper": {
        "versions": VERSIONS_LEAK_CORR,
        "labels": VERSION_LABELS,
        "fiducial": config["plotting"].get("fiducial_for_comparison", config["fiducial"]["version"]),
    },
    "ecut": {
        "versions": ECUT_VERSIONS,
        "labels": ECUT_VERSION_LABELS,
        "fiducial": ECUT_FIDUCIAL,
    },
}


def _comparison_versions(comparison):
    return COMPARISON_SETS[comparison]["versions"]


def _comparison_labels(comparison):
    return COMPARISON_SETS[comparison]["labels"]


def _comparison_fiducial(comparison):
    return COMPARISON_SETS[comparison]["fiducial"]


# --- Pure E/B version comparison ---

def _pure_eb_comparison_inputs(wildcards):
    versions = _comparison_versions(wildcards.comparison)
    return {
        "specs": [
            f"{CONFIG_DIR}/pure_eb_version_comparison.md",
            f"{CONFIG_DIR}/pure_eb.md",
            f"{CONFIG_DIR}/1d_plots.md",
        ],
        "config": f"{CONFIG_DIR}/config.yaml",
        "pure_eb_data": [
            f"results/paper_plots/intermediate/{ver}_A_pure_eb_semianalytic.npz"
            for ver in versions
        ],
    }


rule pure_eb_comparison:
    """B-mode visualization: Pure E/B across catalog versions (parameterized)."""
    input:
        unpack(_pure_eb_comparison_inputs),
    params:
        version_labels=lambda w: _comparison_labels(w.comparison),
        versions=lambda w: _comparison_versions(w.comparison),
        fiducial_for_comparison=lambda w: _comparison_fiducial(w.comparison),
    output:
        evidence=f"{CLAIMS_DIR}/{{comparison}}_pure_eb_version_comparison/evidence.json",
        figure=f"{CLAIMS_DIR}/{{comparison}}_pure_eb_version_comparison/figure.png",
        paper_figure=f"{CLAIMS_DIR}/{{comparison}}_pure_eb_version_comparison/paper_figure.pdf",
    wildcard_constraints:
        comparison=r"(paper|ecut)",
    script:
        "../scripts/pure_eb_version_comparison.py"


# --- COSEBIS version comparison ---

def _cosebis_comparison_inputs(wildcards):
    versions = _comparison_versions(wildcards.comparison)
    return {
        "specs": [
            f"{CONFIG_DIR}/cosebis_version_comparison.md",
            f"{CONFIG_DIR}/cosebis.md",
            f"{CONFIG_DIR}/1d_plots.md",
        ],
        "config": f"{CONFIG_DIR}/config.yaml",
        "xi_integration": [_xi_integration_path(ver) for ver in versions],
        "cov_integration": [_cov_integration_path(ver, "A") for ver in versions],
    }


rule cosebis_comparison:
    """B-mode visualization: COSEBIS across catalog versions (parameterized)."""
    input:
        unpack(_cosebis_comparison_inputs),
    params:
        version_labels=lambda w: _comparison_labels(w.comparison),
        versions=lambda w: _comparison_versions(w.comparison),
        fiducial_for_comparison=lambda w: _comparison_fiducial(w.comparison),
    output:
        evidence=f"{CLAIMS_DIR}/{{comparison}}_cosebis_version_comparison/evidence.json",
        figure_stacked=f"{CLAIMS_DIR}/{{comparison}}_cosebis_version_comparison/figure_stacked.png",
    wildcard_constraints:
        comparison=r"(paper|ecut)",
    script:
        "../scripts/cosebis_version_comparison.py"


# --- Cl version comparison ---

def _cl_comparison_inputs(wildcards):
    versions = _comparison_versions(wildcards.comparison)
    return {
        "specs": [
            f"{CONFIG_DIR}/cl_version_comparison.md",
            f"{CONFIG_DIR}/cl.md",
            f"{CONFIG_DIR}/cl_data_vector.md",
        ],
        "config": f"{CONFIG_DIR}/config.yaml",
        "pseudo_cl": [_pseudo_cl_path(ver) for ver in versions],
        "pseudo_cl_cov": [_pseudo_cl_cov_path(ver) for ver in versions],
    }


rule cl_comparison:
    """C_ell^BB version comparison (parameterized)."""
    input:
        unpack(_cl_comparison_inputs),
    params:
        version_labels=lambda w: _comparison_labels(w.comparison),
        versions=lambda w: _comparison_versions(w.comparison),
        fiducial_for_comparison=lambda w: _comparison_fiducial(w.comparison),
        ell_min_cut=config["cl"]["fiducial_ell_min"],
        ell_max_cut=config["cl"]["fiducial_ell_max"],
    output:
        evidence=f"{CLAIMS_DIR}/{{comparison}}_cl_version_comparison/evidence.json",
        figure=f"{CLAIMS_DIR}/{{comparison}}_cl_version_comparison/figure.png",
        paper_figure=f"{CLAIMS_DIR}/{{comparison}}_cl_version_comparison/paper_figure.pdf",
    wildcard_constraints:
        comparison=r"(paper|ecut)",
    script:
        "../scripts/cl_version_comparison.py"


localrules: pure_eb_comparison, cosebis_comparison, cl_comparison


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Convenience targets
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rule ecut_version_comparisons:
    """All three ecut version comparison figures."""
    input:
        rules.pure_eb_comparison.output[0].format(comparison="ecut"),
        rules.cosebis_comparison.output[0].format(comparison="ecut"),
        rules.cl_comparison.output[0].format(comparison="ecut"),
