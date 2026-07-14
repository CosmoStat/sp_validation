"""Shared helpers for the B-modes Snakemake workflow."""

import json
import os
import re
from pathlib import Path

# Output roots are env-overridable so a reproduction run can write into a
# fresh tree without clobbering (or silently reusing) prior products.
SP_VALIDATION = Path(__file__).resolve().parents[1]
COSMO_VAL = Path(os.environ.get("COSMO_VAL", SP_VALIDATION / "results/cosmo_val"))
COSMO_INFERENCE = Path(
    os.environ.get("COSMO_INFERENCE", SP_VALIDATION / "cosmo_inference")
)
CAT_CONFIG = SP_VALIDATION / "cosmo_val/cat_config.yaml"

BLINDS = ["A", "B", "C"]
BLOCK_PAIRS = [("++", "1"), ("--", "2"), ("+-", "3")]

# Fiducial cosmology: Planck 2018 (astropy Planck18, Table 2 + BAO)
# Source of truth: cs_util.cosmo.PLANCK18
# Regenerate with: snakemake results/cosmology/planck18.json
# Resolved relative to the run directory at configure() time.
COSMOLOGY_PARAMS = "/n23data1/n06data/lgoh/scratch/UNIONS/sp_validation/results/cosmology/planck18.json"

# Wildcard constraints shared by every Snakefile that composes these rules.
# Patterns must match all expected values; overly restrictive patterns cause
# silent failures. Apply with: wildcard_constraints: **WILDCARD_CONSTRAINTS
WILDCARD_CONSTRAINTS = {
    "version": r"SP_v[\d.]+(_w_iv)?(_ecut\d+)?(_leak_corr)?",
    "blind": r"[ABC]",
    "nbins": r"\d+",
    "min_sep": r"[0-9.]+",
    "max_sep": r"[0-9.]+",
    "gaussian": r"(g|ng)",
    "probe": r"(wl|ggl|3x2pt)",
    "block_pm": r"(\+\+|--|\+-)",
    "block_i": r"[123]",
    "mask_suffix": r"(_masked)?",
    "mock_id": r"\d{5}",
    "nside": r"\d+",
}

FIDUCIAL = None
DEFAULT_MASK_SUFFIX = ""
DEFAULT_PROBE = "wl"
CATALOG_CONFIG = None
PLANCK18 = None


def configure(workflow_config):
    """Install config-derived values after Snakemake has loaded configfiles."""
    global CATALOG_CONFIG, DEFAULT_MASK_SUFFIX, DEFAULT_PROBE, FIDUCIAL, PLANCK18
    CATALOG_CONFIG = workflow_config
    FIDUCIAL = workflow_config["fiducial"]
    DEFAULT_PROBE = workflow_config.get("probe") or "wl"
    DEFAULT_MASK_SUFFIX = (
        "_masked" if workflow_config["covariance"].get("default_masked", False) else ""
    )
    with open(COSMOLOGY_PARAMS) as f:
        PLANCK18 = json.load(f)


def fiducial_binning_suffix(fiducial=None):
    """Return binning suffix for fiducial parameters."""
    fiducial = fiducial or FIDUCIAL
    return (
        f"_minsep={fiducial['min_sep']}_maxsep={fiducial['max_sep']}"
        f"_nbins={fiducial['nbins']}_npatch={fiducial['npatch']}"
    )


def resolve_covariance_version(version):
    """Map version to its covariance version."""
    return version


# Single source of truth for the covariance naming scheme. Rule outputs use
# this with wildcards left in braces; covariance_base() fills concrete values.
# Keep the two in sync by construction: covariance_base() formats this string.
COV_BASE_TEMPLATE = (
    "covariance_{version}_{blind}_{gaussian}"
    "_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_{probe}{mask_suffix}"
)


def cov_output(suffix):
    """Wildcard-bearing output path under the covariance tree.

    E.g. cov_output(".ini") ->
      <COSMO_INFERENCE>/data/covariance/<BASE>/<BASE>.ini
    with {version}, {blind}, ... left as Snakemake wildcards.
    """
    return str(
        COSMO_INFERENCE
        / f"data/covariance/{COV_BASE_TEMPLATE}/{COV_BASE_TEMPLATE}{suffix}"
    )


def cov_output_dirfile(filename):
    """Wildcard-bearing path to a fixed-name file inside the covariance dir."""
    return str(COSMO_INFERENCE / f"data/covariance/{COV_BASE_TEMPLATE}/{filename}")


def covariance_base(
    version,
    blind,
    gaussian=None,
    min_sep=None,
    max_sep=None,
    nbins=None,
    probe=None,
    mask_suffix=None,
    resolve_version=True,
    fiducial=None,
    default_mask_suffix=None,
):
    """Construct covariance base name (concrete values, no wildcards)."""
    fiducial = fiducial or FIDUCIAL
    gaussian = gaussian if gaussian is not None else fiducial["gaussian"]
    min_sep = min_sep if min_sep is not None else fiducial["min_sep"]
    max_sep = max_sep if max_sep is not None else fiducial["max_sep"]
    nbins = nbins if nbins is not None else fiducial["nbins"]
    probe = probe if probe is not None else DEFAULT_PROBE
    if mask_suffix is None:
        mask_suffix = (
            DEFAULT_MASK_SUFFIX if default_mask_suffix is None else default_mask_suffix
        )
    cov_version = resolve_covariance_version(version) if resolve_version else version
    return COV_BASE_TEMPLATE.format(
        version=cov_version,
        blind=blind,
        gaussian=gaussian,
        min_sep=min_sep,
        max_sep=max_sep,
        nbins=nbins,
        probe=probe,
        mask_suffix=mask_suffix,
    )


def covariance_dir(
    version,
    blind,
    gaussian=None,
    min_sep=None,
    max_sep=None,
    nbins=None,
    probe=None,
    mask_suffix=None,
    resolve_version=True,
):
    """Construct covariance directory path."""
    base = covariance_base(
        version,
        blind,
        gaussian,
        min_sep,
        max_sep,
        nbins,
        probe,
        mask_suffix,
        resolve_version=resolve_version,
    )
    return str(COSMO_INFERENCE / f"data/covariance/{base}")


def covariance_path(
    version,
    blind,
    gaussian=None,
    min_sep=None,
    max_sep=None,
    nbins=None,
    probe=None,
    mask_suffix=None,
    suffix=".dat",
    resolve_version=True,
):
    """Construct covariance file path."""
    base = covariance_base(
        version,
        blind,
        gaussian,
        min_sep,
        max_sep,
        nbins,
        probe,
        mask_suffix,
        resolve_version=resolve_version,
    )
    return str(COSMO_INFERENCE / f"data/covariance/{base}/{base}{suffix}")


def build_redshift_dir(version):
    """Construct n(z) filepath for given catalog version and blind."""
    base_version = re.sub(r"_leak_corr$", "", version)
    base_version = re.sub(r"_ecut\d+", "", base_version)
    if "v1.4.11" in base_version:
        base_version = "SP_v1.4.6"
    version_dir = base_version.replace("SP_", "")
    return f"/n17data/sguerrini/UNIONS/WL/nz/{version_dir}/"


def build_redshift_path_lens(version, blind):
    """Construct n(z) filepath for given catalog version and blind."""
    base_version = re.sub(r"_leak_corr$", "", version)
    base_version = re.sub(r"_ecut\d+", "", base_version)
    if "v1.4.11" in base_version:
        base_version = "SP_v1.4.6"
    return f"nz_{base_version}_{blind}_lens.txt"


def build_redshift_path_source(version, blind):
    """Construct n(z) filepath for given catalog version and blind."""
    base_version = re.sub(r"_leak_corr$", "", version)
    base_version = re.sub(r"_ecut\d+", "", base_version)
    if "v1.4.11" in base_version:
        base_version = "SP_v1.4.6"
    return f"nz_{base_version}_{blind}_clust.txt"  # TO DO: check lens and source file format


def get_shear_catalog(wildcards):
    """Resolve shear catalog path from config for a given version."""
    base_version = wildcards.version.replace("_leak_corr", "")
    cat_config = CATALOG_CONFIG[base_version]
    shear_path = cat_config["shear"]["path"]
    if shear_path.startswith("/"):
        return shear_path
    subdir = cat_config.get("subdir", "")
    return str(Path(subdir) / shear_path)


# ---------------------------------------------------------------------------
# CosmologyValidation diagnostic suite (cosmo_val.py)
# ---------------------------------------------------------------------------
# The diagnostics in `sp_validation.cosmo_val.CosmologyValidation` share one
# in-memory `cv` object across methods, linked by lazy properties. The
# Snakemake decomposition (workflow/rules/cosmo_val.smk + papers/cosmo_val)
# turns each diagnostic into a rule keyed on the real data products it writes
# under COSMO_VAL. Where a method only emits a figure (no data product), the
# rule declares a sentinel under CV_SENTINELS so the DAG stays trackable.
#
# COSMO_VAL is the cosmo_val/output directory (already defined above), the same
# location every `cv.*` method writes to via `cc["paths"]["output"]`.

# Sentinel directory for pure-plot leaf rules (no natural data-product output).
CV_SENTINELS = COSMO_VAL / "snakemake_sentinels"

# Working directory in which `CosmologyValidation` must be instantiated: it
# reads `./cat_config.yaml` and writes to `./output` by default. Resolved to
# the live (non-worktree) checkout so rules find the catalog config and share
# the output tree with interactive runs.
CV_RUNDIR = "/n17data/cdaley/unions/code/sp_validation/cosmo_val"


def cv_basename(version, fiducial=None):
    """Reproduce CosmologyValidation.basename() for a version.

    Mirrors the f-string in cosmo_val.py so rule outputs match exactly what the
    method writes. Uses fiducial binning (min_sep/max_sep/nbins/npatch).
    """
    fiducial = fiducial or FIDUCIAL
    return (
        f"{version}_minsep={fiducial['min_sep']}"
        f"_maxsep={fiducial['max_sep']}"
        f"_nbins={fiducial['nbins']}"
        f"_npatch={fiducial['npatch']}"
    )


def cv_init_params(config, version_list=None):
    """Assemble the CosmologyValidation(...) constructor kwargs from config.

    Centralizes the run-specific instantiation so every cosmo_val rule script
    builds an identical `cv`. `version_list` overrides config["versions"] (used
    by per-version rules that pass a single version).
    """
    cv = config["cosmo_val"]
    params = dict(
        versions=version_list if version_list is not None else config["versions"],
        npatch=cv["npatch"],
        theta_min=cv["theta_min"],
        theta_max=cv["theta_max"],
        nbins=cv["nbins"],
        theta_min_plot=cv["theta_min_plot"],
        theta_max_plot=cv["theta_max_plot"],
        ylim_alpha=cv["ylim_alpha"],
        nrandom_cell=cv["nrandom_cell"],
        cell_method=cv["cell_method"],
        nside_mask=cv["nside_mask"],
    )
    if cv.get("path_onecovariance"):
        params["path_onecovariance"] = cv["path_onecovariance"]
    if cv.get("rho_tau_method"):
        params["rho_tau_method"] = cv["rho_tau_method"]
    if cv.get("cosmo_params"):
        params["cosmo_params"] = cv["cosmo_params"]
    return params
