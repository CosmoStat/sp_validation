"""Shared helpers for the B-modes Snakemake workflow."""

import json
import os
import re
from pathlib import Path

from snakemake.io import temp

# Absolute path to the generic workflow's scripts, for rules that shell out to a
# script directly rather than through Snakemake's `script:` directive. Anchored
# on this module's own location, not workflow.basedir -- under `module`
# composition basedir reflects the composing paper, not the running checkout.
WORKFLOW_SCRIPTS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "scripts")

# Output roots are env-overridable so a reproduction run can write into a
# fresh tree without clobbering (or silently reusing) prior products.
COSMO_VAL = Path(
    os.environ.get(
        "COSMO_VAL", "/n17data/cdaley/unions/code/sp_validation/cosmo_val/output"
    )
)
COSMO_INFERENCE = Path(
    os.environ.get(
        "COSMO_INFERENCE", "/n17data/cdaley/unions/code/sp_validation/cosmo_inference"
    )
)
CAT_CONFIG = "/n17data/cdaley/unions/code/sp_validation/cosmo_val/cat_config.yaml"
# "blind" is the glass-mock A/B/C realisation convention, NOT Smokescreen
# blinding (a separate axis: the concealed=True SACC stamp). The name is baked
# into on-disk filenames we do not own (e.g. nz_{version}_{A|B|C}.txt).
BLINDS = ["A", "B", "C"]
BLOCK_PAIRS = [("++", "1"), ("--", "2"), ("+-", "3")]

# Fiducial cosmology: Planck 2018 (astropy Planck18, Table 2 + BAO)
# Source of truth: cs_util.cosmo.PLANCK18
# Regenerate with: snakemake results/cosmology/planck18.json
# Resolved relative to the run directory at configure() time.
COSMOLOGY_PARAMS = "results/cosmology/planck18.json"

# Wildcard constraints shared by every Snakefile that composes these rules.
# Patterns must match all expected values; overly restrictive patterns cause
# silent failures. Apply with: wildcard_constraints: **WILDCARD_CONSTRAINTS
WILDCARD_CONSTRAINTS = {
    "version": r"SP_v[\d.]+(_w_iv)?(_ecut\d+)?(_leak_corr)?",
    "blind": r"[ABC]",
    "nbins": r"\d+",
    # Constrained so a producer's ξ± output pattern cannot greedily absorb the
    # "_blinded" suffix into npatch (which would make it ambiguous with the
    # blind_part rule's {stem}_blinded output). npatch is always an integer.
    "npatch": r"\d+",
    "min_sep": r"[0-9.]+",
    "max_sep": r"[0-9.]+",
    "gaussian": r"(g|ng)",
    "block_pm": r"(\+\+|--|\+-)",
    "block_i": r"[123]",
    "mask_suffix": r"(_masked)?",
    "mock_id": r"\d{5}",
    "nside": r"\d+",
}

FIDUCIAL = None
DEFAULT_MASK_SUFFIX = ""
CATALOG_CONFIG = None
PLANCK18 = None
# Run type gates Smokescreen blind-at-birth (see the blind custody section
# below): "data" blinds the three blindable parts and binds ξ-derived consumers
# to the blinded siblings; "mock" bypasses blinding entirely. Set from
# config["cosmo_val"]["type"] in configure(); "data" is the production default.
RUN_TYPE = "data"


def configure(workflow_config):
    """Install config-derived values after Snakemake has loaded configfiles."""
    global CATALOG_CONFIG, DEFAULT_MASK_SUFFIX, FIDUCIAL, PLANCK18, RUN_TYPE
    CATALOG_CONFIG = workflow_config
    FIDUCIAL = workflow_config["fiducial"]
    DEFAULT_MASK_SUFFIX = (
        "_masked" if workflow_config["covariance"].get("default_masked", False) else ""
    )
    RUN_TYPE = workflow_config.get("cosmo_val", {}).get("type", "data")
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


def covariance_base(
    version,
    blind,
    gaussian="ng",
    min_sep=None,
    max_sep=None,
    nbins=None,
    mask_suffix=None,
    resolve_version=True,
    fiducial=None,
    default_mask_suffix=None,
):
    """Construct covariance base name."""
    fiducial = fiducial or FIDUCIAL
    min_sep = min_sep if min_sep is not None else fiducial["min_sep"]
    max_sep = max_sep if max_sep is not None else fiducial["max_sep"]
    nbins = nbins if nbins is not None else fiducial["nbins"]
    mask_suffix = (
        mask_suffix
        if mask_suffix is not None
        else (
            DEFAULT_MASK_SUFFIX if default_mask_suffix is None else default_mask_suffix
        )
    )
    cov_version = resolve_covariance_version(version) if resolve_version else version
    return (
        f"covariance_{cov_version}_{blind}_{gaussian}"
        f"_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}{mask_suffix}"
    )


def covariance_dir(
    version,
    blind,
    gaussian="ng",
    min_sep=None,
    max_sep=None,
    nbins=None,
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
        mask_suffix,
        resolve_version=resolve_version,
    )
    return str(COSMO_INFERENCE / f"data/covariance/{base}")


def covariance_path(
    version,
    blind,
    gaussian="ng",
    min_sep=None,
    max_sep=None,
    nbins=None,
    mask_suffix=None,
    suffix="_processed.txt",
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
        mask_suffix,
        resolve_version=resolve_version,
    )
    return str(COSMO_INFERENCE / f"data/covariance/{base}/{base}{suffix}")


def base_version(version):
    """Strip the derived-catalogue suffixes to the base catalogue version.

    The `_leak_corr` / `_ecut{N}` variants share their parent's n(z) and
    `cov_th` survey parameters, so lookups keyed on either must strip both.
    """
    return re.sub(r"_ecut\d+", "", re.sub(r"_leak_corr$", "", version))


def build_redshift_path(version, blind):
    """Construct n(z) filepath for given catalog version and blind."""
    base = base_version(version)
    if "v1.4.11" in base:
        base = "SP_v1.4.6"
    version_dir = base.replace("SP_", "")
    return f"/n17data/sguerrini/UNIONS/WL/nz/{version_dir}/nz_{base}_{blind}.txt"


def pseudo_cl_tag(config):
    """Fiducial harmonic-binning tag the pseudo-Cl producers stamp into filenames.

    Single definition shared by the producer (twopoint.smk) and the consumers
    (cosmo_val.smk, inference.smk), which reconstruct the name from config.
    """
    fiducial = config["harmonic"]["fiducial"]
    return f"blind={fiducial['blind']}_{fiducial['binning']}_nbins={fiducial['nbins']}"


def get_shear_catalog(wildcards):
    """Resolve shear catalog path from config for a given version."""
    cat_config = CATALOG_CONFIG[wildcards.version.replace("_leak_corr", "")]
    shear_path = cat_config["shear"]["path"]
    if shear_path.startswith("/"):
        return shear_path
    subdir = cat_config.get("subdir", "")
    return str(Path(subdir) / shear_path)


# ---------------------------------------------------------------------------
# Smokescreen blind-at-birth custody (issues #247/#252, PR #253)
# ---------------------------------------------------------------------------
# Distinct from the glass-mock A/B/C `blind` wildcard above: this is Smokescreen
# concealment (the concealed=True SACC stamp). Blinding is per part, at birth.
# The three blindable parts (reporting ξ±, integration ξ±, pseudo-Cℓ) are each
# concealed the moment they are computed, so only blinded parts persist on disk.
# A `data` run binds every ξ-derived consumer (the terminal assemble, and the
# born-blinded COSEBIs / pure-E/B) to the *_blinded parts, which pulls the
# blind_part → blind_init subgraph into the DAG. A `mock` run bypasses blinding
# entirely and binds to the plaintext parts, so the subgraph never appears.
# RUN_TYPE is the single switch that flips which files exist.
#
# The path helpers below MIRROR sp_validation.blinding.init_paths / part_paths
# by hand rather than importing blinding (which pulls in numpy + smokescreen) at
# DAG-build time. test_blinding_wiring asserts the two stay in lockstep.


def run_type():
    """The campaign's run type, ``"data"`` or ``"mock"``.

    Every part writer stamps this as the SACC ``type`` metadata, which is what
    ``blinding.assert_consistent_blind`` reads at assembly. A function
    rather than the ``RUN_TYPE`` global because ``from common import *`` binds
    names before ``configure()`` runs, so only a call reads the configured
    value.
    """
    return RUN_TYPE


def is_data_run():
    """True when blinding is active (production data runs); False for mocks."""
    return run_type() == "data"


def blind_state_dir(version):
    """Per-version blind-init custody directory (commitment + encrypted seed)."""
    return str(COSMO_VAL / "blind" / version)


def blind_state_paths(version):
    """The fixed custody-state files blind_init writes for a version.

    Mirrors sp_validation.blinding.init_paths(blind_state_dir(version)).
    """
    d = blind_state_dir(version)
    return {
        "commitment": os.path.join(d, "commitment.json"),
        "bundle": os.path.join(d, "blind_seed.encrpt"),
        "key": os.path.join(d, "blind_seed.key"),
    }


def commitment_input(version):
    """Input mapping binding a version's commitment.json, on data runs only.

    A part writer stamps its output concealed from that file (sacc_io.save's
    `commitment=`), which is what lets a born-blinded or blind-irrelevant part
    clear the fail-closed load gate the terminal assembly opens every part
    through. A mock run binds nothing and the part stays plaintext.
    """
    if not is_data_run():
        return {}
    return {"commitment": blind_state_paths(version)["commitment"]}


def blinded_path(part_path):
    """The *_blinded sibling blind_part writes beside a plaintext part.

    Mirrors sp_validation.blinding.part_paths(part_path)["blinded"].
    """
    stem, ext = os.path.splitext(str(part_path))
    return f"{stem}_blinded{ext or '.fits'}"


def version_of(stem):
    """Extract the catalogue version embedded in a blindable part's stem.

    Every blindable stem carries the version (as {version}_xi_… or
    pseudo_cl_{version}_…); blind_part needs it to locate the version's blind
    state. Matches the shared `version` wildcard pattern.
    """
    m = re.search(WILDCARD_CONSTRAINTS["version"], stem)
    if m is None:
        raise ValueError(f"no catalogue version found in part stem {stem!r}")
    return m.group(0)


def blindable_part(part_path):
    """On-disk path a run persists for one blindable part.

    Data run -> the blinded sibling (binding it pulls blind_part + blind_init
    into the DAG); mock run -> the plaintext part (blinding bypassed).
    """
    return blinded_path(part_path) if is_data_run() else str(part_path)


def maybe_temp(part_path):
    """Wrap a producer's blindable plaintext part temp() on data runs.

    On a data run the plaintext part's only consumer is blind_part, which
    escrows the true vector before Snakemake removes the temp file — so no
    plaintext blindable part persists. On a mock run the part is the real
    product downstream binds to, so it is left persistent.
    """
    return temp(str(part_path)) if is_data_run() else str(part_path)


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
        # Stamped as the SACC `type` of every part the cv writes; RUN_TYPE reads
        # from this same key (see configure()).
        run_type=cv.get("type", "data"),
    )
    if cv.get("path_onecovariance"):
        params["path_onecovariance"] = cv["path_onecovariance"]
    if cv.get("rho_tau_method"):
        params["rho_tau_method"] = cv["rho_tau_method"]
    if cv.get("cosmo_params"):
        params["cosmo_params"] = cv["cosmo_params"]
    return params
