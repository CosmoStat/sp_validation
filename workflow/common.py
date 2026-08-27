"""Shared helpers for the B-modes Snakemake workflow."""

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

# This checkout's importable source tree: workflow/common.py -> <repo>/src.
REPO_SRC = Path(__file__).resolve().parent.parent / "src"

# The container model lives in the package (``sp_validation/container.py``): the
# registry tag, this user's canonical image paths, and the ``spv-container`` CLI
# that fills them. Taken from *this checkout's* src/, so the workflow and the CLI
# can never disagree -- in particular they share one resolution order, which is
# what lets a package installed into a sandbox ride along into workflow jobs.
#
# Loaded by file path rather than as ``sp_validation.container``: snakemake runs
# on the host, where sp_validation is usually not installed, and importing the
# package would drag in ``__init__`` -> ``version`` -> a metadata warning on
# every launch. The module itself is stdlib-only, so this costs nothing.
_container = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location(
        "_spv_container", REPO_SRC / "sp_validation" / "container.py"
    )
)
sys.modules["_spv_container"] = _container
_container.__loader__.exec_module(_container)

CONTAINER_URI = _container.CONTAINER_URI
compare_revision = _container.compare_revision
image_revision = _container.image_revision
local_sandbox = _container.local_sandbox
local_sif = _container.local_sif
resolve_image = _container.resolve_image


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


def inject_checkout_pythonpath(workflow_config):
    """Make the launched checkout's ``src`` win over the image's baked copy.

    Snakemake's ``script:`` directive already runs the *checkout's* script
    files, so without this a rule executes new script code against an old
    ``import sp_validation`` -- the two halves of one commit, split. Prepending
    ``REPO_SRC`` closes that: the image stays the frozen dependency stack, the
    checkout supplies sp_validation. This mirrors what the image-sims chain has
    always done for both repos (``_ENV_PREFIX`` in rules/image_sims.smk).

    Apptainer forwards ``APPTAINERENV_``-prefixed host variables into the job as
    their unprefixed names, surviving the profile's ``--cleanenv``; setting it
    here on the driver reaches every containerized rule. Any value the user
    already exported is preserved behind ours.

    Opt out with ``--config checkout_pythonpath=false`` to reproduce a run from
    the image alone. Note the caveat: ``rerun-triggers: code`` watches rule and
    script files, not ``src/``, so editing a module under ``src/`` does not by
    itself mark outputs stale -- force with ``-F``/``--forcerun``.
    """
    flag = workflow_config.get("checkout_pythonpath", True)
    # `--config key=false` can arrive as the *string* "false" depending on how
    # Snakemake parses the value, so don't lean on truthiness alone.
    if isinstance(flag, str):
        flag = flag.strip().lower() not in ("false", "no", "0", "off", "")
    if not flag:
        return
    if not REPO_SRC.is_dir():
        return
    existing = os.environ.get("APPTAINERENV_PYTHONPATH", "")
    parts = [str(REPO_SRC)] + [p for p in existing.split(":") if p]
    os.environ["APPTAINERENV_PYTHONPATH"] = ":".join(parts)


def resolve_container(workflow_config):
    """Return the image every rule should run in.

    The same order ``spv-container`` uses, so jobs run what interactive work
    runs: your writable sandbox if you have built one, else your pristine
    ``.sif`` if you have pulled one, else the registry tag -- which Snakemake
    autopulls into ``.snakemake/singularity`` under the working directory.
    Snakemake's ``container:`` accepts all three (a sandbox directory included).
    ``--config container=...`` overrides everything and takes a ``docker://``
    tag, a ``.sif`` path, or a sandbox directory.
    """
    override = workflow_config.get("container")
    if override:
        return str(override)
    return resolve_image()[0]


def warn_if_image_stale():
    """Print one advisory line about a local image that is not pristine or current.

    Never fatal. Two things worth saying at launch:

    * a sandbox is in play, so what jobs run is not fully described by any
      revision label -- somebody installed into it on purpose, and that is the
      point, but it should not be a silent difference from a clean run;
    * the image predates the checkout. Usually fine, because the checkout's
      ``src/`` is what rules import (inject_checkout_pythonpath); it matters when
      the *dependency stack* moved -- a new package, a lockfile bump.

    Silent when there is no local image, no apptainer, or no revision label.
    """
    image, kind = resolve_image()
    if kind == "tag":
        return
    revision = image_revision(image)
    if kind == "sandbox":
        built = f"built from {revision[:12]}" if revision else "revision unknown"
        print(
            f"[container] running the writable sandbox at {image} ({built}). "
            "Anything installed into it is part of this run; "
            "`spv-container status` for detail.",
            file=sys.stderr,
        )
    if compare_revision(revision) == "behind":
        print(
            f"[container] image was built from {revision[:12]}, which is behind this "
            "checkout. Fine unless the dependency stack moved; refresh with "
            "`spv-container pull`.",
            file=sys.stderr,
        )


def configure(workflow_config):
    """Install config-derived values after Snakemake has loaded configfiles."""
    global CATALOG_CONFIG, DEFAULT_MASK_SUFFIX, FIDUCIAL, PLANCK18
    inject_checkout_pythonpath(workflow_config)
    warn_if_image_stale()
    CATALOG_CONFIG = workflow_config
    FIDUCIAL = workflow_config["fiducial"]
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


def build_redshift_path(version, blind):
    """Construct n(z) filepath for given catalog version and blind."""
    base_version = re.sub(r"_leak_corr$", "", version)
    base_version = re.sub(r"_ecut\d+", "", base_version)
    if "v1.4.11" in base_version:
        base_version = "SP_v1.4.6"
    version_dir = base_version.replace("SP_", "")
    return (
        f"/n17data/sguerrini/UNIONS/WL/nz/{version_dir}/nz_{base_version}_{blind}.txt"
    )


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
