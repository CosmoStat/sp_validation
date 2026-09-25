"""Shared helpers for every Snakefile that composes workflow/.

Imported by the host Snakemake, where sp_validation is not installed: this
module imports only the standard library and snakemake, and loads the
stdlib-only project modules it needs by file path.
"""

import functools
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

from snakemake.io import temp

# The running checkout. Anchored on this module's own location, not
# workflow.basedir — under `module` composition basedir reflects the composing
# paper, not the running checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"
# For rules that shell out to a script directly rather than through Snakemake's
# `script:` directive.
WORKFLOW_SCRIPTS = str(REPO_ROOT / "workflow" / "scripts")
REPO_SCRIPTS = str(REPO_ROOT / "scripts")


def _load_checkout_module(name):
    """Load the stdlib-only module ``sp_validation/<name>.py`` from this checkout.

    Loaded by file path rather than imported: snakemake runs on the host, where
    sp_validation is usually not installed, and importing the package would drag
    in ``__init__`` -> ``version`` -> a metadata warning on every launch. Taking
    it from *this checkout's* src/ also means the workflow and the package can
    never disagree about what the module says.
    """
    alias = f"_spv_{name}"
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location(
            alias, REPO_SRC / "sp_validation" / f"{name}.py"
        )
    )
    sys.modules[alias] = module
    module.__loader__.exec_module(module)
    return module


# The container model, shared with the ``spv-container`` CLI.
_container = _load_checkout_module("container")
compare_revision = _container.compare_revision
image_revision = _container.image_revision
image_runtime = _container.image_runtime
resolve_image = _container.resolve_image

# The blinding file-name conventions, shared with ``sp_validation.blinding``.
_blinding_paths = _load_checkout_module("blinding_paths")
init_paths = _blinding_paths.init_paths
part_paths = _blinding_paths.part_paths

# Every job inherits this launch's environment (the slurm executor submits with
# --export=ALL), and the Snakemake each job step starts keeps its source cache
# under XDG_CACHE_HOME, which a login shell may point at node-local storage.
# Without it, jobs use the home directory's cache, which every node mounts.
os.environ.pop("XDG_CACHE_HOME", None)


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
# The catalogue config of the launched checkout: the one file both the host
# (CATALOG_CONFIG, loaded in configure) and every job read catalogues from.
CAT_CONFIG = str(REPO_ROOT / "cosmo_val" / "cat_config.yaml")
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


def inject_checkout_pythonpath(workflow_config):
    """Make the launched checkout's ``src`` win over the image's baked copy.

    Snakemake's ``script:`` directive already runs the *checkout's* script
    files, so without this a rule executes new script code against an old
    ``import sp_validation`` -- the two halves of one commit, split. Prepending
    ``REPO_SRC`` closes that: the image stays the frozen dependency stack, the
    checkout supplies sp_validation.

    Apptainer forwards ``APPTAINERENV_``-prefixed host variables into the job as
    their unprefixed names, surviving the profile's ``--cleanenv``; setting it
    here on the driver reaches every containerized rule. Any value the user
    already exported is preserved behind ours.

    Opt out with ``--config checkout_pythonpath=false`` to reproduce a run from
    the image alone.
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


def resolve_container(override=None):
    """Return the image every rule should run in.

    ``override`` wins if set (a ``docker://`` tag, a ``.sif`` path or a sandbox
    directory -- Snakemake's ``container:`` accepts all three); otherwise
    ``resolve_image()``, so jobs run what interactive ``spv-container`` work
    runs.
    """
    if override:
        return str(override)
    return resolve_image()[0]


def warn_if_image_stale():
    """Print one advisory line about a local image that is not pristine or current.

    Never fatal. Two things worth saying at launch:

    * a sandbox is in play, so what jobs run is not fully described by any
      revision label -- deliberate, but it should not be a silent difference
      from a clean run;
    * the image predates the checkout. Usually fine, because the checkout's
      ``src/`` is what rules import; it matters when the *dependency stack*
      moved -- a new package, a lockfile bump.

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


@functools.cache
def check_host_parity(image):
    """Stop the launch unless this Snakemake matches the image's Python and Snakemake.

    A ``script:`` job appends the host's ``sys.path`` -- standard library
    included -- to its own, as the fallback that lets it unpickle the host's
    ``snakemake`` object. So the image and the host Snakemake must share a
    Python minor (else a module the image lacks loads from the host's stdlib)
    and a Snakemake version (else the pickle does not match its reader).

    An image that cannot be read (a registry tag, no apptainer) is named in one
    line and passes.
    """
    import snakemake
    from snakemake.exceptions import WorkflowError

    runtime = image_runtime(image)
    if runtime is None:
        print(
            f"[container] cannot read the Python of {image}; host/image parity "
            "unchecked.",
            file=sys.stderr,
        )
        return
    python, image_snakemake = runtime
    host_python = ".".join(str(v) for v in sys.version_info[:2])
    host = (host_python, snakemake.__version__)
    if (python, image_snakemake or snakemake.__version__) == host:
        return
    raise WorkflowError(
        f"host Snakemake {snakemake.__version__} on Python {host_python} does not "
        f"match the image {image} (Snakemake {image_snakemake}, Python {python}).\n"
        "Reinstall the host Snakemake to match:\n"
        f"  uv tool install --force --python {python} "
        f"snakemake=={image_snakemake or snakemake.__version__} "
        "--with snakemake-executor-plugin-slurm"
    )


def configure(workflow_config):
    """Install config-derived values after Snakemake has loaded configfiles."""
    global CATALOG_CONFIG, DEFAULT_MASK_SUFFIX, FIDUCIAL, PLANCK18, RUN_TYPE
    from snakemake.common.configfile import load_configfile

    inject_checkout_pythonpath(workflow_config)
    warn_if_image_stale()
    check_host_parity(resolve_container(workflow_config.get("container")))
    CATALOG_CONFIG = load_configfile(CAT_CONFIG)
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


# ---------------------------------------------------------------------------
# ξ± angular grids
# ---------------------------------------------------------------------------
# A grid is a binning: (min_sep, max_sep, nbins, npatch). `reporting` is the
# analysis grid, `integration` the fine one both B-mode statistics (COSEBIs,
# pure-E/B) integrate over.
XI_KEYS = ("min_sep", "max_sep", "nbins", "npatch")


def xi_grids(config, fiducial):
    """The named ξ± grids of a workflow, canonicalised.

    Workflows carrying no cosmo_val block (e.g. papers/bmodes) fall back to
    ``fiducial``. Values are coerced here — separations to float, counts to int
    — so the tag the table stamps into a filename is the one the measurement
    writes: the separations pass through float() on the way to TreeCorr, so a
    YAML ``300`` must become ``300.0`` before it names a file, or producer and
    consumer ask for different paths.
    """
    cv = config.get("cosmo_val", {})
    grids = {
        "reporting": (
            {
                "min_sep": cv["theta_min"],
                "max_sep": cv["theta_max"],
                "nbins": cv["nbins"],
                "npatch": cv["npatch"],
            }
            if cv
            else {k: fiducial[k] for k in XI_KEYS}
        ),
        "integration": dict(
            cv.get("integration")
            or {
                "min_sep": fiducial["min_sep_int"],
                "max_sep": fiducial["max_sep_int"],
                "nbins": fiducial["nbins_int"],
            }
        ),
    }
    grids["integration"].setdefault("npatch", 1)
    for grid in grids.values():
        for key in ("min_sep", "max_sep"):
            grid[key] = float(grid[key])
        for key in ("nbins", "npatch"):
            grid[key] = int(grid[key])
    return grids


def grid_binning(grid):
    """The `minsep=..._maxsep=..._nbins=..._npatch=...` tag of one grid."""
    return (
        f"minsep={grid['min_sep']}_maxsep={grid['max_sep']}"
        f"_nbins={grid['nbins']}_npatch={grid['npatch']}"
    )


def grid_of(grids, binning):
    """Name of the grid a binning belongs to, compared numerically.

    A "300" wildcard matches a 300.0 grid value. Binnings matching no named
    grid (e.g. papers/bmodes' nbins=10000 convergence check) are reporting-style
    measurements.
    """
    key = tuple(float(binning[k]) for k in XI_KEYS)
    for name, grid in grids.items():
        if tuple(float(grid[k]) for k in XI_KEYS) == key:
            return name
    return "reporting"


def pseudo_cl_tag(config):
    """Fiducial harmonic-binning tag stamped into pseudo-Cl filenames."""
    return f"blind={config['harmonic']['fiducial']['blind']}_{pseudo_cl_binning_tag(config)}"


def pseudo_cl_binning_tag(config):
    """The `{binning}_nbins={n}` half of the tag — the binning alone."""
    fiducial = config["harmonic"]["fiducial"]
    return f"{fiducial['binning']}_nbins={fiducial['nbins']}"


def pseudo_cl_analysis_stem(config, version):
    """Stem of the analysis pseudo-Cℓ part for a version.

    Its own name (and its own producing rule, twopoint.smk), distinct from the
    generic `pseudo_cl` variants the bmodes and mock workflows request: only
    this part is blindable, so only it can be temp()'d on a data run. Single
    definition shared by the producer, the assembler (cosmo_val.smk) and the
    blindable-stem regex (blinding.smk).

    Carries the binning but no `blind=` field: the A/B/C blind is the legacy
    n(z) vocabulary (#312), not this part's concealment.
    """
    return f"pseudo_cl_analysis_{version}_{pseudo_cl_binning_tag(config)}"


def get_shear_catalog(wildcards):
    """Resolve shear catalog path from config for a given version."""
    cat_config = CATALOG_CONFIG[wildcards.version.replace("_leak_corr", "")]
    shear_path = cat_config["shear"]["path"]
    if shear_path.startswith("/"):
        return shear_path
    subdir = cat_config.get("subdir", "")
    return str(Path(subdir) / shear_path)


# ---------------------------------------------------------------------------
# Smokescreen blind-at-birth custody
# ---------------------------------------------------------------------------
# Distinct from the glass-mock A/B/C `blind` wildcard above: this is Smokescreen
# concealment (see sp_validation.blinding). RUN_TYPE is the single switch: a
# `data` run binds every ξ-derived consumer to the *_blinded parts, pulling the
# blind_part → blind_init subgraph into the DAG; a `mock` run binds the
# plaintext parts and the subgraph never appears.


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


def blind_root():
    """Root holding one blind-init directory per version, or None on mock runs.

    What `CosmologyValidation(blind_root=...)` takes, so its part writers can
    resolve each version's commitment.json themselves.
    """
    return str(COSMO_VAL / "blind") if is_data_run() else None


def blind_state_dir(version):
    """Per-version blind-init custody directory (commitment + encrypted seed)."""
    return str(COSMO_VAL / "blind" / version)


def blind_state_paths(version):
    """The fixed custody-state files blind_init writes for a version."""
    return init_paths(blind_state_dir(version))


def commitment_input(version):
    """Input mapping binding a version's commitment.json, on data runs only.

    A part writer stamps its output concealed from that file (sacc_io.save's
    `commitment=`), which is what lets a born-blinded or blind-irrelevant part
    clear the fail-closed load gate at assembly.
    """
    if not is_data_run():
        return {}
    return {"commitment": blind_state_paths(version)["commitment"]}


def blinded_path(part_path):
    """The *_blinded sibling blind_part writes beside a plaintext part."""
    return part_paths(part_path)["blinded"]


def version_of(stem):
    """Catalogue version embedded in a blindable part's stem.

    blind_part needs it to locate the version's blind state.
    """
    m = re.search(WILDCARD_CONSTRAINTS["version"], stem)
    if m is None:
        raise ValueError(f"no catalogue version found in part stem {stem!r}")
    return m.group(0)


def blindable_part(part_path):
    """On-disk path a run persists for one blindable part.

    Data run -> the blinded sibling (binding it pulls blind_part + blind_init
    into the DAG); mock run -> the plaintext part.
    """
    return blinded_path(part_path) if is_data_run() else str(part_path)


def maybe_temp(part_path):
    """temp() a producer's blindable plaintext part on data runs.

    Its only consumer there is blind_part, which escrows the true vector before
    Snakemake removes the file, so no plaintext blindable part persists.
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

# Sentinel directory for pure-plot leaf rules (no natural data-product output).
CV_SENTINELS = COSMO_VAL / "snakemake_sentinels"


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


# CosmologyValidation constructor kwargs read from config["cosmo_val"]. Every
# keyword with a default is either here or explicitly exempted in
# src/sp_validation/tests/test_cv_init_params.py, so no default applies silently.
CV_INIT_KEYS = (
    "rho_tau_method",
    "cov_estimate_method",
    "compute_cov_rho",
    "n_cov",
    "theta_min",
    "theta_max",
    "nbins",
    "var_method",
    "npatch",
    "quantile",
    "theta_min_plot",
    "theta_max_plot",
    "ylim_alpha",
    "ylim_xi_sys_ratio",
    "nside",
    "nside_mask",
    "binning",
    "power",
    "n_ell_bins",
    "ell_step",
    "pol_factor",
    "cell_method",
    "noise_bias_method",
    "fiducial_input_inka",
    "nrandom_cell",
    "cell_seed",
    "path_onecovariance",
    "cosmo_params",
)


def cv_init_params(config):
    """Assemble the CosmologyValidation(...) constructor kwargs from config.

    Centralizes the run-specific instantiation so every cosmo_val rule script
    builds an identical `cv`: catalogues from CAT_CONFIG and products under
    COSMO_VAL, never the constructor's cwd-relative defaults.
    """
    cv = config["cosmo_val"]
    return dict(
        versions=config["versions"],
        catalog_config=CAT_CONFIG,
        output_dir=str(COSMO_VAL),
        # Custody state the cv's part writers need: the SACC `type` they stamp,
        # and the blind whose commitment born-blinded parts are stamped under.
        run_type=run_type(),
        blind_root=blind_root(),
        **{key: cv[key] for key in CV_INIT_KEYS},
    )
