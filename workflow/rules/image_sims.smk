"""Image-simulation orchestration: raw SKiLLS sim images -> shear m/c bias.

The sp_validation-side half of the split described in
``UNIONS-WL/MultiBand_ImSim#1``: ShapePipe turns the simulated tiles into
per-tile shape catalogues (``pipeline``, ``merge``), then sp_validation
extracts, calibrates and measures the multiplicative/additive shear bias
(``manifest``, ``extract``, ``calibrate``, ``m_bias``).

One image runs the whole chain: the sp_validation image is built ``FROM`` the
ShapePipe image, so it carries both stacks.  Everything else is parameterised
under ``config["image_sims"]`` -- repository roots, data roots, the PSF
dictionary, the explicit ``tile_ids`` list and the sim/calibration knobs -- so
a fresh user drives it from config alone.  Configuration is fail-fast: a schema
check at load rejects an unknown key (typo) and a missing science key (see
``workflow/image_sims/config.yaml`` for the operational/science split).

The five simulations per grid are the reference ``1z2z`` (no input shear) plus
the ``+/-`` shear pairs ``1p2z``/``1m2z`` (g1) and ``1z2p``/``1z2m`` (g2); the
modern m-bias estimator uses full populations with separately normalized
branch weights (pair_match: false), matching the stored response.
"""

import os

IMSIM = config["image_sims"]

# --- fail-fast schema check ----------------------------------------------
# An unknown key under ``image_sims:`` is a hard error (typo protection); a
# missing key is a hard error naming it.  Both fire at DAG parse, before compute.
#
# Science keys: required from the *run* config; no default here or in
# config.yaml (only a commented template line).  These fix the estimator's
# scientific behaviour, so they must be stated per run, never inherited.
_SCIENCE_KEYS = {
    "w_cols",
    "pair_match",
    "match_radius_deg",
    "n_bootstrap",
    "bootstrap_seed",
    "mask_config",
}
# Deprecated but still accepted, so a pre-``w_cols`` run config keeps parsing
# into the estimator's back-compat path.
_DEPRECATED_KEYS = {
    "w_col",
}
# Operational keys: default in the workflow config.yaml; the .smk reads them
# bare (never ``.get`` with a literal), so config.yaml is their one home.
_OPERATIONAL_KEYS = {
    "sims_type",
    "branches",
    "shape",
    "psf_model",
    "sp_profile",
    "sp_jobs",
    "clean_exposures",
    "tile_store_root",
}
# Optional keys: recognized but not required; defaults applied in code below.
_OPTIONAL_KEYS = {
    # Overrides only: each has a working default derived below from
    # sp_validation_repo, so pointing at a different checkout is opt-in and a
    # missing key cannot silently select someone else's code.
    "extract_script",
    "calibrate_script",
    # DES weight table policy; defaults in config.yaml.
    "des_weight_source",
    "des_weight_product",
    # The container image. Optional because it has a working default chain
    # (sp_validation.container.resolve_image: local sandbox, then local sif,
    # then the registry tag), unlike the structural keys below, which have no
    # sensible default at all. On candide it comes from the profile, whose
    # `config:` sets the cluster's image -- a machine-specific path belongs
    # with the machine, not in this shared file.
    "sif",
    # Exposures from another realization than the tiles (non-grid sims only;
    # default: num). See EXP_NUM below and config.yaml.
    "exp_num",
}
# Structural keys: paths/identifiers the run must supply (no sensible default).
_STRUCTURAL_KEYS = {
    "shapepipe_repo",
    "sp_validation_repo",
    "grids_base",
    "input_sims_base",
    "psf_dict",
    "num",
    "tile_ids",
}
_ALLOWED_KEYS = (
    _SCIENCE_KEYS
    | _DEPRECATED_KEYS
    | _OPERATIONAL_KEYS
    | _OPTIONAL_KEYS
    | _STRUCTURAL_KEYS
)

_unknown = set(IMSIM) - _ALLOWED_KEYS
if _unknown:
    raise ValueError(
        "image_sims: unknown config key(s) "
        f"{sorted(_unknown)} -- check for a typo (allowed keys: "
        f"{sorted(_ALLOWED_KEYS)})"
    )
_missing_science = sorted(_SCIENCE_KEYS - set(IMSIM))
if _missing_science:
    raise ValueError(
        "image_sims: missing required science key(s) "
        f"{_missing_science} -- these have no default and must be set in the "
        "run config (see the commented template in workflow/image_sims/config.yaml)"
    )
_missing_structural = sorted(_STRUCTURAL_KEYS - set(IMSIM))
if _missing_structural:
    raise ValueError(
        "image_sims: missing required key(s) "
        f"{_missing_structural} -- set them in the run config"
    )

if "w_des" in IMSIM["w_cols"]:
    if IMSIM["des_weight_source"] not in ("external", "self"):
        raise ValueError("image_sims: des_weight_source must be external or self")
    _product = IMSIM["des_weight_product"]
    if IMSIM["des_weight_source"] == "external" and (
        not isinstance(_product, dict) or not _product.get("name") or not _product.get("path")
    ):
        raise ValueError(
            "DES image sims require des_weight_product with name and path; "
            "des_weight_source: self is diagnostic only"
        )

# --- container ------------------------------------------------------------
# Every compute rule carries ``container: SIF`` rather than inheriting a
# module-level default: these rules are also included from the top-level
# workflow/Snakefile, whose module default is the cosmology image (no ShapePipe
# stack).  Binds come from the driving profile's ``apptainer-args``.  A null
# ``sif`` resolves to the workflow's one image (see workflow/image_sims/config.yaml).
SIF = common.resolve_container(IMSIM.get("sif"))

# --- repositories (bound into the image; branch code overrides) -----------
SHAPEPIPE_REPO = IMSIM["shapepipe_repo"]
SPV_REPO = IMSIM["sp_validation_repo"]

# --- data and run directories --------------------------------------------
GRIDS_BASE = IMSIM["grids_base"]  # run/output root; one sub-dir per sim
INPUT_SIMS_BASE = IMSIM["input_sims_base"]  # SKiLLS sim images
PSF_DICT = IMSIM["psf_dict"]  # Herve's Full_psf_dict.pickle

# --- simulation grid ------------------------------------------------------
# SIM_BASES is the set of branches *this run requests* -- the reference plus the
# four +/- sheared branches.  Their injected shear (amplitude, per-branch
# (g1,g2), pairing) is NOT a literal here: it lives only in manifest.yaml, built
# by im_manifest from each branch's basic_info.txt and read back by im_mbias.
NUM = IMSIM["num"]
SIMS_TYPE = IMSIM["sims_type"]
_SUFFIX = f"_{SIMS_TYPE}_{NUM}" if SIMS_TYPE == "grid" else f"_{NUM}"
SIM_BASES = list(IMSIM["branches"])
SIMS = [f"{base}{_SUFFIX}" for base in SIM_BASES]
# Optional ``exp_num`` pulls exposures from a different realization than tiles:
# tiles stay on ``{branch}_{num}``, exposures come from ``{branch}_{exp_num}``.
# Non-grid sims only (grid names embed sims_type); see config.yaml for the
# blended use case.
EXP_NUM = IMSIM.get("exp_num", NUM)
if SIMS_TYPE == "grid":
    SIM_EXP_BASE = {sim: sim for sim in SIMS}
else:
    SIM_EXP_BASE = {f"{base}{_SUFFIX}": f"{base}_{EXP_NUM}" for base in SIM_BASES}
MANIFEST = f"{GRIDS_BASE}/manifest.yaml"
BUILD_MANIFEST = f"{SPV_REPO}/workflow/scripts/im_build_manifest.py"

# --- tiles ----------------------------------------------------------------
# tile_ids is the one tile-input mechanism: an explicit list in the run config.
TILE_IDS = list(IMSIM["tile_ids"])

# --- calibration / m-bias knobs ------------------------------------------
SHAPE = IMSIM["shape"]
MASK_CONFIG = IMSIM["mask_config"]  # e.g. config/calibration/mask_v1.X.9_im_sim.yaml
PARAMS_TEMPLATE = f"{SPV_REPO}/workflow/image_sims/params_im_sim.py"

# --- ShapePipe processing: ShapePipe's own workflow ------------------------
# ``sp`` is ShapePipe's workflow launcher; SP_CONFIG_DIR is the config chain it
# selects for ``input_type: image_sims``.  The merge and extract steps read that
# dir's ``final_cat.param`` (through the per-sim ``cfis`` link), so the column
# list and the processing that wrote the columns come from one checkout.
SP_LAUNCHER = f"{SHAPEPIPE_REPO}/workflow/bin/sp"
SP_CONFIG_DIR = f"{SHAPEPIPE_REPO}/workflow/config/cfis_image_sims"
PSF_MODEL = IMSIM["psf_model"]
SP_PROFILE = IMSIM["sp_profile"]
SP_JOBS = IMSIM["sp_jobs"]
CLEAN_EXPOSURES = bool(IMSIM["clean_exposures"])
# Where each tile_shape group's vignette store lives (ShapePipe's
# ``tile_store_root``; ``sp run`` binds it to /local/scratch).  None keeps the
# profile's node-local bind -- and keeps the key out of the run config, so
# campaigns launched before it existed see the same params.
TILE_STORE_ROOT = IMSIM["tile_store_root"]

# ShapePipe scripts live in the ShapePipe repo (also baked into its image).
# Extract/calibrate run from the sp_validation *repo* checkout (bind-mounted),
# not the baked copies: the container tracks the branch but lags it, and the
# image-sims path needs branch-only fixes (star-catalogue-optional extract,
# FITS-aware CalibrateCat.read_cat). Overridable for a different checkout.
EXTRACT_INFO = IMSIM.get(
    "extract_script", f"{SPV_REPO}/scripts/calibration/extract_info.py"
)
CALIBRATE = IMSIM.get(
    "calibrate_script",
    f"{SPV_REPO}/scripts/calibration/calibrate_comprehensive_cat.py",
)
# m-bias is *this branch's* extracted core, injected on PYTHONPATH.
COMPUTE_M_BIAS = f"{SPV_REPO}/scripts/compute_m_bias_image_sims.py"

# --- in-command env prefix -------------------------------------------------
# Snakemake wraps each rule's whole ``shell:`` string inside the container, so
# these ``VAR=value`` tokens land inside it. Three settings:
#
#   * PYTHONPATH prepends both repos' ``src`` so Python resolves the worktree
#     build ahead of the copies baked into the image -- the branch's code runs
#     without an image rebuild. Packages only: the bash and python entry points
#     are invoked at the repo paths from config (EXTRACT_INFO,
#     CALIBRATE, ...), not shadowed by PYTHONPATH.
#   * PSF_DICT points the fake_psf module (PSF_DICT_PATH = $PSF_DICT, expanded
#     via getexpanded) at this run's PSF dictionary.
#   * OMP_NUM_THREADS=1 rides here rather than in the SLURM profile: the chain
#     is MPI-free (Snakemake fans out one job per branch x tile; in-job
#     parallelism is ShapePipe's own ``-N n_smp``), so the OpenMP/BLAS pool must
#     be pinned to 1 to avoid oversubscription, and a profile can only set CLI
#     flags, never the driver env the slurm executor's ``--export=ALL``
#     propagates.
_ENV_PREFIX = (
    f"PYTHONPATH={SHAPEPIPE_REPO}/src:{SPV_REPO}/src "
    f"PSF_DICT={PSF_DICT} OMP_NUM_THREADS=1 "
)


def sp_run_config(sim):
    """ShapePipe workflow run config for one shear branch (one ``sp`` campaign).

    It REPLACES ShapePipe's committed ``workflow/config.yaml`` (``bin/sp``'s
    ``SP_RUN_CONFIG``), so every key the campaign needs is stated; processing
    knobs this config does not state (``ngmix_chunks``) take the ShapePipe
    Snakefile's defaults, never a copy here.  ``clean_tiles`` stays off: the
    merge reads each tile's make_cat output from the tile store.  ``clean``
    (rolling exposure-store reclamation) is ``clean_exposures`` -- a 40-tile
    branch touches ~300 exposures at ~8 GB of stores each.
    """
    run_dir = f"{GRIDS_BASE}/{sim}"
    cfg = {
        "input_type": "image_sims",
        "psf_model": PSF_MODEL,
        "tile_list": f"{run_dir}/tiles_{sim}.txt",
        "inputs": {
            "tiles": f"{INPUT_SIMS_BASE}/{sim}/images/SP_tiles",
            "exposures": f"{INPUT_SIMS_BASE}/{SIM_EXP_BASE[sim]}/images/SP_exp",
        },
        "container": SIF_PIPELINE,
        "outputs": {
            "run_dir": run_dir,
            "index_db": f"{run_dir}/index/run_index.sqlite",
        },
        "clean": CLEAN_EXPOSURES,
        "clean_tiles": False,
        "clean_ignore_tiles": [],
    }
    if PSF_MODEL == "fake":
        cfg["psf_dict"] = PSF_DICT
    if TILE_STORE_ROOT:
        cfg["tile_store_root"] = TILE_STORE_ROOT
    return cfg


localrules:
    im_pipeline,


wildcard_constraints:
    sim="|".join(SIMS),
    tile="|".join(t.replace(".", r"\.") for t in TILE_IDS),


# ==========================================================================
# Convenience targets (run in order)
# ==========================================================================
rule im_manifest_only:
    input:
        MANIFEST,


rule im_init_all:
    input:
        expand(f"{GRIDS_BASE}/{{sim}}/params.py", sim=SIMS),


rule im_shapepipe_all:
    input:
        expand(f"{GRIDS_BASE}/{{sim}}/logs/shapepipe_campaign.yaml", sim=SIMS),


rule im_extract_all:
    input:
        expand(
            f"{GRIDS_BASE}/{{sim}}/shape_catalog_comprehensive_{SHAPE}.fits",
            sim=SIMS,
        ),


rule im_calibrate_all:
    input:
        expand(
            f"{GRIDS_BASE}/{{sim}}/shape_catalog_cut_{SHAPE}.fits", sim=SIMS
        ),


# ==========================================================================
# Rules
# ==========================================================================
rule im_manifest:
    """Build the campaign manifest at the head of the DAG.

    Parses ``g_cosmic`` from every requested branch's ``basic_info.txt``,
    cross-checks each against its ``1{X}2{Y}`` name and the (0,0) reference,
    derives the single injected amplitude, and writes ``manifest.yaml`` into the
    run root.  This is the one home for the injected-shear facts; im_mbias reads
    the amplitude and branch map from here, nowhere else.  Pure sp_validation
    stage (stdlib parse of basic_info; PyYAML to write).
    """
    input:
        # basic_info.txt for each requested branch, so editing a sim's record
        # rebuilds the manifest (and re-validates) rather than reusing a stale one.
        basic_info=expand(
            f"{INPUT_SIMS_BASE}/{{sim}}/basic_info.txt", sim=SIMS
        ),
    output:
        manifest=MANIFEST,
    params:
        branch_args=lambda wc: " ".join(f"--branch {b}" for b in SIM_BASES),
        input_sims_base=INPUT_SIMS_BASE,
        sims_type=SIMS_TYPE,
        num=NUM,
    container:
        SIF
    shell:
        "{_ENV_PREFIX} python {BUILD_MANIFEST} "
        "--input-sims-base {params.input_sims_base} "
        "--sims-type {params.sims_type} --num {params.num} "
        "{params.branch_args} -o {output.manifest}"


rule im_init:
    """Stage per-sim run directory: params.py, mask config, tile list, and the
    ``cfis`` link to ShapePipe's image-sims config dir.

    ``params_im_sim.py`` derives the field name from the directory basename, so
    the same template serves every sim; ``config_mask.yaml`` is a staged copy
    with the image-sim DES-table policy, and ``cfis`` is a symlink. Both are
    what the downstream calibration and merge steps read from cwd. The mask
    config is copied, not linked: a link into the sp_validation checkout or
    spv's code snapshot breaks when that is removed or refreshed, and the run
    dir should keep the mask it was calibrated with. ``tiles_{sim}.txt`` is
    both the ShapePipe campaign's ``tile_list`` and the tile-ID file
    ``params.py`` names for the found/missing-tile check.
    """
    input:
        # Tracked so that editing the params template or mask config re-stages
        # them into every run dir (a plain params: value would not retrigger,
        # silently leaving stale params.py behind after a grammar change).
        template=PARAMS_TEMPLATE,
        mask_src=os.path.join(SPV_REPO, MASK_CONFIG),
        weight_product=(
            [IMSIM["des_weight_product"]["path"]]
            if "w_des" in IMSIM["w_cols"]
            and IMSIM["des_weight_source"] == "external"
            and IMSIM["des_weight_product"] else []
        ),
    output:
        params=f"{GRIDS_BASE}/{{sim}}/params.py",
        mask=f"{GRIDS_BASE}/{{sim}}/config_mask.yaml",
        tiles=f"{GRIDS_BASE}/{{sim}}/tiles_{{sim}}.txt",
    params:
        config_dir=SP_CONFIG_DIR,
        cfis=lambda wc: f"{GRIDS_BASE}/{wc.sim}/cfis",
        tile_ids=" ".join(TILE_IDS),
        source=IMSIM["des_weight_source"],
        product_name=(IMSIM["des_weight_product"] or {}).get("name", "-"),
        product_path=(IMSIM["des_weight_product"] or {}).get("path", "-"),
        w_cols=IMSIM["w_cols"],
    container:
        SIF
    shell:
        # cfis is a stable read-only symlink (used by merge and extract);
        # created here but not tracked as an output, which snakemake will not
        # accept for a symlink to a directory.
        "mkdir -p $(dirname {output.params}) && "
        "cp {input.template} {output.params} && "
        "{_ENV_PREFIX} python {SPV_REPO}/scripts/calibration/stage_image_sim_config.py "
        "--input {input.mask_src:q} --output {output.mask:q} "
        "--w-cols {params.w_cols} --source {params.source:q} "
        "--product-name {params.product_name:q} --product-path {params.product_path:q} && "
        "ln -sfT {params.config_dir} {params.cfis} && "
        "printf '%s\\n' {params.tile_ids} > {output.tiles}"


rule im_pipeline:
    """Run ShapePipe on one simulated tile.

    Delegates the module DAG to ShapePipe's own job runner; the sentinel log
    marks tile completion for the merge step.  The compute-heavy stage.
    """
    input:
        # params.py alone supplies the im_init -> im_pipeline edge.  The `cfis`
        # symlink {RUN_JOB} also reads is an untracked side effect of the same
        # im_init shell (snakemake will not track a symlink output), so it must
        # not be declared here -- doing so asks the DAG for a file no rule
        # produces and aborts on a fresh grids_base.
        params=f"{GRIDS_BASE}/{{sim}}/params.py",
    output:
        record=f"{GRIDS_BASE}/{{sim}}/logs/shapepipe_campaign.yaml",
    log:
        f"{GRIDS_BASE}/{{sim}}/logs/sp_run.log",
    resources:
        mem_mb=16000,
        runtime=720,
    container:
        SIF
    shell:
        "cd {params.run_dir} && "
        "{_ENV_PREFIX} bash {RUN_JOB} "
        "-e {wildcards.tile} -t image_sims -j {JOB_MASK} "
        "-p {params.psf} -N {params.n_smp}"

def merged_cat(wc):
    """This sim's merged catalogue.

    ShapePipe's own workflow (``merge_final_cats``, shapepipe #891) publishes it
    under ``product/`` with the rest of the campaign's products, and that is the
    only copy surviving ``clean_tile``. This workflow no longer merges -- the
    ``im_merge`` rule that wrote one at the branch root was dropped once the
    ShapePipe side gained the step.

    The branch-root fallback is kept for a catalogue merged by hand or by an
    older campaign. NOTHING PRODUCES IT any more, so a sim with neither copy
    fails at DAG build on a missing input instead of silently re-merging --
    the honest outcome, since the campaign that should have published it did
    not.
    """
    prod = f"{GRIDS_BASE}/{wc.sim}/product/final_cat_{wc.sim}.hdf5"
    if os.path.exists(prod):
        return prod
    return f"{GRIDS_BASE}/{wc.sim}/final_cat_{wc.sim}.hdf5"


rule im_extract:
    """Extract the comprehensive ngmix catalogue (sp_validation stage).

    ``extract_info.py`` reads ``params.py`` from cwd and the merged catalogue,
    writing ``shape_catalog_comprehensive_{shape}``.

    Stages the catalogue itself rather than assuming a producer left it in the
    right place: ``params_im_sim.py`` sets ``data_dir = "."`` and names
    ``final_cat_{name}.hdf5`` with ``name`` from the directory basename, so the
    file has to be reachable from the branch dir under exactly that name. A
    catalogue published under ``product/`` is linked in; one already at the
    root is left alone.
    """
    input:
        cat=merged_cat,
        params=f"{GRIDS_BASE}/{{sim}}/params.py",
    output:
        cat=f"{GRIDS_BASE}/{{sim}}/shape_catalog_comprehensive_{SHAPE}.fits",
    params:
        run_dir=lambda wc: f"{GRIDS_BASE}/{wc.sim}",
        staged=lambda wc: f"{GRIDS_BASE}/{wc.sim}/final_cat_{wc.sim}.hdf5",
    container:
        SIF
    shell:
        # Compare resolved paths: when the catalogue is already at the branch
        # root, source and destination are the same file and `ln -sf` would
        # replace it with a link to itself.
        '[ "$(readlink -f {input.cat})" = "$(readlink -f {params.staged})" ] '
        "|| ln -sf {input.cat} {params.staged}; "
        "cd {params.run_dir} && {_ENV_PREFIX} python {EXTRACT_INFO}"


rule im_calibrate:
    """Calibrate and cut the comprehensive catalogue (sp_validation stage).

    ``calibrate_comprehensive_cat.py`` reads ``config_mask.yaml`` from cwd,
    applies the metacal calibration and selection, and writes
    ``shape_catalog_cut_{shape}.fits``.
    """
    input:
        cat=f"{GRIDS_BASE}/{{sim}}/shape_catalog_comprehensive_{SHAPE}.fits",
        mask=f"{GRIDS_BASE}/{{sim}}/config_mask.yaml",
    output:
        cat=f"{GRIDS_BASE}/{{sim}}/shape_catalog_cut_{SHAPE}.fits",
    params:
        run_dir=lambda wc: f"{GRIDS_BASE}/{wc.sim}",
    container:
        SIF
    shell:
        "cd {params.run_dir} && "
        "{_ENV_PREFIX} python {CALIBRATE} -s calibrate"


def campaign_records():
    """Per-branch evidence that the ShapePipe campaign finished, for ordering.

    WHICH file that is depends on which workflow ran the campaign, and both
    shapes are in use: ShapePipe's unified workflow publishes
    product/index/run_report.json, while the older sp_validation-driven runs
    left n_tiles_final.txt beside the merged catalogue at the branch root.
    Pick whichever exists, so a grid processed years apart still gets an edge.

    Nothing READS these -- im_mbias_config.py takes no input.* -- so the path
    only has to name a file that really exists for that branch. A branch with
    none contributes nothing rather than blocking the DAG on a file no rule
    produces; its merged catalogue is already a declared input elsewhere.
    """
    found = []
    for sim in SIMS:
        for cand in (
            f"{GRIDS_BASE}/{sim}/product/index/run_report.json",
            f"{GRIDS_BASE}/{sim}/product/n_tiles_final.txt",
            f"{GRIDS_BASE}/{sim}/n_tiles_final.txt",
        ):
            if os.path.exists(cand):
                found.append(cand)
                break
    return found


rule im_mbias_config:
    """Assemble ``m_bias_config.yaml`` for the m-bias step: the manifest's
    shear/branch facts, this run's science knobs, and git/container provenance.
    """
    input:
        manifest=MANIFEST,
        cats=expand(
            f"{GRIDS_BASE}/{{sim}}/shape_catalog_cut_{SHAPE}.fits", sim=SIMS
        ),
        # The per-branch ShapePipe campaign records, for provenance: the
        # edge that says every campaign finished before this config was
        # written. ShapePipe's own workflow publishes run_report.json with the
        # products; the old logs/shapepipe_campaign.yaml was im_pipeline's
        # output and is never written now that `sp run` drives the campaigns.
        # Nothing READS these -- im_mbias_config.py takes no input.* -- so the
        # path only has to name something the campaign really produced.
        campaigns=campaign_records(),
    output:
        cfg=f"{GRIDS_BASE}/results/m_bias_config.yaml",
    params:
        grids_base=GRIDS_BASE,
        num=NUM,
        sims_type=SIMS_TYPE,
        cat_name=f"shape_catalog_cut_{SHAPE}.fits",
        sif=SIF,
        shapepipe_repo=SHAPEPIPE_REPO,
        sp_validation_repo=SPV_REPO,
        results_dir=f"{GRIDS_BASE}/results",
        results=f"{GRIDS_BASE}/results/m_bias_results.yaml",
        # Science knobs, read bare from the run config (no default here).
        match_radius_deg=IMSIM["match_radius_deg"],
        w_cols=IMSIM["w_cols"],
        n_bootstrap=IMSIM["n_bootstrap"],
        pair_match=IMSIM["pair_match"],
        bootstrap_seed=IMSIM["bootstrap_seed"],
    # NO container: -- deliberately. This is snakemake's `script:` directive,
    # which pickles the `snakemake` object on the host and unpickles it inside
    # the job, so host and job need compatible snakemake versions. They are not:
    # the host has 9.16.3 and the image 9.22.0, and the pickle fails with
    # "Can't get attribute 'InputFiles' on module snakemake.io". Nothing here
    # needs the image anyway -- the script imports only hashlib, os, re,
    # subprocess and yaml, and reads git metadata from the checkouts, which is
    # easier outside the container than in. The `shell:` rules keep theirs;
    # they pass strings, not pickles.
    script:
        "../scripts/im_mbias_config.py"


rule im_mbias:
    """Multiplicative/additive shear bias from the calibrated grids.

    Produces the workflow's headline artifact, ``m_bias_results.yaml``, running
    the estimator against the config ``im_mbias_config`` assembled.
    """
    input:
        cfg=f"{GRIDS_BASE}/results/m_bias_config.yaml",
    output:
        results=f"{GRIDS_BASE}/results/m_bias_results.yaml",
    container:
        SIF
    shell:
        "{_ENV_PREFIX} python {COMPUTE_M_BIAS} -c {input.cfg} -v"
