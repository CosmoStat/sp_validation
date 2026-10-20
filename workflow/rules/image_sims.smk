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
m-bias estimator matches each to the reference by RA/Dec.
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
    "config_dir",
    "psf_model",
    "n_smp",
    "extract_script",
    "calibrate_script",
}
# Structural keys: paths/identifiers the run must supply (no sensible default).
_STRUCTURAL_KEYS = {
    "sif",
    "shapepipe_repo",
    "sp_validation_repo",
    "grids_base",
    "input_sims_base",
    "psf_dict",
    "num",
    "tile_ids",
}
_ALLOWED_KEYS = (
    _SCIENCE_KEYS | _DEPRECATED_KEYS | _OPERATIONAL_KEYS | _STRUCTURAL_KEYS
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

# --- container ------------------------------------------------------------
# Every compute rule carries ``container: SIF`` rather than inheriting a
# module-level default: these rules are also included from the top-level
# workflow/Snakefile, whose module default is the cosmology image (no ShapePipe
# stack).  Binds come from the driving profile's ``apptainer-args``.  A null
# ``sif`` resolves to the workflow's one image (see workflow/image_sims/config.yaml).
SIF = common.resolve_container(IMSIM["sif"])

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
MANIFEST = f"{GRIDS_BASE}/manifest.yaml"
BUILD_MANIFEST = f"{SPV_REPO}/workflow/scripts/im_build_manifest.py"

# --- tiles ----------------------------------------------------------------
# tile_ids is the one tile-input mechanism: an explicit list in the run config.
TILE_IDS = list(IMSIM["tile_ids"])

# --- calibration / m-bias knobs ------------------------------------------
SHAPE = IMSIM["shape"]
MASK_CONFIG = IMSIM["mask_config"]  # e.g. config/calibration/mask_v1.X.9_im_sim.yaml
PARAMS_TEMPLATE = f"{SPV_REPO}/workflow/image_sims/params_im_sim.py"
# ShapePipe cfis_image_sims config dir (per-tile/exposure configs + final_cat.param).
CONFIG_DIR = IMSIM["config_dir"]

# ShapePipe scripts live in the ShapePipe repo (also baked into its image).
CREATE_FINAL_CAT = f"{SHAPEPIPE_REPO}/scripts/python/create_final_cat.py"
RUN_JOB = f"{SHAPEPIPE_REPO}/scripts/sh/run_job_sp_canfar_v2.0.bash"
# Extract/calibrate run from the sp_validation *repo* checkout (bind-mounted),
# not the baked copies: the container tracks the branch but lags it, and the
# image-sims path needs branch-only fixes (star-catalogue-optional extract,
# FITS-aware CalibrateCat.read_cat). Overridable for a different checkout.
EXTRACT_INFO = IMSIM["extract_script"]
CALIBRATE = IMSIM["calibrate_script"]
# m-bias is *this branch's* extracted core, injected on PYTHONPATH.
COMPUTE_M_BIAS = f"{SPV_REPO}/scripts/compute_m_bias_image_sims.py"

# --- in-command env prefix -------------------------------------------------
# Snakemake wraps each rule's whole ``shell:`` string inside the container, so
# these ``VAR=value`` tokens land inside it. Three settings:
#
#   * PYTHONPATH prepends both repos' ``src`` so Python resolves the worktree
#     build ahead of the copies baked into the image -- the branch's code runs
#     without an image rebuild. Packages only: the bash and python entry points
#     are invoked at the repo paths from config (RUN_JOB, CREATE_FINAL_CAT,
#     EXTRACT_INFO, ...), not shadowed by PYTHONPATH.
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

JOB_MASK = sum([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])


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


rule im_pipeline_all:
    input:
        expand(
            f"{GRIDS_BASE}/{{sim}}/logs/pipeline_{{tile}}.done",
            sim=SIMS,
            tile=TILE_IDS,
        ),


rule im_merge_all:
    input:
        expand(f"{GRIDS_BASE}/{{sim}}/final_cat_{{sim}}.hdf5", sim=SIMS),


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
    """Stage per-sim run directory: params.py, mask config, ShapePipe configs,
    and the raw SKiLLS image inputs.

    ``params_im_sim.py`` derives the field name from the directory basename, so
    the same template serves every sim; ``config_mask.yaml`` and ``cfis`` are
    symlinks the downstream calibration and merge steps read from cwd.

    ``input_tiles``/``input_exp`` are top-level symlinks to the raw SKiLLS tile
    and exposure images; ShapePipe's ``get_images_runner`` resolves them via
    ``$SP_DIR/input_{tiles,exp}`` (``$SP_DIR`` is the run dir). ``run_job`` does
    not stage these, so ``im_init`` must -- this is what makes ``im_pipeline``
    runnable from raw images, not just from pre-staged intermediates.
    """
    input:
        # Tracked so that editing the params template or mask config re-stages
        # them into every run dir (a plain params: value would not retrigger,
        # silently leaving stale params.py behind after a grammar change).
        template=PARAMS_TEMPLATE,
        mask_src=os.path.join(SPV_REPO, MASK_CONFIG),
    output:
        params=f"{GRIDS_BASE}/{{sim}}/params.py",
        mask=f"{GRIDS_BASE}/{{sim}}/config_mask.yaml",
    params:
        config_dir=CONFIG_DIR,
        run_dir=lambda wc: f"{GRIDS_BASE}/{wc.sim}",
        cfis=lambda wc: f"{GRIDS_BASE}/{wc.sim}/cfis",
        sim_tiles=lambda wc: f"{INPUT_SIMS_BASE}/{wc.sim}/images/SP_tiles",
        sim_exp=lambda wc: f"{INPUT_SIMS_BASE}/{wc.sim}/images/SP_exp",
    shell:
        # cfis / input_tiles / input_exp are stable read-only symlinks (used by
        # get_images, merge, extract); created here but not tracked as outputs,
        # which snakemake will not accept for a symlink/directory.
        "mkdir -p $(dirname {output.params}) && "
        "cp {input.template} {output.params} && "
        "ln -sf {input.mask_src} {output.mask} && "
        "ln -sfT {params.config_dir} {params.cfis} && "
        "ln -sfT {params.sim_tiles} {params.run_dir}/input_tiles && "
        "ln -sfT {params.sim_exp} {params.run_dir}/input_exp"


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
        done=touch(f"{GRIDS_BASE}/{{sim}}/logs/pipeline_{{tile}}.done"),
    params:
        run_dir=lambda wc: f"{GRIDS_BASE}/{wc.sim}",
        psf=IMSIM["psf_model"],
        n_smp=IMSIM["n_smp"],
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


rule im_merge:
    """Merge per-tile ShapePipe catalogues into final_cat_{sim}.hdf5.

    ``create_final_cat.py`` lives in the ShapePipe repo/image; run in image_sims
    mode (``-I``) it walks the per-tile output under the run directory.
    """
    input:
        tiles=expand(
            f"{GRIDS_BASE}/{{{{sim}}}}/logs/pipeline_{{tile}}.done",
            tile=TILE_IDS,
        ),
    output:
        cat=f"{GRIDS_BASE}/{{sim}}/final_cat_{{sim}}.hdf5",
    params:
        run_dir=lambda wc: f"{GRIDS_BASE}/{wc.sim}",
    container:
        SIF
    shell:
        "cd {params.run_dir} && "
        "{_ENV_PREFIX} python {CREATE_FINAL_CAT} "
        "-I -m final_cat_{wildcards.sim}.hdf5 -i .. "
        "-p cfis/final_cat.param -P {wildcards.sim} "
        "-o n_tiles_final.txt -v"


rule im_extract:
    """Extract the comprehensive ngmix catalogue (sp_validation stage).

    ``extract_info.py`` reads ``params.py`` from cwd and the merged catalogue,
    writing ``shape_catalog_comprehensive_{shape}``.
    """
    input:
        cat=f"{GRIDS_BASE}/{{sim}}/final_cat_{{sim}}.hdf5",
        params=f"{GRIDS_BASE}/{{sim}}/params.py",
    output:
        cat=f"{GRIDS_BASE}/{{sim}}/shape_catalog_comprehensive_{SHAPE}.fits",
    params:
        run_dir=lambda wc: f"{GRIDS_BASE}/{wc.sim}",
    container:
        SIF
    shell:
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


rule im_mbias_config:
    """Assemble ``m_bias_config.yaml`` for the m-bias step: the manifest's
    shear/branch facts, this run's science knobs, and git/container provenance.
    """
    input:
        manifest=MANIFEST,
        cats=expand(
            f"{GRIDS_BASE}/{{sim}}/shape_catalog_cut_{SHAPE}.fits", sim=SIMS
        ),
    output:
        cfg=f"{GRIDS_BASE}/results/m_bias_config.yaml",
    params:
        grids_base=GRIDS_BASE,
        num=NUM,
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
    container:
        SIF
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
