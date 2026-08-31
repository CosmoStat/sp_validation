"""Image-simulation orchestration: raw SKiLLS sim images -> shear m/c bias.

This rule set drives the image-simulation validation chain end to end and is
the sp_validation-side half of the split described in
``UNIONS-WL/MultiBand_ImSim#1``: ShapePipe turns the simulated tiles into
per-tile shape catalogues, then sp_validation merges, extracts, calibrates and
finally measures the multiplicative/additive shear bias.

Two images, one prefix shape.  Architecturally one image could run every
stage -- the sp_validation image is built ``FROM`` the ShapePipe image, so it
carries both stacks -- but the *published* sp_validation image's environment
is not yet trustworthy for the ShapePipe half: sp_validation has no lockfile
and does not declare its numba-bearing dependency (``cosmo_numba``), so
unpinned install layers can drift NumPy past numba's window (a 2026-07-11
gate run hit exactly this: ``Numba needs NumPy 2.4 or less. Got NumPy 2.5``
at the ngmix stage).  PYTHONPATH shadowing covers pure-Python *code*, never
binary deps, so until sp_validation is uv-locked with its deps declared
(spun off as its own task), each half runs in its own repo's image -- the
same split the gate766 baseline ran:

* ShapePipe stages -> ``pipeline`` (raw images -> per-tile cats) and ``merge``
  (``create_final_cat`` -> ``final_cat_{sim}.hdf5``) run in ``sif_pipeline``
  (the ShapePipe image).
* sp_validation stages -> ``manifest``, ``extract`` (-> comprehensive cat),
  ``calibrate`` (-> cut cat) and ``m_bias`` (-> ``m_bias_results.yaml``) run
  in ``sif`` (the sp_validation image).

Every rule sets ``container: None`` and calls ``apptainer exec`` explicitly
through a shared prefix template (``EXEC_PIPELINE`` / ``EXEC`` -- identical
env injections, different image), because the images are not the workflow's
top-level container.  Everything is parameterised under
``config["image_sims"]`` -- the two ``sif`` keys, repository roots, data roots,
the PSF dictionary, the explicit ``tile_ids`` list and the sim/calibration
knobs -- so a fresh user drives it from config alone, with no hard-coded clone
layout.  Configuration is fail-fast: a schema check at load rejects an unknown
key (typo) and a missing science key (see ``workflow/image_sims/config.yaml``
for the operational/science split).  The ``PYTHONPATH`` override injects both
repos' ``src`` so the *branch* source (ShapePipe's ``#766`` build;
sp_validation's ``image_sims.py``, ``catalog.match_catalogs_radec``) wins over
whatever is baked into the image.

The five simulations per grid are the reference ``1z2z`` (no input shear) plus
the ``+/-`` shear pairs ``1p2z``/``1m2z`` (g1) and ``1z2p``/``1z2m`` (g2); the
m-bias estimator matches each to the reference by RA/Dec.
"""

import os
from pathlib import Path

IMSIM = config["image_sims"]

# --- fail-fast schema check ----------------------------------------------
# One home for every fact: the run config carries the science knobs, the
# workflow config.yaml carries the operational defaults, and *this* block is
# where a typo or a missing knob dies -- at DAG parse, before any compute.
#
# Every key must be declared below.  An unknown key under ``image_sims:`` is a
# hard error (typo protection); a missing *science* key is a hard error naming
# the key (no silent code default anywhere).  Operational keys default in the
# workflow config.yaml and nowhere else: the .smk reads them as bare
# ``IMSIM[key]`` (never ``.get`` with a second literal), so their value comes
# from config.yaml alone -- the single home for an operational default.
#
# Science keys: required from the *run* config; no default in config.yaml (only
# a commented template line) and no default in code.  These fix the estimator's
# scientific behaviour, so they must be stated per run, never inherited.
_SCIENCE_KEYS = {
    "w_cols",
    "pair_match",
    "match_radius_deg",
    "n_bootstrap",
    "bootstrap_seed",
    "mask_config",
}
# Deprecated science keys: accepted (so a pre-``w_cols`` run config still parses
# and the estimator's back-compat path runs) but not *required* -- our configs
# state ``w_cols``.  Listed here only to keep them out of the unknown-key error.
_DEPRECATED_KEYS = {
    "w_col",
}
# Operational keys: default (visibly) in the workflow config.yaml; the .smk
# reads them bare, so config.yaml is their one home.
_OPERATIONAL_KEYS = {
    "binds",
    "sims_type",
    "branches",
    "shape",
    "config_dir",
    "pipeline_config_dir",
    "psf_model",
    "n_smp",
    "extract_script",
    "calibrate_script",
}
# Optional keys: recognized but not required; defaults applied in code below.
_OPTIONAL_KEYS = {
    "exp_num",
}
# Structural keys: paths/identifiers the run must supply (no sensible default).
_STRUCTURAL_KEYS = {
    "sif",
    "sif_pipeline",
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

# --- containers -----------------------------------------------------------
# Two images (see module docstring): the ShapePipe image for the pipeline and
# merge stages, the sp_validation image for everything downstream.  Collapse
# back to one image once sp_validation's env is lock-managed.
SIF = IMSIM["sif"]  # sp_validation stages
SIF_PIPELINE = IMSIM["sif_pipeline"]  # ShapePipe stages
BINDS = IMSIM["binds"]

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
# ShapePipe cfis_image_sims config dir (per-tile/exposure configs + final_cat.param).
CONFIG_DIR = IMSIM["config_dir"]
# ShapePipe config dir for the pipeline stage's run_job (``-c`` override; owns
# the ngmix ini and thus METACAL_PSF).  Empty -> run_job's default.  See
# config.yaml.
PIPELINE_CONFIG_DIR = IMSIM["pipeline_config_dir"]

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

# --- container exec prefixes ----------------------------------------------
# One prefix *shape* for every stage -- two instances, one per image.  Three
# env injections make the on-disk branch
# code and the sim PSF win over the image's baked copies:
#
#   * PYTHONPATH prepends BOTH repos' ``src`` (ShapePipe first, then
#     sp_validation), so Python resolves the worktree build before
#     ``/app``/``/sp_validation`` -- the local-testing counterpart of the
#     git-ref deps, letting the branch code run without an image rebuild.  This
#     covers the Python *packages* only: the bash entry points (run_job) and
#     the ShapePipe/sp_validation *scripts* are still invoked at the repo paths
#     resolved from config (RUN_JOB, CREATE_FINAL_CAT, EXTRACT_INFO, ...), not
#     shadowed by PYTHONPATH.
#   * PSF_DICT points the fake_psf module (PSF_DICT_PATH = $PSF_DICT, expanded
#     via getexpanded) at this run's PSF dictionary.
#
# The SLURM env vars are stripped (``env -u ...``) so that when the ShapePipe
# pipeline stage's OpenMPI initialises inside the image it does not try to
# attach to the host SLURM launcher (cf. apptainer_noslurm.sh).  The strip is
# harmless for the pure-Python sp_validation stages, so one prefix serves all.
#
# ``OMP_NUM_THREADS=1`` is injected here, at the ``apptainer exec`` call, and
# not left to the SLURM profile.  The chain is MPI-free: Snakemake fans out one
# job per branch x tile and each job's parallelism is ShapePipe's own internal
# multiprocessing (``-N n_smp``), so the OpenMP/BLAS thread pool inside the
# container must be pinned to 1 to avoid oversubscription.  The SLURM profile
# cannot pin it reliably: the slurm executor submits with ``--export=ALL``,
# which propagates the *driver's* ambient environment -- but a Snakemake
# profile only sets CLI flags, never the driver's own env, so an
# ``OMP_NUM_THREADS`` there would depend on the operator having exported it by
# hand (the implicit, uncommitted state the "one run command" is meant to
# retire).  Injecting it on the ``apptainer exec`` line puts it where the
# compute actually runs -- inside the container, independent of the driver's
# env -- the same lever this prefix already uses for PYTHONPATH/PSF_DICT.
_EXEC_PREFIX = (
    "env -u SLURM_JOBID -u SLURM_JOB_ID -u SLURM_PROCID "
    f"apptainer exec --bind {BINDS} "
    f"--env PYTHONPATH={SHAPEPIPE_REPO}/src:{SPV_REPO}/src "
    f"--env PSF_DICT={PSF_DICT} --env OMP_NUM_THREADS=1 "
)
EXEC = _EXEC_PREFIX + SIF  # sp_validation stages
EXEC_PIPELINE = _EXEC_PREFIX + SIF_PIPELINE  # ShapePipe stages

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
    shell:
        "{EXEC} python {BUILD_MANIFEST} "
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
        # SIM_EXP_BASE maps the tile sim to its exposure-source sim (see above).
        sim_exp=lambda wc: f"{INPUT_SIMS_BASE}/{SIM_EXP_BASE[wc.sim]}/images/SP_exp",
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
    """Run ShapePipe on one simulated tile (ShapePipe stage).

    Delegates the module DAG to ShapePipe's own job runner; the sentinel log
    marks tile completion for the merge step.  This is the compute-heavy,
    MPI-bearing stage.
    """
    input:
        # ``params.py`` is a *tracked* output of ``im_init``, so this one input
        # supplies the im_init -> im_pipeline edge. The ``cfis`` symlink the
        # shell reads (via {RUN_JOB}) is created by that same im_init shell block
        # as an *untracked* side effect -- no rule declares it as an output
        # (snakemake will not track a symlink/directory output). Declaring it an
        # input here therefore asked the DAG for a file no rule produces: on a
        # fresh grids_base it aborted the build with MissingInputException before
        # any job ran. It is safe to drop -- cfis exists whenever params does,
        # since im_init stages both together.
        params=f"{GRIDS_BASE}/{{sim}}/params.py",
    output:
        done=touch(f"{GRIDS_BASE}/{{sim}}/logs/pipeline_{{tile}}.done"),
    params:
        run_dir=lambda wc: f"{GRIDS_BASE}/{wc.sim}",
        psf=IMSIM["psf_model"],
        n_smp=IMSIM["n_smp"],
        config_flag=f"-c {PIPELINE_CONFIG_DIR}" if PIPELINE_CONFIG_DIR else "",
    resources:
        # measured: job uses ~1 core (eff 0.9); post-#843 MaxRSS ~72MB accounting / 2.3GB live-peak; 2cpu/4GB packs ~512 jobs on the usable partition
        mem_mb=4000,
        cpus_per_task=2,
        runtime=720,
    shell:
        "cd {params.run_dir} && "
        "{EXEC_PIPELINE} bash {RUN_JOB} "
        "-e {wildcards.tile} -t image_sims -j {JOB_MASK} "
        "-p {params.psf} -N {params.n_smp} {params.config_flag}"


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
    shell:
        "cd {params.run_dir} && "
        "{EXEC_PIPELINE} python {CREATE_FINAL_CAT} "
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
    shell:
        "cd {params.run_dir} && {EXEC} python {EXTRACT_INFO}"


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
    shell:
        "cd {params.run_dir} && "
        "{EXEC} python {CALIBRATE} -s calibrate"


rule im_mbias:
    """Multiplicative/additive shear bias from the calibrated grids.

    Produces the workflow's headline artifact, ``m_bias_results.yaml``.  The
    injected shear (``shear_amplitude`` and the branch map) comes from
    ``manifest.yaml`` alone -- no literal amplitude here or in config.yaml.  The
    generated ``m_bias_config.yaml`` carries the manifest's ``branches`` and
    ``pairs``, so the estimator's sim list and pairing are the campaign's, not a
    hard-coded default.
    """
    input:
        manifest=MANIFEST,
        cats=expand(
            f"{GRIDS_BASE}/{{sim}}/shape_catalog_cut_{SHAPE}.fits", sim=SIMS
        ),
    output:
        results=f"{GRIDS_BASE}/results/m_bias_results.yaml",
    resources:
        # Blended catalogues are ~5x grid size; the snakemake 1000M default OOMs.
        mem_mb=8000,
    params:
        cfg=f"{GRIDS_BASE}/results/m_bias_config.yaml",
        grids_base=GRIDS_BASE,
        num=NUM,
        sims_type=SIMS_TYPE,
        cat_name=f"shape_catalog_cut_{SHAPE}.fits",
        sif=SIF,
        sif_pipeline=SIF_PIPELINE,
        shapepipe_repo=SHAPEPIPE_REPO,
        sp_validation_repo=SPV_REPO,
        # Science knobs, read bare from the run config (no default here).
        match_radius_deg=IMSIM["match_radius_deg"],
        w_cols=IMSIM["w_cols"],
        n_bootstrap=IMSIM["n_bootstrap"],
        pair_match=IMSIM["pair_match"],
        bootstrap_seed=IMSIM["bootstrap_seed"],
    run:
        import hashlib
        import re
        import subprocess

        import yaml

        with open(input.manifest) as fh:
            manifest = yaml.safe_load(fh)

        def _git(repo, *args):
            """Read a git fact from ``repo``; ``None`` if it is not a checkout."""
            try:
                return subprocess.run(
                    ["git", "-C", repo, *args],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                return None

        def _sif_revision(sif_path):
            """GHCR revision baked into the SIF's OCI labels.

            A plain-text scan of the image file (login-safe: no exec, no
            container start), reading org.opencontainers.image.revision -- the
            source commit GHCR built the image from.  ``None`` if absent.
            """
            try:
                with open(sif_path, "rb") as fh:
                    blob = fh.read()
            except OSError:
                return None
            m = re.search(
                rb'org\.opencontainers\.image\.revision"?[:=]"?([0-9a-f]{7,40})',
                blob,
            )
            return m.group(1).decode() if m else None

        # Manifest hash: sha256 of the exact bytes im_manifest wrote, so the
        # result records which injected-shear facts it was computed against.
        with open(input.manifest, "rb") as fh:
            manifest_sha256 = hashlib.sha256(fh.read()).hexdigest()

        provenance = {
            "manifest_sha256": manifest_sha256,
            "sp_validation": {
                "branch": _git(params.sp_validation_repo, "rev-parse", "--abbrev-ref", "HEAD"),
                "commit": _git(params.sp_validation_repo, "rev-parse", "HEAD"),
            },
            "shapepipe": {
                "branch": _git(params.shapepipe_repo, "rev-parse", "--abbrev-ref", "HEAD"),
                "commit": _git(params.shapepipe_repo, "rev-parse", "HEAD"),
            },
            "containers": {
                "sif": params.sif,
                "ghcr_revision": _sif_revision(params.sif),
                "sif_pipeline": params.sif_pipeline,
                "ghcr_revision_pipeline": _sif_revision(params.sif_pipeline),
            },
        }

        os.makedirs(os.path.dirname(output.results), exist_ok=True)
        # Emit *every* key the estimator requires -- pair_match and
        # bootstrap_seed included.  Requiring a key without emitting it would
        # be a KeyError at run time, so the generated config is the complete
        # contract between rule and estimator.  ``provenance`` rides along as a
        # top-level block: the compute script copies it verbatim into the output
        # results yaml, so a result file is self-describing (which manifest,
        # which repo commits, which container built the number).
        mbias_cfg = {
            "grids_dir": params.grids_base,
            "num": params.num,
            "sims_type": params.sims_type,
            "catalog_name": params.cat_name,
            # Injected shear: from the manifest, the single source of truth.
            "shear_amplitude": manifest["shear_amplitude"],
            "branches": list(manifest["branches"]),
            "pairs": manifest["pairs"],
            "match_radius_deg": params.match_radius_deg,
            "w_cols": list(params.w_cols),
            "pair_match": params.pair_match,
            "n_bootstrap": params.n_bootstrap,
            "bootstrap_seed": params.bootstrap_seed,
            "results_dir": os.path.dirname(output.results),
            "output_path": output.results,
            "provenance": provenance,
        }
        with open(params.cfg, "w") as fh:
            yaml.safe_dump(mbias_cfg, fh)
        shell(
            "{EXEC} python {COMPUTE_M_BIAS} -c {params.cfg} -v"
        )
