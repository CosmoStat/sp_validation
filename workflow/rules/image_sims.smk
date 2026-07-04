"""Image-simulation orchestration: raw SKiLLS sim images -> shear m/c bias.

This rule set drives the image-simulation validation chain end to end and is
the sp_validation-side half of the split described in
``UNIONS-WL/MultiBand_ImSim#1``: ShapePipe (in *its* container) turns the
simulated tiles into per-tile shape catalogues, then sp_validation (in *its*
container) merges, extracts, calibrates and finally measures the
multiplicative/additive shear bias.

Two containers are wired through one workflow.  Every rule sets
``container: None`` and calls ``apptainer exec`` explicitly, because neither
image is the workflow's top-level container and the two halves run in
different images:

* ShapePipe container  -> ``pipeline`` (raw images -> per-tile cats) and
  ``merge`` (``create_final_cat`` -> ``final_cat_{sim}.hdf5``).
* sp_validation container -> ``extract`` (-> comprehensive cat),
  ``calibrate`` (-> cut cat) and ``m_bias`` (-> ``m_bias_results.yaml``).

Everything is parameterised under ``config["image_sims"]`` -- container paths,
repository roots, data roots, the PSF dictionary, tile list and sim/calibration
knobs -- so a fresh user drives it from config alone, with no hard-coded clone
layout.  The ``PYTHONPATH`` override on the sp_validation exec makes the
*branch* source (``image_sims.py``, ``catalog.match_catalogs_radec``) win over
whatever is baked into the image.

The five simulations per grid are the reference ``1z2z`` (no input shear) plus
the ``+/-`` shear pairs ``1p2z``/``1m2z`` (g1) and ``1z2p``/``1z2m`` (g2); the
m-bias estimator matches each to the reference by RA/Dec.
"""

import os
from pathlib import Path

IMSIM = config["image_sims"]

# --- containers -----------------------------------------------------------
SHAPEPIPE_SIF = IMSIM["shapepipe_sif"]
SPV_SIF = IMSIM["sp_validation_sif"]
BINDS = IMSIM.get("binds", "/n17data,/n09data,/home,/automnt")

# --- repositories (bound into the images; branch code overrides) ----------
SHAPEPIPE_REPO = IMSIM["shapepipe_repo"]
SPV_REPO = IMSIM["sp_validation_repo"]

# --- data and run directories --------------------------------------------
GRIDS_BASE = IMSIM["grids_base"]  # run/output root; one sub-dir per sim
INPUT_SIMS_BASE = IMSIM["input_sims_base"]  # SKiLLS sim images
PSF_DICT = IMSIM["psf_dict"]  # Herve's Full_psf_dict.pickle

# --- simulation grid ------------------------------------------------------
NUM = IMSIM["num"]
SIMS_TYPE = IMSIM.get("sims_type", "grid")
_SUFFIX = f"_{SIMS_TYPE}_{NUM}" if SIMS_TYPE == "grid" else f"_{NUM}"
SIM_BASES = ["1z2z", "1p2z", "1m2z", "1z2p", "1z2m"]
SIMS = [f"{base}{_SUFFIX}" for base in SIM_BASES]

# --- tiles ----------------------------------------------------------------
if IMSIM.get("tile_ids"):
    TILE_IDS = list(IMSIM["tile_ids"])
else:
    with open(IMSIM["tile_ids_file"]) as fh:
        TILE_IDS = [line.strip() for line in fh if line.strip()]

# --- calibration / m-bias knobs ------------------------------------------
SHAPE = IMSIM.get("shape", "ngmix")
MASK_CONFIG = IMSIM["mask_config"]  # e.g. config/calibration/mask_v1.X.9_im_sim.yaml
PARAMS_TEMPLATE = f"{SPV_REPO}/workflow/image_sims/params_im_sim.py"
# ShapePipe cfis_image_sims config dir (per-tile/exposure configs + final_cat.param).
CONFIG_DIR = IMSIM.get(
    "config_dir", f"{SHAPEPIPE_REPO}/example/cfis_image_sims"
)

# ShapePipe scripts live in the ShapePipe repo (also baked into its image).
CREATE_FINAL_CAT = f"{SHAPEPIPE_REPO}/scripts/python/create_final_cat.py"
RUN_JOB = f"{SHAPEPIPE_REPO}/scripts/sh/run_job_sp_canfar_v2.0.bash"
# Extract/calibrate run from the sp_validation *repo* checkout (bind-mounted),
# not the baked copies: the container tracks the branch but lags it, and the
# image-sims path needs branch-only fixes (star-catalogue-optional extract,
# FITS-aware CalibrateCat.read_cat). Overridable for a different checkout.
EXTRACT_INFO = IMSIM.get(
    "extract_script", f"{SPV_REPO}/scripts/calibration/extract_info.py"
)
CALIBRATE = IMSIM.get(
    "calibrate_script", f"{SPV_REPO}/scripts/calibration/calibrate_comprehensive_cat.py"
)
# m-bias is *this branch's* extracted core, injected on PYTHONPATH.
COMPUTE_M_BIAS = f"{SPV_REPO}/scripts/compute_m_bias_image_sims.py"

# --- container exec prefixes ---------------------------------------------
# ShapePipe stages. MPI/SLURM env vars are stripped so OpenMPI inside the
# image does not try to attach to the host launcher (cf. apptainer_noslurm.sh).
SP_EXEC = f"env -u SLURM_JOBID -u SLURM_JOB_ID -u SLURM_PROCID apptainer exec --bind {BINDS} {SHAPEPIPE_SIF}"
# sp_validation calibration stages: inject the branch source on PYTHONPATH so
# the repo's sp_validation package (newer than the baked one) wins -- the
# image-sims path depends on branch-only fixes to catalog_builders/extract.
SPV_EXEC = f"apptainer exec --bind {BINDS} --env PYTHONPATH={SPV_REPO}/src {SPV_SIF}"
# m-bias stage uses the same injected environment.
SPV_EXEC_MBIAS = SPV_EXEC

JOB_MASK = sum([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])


wildcard_constraints:
    sim="|".join(SIMS),
    tile="|".join(t.replace(".", r"\.") for t in TILE_IDS),


# ==========================================================================
# Convenience targets (run in order)
# ==========================================================================
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
    output:
        params=f"{GRIDS_BASE}/{{sim}}/params.py",
        mask=f"{GRIDS_BASE}/{{sim}}/config_mask.yaml",
    params:
        template=PARAMS_TEMPLATE,
        mask_src=lambda wc: os.path.join(SPV_REPO, MASK_CONFIG),
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
        "cp {params.template} {output.params} && "
        "ln -sf {params.mask_src} {output.mask} && "
        "ln -sfT {params.config_dir} {params.cfis} && "
        "ln -sfT {params.sim_tiles} {params.run_dir}/input_tiles && "
        "ln -sfT {params.sim_exp} {params.run_dir}/input_exp"


rule im_pipeline:
    """Run ShapePipe on one simulated tile (ShapePipe container).

    Delegates the module DAG to ShapePipe's own job runner; the sentinel log
    marks tile completion for the merge step.  This is the compute-heavy,
    MPI-bearing stage.
    """
    input:
        params=f"{GRIDS_BASE}/{{sim}}/params.py",
        cfis=f"{GRIDS_BASE}/{{sim}}/cfis",
    output:
        done=touch(f"{GRIDS_BASE}/{{sim}}/logs/pipeline_{{tile}}.done"),
    params:
        run_dir=lambda wc: f"{GRIDS_BASE}/{wc.sim}",
        psf=IMSIM.get("psf_model", "psfex"),
        n_smp=IMSIM.get("n_smp", -1),
    resources:
        mem_mb=16000,
        runtime=720,
    shell:
        "cd {params.run_dir} && "
        "{SP_EXEC} bash {RUN_JOB} "
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
    shell:
        "cd {params.run_dir} && "
        "{SP_EXEC} python {CREATE_FINAL_CAT} "
        "-I -m final_cat_{wildcards.sim}.hdf5 -i .. "
        "-p cfis/final_cat.param -P {wildcards.sim} "
        "-o n_tiles_final.txt -v"


rule im_extract:
    """Extract the comprehensive ngmix catalogue (sp_validation container).

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
        "cd {params.run_dir} && {SPV_EXEC} python {EXTRACT_INFO}"


rule im_calibrate:
    """Calibrate and cut the comprehensive catalogue (sp_validation container).

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
        "{SPV_EXEC} python {CALIBRATE} -s calibrate"


rule im_mbias:
    """Multiplicative/additive shear bias from the five calibrated grids.

    Produces the workflow's headline artifact, ``m_bias_results.yaml``.
    """
    input:
        cats=expand(
            f"{GRIDS_BASE}/{{sim}}/shape_catalog_cut_{SHAPE}.fits", sim=SIMS
        ),
    output:
        results=f"{GRIDS_BASE}/results/m_bias_results.yaml",
    params:
        cfg=f"{GRIDS_BASE}/results/m_bias_config.yaml",
        grids_base=GRIDS_BASE,
        num=NUM,
        cat_name=f"shape_catalog_cut_{SHAPE}.fits",
        shear_amplitude=IMSIM.get("shear_amplitude", 0.02),
        match_radius_deg=IMSIM.get("match_radius_deg", 0.0002),
        w_col=IMSIM.get("w_col", "w_des"),
        n_bootstrap=IMSIM.get("n_bootstrap", 500),
    run:
        import yaml

        os.makedirs(os.path.dirname(output.results), exist_ok=True)
        mbias_cfg = {
            "grids_dir": params.grids_base,
            "num": params.num,
            "catalog_name": params.cat_name,
            "shear_amplitude": params.shear_amplitude,
            "match_radius_deg": params.match_radius_deg,
            "w_col": params.w_col,
            "n_bootstrap": params.n_bootstrap,
            "results_dir": os.path.dirname(output.results),
            "output_path": output.results,
        }
        with open(params.cfg, "w") as fh:
            yaml.safe_dump(mbias_cfg, fh)
        shell(
            "{SPV_EXEC_MBIAS} python {COMPUTE_M_BIAS} -c {params.cfg} -v"
        )
