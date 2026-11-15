"""Image simulations -> shear m/c bias, from ShapePipe's merged catalogues.

ShapePipe's own workflow processes each simulation (one shear branch, one
campaign: ``sp run`` with ``input_type: image_sims``) into its merged
catalogue ``final_cat_<sim>.hdf5``; ``catalogues`` names where. This chain
reads those catalogues and nothing else of ShapePipe's: it extracts,
calibrates and measures the multiplicative/additive shear bias
(``manifest``, ``extract``, ``calibrate``, ``m_bias``, ``measure``).

One image runs the whole chain. Everything is parameterised under
``config["image_sims"]``; a schema check at load rejects an unknown key (typo)
and a missing science key (see ``workflow/image_sims/config.yaml`` for the
operational/science split).

The five simulations per grid are the reference ``1z2z`` (no input shear) plus
the ``+/-`` shear pairs ``1p2z``/``1m2z`` (g1) and ``1z2p``/``1z2m`` (g2). The
calibration step leaves each simulation's per-tile response sums beside its
cut catalogue; the m-bias step weights every simulation with one table
(``weight_table``: a data DES-table product, ``self`` or ``unit``) and deletes
each tile from all simulations at once for the covariance. ``im_measure``
repeats m under every weighting (the DES product, unit, inverse variance) and
in S/N bins, and records the sample's response and (S/N, size) distribution.
"""

import os

import yaml

IMSIM = config["image_sims"]

# --- fail-fast schema check ----------------------------------------------
# Science keys: required from the *run* config; no default here or in
# config.yaml (only a commented template line). They fix the estimator's
# scientific behaviour, so they must be stated per run, never inherited.
_SCIENCE_KEYS = {
    "weight_table",
    "des_min_count",
    "mask_config",
}
# Operational keys: default in the workflow config.yaml; the .smk reads them
# bare (never ``.get`` with a literal), so config.yaml is their one home.
_OPERATIONAL_KEYS = {
    "sims_type",
    "branches",
    "shape",
    "snr_edges",
}
# Structural keys: paths/identifiers the run must supply (no sensible default).
_STRUCTURAL_KEYS = {
    "sif",
    "sp_validation_repo",
    "catalogues",
    "input_sims_base",
    "grids_base",
    "num",
}
_ALLOWED_KEYS = _SCIENCE_KEYS | _OPERATIONAL_KEYS | _STRUCTURAL_KEYS

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
# workflow/Snakefile, whose module default may differ. Binds come from the
# driving profile's ``apptainer-args``. A null ``sif`` resolves to the
# workflow's one image (see workflow/image_sims/config.yaml).
SIF = common.resolve_container(IMSIM["sif"])

# --- repository (bound into the image; branch code overrides) -------------
SPV_REPO = IMSIM["sp_validation_repo"]

# --- data and run directories --------------------------------------------
# ShapePipe's merged catalogue of each simulation: a path with ``{sim}``.
CATALOGUES = IMSIM["catalogues"]
# The simulations themselves: <input_sims_base>/<sim>/{basic_info.txt,tiles.txt}.
INPUT_SIMS_BASE = IMSIM["input_sims_base"]
GRIDS_BASE = IMSIM["grids_base"]  # run/output root; one sub-dir per sim

# --- simulation grid ------------------------------------------------------
# SIM_BASES is the set of branches *this run requests* -- the reference plus the
# four +/- sheared branches.  Their injected shear (amplitude, per-branch
# (g1,g2), pairing) is NOT a literal here: it lives only in manifest.yaml, built
# by im_manifest from each branch's basic_info.txt and read back downstream.
NUM = IMSIM["num"]
SIMS_TYPE = IMSIM["sims_type"]
_SUFFIX = f"_{SIMS_TYPE}_{NUM}" if SIMS_TYPE == "grid" else f"_{NUM}"
SIM_BASES = list(IMSIM["branches"])
SIMS = [f"{base}{_SUFFIX}" for base in SIM_BASES]
MANIFEST = f"{GRIDS_BASE}/manifest.yaml"
BUILD_MANIFEST = f"{SPV_REPO}/workflow/scripts/im_build_manifest.py"

# --- calibration / m-bias knobs ------------------------------------------
SHAPE = IMSIM["shape"]
MASK_CONFIG = IMSIM["mask_config"]  # e.g. config/calibration/mask_v2.0_im_sim.yaml
# The table the m-bias step weights every simulation with: a DES weight-table
# product (path), ``self`` or ``unit``. A product is also the table the
# calibration step weights each simulation's catalogue with.
WEIGHT_TABLE = IMSIM["weight_table"]
PARAMS_TEMPLATE = f"{SPV_REPO}/workflow/image_sims/params_im_sim.py"

# The steps run from the sp_validation *repo* checkout (bind-mounted), not the
# copies baked into the image, so the branch's code runs without a rebuild.
EXTRACT_INFO = f"{SPV_REPO}/scripts/calibration/extract_info.py"
CALIBRATE = f"{SPV_REPO}/scripts/calibration/calibrate_comprehensive_cat.py"
COMPUTE_M_BIAS = f"{SPV_REPO}/scripts/compute_m_bias_image_sims.py"
MEASURE = f"{SPV_REPO}/workflow/scripts/im_measure.py"

# --- in-command env prefix -------------------------------------------------
# Snakemake wraps each rule's whole ``shell:`` string inside the container, so
# these tokens land inside it: the checkout's ``src`` ahead of the image's
# copy, and a one-thread OpenMP/BLAS pool (jobs run side by side).
_ENV_PREFIX = f"PYTHONPATH={SPV_REPO}/src OMP_NUM_THREADS=1 "


wildcard_constraints:
    sim="|".join(SIMS),


# ==========================================================================
# Convenience targets (run in order)
# ==========================================================================
rule im_manifest_only:
    input:
        MANIFEST,


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


rule im_results:
    """Everything: m under the configured table, and ``im_measure``'s record."""
    input:
        f"{GRIDS_BASE}/results/m_bias_results.yaml",
        f"{GRIDS_BASE}/results/measurements.yaml",


# ==========================================================================
# Rules
# ==========================================================================
rule im_manifest:
    """Build the campaign manifest at the head of the DAG.

    Parses ``g_cosmic`` from every requested branch's ``basic_info.txt``,
    cross-checks each against its ``1{X}2{Y}`` name and the (0,0) reference,
    derives the single injected amplitude, and writes ``manifest.yaml`` into the
    run root.  This is the one home for the injected-shear facts; the m steps
    read the amplitude and branch map from here, nowhere else.
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
    """Stage a simulation's run directory: params.py, the mask config, and
    links to its merged catalogue and tile list.

    ``params_im_sim.py`` derives the campaign name from the directory basename,
    so one template serves every sim; ``config_mask.yaml`` (the mask config
    naming the ``weight_table`` product, when one is configured) is what the
    calibration step reads from cwd.
    """
    input:
        # Tracked so that editing the params template or mask config re-stages
        # them into every run dir.
        template=PARAMS_TEMPLATE,
        mask_src=os.path.join(SPV_REPO, MASK_CONFIG),
        table=[] if WEIGHT_TABLE in ("self", "unit") else [WEIGHT_TABLE],
        cat=CATALOGUES,
        tiles=f"{INPUT_SIMS_BASE}/{{sim}}/tiles.txt",
    output:
        params=f"{GRIDS_BASE}/{{sim}}/params.py",
        mask=f"{GRIDS_BASE}/{{sim}}/config_mask.yaml",
        cat=f"{GRIDS_BASE}/{{sim}}/final_cat_{{sim}}.hdf5",
        tiles=f"{GRIDS_BASE}/{{sim}}/tiles_{{sim}}.txt",
    run:
        with open(input.mask_src) as f:
            mask = yaml.safe_load(f)
        if input.table:
            mask["metacal"]["weight_table"] = input.table[0]
        with open(output.mask, "w") as f:
            yaml.safe_dump(mask, f)
        shell(
            "cp {input.template} {output.params} && "
            "ln -sfT {input.cat} {output.cat} && "
            "ln -sfT {input.tiles} {output.tiles}"
        )


rule im_extract:
    """Extract the comprehensive ngmix catalogue (sp_validation stage).

    ``extract_info.py`` reads ``params.py`` from cwd and the merged catalogue,
    writing ``shape_catalog_comprehensive_{shape}``.
    """
    input:
        cat=f"{GRIDS_BASE}/{{sim}}/final_cat_{{sim}}.hdf5",
        tiles=f"{GRIDS_BASE}/{{sim}}/tiles_{{sim}}.txt",
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
    ``shape_catalog_cut_{shape}.fits`` with its response record and per-tile
    response sums (``_sums.npz``) beside it.
    """
    input:
        cat=f"{GRIDS_BASE}/{{sim}}/shape_catalog_comprehensive_{SHAPE}.fits",
        mask=f"{GRIDS_BASE}/{{sim}}/config_mask.yaml",
    output:
        cat=f"{GRIDS_BASE}/{{sim}}/shape_catalog_cut_{SHAPE}.fits",
        sums=f"{GRIDS_BASE}/{{sim}}/shape_catalog_cut_{SHAPE}_sums.npz",
    params:
        run_dir=lambda wc: f"{GRIDS_BASE}/{wc.sim}",
    container:
        SIF
    shell:
        "cd {params.run_dir} && "
        "{_ENV_PREFIX} python {CALIBRATE}"


rule im_mbias_config:
    """Assemble ``m_bias_config.yaml`` for the m-bias step: the manifest's
    shear/branch facts, this run's science knobs, and provenance (the
    sp_validation commit, ShapePipe's code stamp on each catalogue, the image).
    """
    input:
        manifest=MANIFEST,
        sums=expand(
            f"{GRIDS_BASE}/{{sim}}/shape_catalog_cut_{SHAPE}_sums.npz", sim=SIMS
        ),
        cats=expand(f"{GRIDS_BASE}/{{sim}}/final_cat_{{sim}}.hdf5", sim=SIMS),
    output:
        cfg=f"{GRIDS_BASE}/results/m_bias_config.yaml",
    params:
        grids_base=GRIDS_BASE,
        num=NUM,
        sums_name=f"shape_catalog_cut_{SHAPE}_sums.npz",
        sims_type=SIMS_TYPE,
        sif=SIF,
        sp_validation_repo=SPV_REPO,
        results=f"{GRIDS_BASE}/results/m_bias_results.yaml",
        # Science knobs, read bare from the run config (no default here).
        results_dir=f"{GRIDS_BASE}/results",
        weight_table=WEIGHT_TABLE,
        des_min_count=IMSIM["des_min_count"],
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


rule im_measure:
    """m and c under every weighting and in S/N bins; each simulation's
    response; the sample's (S/N, size) distribution and per-cell response.

    Reads every simulation's comprehensive catalogue with its run directory's
    ``config_mask.yaml`` (the preselection, the metacal cuts and the DES table
    product), so its DES-weighted m is the one ``im_mbias`` reports.
    """
    input:
        manifest=MANIFEST,
        cats=expand(
            f"{GRIDS_BASE}/{{sim}}/shape_catalog_comprehensive_{SHAPE}.fits",
            sim=SIMS,
        ),
        masks=expand(f"{GRIDS_BASE}/{{sim}}/config_mask.yaml", sim=SIMS),
    output:
        results=f"{GRIDS_BASE}/results/measurements.yaml",
        distributions=f"{GRIDS_BASE}/results/sim_distributions.npz",
    params:
        snr_edges=" ".join(str(e) for e in IMSIM["snr_edges"]),
    resources:
        mem_mb=32000,
    container:
        SIF
    shell:
        "{_ENV_PREFIX} python {MEASURE} --manifest {input.manifest} "
        "--catalogues {input.cats} --masks {input.masks} "
        "--snr-edges {params.snr_edges} "
        "-o {output.results} --distributions {output.distributions}"
