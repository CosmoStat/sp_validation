"""The calibrate stage: the calibrated shear sample of a ShapePipe snapshot.

One rule runs a sample recipe (``config/calibration/sample/``) through
``sp_validation.sample.pipeline`` and declares every product of the build; a
second compares the sample with a reference catalogue when the config names
one. Everything is parameterised under ``config["calibrate"]``:

- ``recipe``: the recipe file, relative to the repository root;
- ``output``: directory to build under, default the recipe's own ``output``
  (the release directory);
- ``reference``: optional HATS catalogue the sample should reproduce.
"""

import common

_REPO = common.REPO_ROOT
CALIBRATE = config["calibrate"]
_unknown = set(CALIBRATE) - {"recipe", "output", "reference"}
if _unknown:
    raise ValueError(f"unknown calibrate keys: {sorted(_unknown)}")
CALIBRATE_RECIPE = _REPO / CALIBRATE["recipe"]
_recipe = common._data_roots.load_yaml(CALIBRATE_RECIPE)
CALIBRATE_OUT = CALIBRATE.get("output") or _recipe["output"]
_NSIDE = _recipe["footprint_nside"]


# @sc [decision:dr6_catalogue.orchestrator]
rule calibrated_sample:
    """Select, weight and calibrate the snapshot's galaxies; write the HATS
    catalogue and the products beside it (sp_validation.sample.pipeline)."""
    input:
        recipe=CALIBRATE_RECIPE,
        selection=_REPO / _recipe["selection"],
        manifest=f"{_recipe['snapshot']}/MANIFEST.json",
    output:
        catalogue=directory(f"{CALIBRATE_OUT}/catalogs/calibrated"),
        response=f"{CALIBRATE_OUT}/catalogs/response.json",
        weights=f"{CALIBRATE_OUT}/catalogs/des_weights.fits",
        survey=f"{CALIBRATE_OUT}/catalogs/survey.json",
        selection=f"{CALIBRATE_OUT}/catalogs/config_mask.yaml",
        footprint=f"{CALIBRATE_OUT}/masks/footprint-nside{_NSIDE}-ring.fits",
        accounting=f"{CALIBRATE_OUT}/provenance/calibrated_tile_accounting.json",
        recipe=f"{CALIBRATE_OUT}/provenance/calibrated_recipe.yaml",
    params:
        script=_REPO / "scripts/calibration/calibrate_sample.py",
        out=CALIBRATE_OUT,
    threads: workflow.cores
    resources:
        # About 2 GiB per worker at the v2 pixel size, plus the survey pass.
        mem_mb=lambda wildcards, threads: 20000 + 2500 * threads,
        runtime=120,
    shell:
        "python {params.script} {input.recipe} --output {params.out}"
        " --workers {threads}"


if CALIBRATE.get("reference"):

    rule compare_calibrated_sample:
        """Compare the sample with the reference object by object; fails
        when they differ (sp_validation.sample.compare)."""
        input:
            new=rules.calibrated_sample.output.catalogue,
        output:
            f"{CALIBRATE_OUT}/provenance/calibrated_comparison.json",
        params:
            script=_REPO / "scripts/calibration/compare_samples.py",
            reference=CALIBRATE["reference"],
        threads: workflow.cores
        resources:
            mem_mb=lambda wildcards, threads: 10000 + 1500 * threads,
            runtime=60,
        shell:
            "python {params.script} {input.new} {params.reference}"
            " --output {output} --workers {threads}"
