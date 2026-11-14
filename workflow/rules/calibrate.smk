"""The calibrate stage: the calibrated shear sample of a ShapePipe snapshot.

One rule runs a sample recipe (``config/calibration/sample/``) through
``sp_validation.sample.pipeline``, which lists the products; a second compares the sample with a reference catalogue when the config names
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


# @sc [decision:dr6_catalogue.orchestrator]
rule calibrated_sample:
    """Select, weight and calibrate the snapshot's galaxies; write the HATS
    catalogue and the products beside it (sp_validation.sample.pipeline)."""
    input:
        recipe=CALIBRATE_RECIPE,
        selection=_REPO / _recipe["selection"],
        manifest=f"{_recipe['snapshot']}/MANIFEST.json",
    output:
        # The last file the build writes, and the only one declared: Snakemake
        # removes a job's declared outputs before it runs, and the catalogue
        # must never be removed that way. The build itself refuses to write
        # over an existing catalogs/calibrated.
        done=f"{CALIBRATE_OUT}/provenance/calibrated_recipe.yaml",
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
            rules.calibrated_sample.output.done,
        output:
            f"{CALIBRATE_OUT}/provenance/calibrated_comparison.json",
        params:
            script=_REPO / "scripts/calibration/compare_samples.py",
            new=f"{CALIBRATE_OUT}/catalogs/calibrated",
            reference=CALIBRATE["reference"],
        threads: workflow.cores
        resources:
            mem_mb=lambda wildcards, threads: 10000 + 1500 * threads,
            runtime=60,
        shell:
            # Snakemake removes a failed job's output; keep the report.
            "python {params.script} {params.new} {params.reference}"
            " --output {output} --workers {threads}"
            " || {{ cp {output} {output}.failed; exit 1; }}"
