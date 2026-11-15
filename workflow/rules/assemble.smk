"""The assemble stage: the v2 HATS collection.

One rule runs a collection recipe (``config/assembly/``) through
``sp_validation.assembly.hats_collection.build``. ``config["assemble"]``
holds ``recipe``, the recipe file relative to the repository root.
"""

import common

_REPO = common.REPO_ROOT
ASSEMBLE = config["assemble"]
_unknown = set(ASSEMBLE) - {"recipe"}
if _unknown:
    raise ValueError(f"unknown assemble keys: {sorted(_unknown)}")
ASSEMBLE_RECIPE = _REPO / ASSEMBLE["recipe"]
_recipe = common._data_roots.load_yaml(ASSEMBLE_RECIPE)


rule collection:
    """Write the shapes with OBJECT_ID, the PhotoPipe and spec-z extensions and
    the OBJECT_ID index, keeping the parts already on disk.

    A part on disk is never rewritten from changed inputs, so the inputs only
    have to exist (``ancient``): to rebuild from a new extract or spec-z match,
    remove ``photopipe/`` and ``specz/`` (or ``shapes/``) and the collection's
    ``collection.properties`` first."""
    input:
        recipe=ASSEMBLE_RECIPE,
        shapes=ancient(f"{_recipe['shapes']}/hats.properties"),
        extract=ancient(f"{_recipe['photopipe_extract']}/COMPLETE"),
        matches=ancient(_recipe["specz_matches"]),
        compilation=ancient(_recipe["specz_compilation"]),
    output:
        # The last file the build writes, and the only one declared: Snakemake
        # removes a job's declared outputs before it runs, and the parts must
        # never be removed that way.
        f"{_recipe['collection']}/collection.properties",
    params:
        script=_REPO / "scripts/assembly/build_collection.py",
    threads: workflow.cores
    resources:
        # About 5 GiB per worker for the PhotoPipe join, the largest part.
        mem_mb=lambda wildcards, threads: 20000 + 5000 * threads,
        # About ten minutes at 60 workers.
        runtime=60,
    shell:
        "python {params.script} {input.recipe} --workers {threads}"
