#!/usr/bin/env python
# /// script
# dependencies = ["hats-import==0.11.1"]
# ///
"""Index the catalogue of step 3 on OBJECT_ID (collection, step 4).

Runs ``hats_import`` over ``WORK/merged/shapes``: writes the index catalogue
``WORK/merged/shapes_OBJECT_ID``, which maps each ``OBJECT_ID`` to its pixel,
and ``WORK/merged/collection.properties``, which makes ``WORK/merged`` a HATS
collection with that index as its default. The container lacks hats-import, so
this runs with ``uv run``.
"""

import argparse
from pathlib import Path

from hats.catalog import CollectionProperties
from hats_import.collection.arguments import CollectionArguments
from hats_import.pipeline import pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path, help="work directory of step 3")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    collection = (
        CollectionArguments(
            output_path=args.work,
            output_artifact_name="merged",
            dask_n_workers=args.workers,
            progress_bar=False,
        )
        .catalog(output_artifact_name="shapes")
        .add_index(
            indexing_column="OBJECT_ID",
            output_artifact_name="shapes_OBJECT_ID",
            drop_duplicates=False,
        )
    )
    pipeline(collection)
    merged = args.work / "merged"
    properties = CollectionProperties.read_from_dir(merged)
    properties.model_copy(update=dict(default_index="OBJECT_ID")).to_properties_file(
        merged
    )


if __name__ == "__main__":
    main()
