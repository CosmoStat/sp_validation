#!/usr/bin/env python
# /// script
# dependencies = ["hats-import==0.11.1", "pyyaml"]
# ///
"""Split the indexed collection of step 4 into shapes and PhotoPipe (collection, step 5).

Runs ``hats_import``'s extension split over ``WORK/merged`` and writes the
collection ``WORK/<name>``:

- ``shapes/``, the primary catalogue: the columns of the shape catalogue
  ``--shapes`` and ``OBJECT_ID``;
- ``shapes_OBJECT_ID/``, the ``OBJECT_ID`` index, the default index;
- ``photopipe/``, the extension: every other column, with ``OBJECT_ID``,
  ``_healpix_29`` and the coordinates, described by ``photopipe.properties`` as
  joined to the primary on ``OBJECT_ID``;
- ``collection.properties``.

The split keeps each pixel file's rows in order, so the same pixel of
``shapes`` and ``photopipe`` holds the same objects in the same order. The
container lacks hats-import, so this runs with ``uv run``, with ``src/`` on
``PYTHONPATH`` for the data roots.
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from hats_import.extension.arguments import ExtensionArguments
from hats_import.pipeline import pipeline

from sp_validation.data_roots import expand

SHAPES = "${unions_data}/v2.x/run1-snap3-hats/catalogue"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path, help="work directory of steps 3 and 4")
    parser.add_argument("--name", default="comprehensive", help="collection name")
    parser.add_argument(
        "--shapes",
        default=SHAPES,
        help="shape HATS catalogue naming the primary's columns (default: %(default)s)",
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    merged = args.work / "merged"
    shape_columns = set(
        pq.read_schema(f"{expand(args.shapes)}/dataset/_common_metadata").names
    )
    columns = pq.read_schema(merged / "shapes/dataset/_common_metadata").names
    split = ExtensionArguments(
        input_catalog_path=merged,
        extension_columns=[
            c for c in columns if c not in shape_columns | {"OBJECT_ID"}
        ],
        primary_column="OBJECT_ID",
        extension_name="photopipe",
        product_type_served="photometry",
        output_path=args.work,
        output_artifact_name=args.name,
        create_metadata=False,
        dask_n_workers=args.workers,
        progress_bar=False,
    )
    pipeline(split)


if __name__ == "__main__":
    main()
