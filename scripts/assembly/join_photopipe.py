#!/usr/bin/env python
"""Give every shape row its PhotoPipe row (PhotoPipe join, step 2).

Reads the shape HATS catalogue and the column extract from step 1, and writes
the PhotoPipe extension of the collection: the catalogue ``OUTPUT/photopipe``
and its provenance beside it (``sp_validation.photopipe_merge``).
Paths may name data roots (``config/data_roots.yaml``), as in ``${unions_data}/...``.
"""

import argparse

from sp_validation.data_roots import expand
from sp_validation.photopipe_merge import merge

SHAPES = "${unions_data}/v2.x/run1-snap3-hats/catalogue"
OUTPUT = "${unions_data}/v2.x/run1-snap3-collection/photopipe"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "extract", help="directory of ra=*.parquet shards and manifests from step 1"
    )
    parser.add_argument(
        "--shapes", default=SHAPES, help="shape HATS catalogue (default: %(default)s)"
    )
    parser.add_argument(
        "--output", default=OUTPUT, help="product directory (default: %(default)s)"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--pixels", type=int, nargs="*", help="restrict to these pixels"
    )
    args = parser.parse_args()
    merge(
        expand(args.shapes),
        args.extract,
        expand(args.output),
        args.workers,
        args.pixels,
    )


if __name__ == "__main__":
    main()
