#!/usr/bin/env python
"""Index the shapes on OBJECT_ID and add PhotoPipe as an extension (collection, step 4).

Needs ``COLLECTION/shapes`` (step 3) and ``COLLECTION/photopipe`` (step 2), and
writes the ``OBJECT_ID`` index and every properties file of the collection
(``sp_validation.hats_collection``). Paths may name data roots
(``config/data_roots.yaml``), as in ``${unions_data}/...``.
"""

import argparse

from sp_validation.data_roots import expand
from sp_validation.hats_collection import build

COLLECTION = "${unions_data}/v2.x/run1-snap3-collection"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection", default=COLLECTION, help="collection (default: %(default)s)"
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    build(expand(args.collection), args.workers)


if __name__ == "__main__":
    main()
