#!/usr/bin/env python
"""Write the shape catalogue into the collection with OBJECT_ID (collection, step 3).

Rewrites each pixel file of the shape HATS catalogue as
``COLLECTION/shapes`` with ``OBJECT_ID`` inserted after ``NUMBER``
(``sp_validation.assembly.object_id``), keeping rows, row groups and metadata. Paths
may name data roots (``config/data_roots.yaml``), as in ``${unions_data}/...``.
"""

import argparse

from sp_validation.assembly.hats_collection import PRIMARY
from sp_validation.assembly.object_id import add_to_catalogue
from sp_validation.data_roots import expand

SHAPES = "${unions_data}/v2.x/run1-snap3-hats/catalogue"
COLLECTION = "${unions_data}/v2.x/run1-snap3-collection"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes", default=SHAPES, help="shape HATS catalogue (default: %(default)s)"
    )
    parser.add_argument(
        "--collection", default=COLLECTION, help="collection (default: %(default)s)"
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    output = f"{expand(args.collection)}/{PRIMARY}"
    add_to_catalogue(expand(args.shapes), output, args.workers, name=PRIMARY)


if __name__ == "__main__":
    main()
