#!/usr/bin/env python
"""Write a v2 HATS catalogue with the DR6 object ID column (collection, step 3).

Rewrites each pixel file of ``SOURCE`` under ``OUTPUT`` with ``OBJECT_ID``
inserted after ``NUMBER`` (``sp_validation.object_id``), keeping rows, row
groups and metadata, and names the result ``shapes``: the primary catalogue of
the collection steps 4 and 5 build. ``SOURCE`` is the catalogue holding the
shape and PhotoPipe columns. Paths may name data roots
(``config/data_roots.yaml``), as in ``${unions_data}/...``.
"""

import argparse

from sp_validation.data_roots import expand
from sp_validation.object_id import add_to_catalogue

SOURCE = "${unions_data}/v2.x/v2.0.0/catalogs/comprehensive"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", help="work directory; writes WORK/merged/shapes")
    parser.add_argument(
        "--source", default=SOURCE, help="HATS catalogue (default: %(default)s)"
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    output = f"{expand(args.work)}/merged/shapes"
    add_to_catalogue(expand(args.source), output, args.workers, name="shapes")


if __name__ == "__main__":
    main()
