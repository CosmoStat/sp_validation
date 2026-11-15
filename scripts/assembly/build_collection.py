#!/usr/bin/env python
"""Build the v2 HATS collection a recipe describes (config/assembly/v2.yaml).

Writes the shapes with OBJECT_ID, the PhotoPipe and spec-z extensions and the
OBJECT_ID index (``sp_validation.assembly.hats_collection.build``). A part
already on disk is kept.
"""

import argparse

from sp_validation.assembly.hats_collection import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recipe", help="collection recipe, e.g. config/assembly/v2.yaml"
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    build(args.recipe, args.workers)


if __name__ == "__main__":
    main()
