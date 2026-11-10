#!/usr/bin/env python
"""Merge the PhotoPipe extract onto the shape HATS catalogue."""

import argparse

from sp_validation.photopipe_merge import merge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shapes", help="shape HATS catalogue directory")
    parser.add_argument(
        "extract", help="directory of ra=*.parquet shards and manifests"
    )
    parser.add_argument(
        "output", help="product directory; the catalogue goes in catalogue/"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--pixels", type=int, nargs="*", help="restrict to these pixels"
    )
    args = parser.parse_args()
    merge(args.shapes, args.extract, args.output, args.workers, args.pixels)


if __name__ == "__main__":
    main()
