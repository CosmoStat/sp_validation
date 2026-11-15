#!/usr/bin/env python
"""Give shape objects their spectroscopic redshift (collection, step 4).

Reads PhotoPipe's match of DR6 objects to spectra (one FITS table over every
tile) and the redshift compilation it used, finds each matched object in the
collection's PhotoPipe extension (step 2), and writes the spec-z extension
``COLLECTION/specz`` (``sp_validation.specz_merge``). Paths may name data roots
(``config/data_roots.yaml``), as in ``${unions_data}/...``.
"""

import argparse

from sp_validation.data_roots import expand
from sp_validation.specz_merge import merge

MATCHES = "${unions_data}/photopipe-dr6-specz-all/UNIONS_DR6_ugriz_specz.cat"
COMPILATION = "${unions_data}/photopipe-dr6-specz-all/asc"
COLLECTION = "${unions_data}/v2.x/run1-snap3-collection"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matches", default=MATCHES, help="matched table (default: %(default)s)"
    )
    parser.add_argument(
        "--compilation",
        default=COMPILATION,
        help="directory of UNIONS.<tile>_redshifts-<date>.asc (default: %(default)s)",
    )
    parser.add_argument(
        "--collection", default=COLLECTION, help="collection (default: %(default)s)"
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    merge(
        expand(args.collection),
        expand(args.matches),
        expand(args.compilation),
        args.workers,
    )


if __name__ == "__main__":
    main()
