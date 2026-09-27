"""Draw a base catalogue's jackknife patch centres into an output tree, once.

    python -m sp_validation.cosmo_val.patch_centers <catalogue> <npatch> \\
        --cat-config cosmo_val/cat_config.yaml --output-dir <COSMO_VAL>

writes ``<COSMO_VAL>/patches/<catalogue>_npatch=<npatch>.dat``
(:meth:`CosmologyValidation.write_patch_centers`). It reads the whole
catalogue, so run it on a compute node.
"""

import argparse

from .core import CosmologyValidation


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m sp_validation.cosmo_val.patch_centers",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument("catalogue", help="a base catalogue of the catalogue config")
    parser.add_argument("npatch", type=int)
    parser.add_argument("--cat-config", required=True)
    parser.add_argument("--output-dir", required=True, help="the output tree")
    a = parser.parse_args(argv)
    cv = CosmologyValidation(
        versions=[a.catalogue], catalog_config=a.cat_config, output_dir=a.output_dir
    )
    print(cv.write_patch_centers(a.catalogue, a.npatch))


if __name__ == "__main__":
    main()
