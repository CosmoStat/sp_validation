#!/usr/bin/env python
"""Prepare the DR6 positional photometry cache used by SOM n(z)."""

import argparse

from sp_validation.data_roots import load_yaml
from sp_validation.nz_photometry import prepare_nz_photometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", help="YAML recipe containing a photometry section")
    args = parser.parse_args()
    photometry = load_yaml(args.recipe)["photometry"]
    selection, dr6_root, output = (
        photometry[key] for key in ("selection", "dr6_root", "output")
    )
    prepare_nz_photometry(selection, dr6_root, output)


if __name__ == "__main__":
    main()
