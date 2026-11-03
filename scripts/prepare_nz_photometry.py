#!/usr/bin/env python
"""Prepare the DR6 positional photometry cache used by SOM n(z)."""

import argparse
from pathlib import Path

import yaml

from sp_validation.nz_photometry import prepare_nz_photometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", help="YAML recipe containing a photometry section")
    args = parser.parse_args()
    photometry = yaml.safe_load(Path(args.recipe).read_text())["photometry"]
    selection, dr6_root, output = (
        photometry[key] for key in ("selection", "dr6_root", "output")
    )
    prepare_nz_photometry(selection, dr6_root, output)


if __name__ == "__main__":
    main()
