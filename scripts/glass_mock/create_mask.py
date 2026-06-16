"""CLI: build a HEALPix footprint mask from a GLASS-mock galaxy catalogue.

Thin wrapper around ``sp_validation.glass_mock.create_mask_from_catalogue``.
"""

import argparse

from sp_validation.glass_mock import create_mask_from_catalogue


def get_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Creates a HEALPix footprint mask from a galaxy catalogue",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "-n", "--nside",
        help="Nside for the simulation. Nside=Lmax", type=int, default=32,
    )
    parser.add_argument(
        "-p", "--path",
        help="Path to the galaxy catalog used to generate the mask",
        type=str, default="./",
    )
    parser.add_argument(
        "-o", "--output", help="Output path for the mask", type=str, default="./",
    )
    parser.add_argument(
        "--ra_col", help="Column name for the right ascension",
        type=str, default="RA",
    )
    parser.add_argument(
        "--dec_col", help="Column name for the declination",
        type=str, default="DEC",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print("-" * 66)
    print(f"Creating the mask with resolution nside={args.nside}")
    print(f"Using the catalog in path: {args.path}")
    print(f"Output path: {args.output}")
    print("-" * 66)

    create_mask_from_catalogue(
        args.nside, args.path, args.output, args.ra_col, args.dec_col
    )
