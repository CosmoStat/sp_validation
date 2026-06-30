"""CLI: two-point statistics (xi, pseudo-Cl, map-Cl) for one GLASS mock.

Thin wrapper around the two-point helpers in ``sp_validation.glass_mock``.
"""

import argparse

import numpy as np
from astropy.io import fits

from sp_validation.glass_mock import (
    compute_two_point_cl,
    compute_two_point_cl_map,
    compute_two_point_xi,
)


def get_parser():
    """Create the parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Selects the simulation on which to run two-point statistics.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("-N", "--number", help="Mock Number", type=int, default=0)
    parser.add_argument(
        "-n",
        "--nside",
        help="Nside for the simulation. Nside=Lmax",
        type=int,
        default=32,
    )
    parser.add_argument(
        "-p",
        "--path",
        help="Path to the simulation data",
        type=str,
        default="/n09data/guerrini/glass_mock_v1.4.6/results/",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    number = str(args.number).zfill(5)

    print("Loading the catalog data...")
    cat = fits.getdata(f"{args.path}/unions_glass_sim_{number}_{args.nside}.fits")
    print("Catalog data loaded successfully.")

    print("Computing two-point statistics in configuration space...")
    xi = compute_two_point_xi(cat)
    print("Two-point statistics computed successfully.")
    xi.write(f"{args.path}/xi_glass_mock_{number}_{args.nside}_nbins=20.fits")

    print("Computing two-point statistics in harmonic space...")
    cl_coupled, cl = compute_two_point_cl(cat)
    print("Two-point statistics in harmonic space computed successfully.")
    np.save(f"{args.path}/cl_glass_mock_{number}_{args.nside}.npy", cl)
    np.save(f"{args.path}/cl_coupled_glass_mock_{number}_{args.nside}.npy", cl_coupled)

    print("Computing two-point statistics in harmonic space (map-based)...")
    cl_coupled_map, cl_map = compute_two_point_cl_map(cat)
    print("Map-based harmonic-space two-point statistics computed successfully.")
    np.save(f"{args.path}/cl_map_glass_mock_{number}_{args.nside}.npy", cl_map)
    np.save(
        f"{args.path}/cl_coupled_map_glass_mock_{number}_{args.nside}.npy",
        cl_coupled_map,
    )

    print(f"Two-point statistics computed and saved for mock {args.number}.")
