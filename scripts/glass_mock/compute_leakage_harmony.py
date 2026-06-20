"""CLI: harmonic-space PSF-leakage rho_0 / tau_0 for one GLASS mock.

Thin wrapper around ``sp_validation.glass_mock.compute_leakage_harmony``.
"""

import argparse

import numpy as np
from astropy.io import fits

from sp_validation.glass_mock import compute_leakage_harmony


def get_parser():
    """Create the parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Selects the simulation on which to run the leakage stats.",
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
    parser.add_argument(
        "-s",
        "--star_cat_path",
        help="Path to the star catalog data",
        type=str,
        default="/n17data/UNIONS/WL/v1.4.x/unions_shapepipe_psf_2024_v1.4.a.fits",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    number = str(args.number).zfill(5)

    print("Loading the catalog data...")
    cat = fits.getdata(f"{args.path}/unions_glass_sim_{number}_{args.nside}.fits")
    print("Catalog data loaded successfully.")

    print("Loading the star catalog data...")
    cat_star = fits.getdata(f"{args.star_cat_path}")
    print("Star catalog data loaded successfully.")

    print("Computing leakage in harmonic space...")
    rho_cl, tau_cl = compute_leakage_harmony(cat, cat_star)
    print("Leakage computation completed.")

    np.save(f"{args.path}/rho_cl_glass_mock_{number}_{args.nside}.npy", rho_cl)
    np.save(f"{args.path}/tau_cl_glass_mock_{number}_{args.nside}.npy", tau_cl)

    print(f"Leakage results computed and saved for mock {args.number}.")
