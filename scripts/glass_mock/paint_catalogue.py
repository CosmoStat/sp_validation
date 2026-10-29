"""Generate projected GLASS maps, then paint calibrated catalogue shapes.

Run ``python scripts/glass_mock/paint_catalogue.py {maps,paint} --help``.
Maps use HEALPix components; catalogue e2 is conjugated for cosmo_val's
``pol_factor=-1``. Only mock catalogues may be analysed without a blind.
"""

import argparse
import json

import numpy as np

from sp_validation.glass_mock import (
    GlassMockConfig,
    generate_lensing_maps,
    paint_catalogue,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    maps = commands.add_parser("maps", help="Generate n(z)-averaged full-sky maps")
    maps.add_argument("--nz", required=True)
    maps.add_argument("--output", required=True)
    maps.add_argument("--cache", required=True)
    maps.add_argument("--seed", type=int, required=True)
    maps.add_argument("--nside", type=int, default=4096)
    maps.add_argument(
        "--lmax", type=int, default=None, help="Default 2*nside; must be <=3*nside-1"
    )
    maps.add_argument("--dx", type=float, default=200)
    maps.add_argument("--zmax", type=float, default=3.5)
    maps.add_argument(
        "--exact-matter-cls",
        action="store_true",
        help="Use non-Limber shell spectra (slower)",
    )
    maps.add_argument("--config", help="YAML overrides, including cosmology")
    paint = commands.add_parser("paint", help="Paint a FITS/HDF5 catalogue in chunks")
    paint.add_argument("--input", required=True)
    paint.add_argument("--output", required=True)
    paint.add_argument("--maps", required=True)
    paint.add_argument("--seed", type=int, required=True, help="Rotation RNG seed")
    for key, default in (
        ("ra", "RA"),
        ("dec", "Dec"),
        ("e1", "e1"),
        ("e2", "e2"),
        ("w", "w_des"),
    ):
        paint.add_argument(f"--{key}-col", default=default)
    paint.add_argument("--psf-cols", nargs=2, default=["e1_PSF", "e2_PSF"])
    paint.add_argument("--alpha", type=float, default=0)
    paint.add_argument("--additive", nargs=2, type=float, default=[0, 0])
    paint.add_argument(
        "--b-amplitude",
        type=float,
        default=0,
        help="C_BB=A exp[-(ell/2000)^2], pixelized target",
    )
    paint.add_argument("--b-cl", help="Two-column ell C_BB text file")
    paint.add_argument("--b-seed", type=int)
    paint.add_argument("--chunk-size", type=int, default=250_000)
    paint.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "maps":
        from dataclasses import replace

        config = (
            GlassMockConfig.from_yaml(args.config, seed=args.seed)
            if args.config
            else GlassMockConfig(seed=args.seed)
        )
        config = replace(
            config,
            nside=args.nside,
            harmonic_lmax=args.lmax or 2 * args.nside,
            dx=args.dx,
            zmax=args.zmax,
            nz_path=args.nz,
            limber=not args.exact_matter_cls,
        )
        result = generate_lensing_maps(config, args.output, args.cache)
    else:
        b_cl = None
        if args.b_amplitude or args.b_cl:
            from pathlib import Path

            import healpy as hp

            nside = hp.npix2nside(
                len(np.load(Path(args.maps) / "kappa.npy", mmap_mode="r"))
            )
            ell = np.arange(2 * nside + 1)
            if args.b_cl:
                supplied = np.loadtxt(args.b_cl)
                b_cl = np.interp(ell, supplied[:, 0], supplied[:, 1], left=0, right=0)
            else:
                b_cl = args.b_amplitude * np.exp(-((ell / 2000) ** 2))
            b_cl[:2] = 0
        result = paint_catalogue(
            args.input,
            args.output,
            args.maps,
            args.seed,
            ra_col=args.ra_col,
            dec_col=args.dec_col,
            e1_col=args.e1_col,
            e2_col=args.e2_col,
            w_col=args.w_col,
            psf_cols=args.psf_cols,
            alpha=args.alpha,
            additive=args.additive,
            b_cl=b_cl,
            b_seed=args.b_seed,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
