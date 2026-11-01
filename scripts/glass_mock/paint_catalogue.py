"""Generate projected GLASS maps, then paint calibrated catalogue shapes.

Run ``python scripts/glass_mock/paint_catalogue.py {maps,paint} --help``.
The sky is a convergence alm, evaluated exactly at each galaxy;
maps use HEALPix components; catalogue e2 is conjugated for cosmo_val's
``pol_factor=-1``. Only mock catalogues may be analysed without a blind.
"""

import argparse
import json

from sp_validation.glass_mock import (
    BLOCK_SIZE,
    GlassMockConfig,
    b_mode_cl,
    generate_lensing_maps,
    paint_catalogue,
    read_sky,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    maps = commands.add_parser("maps", help="Generate n(z)-averaged full-sky maps")
    maps.add_argument("--nz", required=True)
    maps.add_argument("--no-maps", action="store_true", help="Persist only alms and spectra")
    maps.add_argument("--output", required=True)
    maps.add_argument("--cache", required=True)
    maps.add_argument("--seed", type=int, required=True)
    maps.add_argument("--nside", type=int, help="Default 4096")
    maps.add_argument(
        "--lmax", type=int, default=None, help="Default 2*nside; must be <=3*nside-1"
    )
    maps.add_argument("--dx", type=float, help="Default 200 Mpc")
    maps.add_argument("--zmax", type=float, help="Default 3.5")
    maps.add_argument(
        "--exact-matter-cls",
        action="store_true",
        help="Use non-Limber shell spectra (slower)",
    )
    maps.add_argument("--config", help="YAML overrides, including cosmology")
    maps.add_argument(
        "--workers", type=int, help="Spectrum solver processes; default OMP_NUM_THREADS"
    )
    paint = commands.add_parser("paint", help="Paint a FITS/HDF5 catalogue in chunks")
    paint.add_argument("--input", required=True)
    paint.add_argument("--output", required=True)
    paint.add_argument("--maps", required=True)
    paint.add_argument("--seed", type=int, required=True, help="Rotation RNG seed")
    paint.add_argument(
        "--shear",
        choices=["gamma", "reduced"],
        default="gamma",
        help="Default gamma gives a pure-E known-answer sky",
    )
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
        help="C_BB=A exp[-(ell/2000)^2], up to the sky's lmax",
    )
    paint.add_argument("--b-cl", help="Two-column ell C_BB text file")
    paint.add_argument("--b-seed", type=int)
    paint.add_argument(
        "--chunk-size", type=int, default=250_000, help="Rows per read and HDF5 chunk"
    )
    paint.add_argument(
        "--block-size",
        type=int,
        default=BLOCK_SIZE,
        help="Rows per exact-evaluation call; each call costs about one SHT",
    )
    paint.add_argument(
        "--threads", type=int, help="Evaluation threads; default OMP_NUM_THREADS"
    )
    paint.add_argument("--overwrite", action="store_true")
    paint.add_argument("--tomo-bin-col", help="1-based source-bin column; discard other rows")
    paint.add_argument("--keep", nargs="*", default=[], help="Additional safe columns to retain")
    args = parser.parse_args()
    if args.command == "maps":
        from dataclasses import asdict

        settings = asdict(GlassMockConfig(nside=4096, zmax=3.5, limber=True))
        if args.config:
            import yaml

            with open(args.config) as stream:
                supplied = yaml.safe_load(stream)
            settings.update(
                {key: value for key, value in supplied.items() if key in settings}
            )
        for key in ("nside", "dx", "zmax"):
            if getattr(args, key) is not None:
                settings[key] = getattr(args, key)
        settings["harmonic_lmax"] = (
            args.lmax
            if args.lmax is not None
            else settings["harmonic_lmax"] or 2 * settings["nside"]
        )
        settings.update(seed=args.seed, nz_path=args.nz)
        if args.exact_matter_cls:
            settings["limber"] = False
        config = GlassMockConfig(**settings)
        import os

        workers = args.workers or int(os.environ.get("OMP_NUM_THREADS", "1"))
        result = generate_lensing_maps(config, args.output, args.cache, workers=workers,
                                       write_maps=False if args.no_maps else None)
    else:
        import healpy as hp

        kappa_alm, _ = read_sky(args.maps)
        b_cl = b_mode_cl(hp.Alm.getlmax(kappa_alm.shape[-1]), args.b_amplitude, args.b_cl)
        del kappa_alm
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
            block_size=args.block_size,
            nthreads=args.threads,
            overwrite=args.overwrite,
            shear=args.shear,
            tomo_bin_col=args.tomo_bin_col,
            keep=args.keep,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
