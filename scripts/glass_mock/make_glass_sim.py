"""Generate a GLASS mock source catalogue.

Thin CLI wrapper around the generation core in ``sp_validation.glass_mock``:
this script owns argument parsing, the mask downgrade, galaxy sampling, and FITS
catalogue I/O. The reproducibility surface — the fixed cosmology, the
``sigma8``-rescaled CAMB power spectrum, and the lognormal matter / lensing-map
generation — lives in the library and is pinned by
``src/sp_validation/tests/test_glass_mock.py``.

Run inside an image that has GLASS installed (the production sp_validation
container does not yet ship GLASS).
"""

import argparse
import os
import time

import camb
import fitsio

# GLASS modules
import glass
import healpy as hp
import numpy as np
from tqdm import tqdm

from sp_validation.glass_mock import (
    GlassMockConfig,
    build_camb_params,
    build_shells,
    camb_sigma8,
    downgrade_mask,
    ia_convergence,
    matter_shell_cls,
)


def get_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Creates a weak lensing catalogue simulation using GLASS",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("-s", "--seed", help="Random seed", type=int, default=42)
    parser.add_argument("-N", "--number", help="Mock Number", type=int, default=0)
    parser.add_argument("-t", "--test", help="Test run", action="store_true")
    parser.add_argument("-cb", "--camb", help="get Camb C_ell", action="store_true")
    parser.add_argument(
        "-c",
        "--config",
        help="Path to the configuration file to generate the simulation",
        type=str,
        required=True,
    )

    return parser


class Sky:
    """Build a UNIONS GLASS mock from a :class:`GlassMockConfig` + CLI args."""

    def __init__(self):
        parser = get_parser()
        args = parser.parse_args()

        yaml_config = args.config

        # Test mode shrinks the box; otherwise CLI overrides config fields.
        if args.test:
            print("[!!!] Running in test mode: ignore the configuration file [!!!]")
            self.config = GlassMockConfig.from_planck18(
                nside=32,
                seed=args.seed,
                n_arcmin2=0.0824,
                dx=120.0,
                zmax=0.5,
                sigma_e=0.26,
                ia_bias=None,
            )
        else:
            # Load the configuration from the config file in the parser
            self.config = GlassMockConfig.from_yaml(
                yaml_config=yaml_config, seed=args.seed
            )

        # Runtime options
        self.test = args.test
        self.camb = args.camb

        # Number label and random seed
        self.number = args.number
        self.n_sim = str(args.number).zfill(5)
        self.rng = np.random.default_rng(self.config.seed)

        # Input paths
        self.path_mask = args.mask
        self.path_nz = args.pathnz

        # Output path
        self.path = args.path

        print("-" * 66)
        print(
            f"Creating a weak lensing catalogue GLASS simulation with NSide = {self.config.nside}"
        )
        print(f"Mock number: {self.number}")
        print(f"Random seed: {self.config.seed}")
        if self.test:
            print("[!] Running in test mode [!]")
        print("-" * 66)

        self.root = f"{self.path}/results"
        if not os.path.exists(self.root):
            os.makedirs(self.root)
            print("A new directory " + str(self.root) + " is created!")

        # Cosmology + CAMB params come from the library core.
        self.pars = build_camb_params(self.config)
        print("-" * 66)
        print("Parameters used for the simulation:")
        print(f"H0: {self.pars.H0}")
        print(f"omch2: {self.pars.omch2}")
        print(f"ombh2: {self.pars.ombh2}")
        print(f"n_s: {self.pars.InitPower.ns}")
        print(f"As: {self.pars.InitPower.As}")
        print(f"sigma_8: {camb_sigma8(self.pars)}")
        print(f"Omega_m: {self.pars.omegam}")
        print(f"ia_bias: {self.config.ia_bias}")
        print("-" * 66)

        self.z = None
        self.bin_nz = None
        self.camb_cls = None

    def galaxies_simulation(self):
        """Create a source catalogue from the GLASS mock maps."""
        config = self.config

        # Read + downgrade the survey mask.
        unions_mask = hp.read_map(self.path_mask)
        mask_file_name = f"{self.path}/mask_nside{config.nside}.fits"
        if not os.path.isfile(mask_file_name):
            os.makedirs(self.path, exist_ok=True)
            unions_mask = downgrade_mask(unions_mask, config.nside)
            hp.write_map(mask_file_name, unions_mask, overwrite=True)
        else:
            unions_mask = hp.read_map(mask_file_name).astype(np.float64)

        # Shells + matter spectra from the library core (cached cls on disk).
        shells = build_shells(config, self.pars)
        cls_path = f"{self.root}/cls_{config.nside}.npy"
        try:
            cls = np.load(cls_path)
        except FileNotFoundError:
            print(f"[!] cls file {cls_path} missing; computing matter cls ...")
            cls = matter_shell_cls(config, self.pars, shells)
            np.save(cls_path, cls)
            print("[!] Done.")

        fields = glass.lognormal_fields(shells)
        gls = glass.solve_gaussian_spectra(fields, cls)
        matter = glass.generate(fields, gls, config.nside, ncorr=3, rng=self.rng)
        convergence = glass.MultiPlaneConvergence(_cosmology(self.pars))

        # n(z) and per-shell galaxy partition.
        nz = np.loadtxt(self.path_nz)
        nz = nz[nz[:, 0] <= config.zmax]
        self.z = nz[:, 0]
        dndz = nz[:, 1] * config.n_arcmin2
        ngal = glass.partition(self.z, dndz, shells)
        zbins = glass.equal_dens_zbins(self.z, dndz, nbins=config.nbins)

        out_file = f"{self.root}/unions_glass_sim_{self.n_sim}_{config.nside}.fits"
        fits = fitsio.FITS(out_file, "rw", clobber=True)
        fits.write(None)
        fits.create_table_hdu(
            names=[
                "RA",
                "Dec",
                "e1",
                "e2",
                "w",
                "n1",
                "n2",
                "TOM_BIN_ID",
                "TRUE_Z",
                "PHOTO_Z",
            ],
            formats=["D", "D", "E", "E", "D", "E", "E", "J", "D", "D"],
            extname="SOURCE_CATALOGUE",
        )
        cat_dtype = fits["SOURCE_CATALOGUE"].get_rec_dtype()[0]

        ngal_tot = 0
        c = 0
        print("Generating shell ", end="")
        for i, delta_i in tqdm(enumerate(matter)):
            convergence.add_window(delta_i, shells[i])
            kappa_i = convergence.kappa
            if config.ia_bias is not None:
                kappa_i = kappa_i + ia_convergence(delta_i, shells[i], config)
            gamm1_i, gamm2_i = glass.shear_from_convergence(kappa_i)

            for gal_lon, gal_lat, gal_count in glass.points.positions_from_delta(
                ngal[i], delta_i, config.bias, unions_mask, rng=self.rng
            ):
                ngal_tot += gal_count
                gal_z = glass.redshifts(gal_count, shells[i], rng=self.rng)
                gal_phz = glass.gaussian_phz(gal_z, config.phz_sigma_0, rng=self.rng)
                tomo_id = np.digitize(gal_phz, np.unique(zbins)) - 1
                gal_ellip = glass.ellipticity_intnorm(
                    gal_count, config.sigma_e, rng=self.rng
                )
                gal_she = glass.galaxy_shear(
                    gal_lon, gal_lat, gal_ellip, kappa_i, gamm1_i, gamm2_i
                )
                noise_she = glass.galaxies.galaxy_shear(
                    gal_lon,
                    gal_lat,
                    gal_ellip,
                    np.zeros(np.shape(kappa_i)),
                    np.zeros(np.shape(gamm1_i)),
                    np.zeros(np.shape(gamm2_i)),
                )

                catalogue = np.empty(gal_count, dtype=cat_dtype)
                catalogue["RA"] = gal_lon
                catalogue["Dec"] = gal_lat
                catalogue["e1"] = gal_she.real
                catalogue["e2"] = -gal_she.imag
                catalogue["w"] = np.ones_like(gal_lon)
                catalogue["n1"] = noise_she.real
                catalogue["n2"] = noise_she.imag
                catalogue["TOM_BIN_ID"] = tomo_id
                catalogue["TRUE_Z"] = gal_z
                catalogue["PHOTO_Z"] = gal_phz
                fits["SOURCE_CATALOGUE"].append(catalogue)
                c += 1

        print("[DONE] \n")
        print(f"Total number of galaxies sampled: {ngal_tot} using {c} z-shells")
        fits.close()
        print("Saved simulation to: ", out_file)

        if self.camb:
            self.get_camb_cls()

    def get_camb_cls(self, sav=True):
        """Lensing C_ell from CAMB source windows for the mock n(z)."""
        if self.bin_nz is None or self.z is None:
            print("ERROR: run galaxies_simulation() first to populate n(z)")
            return None
        lmax = self.config.lmax
        sources = [
            camb.sources.SplinedSourceWindow(
                z=self.z, W=self.bin_nz[i], source_type="lensing"
            )
            for i in range(self.config.nbins)
        ]
        self.pars.SourceWindows = sources
        self.pars.InitPower.set_params(As=2.1e-9, ns=0.965)
        results = camb.get_results(self.pars)
        cl_camb = results.get_source_cls_dict(lmax=lmax, raw_cl=True)

        dic = {"ell": np.arange(lmax + 1) + 1}
        for key in cl_camb:
            if "P" not in key:
                a, b = key.replace("W", "").replace("x", "-").split("-")
                dic[f"{int(a) - 1}-{int(b) - 1}"] = cl_camb[key]
        if sav:
            out = f"{self.root}/camb_cls.fits"
            fits = fitsio.FITS(out, "rw", clobber=True)
            fits.write(dic)
            fits.close()
        self.camb_cls = dic
        return dic


def _cosmology(pars):
    """GLASS cosmology wrapper from CAMB params (lazy import)."""
    from cosmology import Cosmology

    return Cosmology.from_camb(pars)


if __name__ == "__main__":
    print("Starting the simulation")
    start_time = time.time()
    new_sky = Sky()
    print("--- init %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    new_sky.galaxies_simulation()
    print("--- simulation %s seconds ---" % (time.time() - start_time))
