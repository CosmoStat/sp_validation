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
    Cosmology_from_camb,
    GlassMockConfig,
    build_camb_params,
    build_shells,
    camb_sigma8,
    downgrade_mask,
    ia_convergence,
    matter_shell_cls,
    validate_number_density,
    validate_shape_noise,
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
        "-v", "--validation", help="Run some validation checks", action="store_true"
    )
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
            if not os.path.exists("config/glass_mock/test_data/"):
                raise FileNotFoundError(
                    "The test must be run from the root of the repository."
                )
            if not os.path.exists("config/glass_mock/test_data/mask.fits"):
                raise FileNotFoundError(
                    "Check that you downloaded the test mask from the repository."
                )
            if not os.path.exists(
                "config/glass_mock/test_data/redshift_distribution.txt"
            ):
                raise FileNotFoundError(
                    "Check that you downloaded the test redshift distribution from the repository."
                )

            self.config = GlassMockConfig.from_planck18(
                nside=32,
                seed=args.seed,
                n_arcmin2=0.0824,
                dx=120.0,
                zmax=0.5,
                sigma_e=0.26,
                ia_bias=None,
                mask_path="config/glass_mock/test_data/mask.fits",
                nz_path="config/glass_mock/test_data/redshift_distribution.txt",
                output_path="config/glass_mock/test_data",
                output_prefix="test",
            )
        else:
            # Load the configuration from the config file in the parser
            self.config = GlassMockConfig.from_yaml(
                yaml_config=yaml_config, seed=args.seed
            )

        # Check consistency of the config for the tomographic bins
        self.config.check_consistency()

        # Runtime options
        self.test = args.test
        self.camb = args.camb
        self.validation = args.validation
        self.limber = self.config.limber

        # Number label and random seed
        self.number = args.number
        self.n_sim = str(args.number).zfill(5)
        self.rng = np.random.default_rng(self.config.seed)

        # Input paths
        self.path_mask = self.config.mask_path
        self.path_nz = self.config.nz_path

        # Output path
        self.path = self.config.output_path
        self.prefix = self.config.output_prefix

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
        print(f"number density: {self.config.n_arcmin2}")
        print(f"shape noise: {self.config.sigma_e}")
        print(f"galaxy bias: {self.config.bias}")
        print("-" * 66)

        self.z = None
        self.bin_nz = []
        self.ngal_per_bin = []
        self.camb_cls = None

    def read_mask(self):
        """Read the survey mask from the specified path."""
        mask = hp.read_map(self.path_mask)
        mask_file_name = f"{self.path}/mask_nside_{self.config.nside}.fits"
        if not os.path.isfile(mask_file_name):
            os.makedirs(self.path, exist_ok=True)
            mask = downgrade_mask(mask, self.config.nside)
            hp.write_map(mask_file_name, mask, overwrite=True)
        else:
            mask = hp.read_map(mask_file_name).astype(np.float64)
        return mask

    def get_cl_matter_shells(self, shells):
        """Get the matter power spectrum for the simulation."""
        cls_path = f"{self.root}/cls_{self.config.nside}_limber_{self.limber}.npy"
        try:
            cls = np.load(cls_path)
        except FileNotFoundError:
            print(f"[!] cls file {cls_path} missing; computing matter cls ...")
            cls = matter_shell_cls(self.config, self.pars, shells, limber=self.limber)
            np.save(cls_path, cls)
            print("[!] Done.")
        return cls

    def read_redshift_distributions(self, shells):
        """Read the redshift distributions from the specified path.

        The convention used here is that the non-tomographic redshift distribution sums to one.
        All the per bin distributions will sum to a lower value than 1.
        """
        nz = np.loadtxt(self.path_nz)
        nz = nz[nz[:, 0] <= self.config.zmax]
        self.z = nz[:, 0]

        # Check that the number of nz columns corresponds to what is expected.
        dndz_cols = nz[:, 1:]

        # Check that the per bin integrals sum together to one
        per_bin_integral = np.trapezoid(dndz_cols, self.z.T, axis=0)
        assert np.isclose(per_bin_integral.sum(), 1.0, atol=1e-2), (
            f"Per bin integrals do not sum to one: {per_bin_integral.sum()}. Integrals per bin: {per_bin_integral}"
        )

        for b in range(self.config.nbins):
            n_arcmin2 = (
                self.config.n_arcmin2[b]
                if isinstance(self.config.n_arcmin2, list)
                else self.config.n_arcmin2
            )
            dndz_b = (
                dndz_cols[:, b]
                * n_arcmin2
                / per_bin_integral[
                    b
                ]  # Normalize the per bin redshift distribution to have the correct number density
            )
            ngal_b = glass.partition(self.z, dndz_b, shells)
            self.ngal_per_bin.append(ngal_b)
            self.bin_nz.append(dndz_b)

    def galaxies_simulation(self):
        """Create a source catalogue from the GLASS mock maps."""
        config = self.config

        # Read + downgrade the survey mask.
        mask = self.read_mask()

        # Shells + matter spectra from the library core (cached cls on disk).
        shells = build_shells(config, self.pars)

        # Generate the file for the matter shells if not done already
        cls = self.get_cl_matter_shells(shells)

        # Apply discretisation to the full set of spectra
        cls = glass.discretized_cls(cls, nside=config.nside, lmax=config.lmax, ncorr=3)

        # Generate the matter maps
        fields = glass.lognormal_fields(shells)
        gls = glass.solve_gaussian_spectra(fields, cls)
        matter = glass.generate(fields, gls, config.nside, ncorr=3, rng=self.rng)
        convergence = glass.MultiPlaneConvergence(Cosmology_from_camb(self.pars))

        # n(z) and per-shell galaxy partition.
        self.read_redshift_distributions(shells)

        # Name of the output file
        out_file = (
            f"{self.root}/{self.prefix}_glass_sim_{self.n_sim}_{config.nside}.fits"
        )

        # Open the FITS file to save the catalogue
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

            # Sample each tomographic bin against the same matter/convergence
            # realisation for this shell
            for b in range(config.nbins):
                bias_b = (
                    config.bias[b] if isinstance(config.bias, list) else config.bias
                )
                sigma_e = (
                    config.sigma_e[b]
                    if isinstance(config.sigma_e, list)
                    else config.sigma_e
                )

                for gal_lon, gal_lat, gal_count in glass.points.positions_from_delta(
                    self.ngal_per_bin[b][i], delta_i, bias_b, mask, rng=self.rng
                ):
                    ngal_tot += gal_count
                    gal_z = glass.redshifts(gal_count, shells[i], rng=self.rng)
                    gal_phz = glass.gaussian_phz(
                        gal_z, config.phz_sigma_0, rng=self.rng
                    )

                    gal_ellip = glass.ellipticity_intnorm(
                        gal_count,
                        sigma_e,
                        rng=self.rng,
                        xp=np,
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
                    catalogue["TOM_BIN_ID"] = b + 1
                    catalogue["TRUE_Z"] = gal_z
                    catalogue["PHOTO_Z"] = gal_phz
                    fits["SOURCE_CATALOGUE"].append(catalogue)

            c += 1

        print("[DONE] \n")
        print(f"Total number of galaxies sampled: {ngal_tot} using {c} z-shells")
        fits.close()
        print("Saved simulation to: ", out_file)

        if self.camb:
            print("-" * 66)
            print("Compute the associated theory power spectra...")
            self.get_camb_cls()
            print("-" * 66)

        if self.validation:
            print("-" * 66)
            self.run_validation_checks(out_file, mask)
            print("-" * 66)

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

        results = camb.get_results(self.pars)
        cl_camb = results.get_source_cls_dict(lmax=lmax, raw_cl=True)

        dic = {"ell": np.arange(lmax + 1) + 1}
        for key in cl_camb:
            if "P" not in key:
                a, b = key.replace("W", "").replace("x", "-").split("-")
                dic[f"{int(a) - 1}-{int(b) - 1}"] = cl_camb[key]
        if sav:
            out = f"{self.root}/{self.prefix}_camb_cls_{self.n_sim}_{self.config.nside}.fits"
            fits = fitsio.FITS(out, "rw", clobber=True)
            fits.write(dic)
            print("Saved CAMB power spectra to: ", out)
            fits.close()
        self.camb_cls = dic
        return dic

    def run_validation_checks(self, out_file, mask):
        print("Running validation checks...")

        cat_glass = fitsio.FITS(out_file)["SOURCE_CATALOGUE"].read()

        # First estimate the non-tomographic and tomographic shape noise.
        validate_shape_noise(cat_glass)

        # Next, validate the number density.
        validate_number_density(cat_glass, mask)

        print("Validation checks completed.")


if __name__ == "__main__":
    print("Starting the simulation")
    start_time = time.time()
    new_sky = Sky()
    print("--- init %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    new_sky.galaxies_simulation()
    print("--- simulation %s seconds ---" % (time.time() - start_time))
