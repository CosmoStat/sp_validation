"""Pseudo-Cl / harmonic-space diagnostic for cosmology validation.

This mixin holds the pseudo-Cl (harmonic-space) machinery: NaMaster
field/workspace construction, the analytic (iNKA) Gaussian covariance, the
OneCovariance Gaussian + non-Gaussian covariance, map- and catalog-based
pseudo-Cl estimation, random-rotation noise debiasing, and plotting. It depends
on pymaster (NaMaster), healpy, and OneCovariance.
"""

import configparser
import os

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt
from astropy.io import fits
from cs_util.cosmo import get_theo_c_ell

from .. import sacc_io
from ..pseudo_cl import (
    apply_random_rotation,
    get_n_gal_map,
    get_pseudo_cls_catalog,
    get_pseudo_cls_map,
    make_namaster_bin,
)
from ..rho_tau import get_params_rho_tau
from ..statistics import chi2_and_pte, cov_from_one_covariance
from .sacc_writers import BIN as SACC_BIN
from .sacc_writers import pseudo_cl_to_sacc


class PseudoClMixin:
    @property
    def pseudo_cls(self):
        if not hasattr(self, "_pseudo_cls"):
            self.calculate_pseudo_cl()
            self.calculate_pseudo_cl_eb_cov()
        return self._pseudo_cls

    @property
    def pseudo_cls_onecov(self):
        if not hasattr(self, "_pseudo_cls_onecov"):
            self.calculate_pseudo_cl_onecovariance()
        return self._pseudo_cls_onecov

    def get_namaster_bin(self, lmin, lmax, b_lmax):
        """Build NaMaster binning object (thin wrapper, state -> primitive)."""
        return make_namaster_bin(
            lmin,
            lmax,
            b_lmax,
            self.binning,
            ell_step=self.ell_step,
            n_ell_bins=self.n_ell_bins,
            power=self.power,
        )

    def get_variance_map(self, nside, e1, e2, w, unique_pix, idx_rep):
        """
        Create a variance map from the input catalog.
        """

        variance_map = np.zeros(hp.nside2npix(nside))

        variance_map[unique_pix] = np.bincount(
            idx_rep, weights=(e1**2 + e2**2) / 2 * w**2
        )

        return variance_map

    def get_field_and_workspace_from_map(self, mask, lmax, b):
        """
        Create a NaMaster field and workspace from the input map.
        """

        nside = hp.npix2nside(len(mask))

        # Create NaMaster field
        f = nmt.NmtField(
            mask=mask,
            maps=[np.zeros(hp.nside2npix(nside)), np.zeros(hp.nside2npix(nside))],
            lmax=lmax,
        )

        # Create NaMaster workspace
        wsp = nmt.NmtWorkspace.from_fields(f, f, b)

        return f, wsp

    def calculate_pseudo_cl_eb_cov(self):
        """
        Compute a theoretical Gaussian covariance of the Pseudo-Cl for EE, EB and BB.
        """
        self.print_start("Computing Pseudo-Cl covariance")

        nside = self.nside

        try:
            self._pseudo_cls
        except AttributeError:
            self._pseudo_cls = {}
        for ver in self.versions:
            self.print_magenta(ver)

            if ver not in self._pseudo_cls.keys():
                self._pseudo_cls[ver] = {}

            out_path = self._output_path(f"pseudo_cl_cov_{ver}.fits")
            if os.path.exists(out_path):
                self.print_done(
                    f"Skipping Pseudo-Cl covariance calculation, {out_path} exists"
                )
                self._pseudo_cls[ver]["cov"] = fits.open(out_path)
            else:
                params = get_params_rho_tau(self.cc[ver], survey=ver)

                self.print_cyan(f"Extracting the fiducial power spectrum for {ver}")

                lmax = 2 * self.nside
                ell = np.arange(1, lmax + 1)
                pw = hp.pixwin(nside, lmax=lmax)
                if pw.shape[0] != len(ell) + 1:
                    raise ValueError(
                        "Unexpected pixwin length for lmax="
                        f"{lmax}: got {pw.shape[0]}, expected {len(ell) + 1}"
                    )
                pw = pw[1 : len(ell) + 1]

                # Load redshift distribution and calculate theory C_ell
                path_redshift_distr = self.cc[ver]["shear"]["redshift_path"]
                z, dndz = np.loadtxt(path_redshift_distr, unpack=True)
                fiducial_cl = (
                    get_theo_c_ell(
                        ell=ell,
                        z=z,
                        nz=dndz,
                        backend="ccl",
                        cosmo=self.cosmo,
                    )
                    * pw**2
                )

                self.print_cyan("Getting a binning, n_gal_map, field and workspace.")

                lmin = 8
                lmax = 2 * self.nside
                b_lmax = lmax - 1

                b = self.get_namaster_bin(lmin, lmax, b_lmax)

                # Load data and create shear and noise maps
                cat_gal = fits.getdata(self.cc[ver]["shear"]["path"])

                n_gal, unique_pix, _idx, idx_rep = self.get_n_gal_map(
                    params, nside, cat_gal
                )

                f, wsp = self.get_field_and_workspace_from_map(n_gal, b_lmax, b)

                if self.noise_bias_method == "randoms":
                    self.print_cyan("Getting a sample of Cls with noise bias.")

                    cl_noise, f, wsp = self.get_sample(
                        params,
                        self.nside,
                        b_lmax,
                        b,
                        cat_gal,
                        n_gal,
                        n_gal,
                        unique_pix,
                        idx_rep,
                        np.random.default_rng(self.cell_seed),
                    )

                    noise_bias_cl = np.mean(cl_noise, axis=0)

                elif self.noise_bias_method == "analytic":
                    self.print_cyan("Getting analytic noise bias.")

                    e1, e2, w = (
                        cat_gal[self.cc[ver]["shear"]["e1_col"]],
                        cat_gal[self.cc[ver]["shear"]["e2_col"]],
                        cat_gal[self.cc[ver]["shear"]["w_col"]],
                    )
                    variance_map = self.get_variance_map(
                        self.nside, e1, e2, w, unique_pix, idx_rep
                    )

                    noise_bias = hp.nside2pixarea(self.nside) * np.mean(variance_map)

                    noise_bias_cl = np.zeros((4, lmax))
                    noise_bias_cl[0, :] = noise_bias
                    noise_bias_cl[3, :] = noise_bias

                    noise_bias_cl = wsp.decouple_cell(noise_bias_cl)  # Decouple

                else:
                    raise ValueError(
                        f"Noise bias method {self.noise_bias_method} not recognized. It should be 'randoms' or 'analytic'."
                    )

                # Unbin, then fill the data vector below lmin with the lowest-ell value
                noise_bias_cl = b.unbin_cell(noise_bias_cl)
                lowest_ell = b.get_ell_list(0)[0]
                noise_bias_cl[:, :lowest_ell] = noise_bias_cl[:, [lowest_ell]]

                self.print_cyan("Adding noise bias to the fiducial Cls.")

                fiducial_cl = (
                    np.array(
                        [
                            fiducial_cl,
                            0.0 * fiducial_cl,
                            0.0 * fiducial_cl,
                            0.0 * fiducial_cl,
                        ]
                    )
                    + noise_bias_cl
                )

                if self.fiducial_input_inka == "coupled":
                    self.print_cyan("Coupling the fiducial Cls.")

                    coupling_mat = wsp.get_coupling_matrix()
                    coupling_mat_re = np.reshape(
                        coupling_mat, (4, lmax, 4, lmax), order="F"
                    )
                    fiducial_cl = np.tensordot(coupling_mat_re, fiducial_cl) / np.mean(
                        n_gal**2
                    )  # couple and divide by the mean of the mask squared

                self.print_cyan("Computing the Pseudo-Cl covariance")

                cw = nmt.NmtCovarianceWorkspace.from_fields(f, f, f, f)

                # Get actual number of ell bins from binning scheme
                n_ell_actual = b.get_n_bands()

                covar_22_22 = nmt.gaussian_covariance(
                    cw,
                    2,
                    2,
                    2,
                    2,
                    fiducial_cl,
                    fiducial_cl,
                    fiducial_cl,
                    fiducial_cl,
                    wsp,
                    wb=wsp,
                ).reshape([n_ell_actual, 4, n_ell_actual, 4])

                self.print_cyan("Saving Pseudo-Cl covariance")

                # covar_22_22 is indexed [ell, pol_a, ell, pol_b]; store each of the
                # 16 EE/EB/BE/BB cross-blocks as a named HDU (row-major pol order).
                # Append rather than construct from a list so astropy promotes the
                # first HDU to a PrimaryHDU on write.
                pols = ["EE", "EB", "BE", "BB"]
                hdu = fits.HDUList()
                for i, pa in enumerate(pols):
                    for j, pb in enumerate(pols):
                        hdu.append(
                            fits.ImageHDU(
                                covar_22_22[:, i, :, j], name=f"COVAR_{pa}_{pb}"
                            )
                        )

                hdu.writeto(out_path, overwrite=True)

                self._pseudo_cls[ver]["cov"] = hdu

        self.print_done("Done Pseudo-Cl covariance")

    def calculate_pseudo_cl_onecovariance(self):
        """
        Compute the pseudo-Cl covariance using OneCovariance.
        """
        self.print_start("Computing Pseudo-Cl covariance with OneCovariance")

        if self.path_onecovariance is None:
            raise ValueError("path_onecovariance must be provided to use OneCovariance")

        if not os.path.exists(self.path_onecovariance):
            raise ValueError(
                f"OneCovariance path {self.path_onecovariance} does not exist"
            )

        template_config = os.path.join(
            self.path_onecovariance, "config_files", "config_3x2pt_pure_Cell_UNIONS.ini"
        )
        if not os.path.exists(template_config):
            raise ValueError(f"Template config file {template_config} does not exist")

        self._pseudo_cls_onecov = {}
        for ver in self.versions:
            self.print_magenta(ver)

            out_dir = self._output_path(f"pseudo_cl_cov_onecov_{ver}/")
            os.makedirs(out_dir, exist_ok=True)

            if os.path.exists(
                os.path.join(out_dir, "covariance_list_3x2pt_pure_Cell.dat")
            ):
                self.print_done(f"Skipping OneCovariance calculation, {out_dir} exists")
                self._load_onecovariance_cov(out_dir, ver)
            else:
                mask_path = self.cc[ver]["mask"]
                if not os.path.exists(mask_path):
                    print("Mask file does not exist")
                    print("Computing the mask from the binned catalog and saving...")
                    mask = self._get_binned_catalog_mask(ver)
                    hp.write_map(mask_path, mask, overwrite=True)

                redshift_distr_path = os.path.join(
                    self.cc[ver]["shear"]["redshift_path"]
                )

                config_path = os.path.join(out_dir, f"config_onecov_{ver}.ini")

                self.print_cyan(
                    f"Modifying OneCovariance config file and saving it to {config_path}"
                )
                self._modify_onecov_config(
                    template_config,
                    config_path,
                    out_dir,
                    mask_path,
                    redshift_distr_path,
                    ver,
                )

                self.print_cyan("Running OneCovariance...")
                cmd = f"python {os.path.join(self.path_onecovariance, 'covariance.py')} {config_path}"
                self.print_cyan(f"Command: {cmd}")
                ret = os.system(cmd)
                if ret != 0:
                    raise RuntimeError(
                        f"OneCovariance command failed with return code {ret}"
                    )
                self.print_cyan("OneCovariance completed successfully.")
                self._load_onecovariance_cov(out_dir, ver)

        self.print_done("Done Pseudo-Cl covariance with OneCovariance")

    def _modify_onecov_config(
        self, template_config, config_path, out_dir, mask_path, redshift_distr_path, ver
    ):
        """
        Modify OneCovariance configuration file with correct mask, redshift distribution,
        and ellipticity dispersion parameters.

        Parameters
        ----------
        template_config : str
            Path to the template configuration file
        config_path : str
            Path where the modified configuration will be saved
        mask_path : str
            Path to the mask file
        redshift_distr_path : str
            Path to the redshift distribution file
        """
        config = configparser.ConfigParser()
        # Load the template configuration
        config.read(template_config)

        # Update mask path
        mask_base = os.path.basename(os.path.abspath(mask_path))
        mask_folder = os.path.dirname(os.path.abspath(mask_path))
        config["survey specs"]["mask_directory"] = mask_folder
        config["survey specs"]["mask_file_lensing"] = mask_base
        config["survey specs"]["survey_area_lensing_in_deg2"] = str(self.area[ver])
        config["survey specs"]["ellipticity_dispersion"] = str(
            self.ellipticity_dispersion[ver]
        )
        config["survey specs"]["n_eff_lensing"] = str(self.n_eff_gal[ver])

        # Update redshift distribution path
        redshift_distr_base = os.path.basename(os.path.abspath(redshift_distr_path))
        redshift_distr_folder = os.path.dirname(os.path.abspath(redshift_distr_path))
        config["redshift"]["z_directory"] = redshift_distr_folder
        config["redshift"]["zlens_file"] = redshift_distr_base

        # Update output directory
        config["output settings"]["directory"] = out_dir

        # Save the modified configuration
        with open(config_path, "w") as f:
            config.write(f)

    def _load_onecovariance_cov(self, out_dir, ver):
        self.print_cyan(f"Loading OneCovariance results from {out_dir}")
        cov_one_cov = np.genfromtxt(
            os.path.join(out_dir, "covariance_list_3x2pt_pure_Cell.dat")
        )
        gaussian_one_cov = cov_from_one_covariance(cov_one_cov, gaussian=True)
        all_one_cov = cov_from_one_covariance(cov_one_cov, gaussian=False)

        self._pseudo_cls_onecov[ver] = {
            "gaussian_cov": gaussian_one_cov,
            "all_cov": all_one_cov,
        }

    def calculate_pseudo_cl_g_ng_cov(self, gaussian_part="iNKA"):
        assert gaussian_part in ["iNKA", "OneCovariance"], (
            "gaussian_part must be 'iNKA' or 'OneCovariance'"
        )
        self.print_start(
            f"Gaussian and Non-Gaussian covariance of the Pseudo-Cl's using {gaussian_part} for the Gaussian part"
        )

        self._pseudo_cls_cov_g_ng = {}

        for ver in self.versions:
            self.print_magenta(ver)
            out_file = self._output_path(
                f"pseudo_cl_cov_g_ng_{gaussian_part}_{ver}.fits"
            )
            if os.path.exists(out_file):
                self.print_done(
                    f"Skipping Gaussian and Non-Gaussian covariance calculation, {out_file} exists"
                )
                cov_hdu = fits.open(out_file)
                self._pseudo_cls_cov_g_ng[ver] = cov_hdu
                continue
            if gaussian_part == "iNKA":
                gaussian_cov = self.pseudo_cls[ver]["cov"]["COVAR_EE_EE"].data
                non_gaussian_cov = (
                    self.pseudo_cls_onecov[ver]["all_cov"]
                    - self.pseudo_cls_onecov[ver]["gaussian_cov"]
                )
                full_cov = gaussian_cov + non_gaussian_cov
            elif gaussian_part == "OneCovariance":
                gaussian_cov = self.pseudo_cls_onecov[ver]["gaussian_cov"]
                non_gaussian_cov = (
                    self.pseudo_cls_onecov[ver]["all_cov"]
                    - self.pseudo_cls_onecov[ver]["gaussian_cov"]
                )
                full_cov = self.pseudo_cls_onecov[ver]["all_cov"]
            else:
                raise ValueError(f"Unknown gaussian_part: {gaussian_part}")
            self.print_cyan("Saving Gaussian and Non-Gaussian covariance...")
            hdu = fits.HDUList()
            hdu.append(fits.ImageHDU(gaussian_cov, name="COVAR_GAUSSIAN"))
            hdu.append(fits.ImageHDU(non_gaussian_cov, name="COVAR_NON_GAUSSIAN"))
            hdu.append(fits.ImageHDU(full_cov, name="COVAR_FULL"))
            hdu.writeto(out_file, overwrite=True)
            self._pseudo_cls_cov_g_ng[ver] = hdu
        self.print_done(
            f"Done Gaussian and Non-Gaussian covariance of the Pseudo-Cl's using {gaussian_part} for the Gaussian part"
        )

    def calculate_pseudo_cl(self, out_path=None):
        """
        Compute the pseudo-Cl of given catalogs.

        Each version's spectra are born as a SACC part via
        :func:`sacc_writers.pseudo_cl_to_sacc` — EE/BB/EB carrying the shared
        NaMaster bandpower window, with this instance's (blinded) n(z) stamped
        in. The in-memory ``self._pseudo_cls[ver]`` ``"pseudo_cl"`` entry keeps
        the ``ELL``/``EE``/``EB``/``BB`` arrays the plotting and B-mode-summary
        consumers read by column name.

        ``out_path`` is the exact destination the part is *born at* — the
        Snakemake-declared output. It must resolve per version; single-version
        rules (the tagged blinded producer) pass their tagged output directly.
        When ``None`` (multi-version diagnostic / the ``pseudo_cls`` property)
        each part defaults to the untagged native ``pseudo_cl_{ver}.sacc``.
        Skip-if-exists keys on this final path, so no two rules ever share an
        undeclared native basename (a tagged product born at its native name and
        then renamed would let one rule's skip-if-exists silently adopt — and
        the rename delete — another rule's declared, differently-blinded file).
        """
        self.print_start("Computing pseudo-Cl's")

        nside = self.nside

        if out_path is not None and len(self.versions) != 1:
            raise ValueError(
                "calculate_pseudo_cl(out_path=...) writes one part to one path, "
                f"but {len(self.versions)} versions are configured; call per version"
            )

        try:
            self._pseudo_cls
        except AttributeError:
            self._pseudo_cls = {}
        for ver in self.versions:
            self.print_magenta(ver)

            self._pseudo_cls[ver] = {}

            ver_out_path = out_path or self._output_path(f"pseudo_cl_{ver}.sacc")
            if os.path.exists(ver_out_path):
                self.print_done(
                    f"Skipping Pseudo-Cl's calculation, {ver_out_path} exists"
                )
                self._pseudo_cls[ver]["pseudo_cl"] = self._load_pseudo_cl_sacc(
                    ver_out_path
                )
            elif self.cell_method == "map":
                self.calculate_pseudo_cl_map(ver, nside, ver_out_path)
            elif self.cell_method == "catalog":
                self.calculate_pseudo_cl_catalog(ver, ver_out_path)
            else:
                raise ValueError(f"Unknown cell method: {self.cell_method}")

        self.print_done("Done pseudo-Cl's")

    @staticmethod
    def _load_pseudo_cl_sacc(out_path):
        """Read a pseudo-Cl SACC part into the ELL/EE/EB/BB dict consumers use."""
        # Pipeline-internal readback of a part this producer just wrote: the
        # born-as-SACC parts are unblinded real-data measurements (blinding is a
        # downstream Smokescreen step), so the fail-closed load must be told this
        # is a legitimate pre-blind consumer.
        s = sacc_io.load(out_path, allow_unblinded=True)
        ell, ee, bb, eb, _window = sacc_io.get_pseudo_cl(s, SACC_BIN)
        return {"ELL": ell, "EE": ee, "EB": eb, "BB": bb}

    def calculate_pseudo_cl_map(self, ver, nside, out_path):
        params = get_params_rho_tau(self.cc[ver], survey=ver)

        # Load data and create shear and noise maps
        cat_gal = fits.getdata(self.cc[ver]["shear"]["path"])

        w = cat_gal[params["w_col"]]
        self.print_cyan("Creating maps and computing Cl's...")
        n_gal_map, unique_pix, _idx, idx_rep = self.get_n_gal_map(
            params, nside, cat_gal
        )
        mask = n_gal_map != 0

        shear_map_e1 = np.zeros(hp.nside2npix(nside))
        shear_map_e2 = np.zeros(hp.nside2npix(nside))

        e1 = cat_gal[params["e1_col"]]
        e2 = cat_gal[params["e2_col"]]

        del cat_gal

        shear_map_e1[unique_pix] += np.bincount(idx_rep, weights=e1 * w)
        shear_map_e2[unique_pix] += np.bincount(idx_rep, weights=e2 * w)
        shear_map_e1[mask] /= n_gal_map[mask]
        shear_map_e2[mask] /= n_gal_map[mask]

        shear_map = shear_map_e1 + 1j * shear_map_e2

        del shear_map_e1, shear_map_e2

        ell_eff, cl_shear, wsp = self.get_pseudo_cls_map(shear_map, n_gal_map)

        cl_noise = np.zeros_like(cl_shear)
        rng = np.random.default_rng(self.cell_seed)

        for i in range(self.nrandom_cell):
            noise_map_e1 = np.zeros(hp.nside2npix(nside))
            noise_map_e2 = np.zeros(hp.nside2npix(nside))

            e1_rot, e2_rot = self.apply_random_rotation(e1, e2, rng)

            noise_map_e1[unique_pix] += np.bincount(idx_rep, weights=e1_rot * w)
            noise_map_e2[unique_pix] += np.bincount(idx_rep, weights=e2_rot * w)

            noise_map_e1[mask] /= n_gal_map[mask]
            noise_map_e2[mask] /= n_gal_map[mask]

            noise_map = noise_map_e1 + 1j * noise_map_e2
            del noise_map_e1, noise_map_e2

            _, cl_noise_, _ = self.get_pseudo_cls_map(noise_map, n_gal_map, wsp)
            cl_noise += cl_noise_

        cl_noise /= self.nrandom_cell
        del e1, e2, w
        try:
            del e1_rot, e2_rot
        except NameError:  # Continue if the random generation has been skipped.
            pass
        del n_gal_map

        # Noise realizations are now reproducible (seeded rng from self.cell_seed).
        cl_shear = cl_shear - cl_noise

        self.print_cyan("Saving pseudo-Cl's...")
        self.pseudo_cl_to_sacc_part(ver, out_path, ell_eff, cl_shear, wsp)

        self._pseudo_cls[ver]["pseudo_cl"] = self._load_pseudo_cl_sacc(out_path)

    def calculate_pseudo_cl_catalog(self, ver, out_path):
        params = get_params_rho_tau(self.cc[ver], survey=ver)

        # Load data and create shear and noise maps
        cat_gal = fits.getdata(self.cc[ver]["shear"]["path"])

        ell_eff, cl_shear, wsp = self.get_pseudo_cls_catalog(
            catalog=cat_gal, params=params
        )

        self.print_cyan("Saving pseudo-Cl's...")
        self.pseudo_cl_to_sacc_part(ver, out_path, ell_eff, cl_shear, wsp)

        self._pseudo_cls[ver]["pseudo_cl"] = self._load_pseudo_cl_sacc(out_path)

    def get_n_gal_map(self, params, nside, cat_gal):
        """Weighted galaxy number-density map (thin wrapper -> primitive)."""
        return get_n_gal_map(
            nside,
            cat_gal[params["ra_col"]],
            cat_gal[params["dec_col"]],
            weights=cat_gal[params["w_col"]],
        )

    def get_gaussian_real(
        self, params, nside, lmax, cat_gal, n_gal, mask, unique_pix, idx_rep, rng=None
    ):
        e1_rot, e2_rot = self.apply_random_rotation(
            cat_gal[params["e1_col"]], cat_gal[params["e2_col"]], rng
        )
        noise_map_e1 = np.zeros(hp.nside2npix(nside))
        noise_map_e2 = np.zeros(hp.nside2npix(nside))

        w = cat_gal[params["w_col"]]
        noise_map_e1[unique_pix] += np.bincount(idx_rep, weights=e1_rot * w)
        noise_map_e2[unique_pix] += np.bincount(idx_rep, weights=e2_rot * w)
        noise_map_e1[mask] /= n_gal[mask]
        noise_map_e2[mask] /= n_gal[mask]

        return noise_map_e1 + 1j * noise_map_e2

    def get_sample(
        self,
        params,
        nside,
        lmax,
        b,
        cat_gal,
        n_gal,
        mask,
        unique_pix,
        idx_rep,
        rng=None,
    ):
        noise_map = self.get_gaussian_real(
            params, nside, lmax, cat_gal, n_gal, mask, unique_pix, idx_rep, rng
        )

        f = nmt.NmtField(mask=mask, maps=[noise_map.real, noise_map.imag], lmax=lmax)

        wsp = nmt.NmtWorkspace.from_fields(f, f, b)

        cl_noise = nmt.compute_coupled_cell(f, f)
        cl_noise = wsp.decouple_cell(cl_noise)

        return cl_noise, f, wsp

    def get_pseudo_cls_map(self, map, mask, wsp=None):
        """Map-based pseudo-cl (thin wrapper, state -> primitive)."""
        return get_pseudo_cls_map(
            map,
            mask,
            self.nside,
            self.binning,
            pol_factor=self.pol_factor,
            wsp=wsp,
            ell_step=self.ell_step,
            n_ell_bins=self.n_ell_bins,
            power=self.power,
        )

    def get_pseudo_cls_catalog(self, catalog, params, wsp=None):
        """Catalog-based pseudo-cl (thin wrapper, state -> primitive)."""
        return get_pseudo_cls_catalog(
            catalog,
            params,
            self.nside,
            self.binning,
            pol_factor=self.pol_factor,
            wsp=wsp,
            ell_step=self.ell_step,
            n_ell_bins=self.n_ell_bins,
            power=self.power,
        )

    def apply_random_rotation(self, e1, e2, rng=None):
        """Random ellipticity rotation (thin wrapper -> primitive).

        Pass a seeded ``rng`` for reproducible noise realizations.
        """
        return apply_random_rotation(e1, e2, rng)

    def pseudo_cl_to_sacc_part(self, version, out_path, ell_eff, cl_all, wsp):
        """Write the pseudo-Cl SACC part (EE/BB/EB + shared bandpower window).

        ``cl_all`` is NaMaster's decoupled ``(4, nbp)`` array (EE, EB, BE, BB);
        the writer takes the shared bandpower window from ``wsp``. No covariance
        is attached here — the analysis file's pseudo-Cl block is supplied at
        assembly (``assemble_sacc``) from the NaMaster / OneCovariance product.
        """
        s = pseudo_cl_to_sacc(
            self.sacc_nz(version),
            self.sacc_metadata(version),
            ell_eff,
            cl_all,
            wsp,
        )
        sacc_io.save(s, out_path, type="data")

    def plot_pseudo_cl(self):
        """
        Plot pseudo-Cl's for given catalogs.
        """
        self.print_cyan("Plotting pseudo-Cl's")

        # Plotting EE
        out_path = self._output_path("cell_ee.png")
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

        for ver in self.versions:
            ell = self.pseudo_cls[ver]["pseudo_cl"]["ELL"]
            cov = self.pseudo_cls[ver]["cov"]["COVAR_EE_EE"].data
            ax[0].errorbar(
                ell,
                ell * self.pseudo_cls[ver]["pseudo_cl"]["EE"],
                yerr=ell * np.sqrt(np.diag(cov)),
                fmt=self.cc[ver]["marker"],
                label=ver + " EE",
                color=self.cc[ver]["colour"],
                capsize=2,
            )

        ax[0].set_ylabel(r"$\ell C_\ell$")

        ax[0].set_xlim(ell.min() - 10, ell.max() + 100)
        ax[0].set_xscale("squareroot")
        ax[0].set_xticks(np.array([100, 400, 900, 1600]))
        ax[0].minorticks_on()
        ax[0].tick_params(axis="x", which="minor", length=2, width=0.8)
        minor_ticks = [i * 10 for i in range(1, 10)] + [i * 100 for i in range(1, 21)]
        ax[0].xaxis.set_ticks(minor_ticks, minor=True)

        for ver in self.versions:
            ell = self.pseudo_cls[ver]["pseudo_cl"]["ELL"]
            cov = self.pseudo_cls[ver]["cov"]["COVAR_EE_EE"].data
            ax[1].errorbar(
                ell,
                self.pseudo_cls[ver]["pseudo_cl"]["EE"],
                yerr=np.sqrt(np.diag(cov)),
                fmt=self.cc[ver]["marker"],
                label=ver + " EE",
                color=self.cc[ver]["colour"],
            )

        ax[1].set_xlabel(r"$\ell$")
        ax[1].set_ylabel(r"$C_\ell$")

        ax[1].set_xlim(ell.min() - 10, ell.max() + 100)
        ax[1].set_xscale("squareroot")
        ax[1].set_yscale("log")
        ax[1].set_xticks(np.array([100, 400, 900, 1600]))
        ax[1].minorticks_on()
        ax[1].tick_params(axis="x", which="minor", length=2, width=0.8)
        minor_ticks = [i * 10 for i in range(1, 10)] + [i * 100 for i in range(1, 21)]
        ax[1].xaxis.set_ticks(minor_ticks, minor=True)

        plt.suptitle("Pseudo-Cl EE (Gaussian covariance)")
        plt.legend()
        plt.savefig(out_path)

        # Plotting EB
        out_path = self._output_path("cell_eb.png")

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

        for ver in self.versions:
            ell = self.pseudo_cls[ver]["pseudo_cl"]["ELL"]
            cov = self.pseudo_cls[ver]["cov"]["COVAR_EB_EB"].data
            ax[0].errorbar(
                ell,
                ell * self.pseudo_cls[ver]["pseudo_cl"]["EB"],
                yerr=ell * np.sqrt(np.diag(cov)),
                fmt=self.cc[ver]["marker"],
                label=ver + " EB",
                color=self.cc[ver]["colour"],
                capsize=2,
            )

        ax[0].axhline(0, color="black", linestyle="--")
        ax[0].set_ylabel(r"$\ell C_\ell$")

        ax[0].set_xlim(ell.min() - 10, ell.max() + 100)
        ax[0].set_xscale("squareroot")
        ax[0].set_xticks(np.array([100, 400, 900, 1600]))
        ax[0].minorticks_on()
        ax[0].tick_params(axis="x", which="minor", length=2, width=0.8)
        minor_ticks = [i * 10 for i in range(1, 10)] + [i * 100 for i in range(1, 21)]
        ax[0].xaxis.set_ticks(minor_ticks, minor=True)

        for ver in self.versions:
            ell = self.pseudo_cls[ver]["pseudo_cl"]["ELL"]
            cov = self.pseudo_cls[ver]["cov"]["COVAR_EB_EB"].data
            ax[1].errorbar(
                ell,
                self.pseudo_cls[ver]["pseudo_cl"]["EB"],
                yerr=np.sqrt(np.diag(cov)),
                fmt=self.cc[ver]["marker"],
                label=ver + " EB",
                color=self.cc[ver]["colour"],
            )

        ax[1].set_xlabel(r"$\ell$")
        ax[1].set_ylabel(r"$C_\ell$")

        ax[1].set_xlim(ell.min() - 10, ell.max() + 100)
        ax[1].set_xscale("squareroot")
        ax[1].set_yscale("log")
        ax[1].set_xticks(np.array([100, 400, 900, 1600]))
        ax[1].minorticks_on()
        ax[1].tick_params(axis="x", which="minor", length=2, width=0.8)
        minor_ticks = [i * 10 for i in range(1, 10)] + [i * 100 for i in range(1, 21)]
        ax[1].xaxis.set_ticks(minor_ticks, minor=True)

        plt.suptitle("Pseudo-Cl EB (Gaussian covariance)")
        plt.legend()
        plt.savefig(out_path)

        # Plotting BB
        out_path = self._output_path("cell_bb.png")

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

        for ver in self.versions:
            ell = self.pseudo_cls[ver]["pseudo_cl"]["ELL"]
            cov = self.pseudo_cls[ver]["cov"]["COVAR_BB_BB"].data
            ax[0].errorbar(
                ell,
                ell * self.pseudo_cls[ver]["pseudo_cl"]["BB"],
                yerr=ell * np.sqrt(np.diag(cov)),
                fmt=self.cc[ver]["marker"],
                label=ver + " BB",
                color=self.cc[ver]["colour"],
                capsize=2,
            )

        ax[0].axhline(0, color="black", linestyle="--")
        ax[0].set_ylabel(r"$\ell C_\ell$")

        ax[0].set_xlim(ell.min() - 10, ell.max() + 100)
        ax[0].set_xscale("squareroot")
        ax[0].set_xticks(np.array([100, 400, 900, 1600]))
        ax[0].minorticks_on()
        ax[0].tick_params(axis="x", which="minor", length=2, width=0.8)
        minor_ticks = [i * 10 for i in range(1, 10)] + [i * 100 for i in range(1, 21)]
        ax[0].xaxis.set_ticks(minor_ticks, minor=True)

        for ver in self.versions:
            ell = self.pseudo_cls[ver]["pseudo_cl"]["ELL"]
            cov = self.pseudo_cls[ver]["cov"]["COVAR_BB_BB"].data
            ax[1].errorbar(
                ell,
                self.pseudo_cls[ver]["pseudo_cl"]["BB"],
                yerr=np.sqrt(np.diag(cov)),
                fmt=self.cc[ver]["marker"],
                label=ver + " BB",
                color=self.cc[ver]["colour"],
            )

        ax[1].set_xlabel(r"$\ell$")
        ax[1].set_ylabel(r"$C_\ell$")

        ax[1].set_xlim(ell.min() - 10, ell.max() + 100)
        ax[1].set_xscale("squareroot")
        ax[1].set_yscale("log")
        ax[1].set_xticks(np.array([100, 400, 900, 1600]))
        ax[1].minorticks_on()
        ax[1].tick_params(axis="x", which="minor", length=2, width=0.8)
        minor_ticks = [i * 10 for i in range(1, 10)] + [i * 100 for i in range(1, 21)]
        ax[1].xaxis.set_ticks(minor_ticks, minor=True)

        plt.suptitle("Pseudo-Cl BB (Gaussian covariance)")
        plt.legend()
        plt.savefig(out_path)

        # Print C_l^BB PTE for each version and save BB data
        print("\nC_l^BB PTE summary:")
        for ver in self.versions:
            cl_bb = self.pseudo_cls[ver]["pseudo_cl"]["BB"]
            cov_bb = self.pseudo_cls[ver]["cov"]["COVAR_BB_BB"].data
            chi2_bb, _, pte_bb = chi2_and_pte(cl_bb, cov_bb)
            chi2_bb = float(chi2_bb)
            print(
                f"  {ver}: C_l^BB PTE = {pte_bb:.4f} "
                f"(chi2/dof = {chi2_bb:.1f}/{len(cl_bb)})"
            )

            # Save BB data + covariance to .npz
            ell = self.pseudo_cls[ver]["pseudo_cl"]["ELL"]
            bb_out = self._output_path(f"{ver}_cell_bb_data.npz")
            np.savez(
                bb_out,
                ell=ell,
                cl_bb=cl_bb,
                cov_bb=cov_bb,
                chi2_bb=np.array(chi2_bb),
                pte_bb=np.array(pte_bb),
            )
            print(f"  Saved BB data to {bb_out}")
