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
from astropy.io import fits

import sp_validation.pseudo_cl as spv_pseudo_cl

from ..rho_tau import get_params_rho_tau
from ..statistics import chi2_and_pte, cov_from_one_covariance


class PseudoClMixin:
    # ---------------- Pseudo-Cl properties ---------------- #
    @property
    def pseudo_cls(self):
        if not hasattr(self, "_pseudo_cls"):
            self.calculate_pseudo_cl(compute_tomography=False)
            self.calculate_pseudo_cl_inka_cov(compute_tomography=False)
            if self.compute_tomography:
                self.calculate_pseudo_cl(compute_tomography=True)
                self.calculate_pseudo_cl_inka_cov(compute_tomography=True)
        return self._pseudo_cls

    @property
    def pseudo_cls_onecov(self):
        if not hasattr(self, "_pseudo_cls_onecov"):
            self.calculate_pseudo_cl_onecovariance()
        return self._pseudo_cls_onecov

    # ---------------- Pseudo-Cl calculation methods ---------------- #
    # TODO: some cleaning to clearly separate DV, covariance, and utility functions.
    def calculate_pseudo_cl_inka_cov(self, compute_tomography=True):
        """
        Compute a theoretical Gaussian covariance of the Pseudo-Cl for EE, EB and BB.
        """
        self.print_start("Computing Pseudo-Cl covariance")

        nside = self.nside

        self._pseudo_cls = getattr(self, "_pseudo_cls", {})
        for ver in self.versions:
            self.print_magenta(ver)

            if ver not in self._pseudo_cls.keys():
                self._pseudo_cls[ver] = {}

            if compute_tomography:
                tomo_bin_ids, tomo_bin_pairs = self._get_tomo_bins(ver)

                if tomo_bin_ids is None or tomo_bin_pairs is None:
                    raise ValueError(
                        f"Version {ver} does not have tomography information."
                    )

            else:
                tomo_bin_pairs = [("all", "all")]

            # Initialise dictionnary to store field and workspace
            n_gal_map_dict = {}
            field_dict = {}
            wsp_dict = {}

            self.print_cyan(f"Extracting the fiducial power spectrum for {ver}")

            fiducial_cl = self.get_fiducial_cl(ver, compute_tomography)

            self.print_cyan(
                "Estimating and adding the noise bias to the fiducial power spectra"
            )

            params = get_params_rho_tau(self.cc[ver])
            cat_gal = fits.getdata(self.cc[ver]["shear"]["path"])

            for bin_key1, bin_key2 in tomo_bin_pairs:
                if bin_key1 == bin_key2:
                    cat_gal_ = self._get_tomographic_bin(params, cat_gal, bin_key1)

                    noise_bias_cl = self.get_noise_bias(params, nside, cat_gal_)

                else:
                    noise_bias_cl = np.zeros((4, 2 * nside))

                # Update the fiducial_cl dictionnary
                fiducial_cl[f"W{bin_key1}xW{bin_key2}"] = (
                    np.array(
                        [
                            fiducial_cl[f"W{bin_key1}xW{bin_key2}"],
                            0.0 * fiducial_cl[f"W{bin_key1}xW{bin_key2}"],
                            0.0 * fiducial_cl[f"W{bin_key1}xW{bin_key2}"],
                            0.0 * fiducial_cl[f"W{bin_key1}xW{bin_key2}"],
                        ]
                    )
                    + noise_bias_cl
                )

            # Compute the fields and workspaces
            for bin_key1, bin_key2 in tomo_bin_pairs:
                self.print_cyan(
                    f"Computing fields and workspaces for {bin_key1}, {bin_key2}"
                )
                lmin, lmax, b_lmax = spv_pseudo_cl.pseudo_cl_geometry(self.nside)
                b = self.get_namaster_bin(lmin, lmax, b_lmax)

                # Get the tomographic bins
                cat_gal_a = self._get_tomographic_bin(params, cat_gal, bin_key1)
                cat_gal_b = self._get_tomographic_bin(params, cat_gal, bin_key2)

                # Compute the n_gal_maps and the wsp object
                unique_pix_a, idx_a, idx_rep_a = self.get_pixels(
                    params, nside, cat_gal_a
                )
                unique_pix_b, idx_b, idx_rep_b = self.get_pixels(
                    params, nside, cat_gal_b
                )

                # Compute the number density maps
                n_gal_map_a = self.get_n_gal_map(params, nside, cat_gal_a)
                n_gal_map_b = self.get_n_gal_map(params, nside, cat_gal_b)

                # Get the shear maps
                shear_map_a_e1, shear_map_a_e2 = self.get_shear_map(
                    params,
                    self.nside,
                    cat_gal_a,
                    unique_pix=unique_pix_a,
                    idx=idx_a,
                    idx_rep=idx_rep_a,
                )
                shear_map_b_e1, shear_map_b_e2 = self.get_shear_map(
                    params,
                    self.nside,
                    cat_gal_b,
                    unique_pix=unique_pix_b,
                    idx=idx_b,
                    idx_rep=idx_rep_b,
                )

                # Get the fields and workspaces
                field_a, field_b, wsp = spv_pseudo_cl.get_field_and_workspace_from_map(
                    b,
                    mask_a=n_gal_map_a,
                    e1_map_a=shear_map_a_e1,
                    e2_map_a=shear_map_a_e2,
                    mask_b=n_gal_map_b,
                    e1_map_b=shear_map_b_e1,
                    e2_map_b=shear_map_b_e2,
                    pol_factor=self.pol_factor,
                    return_wsp=True,
                )

                # Save in the dictionnaries
                if not f"W{bin_key1}" not in n_gal_map_dict:
                    n_gal_map_dict[f"W{bin_key1}"] = n_gal_map_a
                if f"W{bin_key2}" not in n_gal_map_dict:
                    n_gal_map_dict[f"W{bin_key2}"] = n_gal_map_b
                if f"W{bin_key1}" not in field_dict:
                    field_dict[f"W{bin_key1}"] = field_a
                if f"W{bin_key2}" not in field_dict:
                    field_dict[f"W{bin_key2}"] = field_b
                if bin_key1 <= bin_key2 and f"W{bin_key1}xW{bin_key2}" not in wsp_dict:
                    wsp_dict[f"W{bin_key1}xW{bin_key2}"] = wsp

            for bin_key1, bin_key2 in tomo_bin_pairs:
                # Couple the cell if required
                if self.fiducial_input_inka == "coupled":
                    self.print_cyan("Coupling the fiducial Cls.")
                    # Get the wsp object
                    n_gal_map_a = n_gal_map_dict[f"W{bin_key1}"]
                    n_gal_map_b = n_gal_map_dict[f"W{bin_key2}"]
                    wsp = wsp_dict[f"W{bin_key1}xW{bin_key2}"]

                    coupling_mat = wsp.get_coupling_matrix()
                    coupling_mat_re = np.reshape(
                        coupling_mat, (4, lmax, 4, lmax), order="F"
                    )
                    fiducial_cl[f"W{bin_key1}xW{bin_key2}"] = np.tensordot(
                        coupling_mat_re, fiducial_cl[f"W{bin_key1}xW{bin_key2}"]
                    ) / np.mean(
                        n_gal_map_a * n_gal_map_b
                    )  # couple and divide by the product of the mask

            # Loop on the different tomographic bin pairs to compute the covariance
            for bin_key1, bin_key2 in tomo_bin_pairs:
                self.print_cyan(f"Tomo Bin Pair: ({bin_key1}, {bin_key2})")

                if (
                    f"tomo_bin_{bin_key1}_tomo_bin_{bin_key2}"
                    not in self._pseudo_cls[ver].keys()
                ):
                    self._pseudo_cls[ver][
                        f"tomo_bin_{bin_key1}_tomo_bin_{bin_key2}"
                    ] = {}

                out_path = self._output_path_pseudo_cl_cov(
                    ver, "iNKA", tomo_bin_pair=(bin_key1, bin_key2)
                )

                if os.path.exists(out_path) and not self.force_run:
                    self.print_done(
                        f"Skipping Pseudo-Cl covariance calculation, {out_path} exists"
                    )
                    self._pseudo_cls[ver][f"tomo_bin_{bin_key1}_tomo_bin_{bin_key2}"][
                        "cov"
                    ] = fits.open(out_path)
                    continue

                self.print_cyan("Computing the Pseudo-Cl covariance")

                covar_22_22 = spv_pseudo_cl.get_pseudo_cl_iNKA_covariance(
                    fiducial_cl[f"W{bin_key1}xW{bin_key1}"],
                    fiducial_cl[f"W{bin_key1}xW{bin_key2}"],
                    fiducial_cl[f"W{bin_key2}xW{bin_key1}"],
                    fiducial_cl[f"W{bin_key2}xW{bin_key2}"],
                    field_dict[f"W{bin_key1}"],
                    field_dict[f"W{bin_key2}"],
                    field_dict[f"W{bin_key1}"],
                    field_dict[f"W{bin_key2}"],
                    wsp_a=wsp_dict[f"W{bin_key1}xW{bin_key2}"],
                    wsp_b=wsp_dict[f"W{bin_key2}xW{bin_key1}"],
                    b=b,
                )

                self.print_cyan("Saving Pseudo-Cl covariance")

                self._pseudo_cls[ver][f"tomo_bin_{bin_key1}_tomo_bin_{bin_key2}"][
                    "cov"
                ] = self._save_iNKA_covariance(covar_22_22, out_path)

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

            if (
                os.path.exists(
                    os.path.join(out_dir, "covariance_list_3x2pt_pure_Cell.dat")
                )
                and not self.force_run
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
            if os.path.exists(out_file) and not self.force_run:
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

    def calculate_pseudo_cl(self, compute_tomography=True):
        """
        Compute the pseudo-Cl of a `CosmologyValidation` inputs with tomography.
        """
        if compute_tomography:
            self.print_start("Computing tomographic pseudo-Cl's")
        else:
            self.print_start("Computing non-tomographic pseudo-Cl's")

        self._pseudo_cls = getattr(self, "_pseudo_cls", {})

        for ver in self.versions:
            self.print_magenta(ver)

            if ver not in self.pseudo_cls.keys():
                self._pseudo_cls[ver] = {}

            if compute_tomography:
                tomo_bin_ids, tomo_bin_pairs = self._get_tomo_bins(ver)

                if tomo_bin_ids is None or tomo_bin_pairs is None:
                    raise ValueError(
                        f"Version {ver} does not have tomography information."
                    )

            else:
                tomo_bin_pairs = [("all", "all")]

            # Loop on the different tomographic bin pairs
            for bin_key1, bin_key2 in tomo_bin_pairs:
                self.print_cyan(f"Tomo Bin Pair: ({bin_key1}, {bin_key2})")

                if (
                    f"tomo_bin_{bin_key1}_tomo_bin_{bin_key2}"
                    not in self._pseudo_cls[ver].keys()
                ):
                    self._pseudo_cls[ver][
                        f"tomo_bin_{bin_key1}_tomo_bin_{bin_key2}"
                    ] = {}

                out_path = self._output_path_pseudo_cl(
                    ver, tomo_bin_pair=(bin_key1, bin_key2)
                )
                if os.path.exists(out_path) and not self.force_run:
                    self.print_done(
                        f"Skipping Pseudo-Cl's calculation, {out_path} exists"
                    )
                    cl_shear = fits.getdata(out_path)
                    self._pseudo_cls[ver][f"tomo_bin_{bin_key1}_tomo_bin_{bin_key2}"][
                        "pseudo_cl"
                    ] = cl_shear
                    continue

                if self.cell_method == "map":
                    self.calculate_pseudo_cl_map(
                        ver, self.nside, out_path, bin_key1, bin_key2
                    )
                elif self.cell_method == "catalog":
                    self.calculate_pseudo_cl_catalog(ver, out_path, bin_key1, bin_key2)
                else:
                    raise ValueError(f"Unknown cell method: {self.cell_method}")

    def calculate_pseudo_cl_map(self, ver, nside, out_path, tomo_bin_a, tomo_bin_b):
        assert (tomo_bin_a == "all" and tomo_bin_b == "all") or (
            isinstance(tomo_bin_a, (int, np.integer))
            and isinstance(tomo_bin_b, (int, np.integer))
        ), "tomo_bin_a and tomo_bin_b must be either both 'all' or both integers."

        params = get_params_rho_tau(self.cc[ver])

        self.print_cyan(
            f"Computing pseudo-Cl's for tomographic bins {tomo_bin_a} and {tomo_bin_b}..."
        )

        # Load data and create shear and noise maps
        cat_gal = fits.getdata(self.cc[ver]["shear"]["path"])

        # Get the tomographic bin
        cat_gal_a = self._get_tomographic_bin(params, cat_gal, tomo_bin_a)
        cat_gal_b = self._get_tomographic_bin(params, cat_gal, tomo_bin_b)

        del cat_gal

        self.print_cyan("Creating maps and computing Cl's...")
        # Get the pixels and indices for the catalogs
        unique_pix_a, idx_a, idx_rep_a = self.get_pixels(params, nside, cat_gal_a)
        unique_pix_b, idx_b, idx_rep_b = self.get_pixels(params, nside, cat_gal_b)

        # Create number density maps for each tomographic bin
        n_gal_map_a = self.get_n_gal_map(
            params,
            nside,
            cat_gal_a,
            unique_pix=unique_pix_a,
            idx=idx_a,
            idx_rep=idx_rep_a,
        )
        n_gal_map_b = self.get_n_gal_map(
            params,
            nside,
            cat_gal_b,
            unique_pix=unique_pix_b,
            idx=idx_b,
            idx_rep=idx_rep_b,
        )

        # Create shear maps for each tomographic bin
        shear_map_a_e1, shear_map_a_e2 = self.get_shear_map(
            params,
            nside,
            cat_gal_a,
            unique_pix=unique_pix_a,
            idx=idx_a,
            idx_rep=idx_rep_a,
            n_gal_map=n_gal_map_a,
        )
        shear_map_a = shear_map_a_e1 + 1j * shear_map_a_e2
        del shear_map_a_e1, shear_map_a_e2

        shear_map_b_e1, shear_map_b_e2 = self.get_shear_map(
            params,
            nside,
            cat_gal_b,
            unique_pix=unique_pix_b,
            idx=idx_b,
            idx_rep=idx_rep_b,
            n_gal_map=n_gal_map_b,
        )
        shear_map_b = shear_map_b_e1 + 1j * shear_map_b_e2
        del shear_map_b_e1, shear_map_b_e2

        # Compute the pseudo-Cl's
        ell_eff, cl_shear, wsp = self.get_pseudo_cls_map(
            shear_map_a, n_gal_map_a, shear_map_b=shear_map_b, mask_b=n_gal_map_b
        )

        # Remove the noise bias for auto-correlations.
        if tomo_bin_a == tomo_bin_b:
            # Compute the noise bias using noise_bias_method
            cl_noise = self.get_noise_bias_from_gaussian_real(
                params,
                nside,
                cat_gal_a,
                unique_pix=unique_pix_a,
                idx=idx_a,
                idx_rep=idx_rep_a,
                n_gal_map=n_gal_map_a,
                wsp=wsp,
            )

            # Subtract the noise bias from the pseudo-Cl's
            cl_shear = cl_shear - cl_noise

        self.print_cyan("Saving pseudo-Cl's...")
        self.save_pseudo_cl(ell_eff, cl_shear, out_path)

        cl_shear = fits.getdata(out_path)
        self._pseudo_cls[ver][f"tomo_bin_{tomo_bin_a}_tomo_bin_{tomo_bin_b}"][
            "pseudo_cl"
        ] = cl_shear

    def calculate_pseudo_cl_catalog(self, ver, out_path, tomo_bin_a, tomo_bin_b):
        assert (tomo_bin_a == "all" and tomo_bin_b == "all") or (
            isinstance(tomo_bin_a, int) and isinstance(tomo_bin_b, int)
        ), "tomo_bin_a and tomo_bin_b must be either both 'all' or both integers."

        params = get_params_rho_tau(self.cc[ver])

        # Load data and create shear and noise maps
        cat_gal = fits.getdata(self.cc[ver]["shear"]["path"])

        ell_eff, cl_shear, wsp = self.get_pseudo_cls_catalog(
            catalog=cat_gal, params=params, tomo_bin_a=tomo_bin_a, tomo_bin_b=tomo_bin_b
        )

        self.print_cyan("Saving pseudo-Cl's...")
        self.save_pseudo_cl(ell_eff, cl_shear, out_path)

        cl_shear = fits.getdata(out_path)
        self._pseudo_cls[ver][f"tomo_bin_{tomo_bin_a}_tomo_bin_{tomo_bin_b}"][
            "pseudo_cl"
        ] = cl_shear

    # ---------------- Utility functions for pseudo-Cl calculations ---------------- #
    def get_namaster_bin(self, lmin, lmax, b_lmax):
        """Build NaMaster binning object (thin wrapper, state -> primitive)."""
        return spv_pseudo_cl.make_namaster_bin(
            lmin,
            lmax,
            b_lmax,
            self.binning,
            ell_step=self.ell_step,
            n_ell_bins=self.n_ell_bins,
            power=self.power,
        )

    def get_pixels(self, params, nside, cat_gal):
        """Get unique pixels and indices for a catalog (thin wrapper -> primitive)."""
        return spv_pseudo_cl.get_pixels(
            cat_gal[params["ra_col"]], cat_gal[params["dec_col"]], nside
        )

    def get_n_gal_map(
        self, params, nside, cat_gal, unique_pix=None, idx=None, idx_rep=None
    ):
        """Weighted galaxy number-density map (thin wrapper -> primitive)."""
        return spv_pseudo_cl.get_n_gal_map(
            nside,
            cat_gal[params["ra_col"]],
            cat_gal[params["dec_col"]],
            weights=cat_gal[params["w_col"]],
            unique_pix=unique_pix,
            idx=idx,
            idx_rep=idx_rep,
        )

    def get_shear_map(
        self,
        params,
        nside,
        cat_gal,
        unique_pix=None,
        idx=None,
        idx_rep=None,
        n_gal_map=None,
    ):
        """Weighted shear map (thin wrapper -> primitive)."""
        return spv_pseudo_cl.get_shear_map(
            cat_gal[params["ra_col"]],
            cat_gal[params["dec_col"]],
            cat_gal[params["e1_col"]],
            cat_gal[params["e2_col"]],
            cat_gal[params["w_col"]],
            nside,
            unique_pix=unique_pix,
            idx=idx,
            idx_rep=idx_rep,
            n_gal_map=n_gal_map,
        )

    def get_noise_realisation(
        self,
        params,
        nside,
        cat_gal,
        n_gal=None,
        unique_pix=None,
        idx=None,
        idx_rep=None,
        rng=None,
    ):
        """
        Get a single Gaussian noise realization (thin wrapper -> primitive).
        """
        return spv_pseudo_cl.get_noise_realisation(
            cat_gal[params["ra_col"]],
            cat_gal[params["dec_col"]],
            cat_gal[params["e1_col"]],
            cat_gal[params["e2_col"]],
            cat_gal[params["w_col"]],
            nside,
            n_gal_map=n_gal,
            unique_pix=unique_pix,
            idx=idx,
            idx_rep=idx_rep,
            rng=rng,
        )

    def get_noise_bias_from_gaussian_real(
        self,
        params,
        nside,
        cat_gal,
        unique_pix=None,
        idx=None,
        idx_rep=None,
        n_gal_map=None,
        wsp=None,
    ):
        """Noise-bias from Gaussian realisations (thin wrapper, state -> primitive)"""
        return spv_pseudo_cl.get_noise_bias_from_gaussian_real(
            cat_gal[params["ra_col"]],
            cat_gal[params["dec_col"]],
            cat_gal[params["e1_col"]],
            cat_gal[params["e2_col"]],
            cat_gal[params["w_col"]],
            nside,
            nrandom_cell=self.nrandom_cell,
            binning=self.binning,
            ell_step=self.ell_step,
            n_ell_bins=self.n_ell_bins,
            power=self.power,
            unique_pix=unique_pix,
            idx=idx,
            idx_rep=idx_rep,
            n_gal_map=n_gal_map,
            wsp=wsp,
            seed=self.cell_seed,
        )

    def get_noise_bias_analytical(
        self, params, nside, cat_gal, unique_pix=None, idx=None, idx_rep=None
    ):
        """Noise-bias from analytical prescription (thin wrapper, state -> primitive)"""
        return spv_pseudo_cl.get_noise_bias_analytical(
            cat_gal[params["ra_col"]],
            cat_gal[params["dec_col"]],
            cat_gal[params["e1_col"]],
            cat_gal[params["e2_col"]],
            cat_gal[params["w_col"]],
            lmax=2 * nside,
            nside=nside,
            unique_pix=unique_pix,
            idx=idx,
            idx_rep=idx_rep,
        )

    def get_noise_bias(self, params, nside, cat_gal):
        """Noise-bias estimation (thin wrapper, state -> primitive)"""
        return spv_pseudo_cl.get_noise_bias(
            cat_gal[params["ra_col"]],
            cat_gal[params["dec_col"]],
            cat_gal[params["e1_col"]],
            cat_gal[params["e2_col"]],
            cat_gal[params["w_col"]],
            nside,
            noise_bias_method=self.noise_bias_method,
            binning=self.binning,
            ell_step=self.ell_step,
            n_ell_bins=self.n_ell_bins,
            power=self.power,
            nrandom_cell=self.nrandom_cell,
            seed=self.cell_seed,
        )

    def get_pseudo_cls_map(
        self, map_a, mask_a, wsp=None, shear_map_b=None, mask_b=None
    ):
        """Map-based pseudo-cl (thin wrapper, state -> primitive)."""
        return spv_pseudo_cl.get_pseudo_cls_map(
            map_a,
            mask_a,
            self.nside,
            self.binning,
            shear_map_b=shear_map_b,
            mask_b=mask_b,
            pol_factor=self.pol_factor,
            wsp=wsp,
            ell_step=self.ell_step,
            n_ell_bins=self.n_ell_bins,
            power=self.power,
        )

    def get_pseudo_cls_catalog(
        self, catalog, params, wsp=None, tomo_bin_a=None, tomo_bin_b=None
    ):
        """Catalog-based pseudo-cl (thin wrapper, state -> primitive)."""
        return spv_pseudo_cl.get_pseudo_cls_catalog(
            catalog,
            params,
            self.nside,
            self.binning,
            tomo_bin_a=tomo_bin_a,
            tomo_bin_b=tomo_bin_b,
            pol_factor=self.pol_factor,
            wsp=wsp,
            ell_step=self.ell_step,
            n_ell_bins=self.n_ell_bins,
            power=self.power,
        )

    def read_redshift_distribution(self, ver, is_tomography):
        path_redshift_distr = self.cc[ver]["shear"]["redshift_path"]
        z, dndz = np.loadtxt(path_redshift_distr, unpack=True)

        # Here it is assumed that the tomographic redshift distribution sum to the non-tomographic one and that the latter is normalised
        if not is_tomography:
            dndz = np.sum(dndz, axis=1)

        return z, dndz

    def get_fiducial_cl(self, ver, is_tomography):
        lmax = 2 * self.nside

        z, dndz = self.read_redshift_distribution(ver, is_tomography)

        fiducial_cl = spv_pseudo_cl.get_fiducial_cl(z, dndz, lmax, self.cosmo)

        # If non-tomographic, change the key to 'WallxWall'
        if not is_tomography:
            fiducial_cl = {"WallxWall": fiducial_cl["W1xW1"]}

        return fiducial_cl

    def _get_tomographic_bin(self, params, cat_gal, tomo_bin):
        if tomo_bin == "all":
            return cat_gal
        else:
            tomo_bin_id = cat_gal[params["tomo_bin_col"]]
            mask = tomo_bin_id == tomo_bin
            return cat_gal[mask]

    def _output_path_pseudo_cl(self, ver, tomo_bin_pair=None):
        if tomo_bin_pair is None:
            return self._output_path(
                f"pseudo_cl_from_{self.cell_method}_non_tomo_{ver}_binning_{self.binning}_nbins_{self.n_ell_bins}.fits"
            )
        else:
            bin_key1, bin_key2 = tomo_bin_pair
            return self._output_path(
                f"pseudo_cl_from_{self.cell_method}_tomo_bin_{bin_key1}_tomo_bin_{bin_key2}_{ver}_binning_{self.binning}_nbins_{self.n_ell_bins}.fits"
            )

    def _output_path_pseudo_cl_cov(self, ver, method, tomo_bin_pair=None):
        if tomo_bin_pair is None:
            return self._output_path(
                f"pseudo_cl_cov_from_{method}_non_tomo_{ver}_binning_{self.binning}_nbins_{self.n_ell_bins}.fits"
            )
        else:
            bin_key1, bin_key2 = tomo_bin_pair
            return self._output_path(
                f"pseudo_cl_cov_from_{method}_tomo_bin_{bin_key1}_tomo_bin_{bin_key2}_{ver}_binning_{self.binning}_nbins_{self.n_ell_bins}.fits"
            )

    def save_pseudo_cl(self, ell_eff, pseudo_cl, out_path):
        """
        Save pseudo-Cl's to a FITS file.

        Parameters
        ----------
        pseudo_cl : np.array
            Pseudo-Cl's to save.
        out_path : str
            Path to save the pseudo-Cl's to.
        """
        # Create columns of the fits file
        col1 = fits.Column(name="ELL", format="D", array=ell_eff)
        col2 = fits.Column(name="EE", format="D", array=pseudo_cl[0])
        col3 = fits.Column(name="EB", format="D", array=pseudo_cl[1])
        col4 = fits.Column(name="BB", format="D", array=pseudo_cl[3])
        coldefs = fits.ColDefs([col1, col2, col3, col4])
        cell_hdu = fits.BinTableHDU.from_columns(coldefs, name="PSEUDO_CELL")

        cell_hdu.writeto(out_path, overwrite=True)

    def _save_iNKA_covariance(self, covar, out_path):
        # covar_22_22 is indexed [ell, pol_a, ell, pol_b]; store each of the
        # 16 EE/EB/BE/BB cross-blocks as a named HDU (row-major pol order).
        # Append rather than construct from a list so astropy promotes the
        # first HDU to a PrimaryHDU on write.
        pols = ["EE", "EB", "BE", "BB"]
        hdu = fits.HDUList()
        for i, pa in enumerate(pols):
            for j, pb in enumerate(pols):
                hdu.append(fits.ImageHDU(covar[:, i, :, j], name=f"COVAR_{pa}_{pb}"))

        hdu.writeto(out_path, overwrite=True)

        return hdu

    # ---------------- Plotting functions for pseudo-Cl's ---------------- #
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
