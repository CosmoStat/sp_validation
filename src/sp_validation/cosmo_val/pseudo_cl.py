"""Pseudo-Cl / harmonic-space diagnostic for cosmology validation.

This mixin holds the pseudo-Cl (harmonic-space) machinery: NaMaster
field/workspace construction, the analytic (iNKA) Gaussian covariance, the
OneCovariance Gaussian + non-Gaussian covariance, map- and catalog-based
pseudo-Cl estimation, random-rotation noise debiasing, and plotting. It depends
on pymaster (NaMaster), healpy, and OneCovariance.
"""

import colorsys
import configparser
import itertools
import os

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import to_rgb

import sp_validation.pseudo_cl as spv_pseudo_cl

from ..rho_tau import get_params_rho_tau
from ..statistics import cov_from_one_covariance


class PseudoClMixin:
    # ---------------- Pseudo-Cl properties ---------------- #
    @property
    def pseudo_cls(self):
        if not hasattr(self, "_pseudo_cls"):
            self.calculate_pseudo_cl(compute_tomography=False)
            self.calculate_pseudo_cl_inka_cov(
                compute_tomography=False, load_all_block=True
            )
            if self.compute_tomography:
                self.calculate_pseudo_cl(compute_tomography=True)
                self.calculate_pseudo_cl_inka_cov(
                    compute_tomography=True, load_all_block=True
                )
        return self._pseudo_cls

    @property
    def pseudo_cls_onecov(self):
        if not hasattr(self, "_pseudo_cls_onecov"):
            self.calculate_pseudo_cl_onecovariance()
        return self._pseudo_cls_onecov

    # ---------------- Pseudo-Cl calculation methods ---------------- #
    def calculate_pseudo_cl(self, compute_tomography=True):
        """
        Compute the pseudo-Cl of a `CosmologyValidation` inputs with tomography.
        """
        out_dir = self._output_path("pseudo_cl")
        os.makedirs(out_dir, exist_ok=True)

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
            isinstance(tomo_bin_a, (int, np.integer))
            and isinstance(tomo_bin_b, (int, np.integer))
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

    def calculate_pseudo_cl_inka_cov(
        self, compute_tomography=True, load_all_block=False
    ):
        """
        Compute a theoretical Gaussian covariance of the Pseudo-Cl for EE, EB and BB.
        """
        self.print_start("Computing Pseudo-Cl covariance")

        nside = self.nside

        self._pseudo_cls = getattr(self, "_pseudo_cls", {})
        for ver in self.versions:
            self.print_magenta(ver)

            out_dir_block = self._output_path(f"pseudo_cl/iNKA_block_{ver}/")
            os.makedirs(out_dir_block, exist_ok=True)

            if ver not in self._pseudo_cls.keys():
                self._pseudo_cls[ver] = {}

            out_path_merged = self._output_path_pseudo_cl_cov(
                ver, "iNKA", tomography=compute_tomography
            )
            tomo_str = "tomo" if compute_tomography else "non_tomo"

            if compute_tomography:
                tomo_bin_ids, tomo_bin_pairs = self._get_tomo_bins(ver)

                if tomo_bin_ids is None or tomo_bin_pairs is None:
                    raise ValueError(
                        f"Version {ver} does not have tomography information."
                    )

            else:
                tomo_bin_pairs = [("all", "all")]

            if os.path.exists(out_path_merged) and not self.force_run:
                self.print_done(
                    f"Skipping Pseudo-Cl iNKA covariance calculation, {out_path_merged} exists"
                )
                self._pseudo_cls[ver][f"cov_iNKA_{tomo_str}"] = fits.open(
                    out_path_merged
                )

                if load_all_block:
                    self.print_done("Loading all the iNKA covariance blocks")
                    n_spectra = len(tomo_bin_pairs)

                    # indices into tomo_bin_pairs for all unique covariance blocks
                    block_indices = list(
                        itertools.combinations_with_replacement(range(n_spectra), 2)
                    )

                    for index_a, index_b in block_indices:
                        bin_key_a1, bin_key_a2 = tomo_bin_pairs[index_a]
                        bin_key_b1, bin_key_b2 = tomo_bin_pairs[index_b]

                        load_path = self._output_path_iNKA_block_cov(
                            ver,
                            tomo_bin_quad=(
                                bin_key_a1,
                                bin_key_a2,
                                bin_key_b1,
                                bin_key_b2,
                            ),
                        )

                        if os.path.exists(load_path):
                            self._pseudo_cls[ver][
                                f"spectra_{bin_key_a1}{bin_key_a2}_spectra_{bin_key_b1}{bin_key_b2}"
                            ] = fits.open(load_path)

                            if bin_key_a1 == bin_key_b1 and bin_key_a2 == bin_key_b2:
                                self._pseudo_cls[ver][
                                    f"tomo_bin_{bin_key_a1}_tomo_bin_{bin_key_a2}"
                                ]["cov"] = fits.open(load_path)
                        else:
                            raise FileNotFoundError(
                                "The file does not exist, please run `cosmo_val` with `force_run` to True."
                            )

                continue

            # Initialise dictionnary to store field and workspace
            n_gal_map_dict = {}
            field_dict = {}
            wsp_dict = {}

            self.print_cyan(f"Extracting the fiducial power spectrum for {ver}")

            fiducial_cl = self.get_fiducial_cl(ver, compute_tomography)

            # Add a zero multipole and drop the last value to align
            # Namaster and fiducial binning
            fiducial_cl = {
                key: np.concatenate(([0.0], np.asarray(value[:-1], dtype=float)))
                for key, value in fiducial_cl.items()
            }

            self.print_cyan(
                "Estimating and adding the noise bias to the fiducial power spectra"
            )
            self.print_cyan(f"Method used: {self.noise_bias_method}")

            params = get_params_rho_tau(self.cc[ver])
            cat_gal = fits.getdata(self.cc[ver]["shear"]["path"])

            lmin, lmax, b_lmax = spv_pseudo_cl.pseudo_cl_geometry(self.nside)
            b = self.get_namaster_bin(lmin, lmax, b_lmax)

            # Compute the fields and workspaces
            for bin_key1, bin_key2 in tomo_bin_pairs:
                self.print_cyan(
                    f"Computing fields and workspaces for {bin_key1}, {bin_key2}"
                )

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
                if f"W{bin_key1}" not in n_gal_map_dict:
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
                if bin_key1 == bin_key2:
                    self.print_cyan(
                        f"Adding the noise bias for tomographic bins {bin_key1}, {bin_key2}"
                    )

                    cat_gal_ = self._get_tomographic_bin(params, cat_gal, bin_key1)

                    noise_bias_cl = self.get_noise_bias(params, nside, cat_gal_)

                    # If using the analytical noise bias, we need to decouple it
                    if self.noise_bias_method == "analytic":
                        self.print_cyan(
                            "Decoupling the noise bias for the analytic method..."
                        )
                        wsp = wsp_dict[f"W{bin_key1}xW{bin_key2}"]
                        noise_bias_cl = wsp.decouple_cell(noise_bias_cl)

                        # And then unbin it
                        noise_bias_cl = b.unbin_cell(noise_bias_cl)

                        # Force the bins not part of the mask to be zero
                        lowest_ell = b.get_ell_list(0)[0]
                        for i in range(4):
                            noise_bias_cl[i, :lowest_ell] = 0

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

            if self.fiducial_input_inka == "coupled":
                # Couple the cell if required
                self.print_cyan("Coupling the fiducial Cls.")
                for bin_key1, bin_key2 in tomo_bin_pairs:
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

            # To compute all the blocks in the covariance,
            # we must compute all the (i, j, k, l) bin combinations
            # Or equivalently, compute the covariance for all spectra pairs
            n_spectra = len(tomo_bin_pairs)

            # indices into tomo_bin_pairs for all unique covariance blocks
            block_indices = list(
                itertools.combinations_with_replacement(range(n_spectra), 2)
            )
            # Loop on the different tomographic bin pairs to compute the covariance
            for index_a, index_b in block_indices:
                bin_key_a1, bin_key_a2 = tomo_bin_pairs[index_a]
                bin_key_b1, bin_key_b2 = tomo_bin_pairs[index_b]
                self.print_cyan(
                    f"Tomo Bin Quad: ({bin_key_a1}, {bin_key_a2}, {bin_key_b1}, {bin_key_b2})"
                )

                if (
                    bin_key_a1 == bin_key_b1
                    and bin_key_a2 == bin_key_b2
                    and (
                        f"tomo_bin_{bin_key_a1}_tomo_bin_{bin_key_a2}"
                        not in self._pseudo_cls[ver].keys()
                    )
                ):
                    self._pseudo_cls[ver][
                        f"tomo_bin_{bin_key_a1}_tomo_bin_{bin_key_a2}"
                    ] = {}

                out_path = self._output_path_iNKA_block_cov(
                    ver, tomo_bin_quad=(bin_key_a1, bin_key_a2, bin_key_b1, bin_key_b2)
                )

                if os.path.exists(out_path) and not self.force_run:
                    self.print_done(
                        f"Skipping Pseudo-Cl covariance block Cl ({bin_key_a1, bin_key_a2}), Cl ({bin_key_b1, bin_key_b2}) calculation, {out_path} exists"
                    )

                    self._pseudo_cls[ver][
                        f"spectra_{bin_key_a1}{bin_key_a2}_spectra_{bin_key_b1}{bin_key_b2}"
                    ] = fits.open(out_path)

                    if bin_key_a1 == bin_key_b1 and bin_key_a2 == bin_key_b2:
                        self._pseudo_cls[ver][
                            f"tomo_bin_{bin_key_a1}_tomo_bin_{bin_key_a2}"
                        ]["cov"] = fits.open(out_path)

                    continue

                self.print_cyan("Computing the Pseudo-Cl covariance")

                input_cl_a1_b1 = (
                    fiducial_cl[f"W{bin_key_a1}xW{bin_key_b1}"]
                    if bin_key_a1 <= bin_key_b1
                    else fiducial_cl[f"W{bin_key_b1}xW{bin_key_a1}"]
                )
                input_cl_a1_b2 = (
                    fiducial_cl[f"W{bin_key_a1}xW{bin_key_b2}"]
                    if bin_key_a1 <= bin_key_b2
                    else fiducial_cl[f"W{bin_key_b2}xW{bin_key_a1}"]
                )
                input_cl_a2_b1 = (
                    fiducial_cl[f"W{bin_key_a2}xW{bin_key_b1}"]
                    if bin_key_a2 <= bin_key_b1
                    else fiducial_cl[f"W{bin_key_b1}xW{bin_key_a2}"]
                )
                input_cl_a2_b2 = (
                    fiducial_cl[f"W{bin_key_a2}xW{bin_key_b2}"]
                    if bin_key_a2 <= bin_key_b2
                    else fiducial_cl[f"W{bin_key_b2}xW{bin_key_a2}"]
                )

                covar_22_22 = spv_pseudo_cl.get_pseudo_cl_iNKA_covariance(
                    input_cl_a1_b1,
                    input_cl_a1_b2,
                    input_cl_a2_b1,
                    input_cl_a2_b2,
                    field_dict[f"W{bin_key_a1}"],
                    field_dict[f"W{bin_key_a2}"],
                    field_dict[f"W{bin_key_b1}"],
                    field_dict[f"W{bin_key_b2}"],
                    wsp_a=wsp_dict[f"W{bin_key_a1}xW{bin_key_a2}"],
                    wsp_b=wsp_dict[f"W{bin_key_b1}xW{bin_key_b2}"],
                    b=b,
                )

                self.print_cyan("Saving Pseudo-Cl covariance")

                self._pseudo_cls[ver][
                    f"spectra_{bin_key_a1}{bin_key_a2}_spectra_{bin_key_b1}{bin_key_b2}"
                ] = self._save_iNKA_covariance(covar_22_22, out_path)

                if bin_key_a1 == bin_key_b1 and bin_key_a2 == bin_key_b2:
                    self._pseudo_cls[ver][
                        f"tomo_bin_{bin_key_a1}_tomo_bin_{bin_key_a2}"
                    ]["cov"] = self._pseudo_cls[ver][
                        f"spectra_{bin_key_a1}{bin_key_a2}_spectra_{bin_key_b1}{bin_key_b2}"
                    ]

            # Merge the covariance blocks
            self._pseudo_cls[ver][f"cov_iNKA_{tomo_str}"] = self._merge_iNKA_covariance(
                ver, tomography=compute_tomography
            )
            self.print_done(f"Done Pseudo-Cl covariance calculation for {ver}")
        self.print_done("Done Pseudo-Cl covariance")

    def calculate_pseudo_cl_onecovariance(self, tomography=False):
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

        if not hasattr(self, "_pseudo_cls_onecov"):
            self._pseudo_cls_onecov = {}
        for ver in self.versions:
            self.print_magenta(ver)

            out_dir = self._output_path(
                "pseudo_cl/", f"pseudo_cl_cov_onecov_{ver}_tomography_{tomography}"
            )
            os.makedirs(out_dir, exist_ok=True)

            if (
                os.path.exists(
                    os.path.join(out_dir, "covariance_list_3x2pt_pure_Cell.dat")
                )
                and not self.force_run
            ):
                self.print_done(f"Skipping OneCovariance calculation, {out_dir} exists")
                self._load_onecovariance_cov(out_dir, ver, tomography)
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

                config_path = os.path.join(
                    out_dir, f"config_onecov_{ver}_tomography_{tomography}.ini"
                )

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
                    tomography,
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
                self._load_onecovariance_cov(out_dir, ver, tomography)

        self.print_done("Done Pseudo-Cl covariance with OneCovariance")

    def calculate_pseudo_cl_g_ng_cov(self, tomography=False, gaussian_part="iNKA"):
        assert gaussian_part in ["iNKA", "OneCovariance"], (
            "gaussian_part must be 'iNKA' or 'OneCovariance'"
        )
        self.print_start(
            f"Gaussian and Non-Gaussian covariance of the Pseudo-Cl's using {gaussian_part} for the Gaussian part"
        )

        if not hasattr(self, "_pseudo_cls_cov_g_ng"):
            self._pseudo_cls_cov_g_ng = {}

        for ver in self.versions:
            self.print_magenta(ver)
            self._pseudo_cls_cov_g_ng.setdefault(ver, {})
            key_to_use = "tomo" if tomography else "non_tomo"
            out_file = self._output_path(
                f"pseudo_cl_cov_g_ng_{gaussian_part}_{ver}_tomography_{tomography}.fits"
            )
            if os.path.exists(out_file) and not self.force_run:
                self.print_done(
                    f"Skipping Gaussian and Non-Gaussian covariance calculation, {out_file} exists"
                )
                cov_hdu = fits.open(out_file)
                self._pseudo_cls_cov_g_ng[ver][key_to_use] = cov_hdu
                continue
            if gaussian_part == "iNKA":
                gaussian_cov = self.pseudo_cls[ver][f"cov_iNKA_{key_to_use}"][
                    "COVAR_EE_EE"
                ].data
                non_gaussian_cov = (
                    self.pseudo_cls_onecov[ver][key_to_use]["all_cov"]
                    - self.pseudo_cls_onecov[ver][key_to_use]["gaussian_cov"]
                )
                full_cov = gaussian_cov + non_gaussian_cov
            elif gaussian_part == "OneCovariance":
                gaussian_cov = self.pseudo_cls_onecov[ver][key_to_use]["gaussian_cov"]
                non_gaussian_cov = (
                    self.pseudo_cls_onecov[ver][key_to_use]["all_cov"]
                    - self.pseudo_cls_onecov[ver][key_to_use]["gaussian_cov"]
                )
                full_cov = self.pseudo_cls_onecov[ver][key_to_use]["all_cov"]
            else:
                raise ValueError(f"Unknown gaussian_part: {gaussian_part}")
            self.print_cyan("Saving Gaussian and Non-Gaussian covariance...")
            hdu = fits.HDUList()
            hdu.append(fits.ImageHDU(gaussian_cov, name="COVAR_GAUSSIAN"))
            hdu.append(fits.ImageHDU(non_gaussian_cov, name="COVAR_NON_GAUSSIAN"))
            hdu.append(fits.ImageHDU(full_cov, name="COVAR_FULL"))
            hdu.writeto(out_file, overwrite=True)
            self._pseudo_cls_cov_g_ng[ver][key_to_use] = hdu
        self.print_done(
            f"Done Gaussian and Non-Gaussian covariance of the Pseudo-Cl's using {gaussian_part} for the Gaussian part"
        )

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
        redshift_distribution = np.loadtxt(path_redshift_distr)
        z = redshift_distribution[:, 0]
        dndz = redshift_distribution[:, 1:]

        # Here it is assumed that the tomographic redshift distribution sum to the non-tomographic one and that the latter is normalised
        if not is_tomography:
            dndz = np.sum(dndz, axis=1)

        return z, dndz

    def get_fiducial_cl(self, ver, is_tomography):
        """Get a theory prediction for the angular power spectra (thin wrapper, state -> primitive)."""
        lmax = 2 * self.nside

        z, dndz = self.read_redshift_distribution(ver, is_tomography)

        fiducial_cl = spv_pseudo_cl.get_fiducial_cl(z, dndz, lmax, self.cosmo)

        # If non-tomographic, change the key to 'WallxWall'
        if not is_tomography:
            fiducial_cl = {"WallxWall": fiducial_cl["W1xW1"]}

        return fiducial_cl

    def _modify_onecov_config(
        self,
        template_config,
        config_path,
        out_dir,
        mask_path,
        redshift_distr_path,
        ver,
        tomography,
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
        ver : str
            Version identifier for the current analysis
        tomography : bool
            Whether to compute tomography or not
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

        # Update ellipticity dispersion and effective number density
        # Account for tomography if needed
        if tomography:
            tomo_bin_ids, _ = self._get_tomo_bins(ver)
        else:
            tomo_bin_ids = ["all"]
        input_ellipticity_distribution = ", ".join(
            str(self.ellipticity_dispersion[ver][f"tomo_bin_{bin_id}"])
            for bin_id in tomo_bin_ids
        )
        input_n_eff_gal = ", ".join(
            str(self.n_eff_gal[ver][f"tomo_bin_{bin_id}"]) for bin_id in tomo_bin_ids
        )
        config["survey specs"]["ellipticity_dispersion"] = (
            input_ellipticity_distribution
        )
        config["survey specs"]["n_eff_lensing"] = input_n_eff_gal

        # Update redshift distribution path
        redshift_distr_base = os.path.basename(os.path.abspath(redshift_distr_path))
        redshift_distr_folder = os.path.dirname(os.path.abspath(redshift_distr_path))
        config["redshift"]["z_directory"] = redshift_distr_folder
        config["redshift"]["zlens_file"] = redshift_distr_base

        # TODO: add an update of the config file cosmological parameters using the cosmo object
        self._update_onecov_cosmo_params(config, self.cosmo)

        # Update output directory
        config["output settings"]["directory"] = out_dir

        # Save the modified configuration
        with open(config_path, "w") as f:
            config.write(f)

    def _update_onecov_cosmo_params(self, config, cosmo):
        """
        Update the cosmological parameters in the OneCovariance configuration.

        Parameters
        ----------
        config : configparser.ConfigParser
            The configuration object to update.
        cosmo
            The cosmology object containing the parameters to set.
        """
        # Update cosmo params section
        config["cosmo"]["h"] = str(cosmo["H0"] / 100.0)
        config["cosmo"]["omega_m"] = str(cosmo["Omega_m"])
        config["cosmo"]["omega_b"] = str(cosmo["Omega_b"])
        config["cosmo"]["omega_de"] = str(cosmo["Omega_de"])
        config["cosmo"]["sigma8"] = str(cosmo.sigma8())
        config["cosmo"]["ns"] = str(cosmo["n_s"])
        config["cosmo"]["w0"] = str(cosmo["w0"])
        config["cosmo"]["wa"] = str(cosmo["wa"])
        config["cosmo"]["neff"] = str(cosmo["Neff"])
        config["cosmo"]["m_nu"] = str(cosmo["m_nu"])

        # Update powspec evaluation section
        cosmo_dict = cosmo.to_dict()

        config["powspec evaluation"]["non_linear_model"] = str(
            cosmo_dict.get("matter_power_spectrum")
        )
        config["powspec evaluation"]["HMCode_logT_AGN"] = str(
            cosmo_dict.get("extra_parameters", {})
            .get("camb", {})
            .get("HMCode_logT_AGN")
        )

    def _load_onecovariance_cov(self, out_dir, ver, tomography):
        self.print_cyan(f"Loading OneCovariance results from {out_dir}")
        cov_one_cov = np.genfromtxt(
            os.path.join(out_dir, "covariance_list_3x2pt_pure_Cell.dat")
        )
        gaussian_one_cov = cov_from_one_covariance(cov_one_cov, gaussian=True)
        all_one_cov = cov_from_one_covariance(cov_one_cov, gaussian=False)

        key_to_update = "tomo" if tomography else "non_tomo"
        self._pseudo_cls_onecov.setdefault(ver, {}).update(
            {
                key_to_update: {
                    "gaussian_cov": gaussian_one_cov,
                    "all_cov": all_one_cov,
                }
            }
        )

    def _get_tomographic_bin(self, params, cat_gal, tomo_bin):
        """Extract tomographic bin from a given catalogue"""
        if tomo_bin == "all":
            return cat_gal
        else:
            tomo_bin_id = cat_gal[params["tomo_bin_col"]]
            mask = tomo_bin_id == tomo_bin
            return cat_gal[mask]

    def _output_path_pseudo_cl(self, ver, tomo_bin_pair=None):
        if tomo_bin_pair is None:
            return self._output_path(
                "pseudo_cl",
                f"pseudo_cl_from_{self.cell_method}_non_tomo_{ver}_binning_{self.binning}_nbins_{self.n_ell_bins}.fits",
            )
        else:
            bin_key1, bin_key2 = tomo_bin_pair
            return self._output_path(
                "pseudo_cl",
                f"pseudo_cl_from_{self.cell_method}_tomo_bin_{bin_key1}_tomo_bin_{bin_key2}_{ver}_binning_{self.binning}_nbins_{self.n_ell_bins}.fits",
            )

    def _output_path_pseudo_cl_cov(self, ver, method, tomography):
        is_tomo = "tomo" if tomography else "non_tomo"
        return self._output_path(
            "pseudo_cl",
            f"pseudo_cl_cov_{is_tomo}_{ver}_from_{method}_binning_{self.binning}_nbins_{self.n_ell_bins}.fits",
        )

    def _output_path_iNKA_block_cov(self, ver, tomo_bin_quad):
        bin_key_a1, bin_key_a2, bin_key_b1, bin_key_b2 = tomo_bin_quad
        return self._output_path(
            "pseudo_cl",
            f"iNKA_block_{ver}",
            f"pseudo_cl_cov_from_iNKA_tomo_bin_{bin_key_a1}_tomo_bin_{bin_key_a2}_tomo_bin_{bin_key_b1}_tomo_bin_{bin_key_b2}_{ver}_binning_{self.binning}_nbins_{self.n_ell_bins}.fits",
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
        col4 = fits.Column(name="BE", format="D", array=pseudo_cl[2])
        col5 = fits.Column(name="BB", format="D", array=pseudo_cl[3])
        coldefs = fits.ColDefs([col1, col2, col3, col4, col5])
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

    def _merge_iNKA_covariance(self, ver, tomography):
        """
        Merge the iNKA covariance matrices for a given version to get the data vector covariance.
        """
        out_path = self._output_path_pseudo_cl_cov(
            ver, method="iNKA", tomography=tomography
        )

        if tomography:
            # Merge the tomographic covariance matrices
            tomo_bin_ids, tomo_bin_pairs = self._get_tomo_bins(ver)

            # Get the number of bins from the pseudo_cls attribute
            # The non-tomographic pseudo-cl are computed from the call
            # to this attribute if not already computed.
            n_ell = self._pseudo_cls[ver]["tomo_bin_all_tomo_bin_all"]["pseudo_cl"][
                "ELL"
            ].shape[0]

            if tomo_bin_ids is None or tomo_bin_pairs is None:
                raise AssertionError(
                    f"Tomographic bin IDs of version {ver} is not available."
                )

            n_spectra = len(tomo_bin_pairs)
            block_indices = list(
                itertools.combinations_with_replacement(range(n_spectra), 2)
            )

            pols = ["EE", "EB", "BE", "BB"]
            covar = fits.HDUList()
            for pa in pols:
                for pb in pols:
                    full_cov = np.zeros((n_spectra * n_ell, n_spectra * n_ell))
                    for index_a, index_b in block_indices:
                        bin_key_a1, bin_key_a2 = tomo_bin_pairs[index_a]
                        bin_key_b1, bin_key_b2 = tomo_bin_pairs[index_b]

                        block_path = self._output_path_iNKA_block_cov(
                            ver,
                            tomo_bin_quad=(
                                bin_key_a1,
                                bin_key_a2,
                                bin_key_b1,
                                bin_key_b2,
                            ),
                        )
                        block = fits.open(block_path)[f"COVAR_{pa}_{pb}"].data

                        sl_a = slice(index_a * n_ell, (index_a + 1) * n_ell)
                        sl_b = slice(index_b * n_ell, (index_b + 1) * n_ell)

                        full_cov[sl_a, sl_b] = block

                        if index_a != index_b:
                            full_cov[sl_b, sl_a] = block.T

                    covar.append(fits.ImageHDU(full_cov, name=f"COVAR_{pa}_{pb}"))

        else:
            block_path = self._output_path_iNKA_block_cov(
                ver, tomo_bin_quad=("all", "all", "all", "all")
            )
            covar = fits.open(block_path)

        covar.writeto(out_path, overwrite=True)

        return covar

    # ---------------- Plotting functions for pseudo-Cl's ---------------- #
    def plot_pseudo_cl(
        self,
        pol_list,
        versions=None,
        ell_factor="ell",
        cov_type="iNKA",
        offset=0.15,
        tomography=True,
        savefig=None,
        show=True,
    ):
        """
        Plot the pseudo-Cl for EE power spectrum.

        Parameters
        ----------
        pol_list : list
            List of polarization types to plot (e.g., ["EE", "BB"]).
        ell_factor : {"None", "ell", "ell(ell+1)"}
            Factor to multiply the ell values by.
        cov_type : {"iNKA"}
            Type of covariance to use.
        tomography : bool, optional
            Whether to plot tomographic power spectra.
        savefig : str, optional
            Path to save the figure.
        show : bool, optional
            Whether to show the figure.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
            Figure and axes objects for the plot.
        """
        if versions is None:
            versions = self.versions
        else:
            for ver in versions:
                if ver not in self.versions:
                    raise ValueError(
                        f"Version {ver} is not available. Available versions: {self.versions}"
                    )
        # Check that the method for the covariance is valid
        if cov_type not in ["iNKA"]:
            raise ValueError(
                f"Invalid covariance type: {cov_type}. Valid options are: ['iNKA']"
            )

        # Check that the ell_factor is valid
        if ell_factor not in ["None", "ell", "ell(ell+1)"]:
            raise ValueError(
                f"Invalid ell_factor: {ell_factor}. Valid options are: ['None', 'ell', 'ell(ell+1)']"
            )

        def get_ell_factor(ell):
            """Given some ell, return the ell_factor for the plot."""
            if ell_factor == "None":
                return 1
            elif ell_factor == "ell":
                return ell
            elif ell_factor == "ell(ell+1)":
                return ell * (ell + 1)

        # Check that all items in the list are valid polarisation
        valid_pols = ["EE", "BB", "EB", "BE"]
        for pol in pol_list:
            if pol not in valid_pols:
                raise ValueError(
                    f"Invalid polarization type: {pol}. Valid options are: {valid_pols}"
                )

        fmt_dict = {"EE": "o", "BB": "s", "EB": "^", "BE": "v"}
        # From all the versions, get the maximum number of tomo_bin_ids
        tomo_bins = {}
        for ver in versions:
            if tomography:
                tomo_bin_ids, tomo_bin_pairs = self._get_tomo_bins(ver)
            else:
                tomo_bin_ids, tomo_bin_pairs = ["all"], [("all", "all")]

            tomo_bins[ver] = {"ids": tomo_bin_ids, "pairs": tomo_bin_pairs}

        n_tomo_bins_plot = max(len(bins["ids"]) for bins in tomo_bins.values())

        fig, axs = plt.subplots(
            n_tomo_bins_plot,
            n_tomo_bins_plot,
            figsize=(12, 12),
            sharex=True,
            sharey=True,
        )

        for j, ver in enumerate(versions):
            # Plot the pseudo-cl for each tomo bin of the considered version
            for tomo_bin_a, tomo_bin_b in tomo_bins[ver]["pairs"]:
                if tomography:
                    ax = axs[tomo_bin_b - 1, tomo_bin_a - 1]
                else:
                    ax = axs
                ver_tomo_info = self._pseudo_cls[ver][
                    f"tomo_bin_{tomo_bin_a}_tomo_bin_{tomo_bin_b}"
                ]
                ver_label = self.cc[ver]["label"] if "label" in self.cc[ver] else ver
                ver_color = (
                    self.cc[ver]["colour"] if "colour" in self.cc[ver] else "black"
                )

                pseudo_cls = ver_tomo_info["pseudo_cl"]
                cov = ver_tomo_info["cov"]

                ell = pseudo_cls["ell"]

                ell_widths = np.diff(ell)
                ell_widths = np.append(
                    ell_widths, ell_widths[-1]
                )  # Assume last bin width is same as second last

                # Better jittering: symmetric around original ell values
                jitter_fraction = (j - (len(versions) - 1) / 2) * offset
                jittered_ell = ell + jitter_fraction * ell_widths
                ell_factor_ = get_ell_factor(jittered_ell)

                for pol in pol_list:
                    pol_color = self.get_pol_color(ver_color, pol, pol_list)
                    ax.errorbar(
                        jittered_ell,
                        ell_factor_ * pseudo_cls[pol],
                        yerr=np.sqrt(np.diag(cov[f"COVAR_{pol}_{pol}"].data))
                        * ell_factor_,
                        fmt=fmt_dict[pol],
                        label=ver_label + f" {pol}",
                        color=pol_color,
                        capsize=2,
                    )

        # Draw to extract the yaxis text offset
        fig.canvas.draw()

        if ell_factor == "None":
            ell_label = r"$C_\ell$"
        elif ell_factor == "ell":
            ell_label = r"$\ell C_\ell$"
        else:
            ell_label = r"$\ell(\ell+1) C_\ell$"

        for tomo_bin_a, tomo_bin_b in tomo_bin_pairs:
            if tomography:
                ax = axs[tomo_bin_b - 1, tomo_bin_a - 1]
            else:
                ax = axs
            ax.text(
                0.8,
                0.95,
                f"{tomo_bin_a}-{tomo_bin_b}",
                transform=ax.transAxes,
                verticalalignment="top",
            )
            ax.axhline(0, color="black", ls="--")
            ax.set_xlim(ell.min(), ell.max())
            ax.set_xscale("squareroot")
            ax.set_xticks(np.array([100, 400, 900, 1600]))
            ax.minorticks_on()
            minor_ticks = [i * 10 for i in range(1, 10)] + [
                i * 100 for i in range(1, 21)
            ]
            ax.xaxis.set_ticks(minor_ticks, minor=True)
            ax.tick_params(axis="both", which="both", direction="in")
            if tomo_bin_b == 6 or tomo_bin_b == "all":
                ax.set_xlabel(r"$\ell$")
            if tomo_bin_a == 1 or tomo_bin_a == "all":
                text_offset = ax.yaxis.get_offset_text().get_text()
                ax.yaxis.get_offset_text().set_visible(False)
                ax.set_ylabel(f"{ell_label}{text_offset}")
            else:
                ax.yaxis.get_offset_text().set_visible(False)
            if tomo_bin_a != tomo_bin_b:
                ax = axs[tomo_bin_a - 1, tomo_bin_b - 1]
                ax.set_visible(False)

        # Setup the legend
        plt.subplots_adjust(hspace=0.0, wspace=0.0)  # Remove space between subplots

        legend_ax = self._add_grouped_legend(fig, versions, self.cc, pol_list, fmt_dict)

        if savefig is not None:
            plt.savefig(savefig, dpi=300, bbox_inches="tight")
        if show:
            plt.show()

        return fig, axs, legend_ax

    def _add_grouped_legend(
        self,
        fig,
        versions,
        cc,
        pol_list,
        fmt_dict,
        row_height=0.35,
        label_width=2,
        col_width=0.9,
        gap_below=0.10,
        capsize=2,
        fontsize=10,
    ):
        """
        Add a custom legend below the figure with one row per version:
            <version label>   <marker> <pol>   <marker> <pol>  ...

        The box size (in inches) scales with the number of versions (rows)
        and polarizations (columns), then gets converted to figure-fraction
        coordinates so it works for any figsize.

        Parameters
        ----------
        row_height : float
            Height per row (per version), in inches.
        label_width : float
            Width reserved for the version label column, in inches.
        col_width : float
            Width per polarization column (marker + pol label), in inches.
        gap_below : float
            Vertical gap between the bottom of the subplot grid and the
            top of the legend box, in inches.
        """
        n_rows = len(versions)
        n_pol = len(pol_list)

        # Desired box size in inches
        box_width_in = label_width + n_pol * col_width
        box_height_in = n_rows * row_height

        fig_w_in, fig_h_in = fig.get_size_inches()

        # Convert to figure-fraction
        width_frac = box_width_in / fig_w_in
        height_frac = box_height_in / fig_h_in
        gap_frac = gap_below / fig_h_in

        left = 0.5 - width_frac / 2  # centered horizontally
        bottom = -gap_frac - height_frac  # just below the subplot grid (y=0)

        legend_ax = fig.add_axes([left, bottom, width_frac, height_frac])
        legend_ax.axis("off")
        legend_ax.set_xlim(0, 1)
        legend_ax.set_ylim(0, 1)

        # Fractions *within* legend_ax's own 0-1 coordinate system, derived
        # from the same inch-based proportions so columns stay consistent
        label_frac = label_width / box_width_in
        col_frac = col_width / box_width_in

        dummy_yerr = 0.15 / n_rows

        for i, ver in enumerate(versions):
            y = 1.0 - (i + 0.5) / n_rows

            ver_label = cc[ver]["label"] if "label" in cc[ver] else ver
            ver_color = cc[ver]["colour"] if "colour" in cc[ver] else "black"

            legend_ax.text(
                0.02 * label_frac,
                y,
                ver_label,
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=fontsize,
            )

            for k, pol in enumerate(pol_list):
                pol_color = self.get_pol_color(ver_color, pol, pol_list)
                x_marker = label_frac + k * col_frac + 0.15 * col_frac
                x_text = x_marker + 0.15 * col_frac

                legend_ax.errorbar(
                    [x_marker],
                    [y],
                    yerr=dummy_yerr,
                    fmt=fmt_dict[pol],
                    color=pol_color,
                    markersize=6,
                    capsize=capsize,
                    clip_on=False,
                )
                legend_ax.text(
                    x_text,
                    y,
                    pol,
                    ha="left",
                    va="center",
                    fontsize=fontsize - 1,
                )

        return legend_ax

    def get_pol_color(self, base_color, pol, pol_list, lightness_range=(-0.25, 0.25)):
        """
        Given a base version color, return a shade variant for a given
        polarization, by adjusting lightness in HLS space while keeping
        hue and saturation fixed.

        Parameters
        ----------
        base_color : str or tuple
            Any matplotlib-recognized color (hex, name, RGB tuple, etc.)
        pol : str
            The polarization this color is for (e.g. "EE").
        pol_list : list
            Full list of polarizations being plotted, used to compute
            this pol's relative position in the lightness range.
        lightness_range : tuple of float
            (min_offset, max_offset) added to the base lightness, spread
            evenly across pol_list. Negative = darker, positive = lighter.
            Values are in HLS lightness units (0-1 scale), so keep these
            modest (e.g. +/-0.25) to avoid washing out to white or black.

        Returns
        -------
        tuple
            RGB color tuple in [0, 1] range, usable directly in matplotlib.
        """
        r, g, b = to_rgb(base_color)
        h, lightness, s = colorsys.rgb_to_hls(r, g, b)

        n_pol = len(pol_list)
        idx = pol_list.index(pol)

        if n_pol == 1:
            l_offset = 0.0
        else:
            # spread idx evenly across [lightness_range[0], lightness_range[1]]
            frac = idx / (n_pol - 1)  # 0 to 1
            l_offset = lightness_range[0] + frac * (
                lightness_range[1] - lightness_range[0]
            )

        lightness_new = min(
            max(lightness + l_offset, 0.05), 0.95
        )  # clamp to avoid pure black/white

        r_new, g_new, b_new = colorsys.hls_to_rgb(h, lightness_new, s)
        return (r_new, g_new, b_new)
