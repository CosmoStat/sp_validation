"""Catalog characterization diagnostic for cosmology validation.

This mixin holds the survey-statistics and catalog-diagnostic machinery:
effective survey statistics (area, effective number density, shape noise),
the per-version area / n_eff / ellipticity-dispersion calculations, the catalog
diagnostic plots (footprints, ellipticity, weight, and separation histograms),
and the additive-bias (c1/c2) estimation. It depends on healpy, the cs_util
plotting helpers, and astropy.io.fits.
"""

import os
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from cs_util import plots as cs_plots


class CatalogCharacterizationMixin:
    def compute_survey_stats(
        self,
        ver,
        weights_key_override=None,
        mask_path=None,
        nside=None,
        overwrite_config=False,
    ):
        """Compute effective survey statistics for a catalog version.

        Parameters
        ----------
        ver : str
            Version string registered in the catalog config.
        weights_key_override : str, optional
            Override the weight column key (defaults to the configured `w_col`).
        mask_path : str, optional
            Explicit mask path to use when measuring survey area.
        nside : int, optional
            If provided, compute survey area from the catalog using this NSIDE when no
            mask path is available.
        overwrite_config : bool, optional
            If True, persist the derived statistics back to the catalog configuration.

        Returns
        -------
        dict
            Dictionary containing:
            - area_deg2: Survey area in square degrees.
            - n_eff: Effective number density per arcmin^2.
            - sigma_e: Per-component shape noise.
            - sum_w: Sum of weights.
            - sum_w2: Sum of squared weights.
            - catalog_size: Number of galaxies processed.
        """
        if ver not in self.cc:
            raise KeyError(f"Version {ver} not found in catalog configuration")

        shear_cfg = self.cc[ver]["shear"]
        cov_th = self.cc[ver].get("cov_th", {})

        if "path" not in shear_cfg:
            raise KeyError(f"No shear catalog path defined for version {ver}")

        catalog_path = shear_cfg["path"]
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Shear catalog not found: {catalog_path}")

        data = fits.getdata(catalog_path, memmap=True)
        n_rows = len(data)

        e1 = np.asarray(data[shear_cfg["e1_col"]], dtype=float)
        e2 = np.asarray(data[shear_cfg["e2_col"]], dtype=float)

        weight_column = weights_key_override or shear_cfg["w_col"]
        if weight_column not in data.columns.names:
            raise KeyError(f"Weight column '{weight_column}' missing in {catalog_path}")

        w = np.asarray(data[weight_column], dtype=float)

        sum_w = float(np.sum(w))
        sum_w2 = float(np.sum(w**2))
        sum_w2_e2 = float(np.sum((w**2) * (e1**2 + e2**2)))

        if mask_path is not None:
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask path not found: {mask_path}")
            mask_candidate = mask_path
        else:
            mask_candidate = self.cc[ver].get("mask")
            if isinstance(mask_candidate, str) and not os.path.isabs(mask_candidate):
                mask_candidate = str(Path(self.cc[ver]["subdir"]) / mask_candidate)
            if mask_candidate is not None and not os.path.exists(mask_candidate):
                mask_candidate = None

        area_deg2 = None
        if mask_candidate is not None and os.path.exists(mask_candidate):
            area_deg2 = self._area_from_mask(mask_candidate)
        elif cov_th.get("A") is not None:
            area_deg2 = float(cov_th["A"])
        elif nside is not None:
            area_deg2 = self._area_from_catalog(catalog_path, nside)
        else:
            raise ValueError(
                f"Unable to determine survey area for {ver}. Provide mask_path or nside."
            )

        area_arcmin2 = area_deg2 * 3600.0

        n_eff = (sum_w**2) / (area_arcmin2 * sum_w2) if sum_w2 > 0 else 0.0
        sigma_e = np.sqrt(sum_w2_e2 / sum_w2) if sum_w2 > 0 else 0.0

        results = {
            "area_deg2": area_deg2,
            "n_eff": n_eff,
            "sigma_e": sigma_e,
            "sum_w": sum_w,
            "sum_w2": sum_w2,
            "catalog_size": n_rows,
        }

        if overwrite_config:
            self.cc[ver].setdefault("cov_th", {}).update(
                A=float(area_deg2), n_e=float(n_eff), sigma_e=float(sigma_e)
            )
            self._write_catalog_config()

        return results

    def _area_from_catalog(self, catalog_path, nside):
        data = fits.getdata(catalog_path, memmap=True)
        ra = np.asarray(data["RA"], dtype=float)
        dec = np.asarray(data["Dec"], dtype=float)
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        pix = hp.ang2pix(nside, theta, phi, lonlat=False)
        unique_pix = np.unique(pix)
        return float(unique_pix.size * hp.nside2pixarea(nside, degrees=True))

    def _area_from_mask(self, mask_map_path):
        mask = hp.read_map(mask_map_path, dtype=np.float64)
        return float(mask.sum() * hp.nside2pixarea(hp.get_nside(mask), degrees=True))

    @property
    def area(self):
        if not hasattr(self, "_area"):
            self.calculate_area()
        return self._area

    @property
    def n_eff_gal(self):
        if not hasattr(self, "_n_eff_gal"):
            self.calculate_n_eff_gal()
        return self._n_eff_gal

    @property
    def ellipticity_dispersion(self):
        if not hasattr(self, "_ellipticity_dispersion"):
            self.calculate_ellipticity_dispersion()
        return self._ellipticity_dispersion

    def _get_binned_catalog_mask(self, ver):
        with self.results[ver].temporarily_read_data():
            ra = self.results[ver].dat_shear["RA"]
            dec = self.results[ver].dat_shear["Dec"]
            hsp_map = hp.ang2pix(
                self.nside_mask,
                np.radians(90 - dec),
                np.radians(ra),
                lonlat=False,
            )
            mask = np.bincount(hsp_map, minlength=hp.nside2npix(self.nside_mask)) > 0
        return mask

    def calculate_area(self):
        self.print_start("Calculating area")
        area = {}
        for ver in self.versions:
            self.print_magenta(ver)

            if "mask" not in self.cc[ver]:
                print(
                    "Mask not found in config file, calculating area from binned catalog"
                )
                area[ver] = self.calculate_area_from_binned_catalog(ver)
            else:
                mask = hp.read_map(self.cc[ver]["mask"], verbose=False)
                nside_mask = hp.get_nside(mask)
                print(f"nside_mask = {nside_mask}")
                area[ver] = np.sum(mask) * hp.nside2pixarea(nside_mask, degrees=True)
            print(f"Area = {area[ver]:.2f} deg^2")

        self._area = area
        self.print_done("Area calculation finished")

    def calculate_area_from_binned_catalog(self, ver):
        print(f"nside_mask = {self.nside_mask}")
        mask = self._get_binned_catalog_mask(ver)

        area = np.sum(mask) * hp.nside2pixarea(self.nside_mask, degrees=True)
        print(f"Area = {area:.2f} deg^2")

        return area

    def calculate_n_eff_gal(self):
        self.print_start("Calculating effective number of galaxy")
        n_eff_gal = {}
        for ver in self.versions:
            self.print_magenta(ver)
            with self.results[ver].temporarily_read_data():
                w = self.results[ver].dat_shear[self.cc[ver]["shear"]["w_col"]]
                n_eff_gal[ver] = (
                    1 / (self.area[ver] * 60 * 60) * np.sum(w) ** 2 / np.sum(w**2)
                )
                print(f"n_eff_gal = {n_eff_gal[ver]:.2f} gal./arcmin^-2")

        self._n_eff_gal = n_eff_gal
        self.print_done("Effective number of galaxy calculation finished")

    def calculate_ellipticity_dispersion(self):
        self.print_start("Calculating ellipticity dispersion")
        ellipticity_dispersion = {}
        for ver in self.versions:
            self.print_magenta(ver)
            with self.results[ver].temporarily_read_data():
                e1 = self.results[ver].dat_shear[self.cc[ver]["shear"]["e1_col"]]
                e2 = self.results[ver].dat_shear[self.cc[ver]["shear"]["e2_col"]]
                w = self.results[ver].dat_shear[self.cc[ver]["shear"]["w_col"]]
                ellipticity_dispersion[ver] = np.sqrt(
                    0.5
                    * (
                        np.average(e1**2, weights=w**2)
                        + np.average(e2**2, weights=w**2)
                    )
                )
                print(f"Ellipticity dispersion = {ellipticity_dispersion[ver]:.4f}")
        self._ellipticity_dispersion = ellipticity_dispersion

    def plot_footprints(self):
        self.print_start("Plotting footprints:")
        for ver in self.versions:
            self.print_magenta(ver)

            fp = cs_plots.FootprintPlotter()

            for region in fp._regions:
                out_path = os.path.abspath(
                    f"{self.cc['paths']['output']}/footprint_{ver}_{region}.png"
                )
            if os.path.exists(out_path):
                self.print_done(f"Skipping footprint plot, {out_path} exists")
            else:
                with self.results[ver].temporarily_read_data():
                    hsp_map = fp.create_hsp_map(
                        self.results[ver].dat_shear["RA"],
                        self.results[ver].dat_shear["Dec"],
                    )
                fp.plot_region(hsp_map, fp._regions[region], outpath=out_path)
                self.print_done("Footprint plot saved to " + out_path)

    def plot_ellipticity(self, nbins=200):
        out_path = os.path.abspath(f"{self.cc['paths']['output']}/ell_hist.png")
        if os.path.exists(out_path):
            self.print_done(f"Skipping ellipticity histograms, {out_path} exists")
        else:
            self.print_start("Computing ellipticity histograms:")

            _fig, axs = plt.subplots(1, 2, figsize=(22, 7))
            bins = np.linspace(-1.1, 1.1, nbins + 1)
            for ver in self.versions:
                self.print_magenta(ver)
                R = self.cc[ver]["shear"]["R"]
                with self.results[ver].temporarily_read_data():
                    e1 = (
                        self.results[ver].dat_shear[self.cc[ver]["shear"]["e1_col"]] / R
                    )
                    e2 = (
                        self.results[ver].dat_shear[self.cc[ver]["shear"]["e2_col"]] / R
                    )
                    w = self.results[ver].dat_shear[self.cc[ver]["shear"]["w_col"]]

                    axs[0].hist(
                        e1,
                        bins=bins,
                        density=True,
                        histtype="step",
                        weights=w,
                        label=ver,
                        color=self.cc[ver]["colour"],
                    )
                    axs[1].hist(
                        e2,
                        bins=bins,
                        density=True,
                        histtype="step",
                        weights=w,
                        label=ver,
                        color=self.cc[ver]["colour"],
                    )

            for idx in (0, 1):
                axs[idx].set_xlabel(f"$e_{idx}$")
                axs[idx].set_ylabel("normalised count")
                axs[idx].legend()
                axs[idx].set_xlim([-1.5, 1.5])
            cs_plots.savefig(out_path, close_fig=False)
            cs_plots.show()
            self.print_done("Ellipticity histograms saved to " + out_path)

    def plot_weights(self, nbins=200):
        out_path = os.path.abspath(f"{self.cc['paths']['output']}/weight_hist.png")
        if os.path.exists(out_path):
            self.print_done(f"Skipping weight histograms, {out_path} exists")
        else:
            self.print_start("Computing weight histograms:")

            plt.figure(figsize=(10, 7))
            for ver in self.versions:
                self.print_magenta(ver)
                with self.results[ver].temporarily_read_data():
                    w = self.results[ver].dat_shear[self.cc[ver]["shear"]["w_col"]]

                    plt.hist(
                        w,
                        bins=nbins,
                        density=False,
                        histtype="step",
                        weights=w,
                        label=ver,
                        color=self.cc[ver]["colour"],
                    )

            plt.xlabel("$w$")
            plt.ylabel("frequency")
            plt.yscale("log")
            plt.legend()
            # plt.xlim([-0.01, 1.2])
            cs_plots.savefig(out_path, close_fig=False)
            cs_plots.show()
            self.print_done("Weight histograms saved to " + out_path)

    def plot_separation(self, nbins=200):
        self.print_start("Separation histograms")
        if "SP_matched_MP_v1.0" in self.versions:
            _fig, axs = plt.subplots(1, 1, figsize=(10, 7))
            with self.results["SP_matched_MP_v1.0"].temporarily_read_data():
                sep = self.results["SP_matched_MP_v1.0"].dat_shear["Separation"]
            axs.hist(
                sep,
                bins=nbins,
                density=False,
                histtype="step",
                label="SP_matched_MP_v1.0",
                color=self.cc["SP_matched_MP_v1.0"]["colour"],
            )
            print("Max separation: %s arcsec" % max(sep))
            axs.set_xlabel(r"Separation $\theta$ [arcsec]")
            axs.legend()
        else:
            self.print_done("SP_matched_MP_v1.0 not in versions")

    def calculate_additive_bias(self):
        self.print_start("Calculating additive bias:")
        self._c1 = {}
        self._c2 = {}
        for ver in self.versions:
            self.print_magenta(ver)
            R = self.cc[ver]["shear"]["R"]
            e1_col, e2_col, w_col = [
                self.cc[ver]["shear"][k] for k in ["e1_col", "e2_col", "w_col"]
            ]
            with self.results[ver].temporarily_read_data():
                self._c1[ver] = np.average(
                    self.results[ver].dat_shear[e1_col] / R,
                    weights=self.results[ver].dat_shear[w_col],
                )
                self._c2[ver] = np.average(
                    self.results[ver].dat_shear[e2_col] / R,
                    weights=self.results[ver].dat_shear[w_col],
                )
        self.print_done("Finished additive bias calculation.")

    @property
    def c1(self):
        if not hasattr(self, "_c1"):
            self.calculate_additive_bias()
        return self._c1

    @property
    def c2(self):
        if not hasattr(self, "_c2"):
            self.calculate_additive_bias()
        return self._c2
