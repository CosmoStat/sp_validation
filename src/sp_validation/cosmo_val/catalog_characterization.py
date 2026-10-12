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
import treecorr
from cs_util import plots as cs_plots
from uncertainties import ufloat

from ..io import open_entry
from ..survey import (
    additive_bias,
    additive_bias_errors,
    area_from_coords,
    effective_survey_stats,
    n_eff_density,
)
from ..survey import ellipticity_dispersion as ellipticity_dispersion_stat


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

        weight_column = weights_key_override or shear_cfg["w_col"]
        data = open_entry(shear_cfg)
        n_rows = len(data)

        e1 = np.asarray(data[shear_cfg["e1_col"]], dtype=float)
        e2 = np.asarray(data[shear_cfg["e2_col"]], dtype=float)
        w = np.asarray(data[weight_column], dtype=float)

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
            area_deg2 = self._area_from_catalog(shear_cfg, nside)
        else:
            raise ValueError(
                f"Unable to determine survey area for {ver}. Provide mask_path or nside."
            )

        stats = effective_survey_stats(e1, e2, w, area_deg2)

        results = {
            "area_deg2": area_deg2,
            "n_eff": stats["n_eff"],
            "sigma_e": stats["sigma_e"],
            "sum_w": stats["sum_w"],
            "sum_w2": stats["sum_w2"],
            "catalog_size": n_rows,
        }

        if overwrite_config:
            self.cc[ver].setdefault("cov_th", {}).update(
                A=float(area_deg2),
                n_e=float(stats["n_eff"]),
                sigma_e=float(stats["sigma_e"]),
            )
            self._write_catalog_config()

        return results

    def _area_from_catalog(self, shear_cfg, nside):
        ra_col = shear_cfg.get("ra_col", "RA")
        dec_col = shear_cfg.get("dec_col", "Dec")
        data = open_entry(shear_cfg)
        ra = np.asarray(data[ra_col], dtype=float)
        dec = np.asarray(data[dec_col], dtype=float)
        return area_from_coords(ra, dec, nside)

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
            self.calculate_n_eff_gal(tomography=False)
            if self.compute_tomography:
                self.calculate_n_eff_gal(tomography=True)
        return self._n_eff_gal

    @property
    def ellipticity_dispersion(self):
        if not hasattr(self, "_ellipticity_dispersion"):
            self.calculate_ellipticity_dispersion(tomography=False)
            if self.compute_tomography:
                self.calculate_ellipticity_dispersion(tomography=True)
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

    def calculate_n_eff_gal(self, tomography=False):
        self.print_start("Calculating effective number of galaxy")
        if not hasattr(self, "_n_eff_gal"):
            self._n_eff_gal = {}
        for ver in self.versions:
            self.print_magenta(ver)
            if ver not in self._n_eff_gal:
                self._n_eff_gal[ver] = {}

            if tomography:
                tomo_bin_ids, tomo_bin_pairs = self._get_tomo_bins(ver)

                if tomo_bin_ids is None or tomo_bin_pairs is None:
                    raise ValueError(
                        f"Version {ver} does not have tomography information."
                    )

            else:
                tomo_bin_ids, tomo_bin_pairs = ["all"], [("all", "all")]

            with self.results[ver].temporarily_read_data():
                w = self._read_shear_cols(ver, "w_col")
                for tomo_bin_id in tomo_bin_ids:
                    if tomo_bin_id == "all":
                        self._n_eff_gal[ver][f"tomo_bin_{tomo_bin_id}"] = n_eff_density(
                            w, self.area[ver]
                        )
                    else:
                        tomo_bin = self._read_shear_cols(ver, "tomo_bin_col")
                        mask = tomo_bin == tomo_bin_id
                        self._n_eff_gal[ver][f"tomo_bin_{tomo_bin_id}"] = n_eff_density(
                            w[mask], self.area[ver]
                        )
                    print(
                        f"n_eff_gal for tomo_bin_{tomo_bin_id} = {self._n_eff_gal[ver][f'tomo_bin_{tomo_bin_id}']:.2f} gal./arcmin^-2"
                    )

        self.print_done("Effective number of galaxy calculation finished")

    def calculate_ellipticity_dispersion(self, tomography=False):
        self.print_start("Calculating ellipticity dispersion")
        if not hasattr(self, "_ellipticity_dispersion"):
            self._ellipticity_dispersion = {}
        for ver in self.versions:
            self.print_magenta(ver)
            if ver not in self._ellipticity_dispersion:
                self._ellipticity_dispersion[ver] = {}

            if tomography:
                tomo_bin_ids, tomo_bin_pairs = self._get_tomo_bins(ver)

                if tomo_bin_ids is None or tomo_bin_pairs is None:
                    raise ValueError(
                        f"Version {ver} does not have tomography information."
                    )

            else:
                tomo_bin_ids, tomo_bin_pairs = ["all"], [("all", "all")]

            with self.results[ver].temporarily_read_data():
                e1, e2, w = self._read_shear_cols(ver, "e1_col", "e2_col", "w_col")
                for tomo_bin_id in tomo_bin_ids:
                    if tomo_bin_id == "all":
                        self._ellipticity_dispersion[ver][f"tomo_bin_{tomo_bin_id}"] = (
                            ellipticity_dispersion_stat(e1, e2, w)
                        )
                    else:
                        tomo_bin = self._read_shear_cols(ver, "tomo_bin_col")
                        mask = tomo_bin == tomo_bin_id
                        self._ellipticity_dispersion[ver][f"tomo_bin_{tomo_bin_id}"] = (
                            ellipticity_dispersion_stat(e1[mask], e2[mask], w[mask])
                        )
                    print(
                        f"Ellipticity dispersion for tomo_bin_{tomo_bin_id} = {self._ellipticity_dispersion[ver][f'tomo_bin_{tomo_bin_id}']:.4f}"
                    )

    def plot_footprints(self):
        self.print_start("Plotting footprints:")
        for ver in self.versions:
            self.print_magenta(ver)

            fp = cs_plots.FootprintPlotter()

            for region in fp._regions:
                out_path = self._output_path(f"footprint_{ver}_{region}.png")
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
        out_path = self._output_path("ell_hist.png")
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
        out_path = self._output_path("weight_hist.png")
        if os.path.exists(out_path):
            self.print_done(f"Skipping weight histograms, {out_path} exists")
        else:
            self.print_start("Computing weight histograms:")

            plt.figure(figsize=(10, 7))
            for ver in self.versions:
                self.print_magenta(ver)
                with self.results[ver].temporarily_read_data():
                    w = self._read_shear_cols(ver, "w_col")

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
            with self.results[ver].temporarily_read_data():
                e1, e2, w = self._read_shear_cols(ver, "e1_col", "e2_col", "w_col")
                self._c1[ver], self._c2[ver] = additive_bias(e1, e2, w, R)
        self.print_done("Finished additive bias calculation.")

    def calculate_additive_bias_errors(self, npatch=None):
        """Shape-noise and jackknife errors of the additive bias.

        The jackknife deletes one patch at a time; patches are the seeded
        k-means patches of the 2PCF (``_patch_centers``), so it includes
        spatially correlated contributions such as cosmic shear and PSF
        residuals.

        Parameters
        ----------
        npatch : int, optional
            Number of jackknife patches; default ``self.npatch``. With
            ``npatch <= 1`` only the shape-noise errors are computed.
        """
        npatch = int(npatch or self.npatch)
        self.print_start(f"Calculating additive bias errors ({npatch} patches):")
        self._c_err = {}
        for ver in self.versions:
            self.print_magenta(ver)
            R = self.cc[ver]["shear"]["R"]
            with self.results[ver].temporarily_read_data():
                e1, e2, w = self._read_shear_cols(ver, "e1_col", "e2_col", "w_col")
                patch = None
                if npatch > 1:
                    cols = {
                        "ra": np.asarray(self.results[ver].dat_shear["RA"]),
                        "dec": np.asarray(self.results[ver].dat_shear["Dec"]),
                        "w": np.asarray(w),
                    }
                    patch = treecorr.Catalog(
                        ra=cols["ra"],
                        dec=cols["dec"],
                        ra_units=self.treecorr_config["ra_units"],
                        dec_units=self.treecorr_config["dec_units"],
                        patch_centers=self._patch_centers(cols, npatch),
                    ).patch
                self._c_err[ver] = additive_bias_errors(e1, e2, w, R, patch=patch)
        self.print_done("Finished additive bias errors.")

    def print_additive_bias(self, npatch=None, out_base="c_non_tomographic", labels=None):
        """Write the additive bias with its errors to text and LaTeX tables.

        The text table ``<out_base>.txt`` lists c with shape-noise (sn) and
        jackknife (jk) errors; the LaTeX table ``<out_base>.tex`` lists c in
        units of 1e-4 with jackknife errors (shape-noise errors without
        jackknife). LaTeX rows are labelled by ``labels``, else the config's
        ``label``, else the version name.

        Parameters
        ----------
        npatch : int, optional
            Number of jackknife patches; see
            :meth:`calculate_additive_bias_errors`.
        out_base : str, optional
            Output file name base in the output directory.
        labels : dict, optional
            LaTeX row label per version.
        """
        if not hasattr(self, "_c_err"):
            self.calculate_additive_bias_errors(npatch=npatch)

        def fmt(value, error, latex=False):
            return f"${ufloat(value, error):.1uL}$" if latex else f"{ufloat(value, error):.1u}"

        out_path = self._output_path(f"{out_base}.txt")
        with open(out_path, "w") as f:
            print(
                f"{'# Version':30s} {'c_1 (sn)':25s} {'c_2 (sn)':25s}"
                + f" {'c_1 (jk)':25s} {'c_2 (jk)':25s}",
                file=f,
            )
            for ver in self.versions:
                err = self._c_err[ver]
                row = [fmt(self.c1[ver], err["sn"][0]), fmt(self.c2[ver], err["sn"][1])]
                if err["jk"] is not None:
                    row += [fmt(self.c1[ver], err["jk"][0]), fmt(self.c2[ver], err["jk"][1])]
                print(f"{ver:30s} " + " ".join(f"{r:25s}" for r in row), file=f)
        self.print_done(f"Additive bias table written to {out_path}")

        lines = [
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"version & $c_1 / 10^{-4}$ & $c_2 / 10^{-4}$ \\",
            r"\midrule",
        ]
        for ver in self.versions:
            err = self._c_err[ver]["jk"] or self._c_err[ver]["sn"]
            label = (labels or {}).get(ver) or self.cc[ver].get("label", ver)
            label = label.replace("_", r"\_")
            lines.append(
                f"{label} & {fmt(self.c1[ver] * 1e4, err[0] * 1e4, latex=True)}"
                + f" & {fmt(self.c2[ver] * 1e4, err[1] * 1e4, latex=True)} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]
        out_path = self._output_path(f"{out_base}.tex")
        with open(out_path, "w") as f:
            print("\n".join(lines), file=f)
        self.print_done(f"Additive bias table written to {out_path}")

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
