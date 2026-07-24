"""Real-space two-point diagnostics for cosmology validation.

This mixin holds the real-space machinery: the TreeCorr two-point correlation
function (2PCF) ξ± measurement and its plots, the ratio of PSF systematics to
the cosmic-shear signal, and the aperture-mass dispersion ⟨M_ap²⟩ measurement
and plots. It depends on TreeCorr.
"""

import os

import numpy as np
import treecorr
from astropy.io import fits
from cs_util import plots as cs_plots


class RealSpaceMixin:
    def calculate_2pcf_version(
        self,
        ver,
        npatch=None,
        compute_tomography=False,
        **treecorr_config,
    ):
        """
        Calculate the two-point correlation function (2PCF) ξ± for a single catalog
        version with TreeCorr.

        This is the per-version child function. Use :meth:`calculate_2pcf` to run over every
        version in ``self.versions`` in one call.

        By default the class instance's `npatch` and `treecorr_config` entries are
        used to initialize the TreeCorr Catalog and GGCorrelation objects, but may
        be overridden by passing keyword arguments.

        Parameters:
            ver (str): The catalog version to process.

            npatch (int, optional): The number of patches to use for the
            calculation. Defaults to the instance's `npatch` attribute.

            compute_tomography (bool, optional): Whether to compute tomographic
            correlations. Defaults to False.

            **treecorr_config: Additional TreeCorr configuration parameters that
            will override the instance's default `treecorr_config`. For example,
            `min_sep=1`.

        Returns:
            dict: Mapping of ``"tomo_bin_{b1}_tomo_bin_{b2}"`` to the corresponding
            treecorr.GGCorrelation object. For the non-tomographic case the single
            key is ``"tomo_bin_all_tomo_bin_all"``.
        """

        npatch = npatch or self.npatch
        treecorr_config = {
            **self._binning(**treecorr_config),
            "var_method": "jackknife" if int(npatch) > 1 else "shot",
        }

        if compute_tomography:
            tomo_bin_ids, tomo_bin_pairs = self._get_tomo_bins(ver)
            if tomo_bin_ids is None or tomo_bin_pairs is None:
                raise ValueError(f"Version {ver} does not have tomography information.")
            self.print_magenta(
                f"Computing tomographic ξ± for {ver} with {len(tomo_bin_pairs)} bins."
            )
        else:
            self.print_magenta(f"Computing non-tomographic ξ± for {ver}.")
            tomo_bin_pairs = [("all", "all")]

        ggs = {f"tomo_bin_{b1}_tomo_bin_{b2}": None for b1, b2 in tomo_bin_pairs}

        # LG TO-DO: Change to sacc_io method

        patch_file = self._output_path(f"{ver}_patches_npatch={npatch}.dat")

        cat_gal = fits.getdata(self.cc[ver]["shear"]["path"])
        with self.results[ver].temporarily_read_data():
            g1, g2 = self._calibrated_g(ver)
            w = self._read_shear_cols(ver, "w_col")

        for bin1, bin2 in tomo_bin_pairs:
            gg = treecorr.GGCorrelation(treecorr_config)

            # Load data and create a catalog
            if bin1 == "all" and bin2 == "all":
                mask_bin1 = np.ones(len(g1), dtype=bool)
            else:
                mask_bin1 = cat_gal[self.cc[ver]["shear"]["tomo_bin_col"]] == bin1

            cat_gal1 = treecorr.Catalog(
                ra=cat_gal["RA"][mask_bin1],
                dec=cat_gal["Dec"][mask_bin1],
                g1=g1[mask_bin1],
                g2=g2[mask_bin1],
                w=w[mask_bin1],
                ra_units=self.treecorr_config["ra_units"],
                dec_units=self.treecorr_config["dec_units"],
                npatch=npatch,
                patch_centers=patch_file if os.path.exists(patch_file) else None,
            )
            cat_gal2 = None

            if bin1 != bin2:
                mask_bin2 = cat_gal[self.cc[ver]["shear"]["tomo_bin_col"]] == bin2
                cat_gal2 = treecorr.Catalog(
                    ra=cat_gal["RA"][mask_bin2],
                    dec=cat_gal["Dec"][mask_bin2],
                    g1=g1[mask_bin2],
                    g2=g2[mask_bin2],
                    w=w[mask_bin2],
                    ra_units=self.treecorr_config["ra_units"],
                    dec_units=self.treecorr_config["dec_units"],
                    npatch=npatch,
                    patch_centers=patch_file if os.path.exists(patch_file) else None,
                )

            # If no patch file exists, save the current patches
            if not os.path.exists(patch_file):
                cat_gal1.write_patch_centers(patch_file)

            # Process the catalog & write the correlation functions
            gg.process(cat_gal1, cat2=cat_gal2)
            ggs[f"tomo_bin_{bin1}_tomo_bin_{bin2}"] = gg

        self.print_done(f"Done 2PCF for {ver}.")

        return ggs

    def calculate_2pcf(
        self,
        npatch=None,
        compute_tomography=False,
        **treecorr_config,
    ):
        """
        Calculate the 2PCF ξ± for every catalog version in ``self.versions``.

        Parent function that iterates over ``self.versions`` and delegates the
        per-version computation to :meth:`calculate_2pcf_version`. Results are stored
        in ``self.cat_ggs`` keyed by version.

        Parameters:
            npatch (int, optional): Number of patches to use; defaults to the
            instance's `npatch` attribute.

            compute_tomography (bool, optional): Whether to compute tomographic
            correlations. Defaults to False.

            **treecorr_config: Additional TreeCorr configuration parameters passed
            through to each per-version call.

        Returns:
            dict: ``self.cat_ggs``, mapping each version to its
            ``{"tomo_bin_{b1}_tomo_bin_{b2}": treecorr.GGCorrelation}`` dict.
        """
        self.cat_ggs = {}
        for ver in self.versions:
            self.cat_ggs[ver] = self.calculate_2pcf_version(
                ver,
                npatch=npatch,
                compute_tomography=compute_tomography,
                **treecorr_config,
            )

        # LG TO-DO: No longer writing out text file, change to sacc_io method

        return self.cat_ggs

    def calculate_aperture_mass_dispersion(
        self,
        theta_min=0.3,
        theta_max=200,
        nbins=500,
        nbins_map=15,
        npatch=25,
        compute_tomography=False,
    ):

        self._map2 = {}
        theta_map = np.geomspace(theta_min * 5, theta_max / 2, nbins_map)
        self._map2["theta_map"] = theta_map

        treecorr_config = self._binning(theta_min, theta_max, nbins)

        for ver in self.versions:
            if compute_tomography:
                tomo_bin_ids, tomo_bin_pairs = self._get_tomo_bins(ver)
                if tomo_bin_ids is None or tomo_bin_pairs is None:
                    raise ValueError(
                        f"Version {ver} does not have tomography information."
                    )
                self.print_magenta(
                    f"Computing MAP for {ver} with {len(tomo_bin_pairs)} bins."
                )
            else:
                self.print_magenta(f"Computing non-tomographic MAP for {ver}.")

                tomo_bin_pairs = [("all", "all")]

            self._map2.setdefault(ver, {})
            cat_gal = fits.getdata(self.cc[ver]["shear"]["path"])

            # LG TO-DO: Change to sacc_io method
            with self.results[ver].temporarily_read_data():
                g1, g2 = self._calibrated_g(ver)
                w = self._read_shear_cols(ver, "w_col")

            for bin1, bin2 in tomo_bin_pairs:
                gg = treecorr.GGCorrelation(treecorr_config)

                # Load data and create a catalog
                if bin1 == "all" and bin2 == "all":
                    mask_bin1 = np.ones(len(cat_gal), dtype=bool)
                else:
                    mask_bin1 = cat_gal[self.cc[ver]["shear"]["tomo_bin_col"]] == bin1

                cat_gal1 = treecorr.Catalog(
                    ra=cat_gal["RA"][mask_bin1],
                    dec=cat_gal["Dec"][mask_bin1],
                    g1=g1[mask_bin1],
                    g2=g2[mask_bin1],
                    w=w[mask_bin1],
                    ra_units=self.treecorr_config["ra_units"],
                    dec_units=self.treecorr_config["dec_units"],
                    npatch=npatch,
                )
                cat_gal2 = None

                if bin1 != bin2:
                    mask_bin2 = cat_gal[self.cc[ver]["shear"]["tomo_bin_col"]] == bin2
                    cat_gal2 = treecorr.Catalog(
                        ra=cat_gal["RA"][mask_bin2],
                        dec=cat_gal["Dec"][mask_bin2],
                        g1=g1[mask_bin2],
                        g2=g2[mask_bin2],
                        w=w[mask_bin2],
                        ra_units=self.treecorr_config["ra_units"],
                        dec_units=self.treecorr_config["dec_units"],
                        npatch=npatch,
                    )

                gg.process(cat_gal1, cat2=cat_gal2)

                mapsq, mapsq_im, mxsq, mxsq_im, varmapsq = gg.calculateMapSq(
                    R=theta_map,
                    m2_uform="Schneider",
                )
                self._map2[ver][f"tomo_bin_{bin1}_tomo_bin_{bin2}"] = {
                    "mapsq": mapsq,
                    "mapsq_im": mapsq_im,
                    "mxsq": mxsq,
                    "mxsq_im": mxsq_im,
                    "varmapsq": varmapsq,
                }
            self.print_done(f"Done aperture-mass dispersion for {ver}.")

    @property
    def map2(self):
        if not hasattr(self, "_map2"):
            self.calculate_aperture_mass_dispersion()
        return self._map2

    # LG: plotting functions removed, perhaps can use Sacha's implementation in psf_systematics.py instead

    def plot_aperture_mass_dispersion(self):
        for mode in ["mapsq", "mapsq_im", "mxsq", "mxsq_im"]:
            x = [self.map2["theta_map"] for ver in self.versions]
            # LG: FORCE non-tomography for now
            y = [
                self.map2[ver]["tomo_bin_all_tomo_bin_all"][mode]
                for ver in self.versions
            ]
            yerr = [
                np.sqrt(self.map2[ver]["tomo_bin_all_tomo_bin_all"]["varmapsq"])
                for ver in self.versions
            ]
            labels = list(self.versions)
            colors = [self.cc[ver]["colour"] for ver in self.versions]
            linestyles = [self.cc[ver]["ls"] for ver in self.versions]

            xlabel = r"$\theta$ [arcmin]"
            ylabel = "dispersion"
            title = f"Aperture-mass dispersion {mode}"
            out_path = self._output_path(f"{mode}.png")
            cs_plots.plot_data_1d(
                x,
                y,
                yerr,
                title,
                xlabel,
                ylabel,
                out_path=None,
                labels=labels,
                xlog=True,
                xlim=[self.theta_min_plot, self.theta_max_plot],
                ylim=[-2e-6, 5e-6],
                colors=colors,
                linestyles=linestyles,
                shift_x=True,
            )
            cs_plots.savefig(out_path, close_fig=False)
            cs_plots.show()
            self.print_done(f"linear-scale {mode} plot saved to {out_path}")

        for mode in ["mapsq", "mapsq_im", "mxsq", "mxsq_im"]:
            x = [self.map2["theta_map"] for ver in self.versions]
            # FORCE non-tomography for now
            y = [
                np.abs(self.map2[ver]["tomo_bin_all_tomo_bin_all"][mode])
                for ver in self.versions
            ]
            yerr = [
                np.sqrt(self.map2[ver]["tomo_bin_all_tomo_bin_all"]["varmapsq"])
                for ver in self.versions
            ]
            xlabel = r"$\theta$ [arcmin]"
            ylabel = "dispersion"
            title = f"Aperture-mass dispersion mode {mode}"
            out_path = self._output_path(f"{mode}_log.png")
            cs_plots.plot_data_1d(
                x,
                y,
                yerr,
                title,
                xlabel,
                ylabel,
                out_path=None,
                labels=labels,
                xlog=True,
                ylog=True,
                xlim=[self.theta_min_plot, self.theta_max_plot],
                ylim=[1e-8, 1e-5],
                colors=colors,
                linestyles=linestyles,
                shift_x=True,
            )
            cs_plots.savefig(out_path, close_fig=False)
            cs_plots.show()
            self.print_done(f"log-scale {mode} plot saved to {out_path}")
