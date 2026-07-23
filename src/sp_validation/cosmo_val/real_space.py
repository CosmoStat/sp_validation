"""Real-space two-point diagnostics for cosmology validation.

This mixin holds the real-space machinery: the TreeCorr two-point correlation
function (2PCF) ξ± measurement and its plots, the ratio of PSF systematics to
the cosmic-shear signal, and the aperture-mass dispersion ⟨M_ap²⟩ measurement
and plots. It depends on TreeCorr.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import treecorr
from cs_util import plots as cs_plots


class RealSpaceMixin:
    def calculate_2pcf(
        self,
        ver,
        npatch=None,
        compute_tomography=False,
        #    save_fits=False,
        **treecorr_config,
    ):
        """
        Calculate the two-point correlation function (2PCF) ξ± for a given catalog
        version with TreeCorr.

        By default the class instance's `npatch` and `treecorr_config` entries are
        used to
        initialize the TreeCorr Catalog and GGCorrelation objects, but may be
        overridden
        by passing keyword arguments.

        Parameters:
            ver (str): The catalog version to process.

            npatch (int, optional): The number of patches to use for the calculation.
            Defaults to the instance's `npatch` attribute.

            compute_tomography (bool, optional): Whether to compute tomographic correlations.
            Defaults to False.

            save_fits (bool, optional): Whether to save the ξ± results to FITS files.
            Defaults to False.

            **treecorr_config: Additional TreeCorr configuration parameters that will
            override the instance's default `treecorr_config`. For example, `min_sep=1`.

        Returns:
            treecorr.GGCorrelation: The TreeCorr GGCorrelation object containing the
            computed 2PCF results.

        Notes:
            - If the output file for the given configuration already exists, the
              calculation is skipped, and the results are loaded from the file.
            - If a patch file for the given configuration does not exist, it is
              created during the process.
            - FITS files for ξ+ and ξ− are saved with additional metadata in their
              headers if `save_fits` is True.
        """

        npatch = npatch or self.npatch
        treecorr_config = {
            **self._binning(**treecorr_config),
            "var_method": "jackknife" if int(npatch) > 1 else "shot",
        }

        if compute_tomography:
            tomo_bin_ids, tomo_bin_pairs = self._get_tomo_bins(ver)
            self.print_magenta(
                f"Computing tomographic ξ± for {ver} with {len(tomo_bin_pairs)} bins."
            )

            if tomo_bin_ids is None or tomo_bin_pairs is None:
                raise ValueError(f"Version {ver} does not have tomography information.")
        else:
            self.print_magenta(f"Computing non-tomographic ξ± for {ver}.")
            tomo_bin_pairs = [("all", "all")]

        ggs = {f"tomo_bin_{b1}_tomo_bin_{b2}": None for b1, b2 in tomo_bin_pairs}

        # LG TO-DO: Change to sacc_io method
        out_fname = self._output_path(
            f"{ver}_xi_minsep={treecorr_config['min_sep']}_maxsep={treecorr_config['max_sep']}_nbins={treecorr_config['nbins']}_npatch={npatch}.txt"
        )
        if os.path.exists(out_fname):
            self.print_done(f"Skipping 2PCF calculation, {out_fname} exists.")
            for bin1, bin2 in tomo_bin_pairs:
                ggs[f"tomo_bin_{bin1}_tomo_bin_{bin2}"] = treecorr.GGCorrelation(
                    treecorr_config
                )
                ggs[f"tomo_bin_{bin1}_tomo_bin_{bin2}"].read(out_fname)

        else:
            patch_file = self._output_path(f"{ver}_patches_npatch={npatch}.dat")

            for bin1, bin2 in tomo_bin_pairs:
                gg = treecorr.GGCorrelation(treecorr_config)

                # Load data and create a catalog
                with self.results[ver].temporarily_read_data():
                    if bin1 == "all" and bin2 == "all":
                        mask_bin1 = np.ones(
                            len(self.results[ver].dat_shear), dtype=bool
                        )
                    else:
                        mask_bin1 = self.results[ver].dat_shear["tom_bin_id"] == bin1

                    g1, g2 = self._calibrated_g(ver)
                    w = self._read_shear_cols(ver, "w_col")

                    cat_gal1 = treecorr.Catalog(
                        ra=self.results[ver].dat_shear["RA"][mask_bin1],
                        dec=self.results[ver].dat_shear["Dec"][mask_bin1],
                        g1=g1[mask_bin1],
                        g2=g2[mask_bin1],
                        w=w[mask_bin1],
                        ra_units=self.treecorr_config["ra_units"],
                        dec_units=self.treecorr_config["dec_units"],
                        npatch=npatch,
                        patch_centers=patch_file
                        if os.path.exists(patch_file)
                        else None,
                    )
                    cat_gal2 = None

                    if bin1 != bin2:
                        mask_bin2 = self.results[ver].dat_shear["tom_bin_id"] == bin2
                        cat_gal2 = treecorr.Catalog(
                            ra=self.results[ver].dat_shear["RA"][mask_bin2],
                            dec=self.results[ver].dat_shear["Dec"][mask_bin2],
                            g1=g1[mask_bin2],
                            g2=g2[mask_bin2],
                            w=w[mask_bin2],
                            ra_units=self.treecorr_config["ra_units"],
                            dec_units=self.treecorr_config["dec_units"],
                            npatch=npatch,
                            patch_centers=patch_file
                            if os.path.exists(patch_file)
                            else None,
                        )

                    # If no patch file exists, save the current patches
                    if not os.path.exists(patch_file):
                        cat_gal1.write_patch_centers(patch_file)

                # Process the catalog & write the correlation functions
                gg.process(cat_gal1, cat2=cat_gal2)
                ggs[f"tomo_bin_{bin1}_tomo_bin_{bin2}"] = gg
                # LG TO-DO: No longer writing out text file, change to sacc_io method
                # gg.write(out_fname, write_patch_results=True, write_cov=True)

        # LG: FITS writeout function deprecated, now writing to SACC format

        # Save xi_p and xi_m results to fits file
        # (moved outside so it runs even if txt exists)
        # if save_fits:
        #     lst = np.arange(1, treecorr_config["nbins"] + 1)

        #     col1 = fits.Column(name="BIN1", format="K", array=np.ones(len(lst)))
        #     col2 = fits.Column(name="BIN2", format="K", array=np.ones(len(lst)))
        #     col3 = fits.Column(name="ANGBIN", format="K", array=lst)
        #     col4 = fits.Column(name="VALUE", format="D", array=gg.xip)
        #     col5 = fits.Column(name="ANG", format="D", unit="arcmin", array=gg.meanr)
        #     coldefs = fits.ColDefs([col1, col2, col3, col4, col5])
        #     xiplus_hdu = fits.BinTableHDU.from_columns(coldefs, name="XI_PLUS")

        #     col4 = fits.Column(name="VALUE", format="D", array=gg.xim)
        #     coldefs = fits.ColDefs([col1, col2, col3, col4, col5])
        #     ximinus_hdu = fits.BinTableHDU.from_columns(coldefs, name="XI_MINUS")

        #     # append xi_plus header info
        #     xiplus_dict = {
        #         "2PTDATA": "T",
        #         "QUANT1": "G+R",
        #         "QUANT2": "G+R",
        #         "KERNEL_1": "NZ_SOURCE",
        #         "KERNEL_2": "NZ_SOURCE",
        #         "WINDOWS": "SAMPLE",
        #     }
        #     for key in xiplus_dict:
        #         xiplus_hdu.header[key] = xiplus_dict[key]

        #         col1 = fits.Column(name="BIN1", format="K", array=np.ones(len(lst)))
        #         col2 = fits.Column(name="BIN2", format="K", array=np.ones(len(lst)))
        #         col3 = fits.Column(name="ANGBIN", format="K", array=lst)
        #         col4 = fits.Column(name="VALUE", format="D", array=gg.xip)
        #         col5 = fits.Column(name="ANG", format="D", unit="arcmin", array=gg.rnom)
        #         coldefs = fits.ColDefs([col1, col2, col3, col4, col5])
        #         xiplus_hdu = fits.BinTableHDU.from_columns(coldefs, name="XI_PLUS")

        #         col4 = fits.Column(name="VALUE", format="D", array=gg.xim)
        #         coldefs = fits.ColDefs([col1, col2, col3, col4, col5])
        #         ximinus_hdu = fits.BinTableHDU.from_columns(coldefs, name="XI_MINUS")

        #         # append xi_plus header info
        #         xiplus_dict = {
        #             "2PTDATA": "T",
        #             "QUANT1": "G+R",
        #             "QUANT2": "G+R",
        #             "KERNEL_1": "NZ_SOURCE",
        #             "KERNEL_2": "NZ_SOURCE",
        #             "WINDOWS": "SAMPLE",
        #         }
        #         for key in xiplus_dict:
        #             xiplus_hdu.header[key] = xiplus_dict[key]
        #     # Use same naming format as txt output
        #     fits_base = out_fname.replace(".txt", "").replace("_xi_", "_")
        #     xiplus_hdu.writeto(
        #         f"{fits_base.replace(ver, f'xi_plus_{ver}')}.fits",
        #         overwrite=True,
        #     )

        #     # append xi_minus header info
        #     ximinus_dict = {**xiplus_dict, "QUANT1": "G-R", "QUANT2": "G-R"}
        #     for key in ximinus_dict:
        #         ximinus_hdu.header[key] = ximinus_dict[key]
        #     ximinus_hdu.writeto(
        #         f"{fits_base.replace(ver, f'xi_minus_{ver}')}.fits",
        #         overwrite=True,
        #     )

        # Add correlation object to class
        if not hasattr(self, "cat_ggs"):
            self.cat_ggs = {}
        self.cat_ggs[ver] = ggs

        self.print_done(f"Done 2PCF for {ver}.")

        return ggs

    def plot_2pcf(self):
        # Plot of n_pairs
        plt.subplots(ncols=1, nrows=1)
        for ver in self.versions:
            # FORCE non-tomography for now
            self.calculate_2pcf(ver)
            plt.plot(
                self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].meanr,
                self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].npairs,
                label=ver,
                ls=self.cc[ver]["ls"],
                color=self.cc[ver]["colour"],
            )
        plt.xlabel(rf"$\theta$ [{self.treecorr_config['sep_units']}]")
        plt.ylabel(r"$n_{\rm pair}$")
        plt.legend()
        out_path = self._output_path("n_pair.png")
        cs_plots.savefig(out_path, close_fig=False)
        cs_plots.show()
        self.print_done(f"n_pair plot saved to {out_path}")

        # Plot of xi_+
        plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
        for idx, ver in enumerate(self.versions):
            plt.errorbar(
                self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].meanr
                * cs_plots.dx(idx, fx=1.05, nx=len(ver)),
                self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].xip,
                yerr=np.sqrt(self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].varxip),
                label=ver,
                ls=self.cc[ver]["ls"],
                color=self.cc[ver]["colour"],
            )
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.ticklabel_format(axis="y")
        plt.xlabel(rf"$\theta$ [{self.treecorr_config['sep_units']}]")
        plt.xlim([self.theta_min_plot, self.theta_max_plot])
        plt.ylabel(r"$\xi_+(\theta)$")
        out_path = self._output_path("xi_p.png")
        cs_plots.savefig(out_path, close_fig=False)
        cs_plots.show()
        self.print_done(f"xi_plus plot saved to {out_path}")

        # Plot of xi_-
        plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
        for idx, ver in enumerate(self.versions):
            plt.errorbar(
                self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].meanr
                * cs_plots.dx(idx, fx=1.05, nx=len(ver)),
                self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].xim,
                yerr=np.sqrt(self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].varxim),
                label=ver,
                ls=self.cc[ver]["ls"],
                color=self.cc[ver]["colour"],
            )
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.ticklabel_format(axis="y")
        plt.xlabel(rf"$\theta$ [{self.treecorr_config['sep_units']}]")
        plt.xlim([self.theta_min_plot, self.theta_max_plot])
        plt.ylabel(r"$\xi_-(\theta)$")
        out_path = self._output_path("xi_m.png")
        cs_plots.savefig(out_path, close_fig=False)
        cs_plots.show()
        self.print_done(f"xi_minus plot saved to {out_path}")

        # Plot of xi_+(theta) * theta
        plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
        for idx, ver in enumerate(self.versions):
            plt.errorbar(
                self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].meanr,
                self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].xip
                * self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].meanr,
                yerr=np.sqrt(self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].varxip)
                * self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].meanr,
                label=ver,
                ls=self.cc[ver]["ls"],
                color=self.cc[ver]["colour"],
            )
        plt.xscale("log")
        plt.legend()
        plt.ticklabel_format(axis="y")
        plt.xlabel(rf"$\theta$ [{self.treecorr_config['sep_units']}]")
        plt.xlim([self.theta_min_plot, self.theta_max_plot])
        plt.ylabel(r"$\theta \xi_+(\theta)$")
        out_path = self._output_path("xi_p_theta.png")
        cs_plots.savefig(out_path, close_fig=False)
        cs_plots.show()
        self.print_done(f"xi_plus_theta plot saved to {out_path}")

        # Plot of xi_- * theta
        plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
        for idx, ver in enumerate(self.versions):
            plt.errorbar(
                self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].meanr
                * cs_plots.dx(idx, len(ver)),
                self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].xim
                * self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].meanr,
                yerr=np.sqrt(self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].varxim)
                * self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].meanr,
                label=ver,
                ls=self.cc[ver]["ls"],
                color=self.cc[ver]["colour"],
            )
        plt.xscale("log")
        plt.legend()
        plt.ticklabel_format(axis="y")
        plt.xlabel(rf"$\theta$ [{self.treecorr_config['sep_units']}]")
        plt.xlim([self.theta_min_plot, self.theta_max_plot])
        plt.ylabel(r"$\theta \xi_-(\theta)$")
        out_path = self._output_path("xi_m_theta.png")
        cs_plots.savefig(out_path, close_fig=False)
        cs_plots.show()
        self.print_done(f"xi_minus_theta plot saved to {out_path}")

        # Plot of xi_+ with and without xi_psf_sys
        # but skip if xi_psf_sys is not calculated since that takes forever
        if hasattr(self, "_xi_psf_sys"):
            for idx, ver in enumerate(self.versions):
                plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
                plt.errorbar(
                    self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].meanr
                    * cs_plots.dx(idx, len(ver)),
                    self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].xip,
                    yerr=np.sqrt(self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].varxip),
                    label=r"$\xi_+$",
                    ls="solid",
                    color="green",
                )
                plt.errorbar(
                    self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].meanr
                    * cs_plots.dx(idx, len(ver)),
                    self.xi_psf_sys[ver]["mean"],
                    yerr=np.sqrt(self.xi_psf_sys[ver]["var"]),
                    label=r"$\xi^{\rm psf}_{+, {\rm sys}}$",
                    ls="dotted",
                    color="red",
                )
                plt.errorbar(
                    self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].meanr
                    * cs_plots.dx(idx, len(ver)),
                    self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].xip
                    + self.xi_psf_sys[ver]["mean"],
                    yerr=np.sqrt(
                        self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"].varxip
                        + self.xi_psf_sys[ver]["var"]
                    ),
                    label=r"$\xi_+ + \xi^{\rm psf}_{+, {\rm sys}}$",
                    ls="dashdot",
                    color="magenta",
                )
                plt.xscale("log")
                plt.yscale("log")
                plt.legend()
                plt.ticklabel_format(axis="y")
                plt.xlabel(rf"$\theta$ [{self.treecorr_config['sep_units']}]")
                plt.xlim([self.theta_min_plot, self.theta_max_plot])
                plt.ylim(1e-8, 5e-4)
                plt.ylabel(r"$\xi_+(\theta)$")
                out_path = self._output_path(f"xi_p_xi_psf_sys_{ver}.png")
                cs_plots.savefig(out_path, close_fig=False)
                cs_plots.show()
                self.print_done(f"xi_plus_xi_psf_sys {ver} plot saved to {out_path}")

    def plot_ratio_xi_sys_xi(self, threshold=0.1, offset=0.02):

        plt.subplots(ncols=1, nrows=1, figsize=(10, 7))

        for idx, ver in enumerate(self.versions):
            self.calculate_2pcf(ver)
            xi_psf_sys = self.xi_psf_sys[ver]
            gg = self.cat_ggs[ver]["tomo_bin_all_tomo_bin_all"]

            ratio = xi_psf_sys["mean"] / gg.xip
            ratio_err = np.sqrt(
                (np.sqrt(xi_psf_sys["var"]) / gg.xip) ** 2
                + (xi_psf_sys["mean"] * np.sqrt(gg.varxip) / gg.xip**2) ** 2
            )

            theta = gg.meanr
            jittered_theta = theta * (1 + idx * offset)

            plt.errorbar(
                jittered_theta,
                ratio,
                yerr=ratio_err,
                label=ver,
                ls=self.cc[ver]["ls"],
                color=self.cc[ver]["colour"],
                fmt=self.cc[ver].get("marker", None),
                capsize=5,
            )

        plt.fill_between(
            [self.theta_min_plot, self.theta_max_plot],
            -threshold,
            +threshold,
            color="black",
            alpha=0.1,
            label=f"{threshold:.0%} threshold",
        )
        plt.plot(
            [self.theta_min_plot, self.theta_max_plot],
            [threshold, threshold],
            ls="dashed",
            color="black",
        )
        plt.plot(
            [self.theta_min_plot, self.theta_max_plot],
            [-threshold, -threshold],
            ls="dashed",
            color="black",
        )
        plt.xscale("log")
        plt.xlabel(rf"$\theta$ [{self.treecorr_config['sep_units']}]")
        plt.ylabel(r"$\xi^{\rm psf}_{+, {\rm sys}} / \xi_+$")
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        plt.legend()
        plt.title("Ratio of PSF systematics to cosmic shear signal")
        out_path = self._output_path("ratio_xi_sys_xi.png")
        cs_plots.savefig(out_path, close_fig=False)
        cs_plots.show()
        print(f"Ratio of xi_psf_sys to xi plot saved to {out_path}")

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
                self.print_magenta(
                    f"Computing MAP for {ver} with {len(tomo_bin_pairs)} bins."
                )

                if tomo_bin_ids is None or tomo_bin_pairs is None:
                    raise ValueError(
                        f"Version {ver} does not have tomography information."
                    )
            else:
                self.print_magenta(f"Computing non-tomographic MAP for {ver}.")

                tomo_bin_pairs = [("all", "all")]

            self._map2.setdefault(ver, {})

            # LG TO-DO: Change to sacc_io method
            out_fname = self._output_path(f"xi_for_map2_{ver}.txt")
            if os.path.exists(out_fname):
                self.print_green(f"Skipping xi for Map2, {out_fname} exists")
                for bin1, bin2 in tomo_bin_pairs:
                    self._map2[ver][f"tomo_bin_{bin1}_{bin2}"] = treecorr.GGCorrelation(
                        treecorr_config
                    )
                    self._map2[ver][f"tomo_bin_{bin1}_{bin2}"].read(out_fname)

            else:
                for bin1, bin2 in tomo_bin_pairs:
                    gg = treecorr.GGCorrelation(treecorr_config)

                    # Load data and create a catalog
                    with self.results[ver].temporarily_read_data():
                        if bin1 == "all" and bin2 == "all":
                            mask_bin1 = np.ones(
                                len(self.results[ver].dat_shear), dtype=bool
                            )
                        else:
                            mask_bin1 = (
                                self.results[ver].dat_shear["tom_bin_id"] == bin1
                            )

                        g1, g2 = self._calibrated_g(ver)
                        w = self._read_shear_cols(ver, "w_col")

                        cat_gal1 = treecorr.Catalog(
                            ra=self.results[ver].dat_shear["RA"][mask_bin1],
                            dec=self.results[ver].dat_shear["Dec"][mask_bin1],
                            g1=g1[mask_bin1],
                            g2=g2[mask_bin1],
                            w=w[mask_bin1],
                            ra_units=self.treecorr_config["ra_units"],
                            dec_units=self.treecorr_config["dec_units"],
                            npatch=npatch,
                        )
                        cat_gal2 = None

                        if bin1 != bin2:
                            mask_bin2 = (
                                self.results[ver].dat_shear["tom_bin_id"] == bin2
                            )
                            cat_gal2 = treecorr.Catalog(
                                ra=self.results[ver].dat_shear["RA"][mask_bin2],
                                dec=self.results[ver].dat_shear["Dec"][mask_bin2],
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

    def plot_aperture_mass_dispersion(self):
        for mode in ["mapsq", "mapsq_im", "mxsq", "mxsq_im"]:
            x = [self.map2["theta_map"] for ver in self.versions]
            # FORCE non-tomography for now
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
