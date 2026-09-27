"""Real-space two-point diagnostics for cosmology validation.

This mixin holds the real-space machinery: the TreeCorr two-point correlation
function (2PCF) ξ± measurement and its plots, the ratio of PSF systematics to
the cosmic-shear signal, and the aperture-mass dispersion ⟨M_ap²⟩ measurement
and plots. It depends on TreeCorr.
"""

import hashlib
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import treecorr
from cs_util import plots as cs_plots

from .. import sacc_io
from ..custody import base_catalogue
from .sacc_writers import xi_to_sacc


class RealSpaceMixin:
    def calculate_2pcf(
        self,
        ver,
        *,
        grid="reporting",
        npatch=None,
        patch_centers=None,
        out=None,
        **treecorr_config,
    ):
        """ξ± of ``ver`` on one binning, as its SACC part sealed under its custody.

        @sc signal-leaves-sealed
        A catalogue's ξ± leaves this object only as a part sealed under the
        catalogue's custody (:func:`sp_validation.sacc_io.seal`): concealed
        when it is blinded, before it is returned, cached or written. The
        blind is opened first, so a blind that cannot open fails before
        TreeCorr runs.

        @sc patch-centres-are-inputs
        With patches, the catalogue splits at persisted centres, one file per
        base catalogue (:meth:`patch_centers_path`, written by
        :meth:`write_patch_centers`), never at centres drawn here: the
        full-sample ξ± depends on the patch layout, and TreeCorr's k-means on
        the machine. The part names the file by its sha256.

        Parameters:
            ver (str): The catalogue version to measure.
            grid (str): The grid tag the part's points carry.
            npatch (int, optional): Jackknife patches; the instance's
                ``npatch`` by default. With patches the part carries the
                jackknife covariance, without them the shot-noise diagonal.
            patch_centers (str, optional): The centres file; by default
                :meth:`patch_centers_path`.
            out (str, optional): Where to write the part as well.
            **treecorr_config: Overrides of the instance's ``treecorr_config``,
                e.g. ``min_sep=1``.

        Returns:
            sacc.Sacc: The sealed part, also kept in ``self.xi_parts[ver, grid]``.
        """
        self.print_magenta(f"Computing {ver} ξ±")
        npatch = int(npatch or self.npatch)
        custody = self.custody(ver)
        if custody.status == "blinded":
            from .. import blinding

            blinding.open_blind(custody)

        metadata = {**self.sacc_metadata(ver), "npatch": npatch}
        patch_centers = self._patch_centers(ver, npatch, patch_centers)
        if patch_centers is not None:
            with open(patch_centers, "rb") as f:
                metadata["patch_centers_sha256"] = hashlib.sha256(f.read()).hexdigest()
        gg = self._measure_xi(ver, npatch, patch_centers, **treecorr_config)

        jackknife = npatch > 1
        s = xi_to_sacc(
            self.sacc_nz(ver),
            metadata,
            gg.meanr,
            gg.xip,
            gg.xim,
            grid=grid,
            theta_nom=gg.rnom,
            npairs=gg.npairs,
            weight=gg.weight,
            covariance=gg.cov if jackknife else None,
            variances=None if jackknife else np.concatenate([gg.varxip, gg.varxim]),
        )
        part = (
            sacc_io.save(s, out, custody=custody) if out else sacc_io.seal(s, custody)
        )
        self.xi_parts[ver, grid] = part
        self.print_done("Done 2PCF")
        return part

    def _measure_xi(self, ver, npatch, patch_centers=None, **treecorr_config):
        """TreeCorr's ξ± of ``ver``, in plaintext: for this object's use only.

        Jackknife variances with patches, shot noise without. With patches and
        no ``patch_centers``, TreeCorr draws its own centres.
        """
        gg = treecorr.GGCorrelation(
            {
                **self._binning(**treecorr_config),
                "var_method": "jackknife" if npatch > 1 else "shot",
            }
        )
        with self.results[ver].temporarily_read_data():
            g1, g2 = self._calibrated_g(ver)
            catalogue = treecorr.Catalog(
                ra=self.results[ver].dat_shear["RA"],
                dec=self.results[ver].dat_shear["Dec"],
                g1=g1,
                g2=g2,
                w=self._read_shear_cols(ver, "w_col"),
                ra_units=self.treecorr_config["ra_units"],
                dec_units=self.treecorr_config["dec_units"],
                npatch=npatch,
                patch_centers=patch_centers,
            )
        gg.process(catalogue)
        return gg

    def _patch_centers(self, ver, npatch, path=None):
        """The existing centres file ``ver`` splits at, or None without patches."""
        if npatch <= 1:
            return None
        path = path or self.patch_centers_path(ver, npatch)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{ver} has no patch centres at {path}; write them once with "
                f"write_patch_centers({base_catalogue(self._declared, ver)!r}, "
                f"{npatch}) (rule xi_patches)"
            )
        return path

    def write_patch_centers(self, catalogue, npatch, path=None):
        """Draw ``npatch`` jackknife patch centres for a base catalogue, once.

        TreeCorr's k-means over the catalogue's positions and weights, written
        to ``path`` (default :meth:`patch_centers_path`); every measurement of
        the catalogue and its variants splits at them.
        """
        base = base_catalogue(self._declared, catalogue)
        if base != catalogue:
            raise ValueError(
                f"patch centres belong to {base}, the base catalogue of "
                f"{catalogue}; write them from {base}"
            )
        path = path or self.patch_centers_path(catalogue, npatch)
        # A Catalog's k-means runs on TreeCorr's process-wide thread count.
        treecorr.set_omp_threads(self.treecorr_config["num_threads"])
        with self.results[catalogue].temporarily_read_data():
            positions = treecorr.Catalog(
                ra=self.results[catalogue].dat_shear["RA"],
                dec=self.results[catalogue].dat_shear["Dec"],
                w=self._read_shear_cols(catalogue, "w_col"),
                ra_units=self.treecorr_config["ra_units"],
                dec_units=self.treecorr_config["dec_units"],
                npatch=int(npatch),
            )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        positions.write_patch_centers(path)
        return path

    def _reporting_xi(self, ver):
        """``ver``'s reporting-grid ξ±, from its part, measured if not yet held."""
        part = self.xi_parts.get((ver, "reporting"))
        if part is None:
            part = self.calculate_2pcf(ver)
        return sacc_io.xi_correlation(part)

    def plot_2pcf(self):
        """The reporting-grid ξ± of every version, drawn from their parts."""
        xi = {ver: self._reporting_xi(ver) for ver in self.versions}

        # Plot of n_pairs
        plt.subplots(ncols=1, nrows=1)
        for ver in self.versions:
            plt.plot(
                xi[ver].meanr,
                xi[ver].npairs,
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
                xi[ver].meanr * cs_plots.dx(idx, fx=1.05, nx=len(ver)),
                xi[ver].xip,
                yerr=np.sqrt(xi[ver].varxip),
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
                xi[ver].meanr * cs_plots.dx(idx, fx=1.05, nx=len(ver)),
                xi[ver].xim,
                yerr=np.sqrt(xi[ver].varxim),
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
                xi[ver].meanr,
                xi[ver].xip * xi[ver].meanr,
                yerr=np.sqrt(xi[ver].varxip) * xi[ver].meanr,
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
                xi[ver].meanr * cs_plots.dx(idx, len(ver)),
                xi[ver].xim * xi[ver].meanr,
                yerr=np.sqrt(xi[ver].varxim) * xi[ver].meanr,
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
                    xi[ver].meanr * cs_plots.dx(idx, len(ver)),
                    xi[ver].xip,
                    yerr=np.sqrt(xi[ver].varxim),
                    label=r"$\xi_+$",
                    ls="solid",
                    color="green",
                )
                plt.errorbar(
                    xi[ver].meanr * cs_plots.dx(idx, len(ver)),
                    self.xi_psf_sys[ver]["mean"],
                    yerr=np.sqrt(self.xi_psf_sys[ver]["var"]),
                    label=r"$\xi^{\rm psf}_{+, {\rm sys}}$",
                    ls="dotted",
                    color="red",
                )
                plt.errorbar(
                    xi[ver].meanr * cs_plots.dx(idx, len(ver)),
                    xi[ver].xip + self.xi_psf_sys[ver]["mean"],
                    yerr=np.sqrt(xi[ver].varxip + self.xi_psf_sys[ver]["var"]),
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
            xi_psf_sys = self.xi_psf_sys[ver]
            gg = self._reporting_xi(ver)

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
        self, theta_min=0.3, theta_max=200, nbins=500, nbins_map=15, npatch=25
    ):
        self.print_start("Computing aperture-mass dispersion")

        self._map2 = {}
        theta_map = np.geomspace(theta_min * 5, theta_max / 2, nbins_map)
        self._map2["theta_map"] = theta_map

        for ver in self.versions:
            self.print_magenta(ver)
            self._refuse_if_blinded(ver, "The aperture-mass dispersion")
            gg = self._measure_xi(
                ver, npatch, min_sep=theta_min, max_sep=theta_max, nbins=nbins
            )

            mapsq, mapsq_im, mxsq, mxsq_im, varmapsq = gg.calculateMapSq(
                R=theta_map,
                m2_uform="Schneider",
            )
            out_fname_map2 = self._output_path(f"map2_{ver}.txt")
            if os.path.exists(out_fname_map2):
                self.print_green(f"Skipping Map2, {out_fname_map2} exists")
            else:
                print(f"Writing Map2 to output file {out_fname_map2} ")
                gg.writeMapSq(out_fname_map2, R=theta_map, m2_uform="Schneider")
            self._map2[ver] = {
                "mapsq": mapsq,
                "mapsq_im": mapsq_im,
                "mxsq": mxsq,
                "mxsq_im": mxsq_im,
                "varmapsq": varmapsq,
            }

        self.print_done("Done aperture-mass dispersion")

    @property
    def map2(self):
        if not hasattr(self, "_map2"):
            self.calculate_aperture_mass_dispersion()
        return self._map2

    def plot_aperture_mass_dispersion(self):
        for mode in ["mapsq", "mapsq_im", "mxsq", "mxsq_im"]:
            x = [self.map2["theta_map"] for ver in self.versions]
            y = [self.map2[ver][mode] for ver in self.versions]
            yerr = [np.sqrt(self.map2[ver]["varmapsq"]) for ver in self.versions]
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
            y = [np.abs(self.map2[ver][mode]) for ver in self.versions]
            yerr = [np.sqrt(self.map2[ver]["varmapsq"]) for ver in self.versions]
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
