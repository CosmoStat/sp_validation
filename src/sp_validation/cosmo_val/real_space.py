"""Real-space two-point diagnostics for cosmology validation.

This mixin holds the real-space machinery: the TreeCorr two-point correlation
function (2PCF) ξ± measurement, the aperture-mass dispersion ⟨M_ap²⟩
measurement, and the per-bin-pair plots of both. It depends on TreeCorr.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import treecorr

from sp_validation.statistics import jackknife_patch_centers

from .. import sacc_io
from .sacc_writers import xi_to_sacc

ALL_PAIR = "tomo_bin_all_tomo_bin_all"


class RealSpaceMixin:
    def calculate_2pcf_version(
        self,
        ver,
        npatch=None,
        compute_tomography=False,
        *,
        grid="reporting",
        out=None,
        **treecorr_config,
    ):
        """ξ± of one catalogue version on one binning, for each bin pair.

        The non-tomographic ``("all", "all")`` pair is born as its SACC part,
        sealed under the version's blind (:func:`sp_validation.sacc_io.seal`)
        before it is kept in ``self.xi_parts[ver, grid]``, returned or written,
        so on a blinded catalogue its true values never leave this method. A
        part already held for ``(ver, grid)`` is reused when no ``npatch``,
        binning or ``out`` is asked for, so the figure rules draw the parts they
        are handed. Tomographic pairs are measured only on a public catalogue
        (:meth:`_refuse_blinded_tomography`). Use :meth:`calculate_2pcf` to run
        over every version in ``self.versions``.

        Parameters:
            ver (str): The catalogue version to measure.
            npatch (int, optional): Jackknife patches; the instance's
                ``npatch`` by default. With patches the part carries the
                jackknife covariance, without them the shot-noise diagonal.
                Seeded patch centres come from the full catalogue and are
                shared by every bin pair.
            compute_tomography (bool, optional): Measure the tomographic bin
                pairs instead of ``("all", "all")``.
            grid (str): The grid tag the part's points carry.
            out (str, optional): Where to write the part as well.
            **treecorr_config: Overrides of the instance's ``treecorr_config``,
                e.g. ``min_sep=1``.

        Returns:
            dict: ``"tomo_bin_{b1}_tomo_bin_{b2}"`` → ξ± shaped like a TreeCorr
            ``GGCorrelation``: the part's
            :func:`~sp_validation.sacc_io.xi_correlation` view for
            ``("all", "all")``, the TreeCorr object for a tomographic pair.
        """
        reuse = npatch is None and out is None and not treecorr_config
        npatch = int(npatch or self.npatch)
        blind = self.blind(ver)

        if compute_tomography:
            self._refuse_blinded_tomography(ver)
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
        held = self.xi_parts.get((ver, grid)) if reuse else None
        if ALL_PAIR in ggs and held is not None:
            ggs[ALL_PAIR] = sacc_io.xi_correlation(held)
        to_compute = [
            (b1, b2)
            for b1, b2 in tomo_bin_pairs
            if ggs[f"tomo_bin_{b1}_tomo_bin_{b2}"] is None
        ]

        if to_compute:
            jackknife = npatch > 1
            gg_config = {
                **self._binning(**treecorr_config),
                "var_method": "jackknife" if jackknife else "shot",
            }
            cols = self._shear_columns(ver, compute_tomography)
            patch_centers = self._patch_centers(cols, npatch)

            for bin1, bin2 in to_compute:
                gg = treecorr.GGCorrelation(gg_config)

                cat_gal1 = self._bin_catalog(cols, bin1, npatch, patch_centers)
                cat_gal2 = (
                    self._bin_catalog(cols, bin2, npatch, patch_centers)
                    if bin1 != bin2
                    else None
                )

                gg.process(cat_gal1, cat2=cat_gal2)

                if (bin1, bin2) != ("all", "all"):
                    ggs[f"tomo_bin_{bin1}_tomo_bin_{bin2}"] = gg
                    continue
                s = xi_to_sacc(
                    self.sacc_nz(ver),
                    {**self.sacc_metadata(ver), "npatch": npatch},
                    gg.meanr,
                    gg.xip,
                    gg.xim,
                    grid=grid,
                    theta_nom=gg.rnom,
                    npairs=gg.npairs,
                    weight=gg.weight,
                    covariance=gg.cov if jackknife else None,
                    variances=(
                        None if jackknife else np.concatenate([gg.varxip, gg.varxim])
                    ),
                )
                if out:
                    part = sacc_io.save(s, out, blind=blind)
                else:
                    part = sacc_io.seal(s, blind)
                self.xi_parts[ver, grid] = part
                ggs[ALL_PAIR] = sacc_io.xi_correlation(part)

        self.print_done(f"Done 2PCF for {ver}.")

        return ggs

    def _shear_columns(self, ver, compute_tomography):
        """Positions, calibrated shears, weights and bin labels of a version.

        All arrays come from the same rows of the version's shear catalogue, as
        read by ``self.results[ver]``. The bin labels are ``None`` unless
        ``compute_tomography``.
        """
        with self.results[ver].temporarily_read_data():
            dat = self.results[ver].dat_shear
            g1, g2 = self._calibrated_g(ver)
            return {
                "ra": np.asarray(dat["RA"]),
                "dec": np.asarray(dat["Dec"]),
                "g1": np.asarray(g1),
                "g2": np.asarray(g2),
                "w": np.asarray(self._read_shear_cols(ver, "w_col")),
                "tomo_bin": (
                    np.asarray(dat[self.cc[ver]["shear"]["tomo_bin_col"]])
                    if compute_tomography
                    else None
                ),
            }

    def _patch_centers(self, cols, npatch):
        """Seeded patch centres from the full catalogue of one version."""
        if int(npatch) <= 1:
            return None
        cat = treecorr.Catalog(
            ra=cols["ra"],
            dec=cols["dec"],
            w=cols["w"],
            ra_units=self.treecorr_config["ra_units"],
            dec_units=self.treecorr_config["dec_units"],
        )
        return jackknife_patch_centers(cat, int(npatch))

    def _bin_catalog(self, cols, tomo_bin_id, npatch, patch_centers=None):
        """TreeCorr catalogue of one tomographic bin (``"all"``: every row)."""
        mask = slice(None) if tomo_bin_id == "all" else cols["tomo_bin"] == tomo_bin_id
        return treecorr.Catalog(
            ra=cols["ra"][mask],
            dec=cols["dec"][mask],
            g1=cols["g1"][mask],
            g2=cols["g2"][mask],
            w=cols["w"][mask],
            ra_units=self.treecorr_config["ra_units"],
            dec_units=self.treecorr_config["dec_units"],
            npatch=npatch,
            patch_centers=patch_centers,
        )

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
            ``{"tomo_bin_{b1}_tomo_bin_{b2}": ξ±}`` dict.
        """
        self.cat_ggs = {}
        for ver in self.versions:
            self.cat_ggs[ver] = self.calculate_2pcf_version(
                ver,
                npatch=npatch,
                compute_tomography=compute_tomography,
                **treecorr_config,
            )

        return self.cat_ggs

    def calculate_aperture_mass_dispersion(
        self,
        theta_min=0.3,
        theta_max=200,
        nbins=500,
        nbins_map=15,
        npatch=None,
        compute_tomography=False,
    ):
        """⟨M_ap²⟩ and ⟨M_×²⟩ of every version and bin pair, from its ξ±.

        Both are linear in ξ± (TreeCorr's ``calculateMapSq`` sum, Schneider et
        al. 2002 filter), so they and their covariance T·C·Tᵀ come from the ξ±
        of :meth:`calculate_2pcf_version` on a fine grid: for ``("all",
        "all")`` its sealed part, concealed on a blinded catalogue, whose shift,
        the same in every patch, leaves C unchanged.
        """
        theta_map = np.geomspace(theta_min * 5, theta_max / 2, nbins_map)
        self._map2 = {"theta_map": theta_map}
        bin_size = np.log(theta_max / theta_min) / nbins

        for ver in self.versions:
            ggs = self.calculate_2pcf_version(
                ver,
                npatch=npatch,
                compute_tomography=compute_tomography,
                grid="aperture_mass",
                min_sep=theta_min,
                max_sep=theta_max,
                nbins=nbins,
            )
            self._map2[ver] = {}
            for key, gg in ggs.items():
                transform = _map2_transform(theta_map, gg.meanr, bin_size)
                mapsq, mxsq = np.split(transform @ np.concatenate([gg.xip, gg.xim]), 2)
                variances = np.split(np.diag(transform @ gg.cov @ transform.T), 2)
                self._map2[ver][key] = {
                    "mapsq": mapsq,
                    "mxsq": mxsq,
                    "varmapsq": variances[0],
                    "varmxsq": variances[1],
                }
            self.print_done(f"Done aperture-mass dispersion for {ver}.")

    @property
    def map2(self):
        if not hasattr(self, "_map2"):
            self.calculate_aperture_mass_dispersion()
        return self._map2

    def plot_2pcf(
        self, tomography=False, offset=0.02, alpha=1.0, show=True, close=True
    ):
        """Plot ξ± of every version, one panel per bin pair.

        Measures, or reads back, the 2PCF with :meth:`calculate_2pcf` and draws
        it with :meth:`plot_2pcf_tomography`, as ξ± (log-log) and as θ·ξ±.
        Writes ``xi_pm_tomography_{tomography}.png`` and
        ``xi_pm_theta_tomography_{tomography}.png`` under the output directory.
        """
        self.calculate_2pcf(compute_tomography=tomography)

        for times_theta in (False, True):
            prefix = r"$\theta\,$" if times_theta else ""
            suffix = "_theta" if times_theta else ""
            self.plot_2pcf_tomography(
                self._xiplus_ximinus_sample_x_y_plot_function,
                r"$\theta$ [arcmin]",
                prefix + r"$\xi_+(\theta)$",
                prefix + r"$\xi_-(\theta)$",
                (0.05, 0.9) if times_theta else (0.8, 0.95),
                extract_text_offset=times_theta,
                add_index_version_to_kwargs=True,
                x_scale="log",
                y_scale="linear" if times_theta else "log",
                tomography=tomography,
                savefig=self._output_path(f"xi_pm{suffix}_tomography_{tomography}.png"),
                show=show,
                close=close,
                offset=offset,
                times_theta=times_theta,
                alpha=alpha,
            )

    def plot_ratio_xi_sys_xi(
        self, tomography=False, threshold=0.1, offset=0.02, show=True, close=True
    ):
        """Plot ξ^{PSF, sys}_± / ξ± of every version, one panel per bin pair.

        The band marks ``±threshold``. ξ± comes from :meth:`calculate_2pcf` and
        ξ^{PSF, sys} from the ``xi_psf_sys`` property, both on the instance's
        ``treecorr_config`` binning. Writes ``ratio_xi_sys_xi.png`` (non-
        tomographic) or ``ratio_xi_sys_xi_tomography.png`` under the output
        directory.
        """
        if tomography and not self.compute_tomography:
            raise ValueError(
                "plot_ratio_xi_sys_xi(tomography=True) needs the tomographic "
                "xi_psf_sys; construct CosmologyValidation with "
                "compute_tomography=True"
            )
        self.calculate_2pcf(compute_tomography=tomography)

        y_label = r"$\xi^{{\rm PSF, sys}}_{0} / \xi_{0}$"
        self.plot_2pcf_tomography(
            self._ratio_xi_sys_xi_x_y_plot_function,
            r"$\theta$ [arcmin]",
            y_label.format("+"),
            y_label.format("-"),
            (0.8, 0.95),
            extract_text_offset=False,
            add_index_version_to_kwargs=True,
            x_scale="log",
            tomography=tomography,
            savefig=self._output_path(
                "ratio_xi_sys_xi_tomography.png"
                if tomography
                else "ratio_xi_sys_xi.png"
            ),
            show=show,
            close=close,
            offset=offset,
            threshold=threshold,
        )

    def plot_2pcf_tomography(
        self,
        x_y_plot_function,
        x_label,
        y_label_plus,
        y_label_minus,
        tomo_bin_label_position,
        extract_text_offset,
        add_index_version_to_kwargs,
        x_scale=None,
        y_scale=None,
        tomography=False,
        versions=None,
        colors=None,
        savefig=None,
        show=True,
        close=True,
        **kwargs,
    ):
        """
        Standard plot function for 2-point correlation functions with tomographic bins.

        Parameters
        ----------
        x_y_plot_function : callable
            Function to plot the x and y data. Should accept axes for `+' and `-' components, version, tomo_bin_indices, and kwargs.
        x_label : str
            Label for the x-axis.
        y_label_plus : str
            Label for the y-axis of the `+' component.
        y_label_minus : str
            Label for the y-axis of the `-' component.
        tomo_bin_label_position : tuple
            Position to place the tomographic bin labels in axes coordinates (x, y).
        extract_text_offset : bool
            If True, extract the y-axis offset text and include it in the y-label.
        add_index_version_to_kwargs : bool
            If True, add the index of the version to the kwargs for plotting.
        x_scale : str, optional
            Scale for the x-axis ('linear', 'log', etc.). If None, default scale is used.
        y_scale : str, optional
            Scale for the y-axis ('linear', 'log', etc.). If None, default scale is used.
        tomography : bool, optional
            If True, plot the tomographic bins. Default is False.
        versions : list, optional
            List of versions to plot. If None, all versions are plotted.
        colors : list, optional
            List of colors for each version. If None, default colors are used.
        savefig : str, optional
            If provided, save the figure to this file.
        show : bool, optional
            If True, show the figure.
        close : bool, optional
            If True, close the figure after saving or showing.
        kwargs : dict
            Additional keyword arguments to pass to the plotting function.
        """
        versions = versions if versions is not None else self.versions
        colors = (
            colors
            if colors is not None
            else [self.cc[ver]["colour"] for ver in versions]
        )

        # First get the max number of tomo bins among the versions.
        tomo_bins = self._get_tomo_bins_for_versions(versions, tomography=tomography)

        max_key = max(tomo_bins, key=lambda k: len(tomo_bins[k]["ids"]))
        n_tomo_bins_plot = len(tomo_bins[max_key]["ids"])
        reference_tomo_bin_pairs = tomo_bins[max_key]["pairs"]

        n_rows = n_tomo_bins_plot
        n_cols = n_tomo_bins_plot + 2

        # First start with the quantiles plot
        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5 * n_cols, 5 * n_rows),
            sharex=True,
            sharey=True,
            gridspec_kw={"wspace": 0, "hspace": 0},
        )

        for idx, (ver, color) in enumerate(zip(versions, colors)):
            tomo_bin_pairs = tomo_bins[ver]["pairs"]

            kwargs["color"] = color
            if add_index_version_to_kwargs:
                kwargs["idx"] = idx
                kwargs["versions"] = versions

            for tomo_bin_a, tomo_bin_b in tomo_bin_pairs:
                # Plot the nsamples last samples
                ax_plus = self._get_ax_plus(axs, tomo_bin_a, tomo_bin_b)
                ax_minus = self._get_ax_minus(axs, tomo_bin_a, tomo_bin_b)

                # Apply the x_y_plot_function to plot the data
                x_y_plot_function(
                    ax_plus, ax_minus, ver, tomo_bin_a, tomo_bin_b, **kwargs
                )

        # Draw to extract the y-axis text offset
        fig.canvas.draw()

        # Set the visibility to false where necessary
        self._set_ax_visibility_to_false(axs, n_tomo_bins_plot)

        # Set the labels and scales for the plots
        for tomo_bin_a, tomo_bin_b in reference_tomo_bin_pairs:
            ax_plus = self._get_ax_plus(axs, tomo_bin_a, tomo_bin_b)
            ax_minus = self._get_ax_minus(axs, tomo_bin_a, tomo_bin_b)

            ax_plus.tick_params(
                axis="both",
                which="both",
                direction="in",
                bottom=True,
                top=False,
                labelbottom=tomo_bin_b == 1 or tomo_bin_b == "all",
                left=True,
                right=False,
                labelleft=tomo_bin_a == 1 or tomo_bin_a == "all",
            )
            if x_scale is not None:
                ax_plus.set_xscale(x_scale)

            ax_plus.text(
                tomo_bin_label_position[0],
                tomo_bin_label_position[1],
                f"{tomo_bin_a}-{tomo_bin_b}",
                transform=ax_plus.transAxes,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="square",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.8,
                ),
            )
            ax_plus.axhline(0, color="k", linestyle="--")

            if y_scale is not None:
                ax_plus.set_yscale(y_scale)
            if tomo_bin_b == 1 or tomo_bin_b == "all":
                ax_plus.set_xlabel(r"$\theta$ [arcmin]")
            if tomo_bin_a == 1 or tomo_bin_a == "all":
                text_offset = (
                    ax_plus.yaxis.get_offset_text().get_text()
                    if extract_text_offset
                    else ""
                )
                ax_plus.set_ylabel(y_label_plus + text_offset)
            ax_plus.yaxis.get_offset_text().set_visible(
                False
            )  # Hide the offset text for the plus ax

            # Move the ticks to the right for the minus ax
            ax_minus.yaxis.tick_right()
            ax_minus.yaxis.set_label_position("right")
            ax_minus.tick_params(
                axis="both",
                which="both",
                direction="in",
                bottom=True,
                top=False,
                labelbottom=tomo_bin_b == n_tomo_bins_plot or tomo_bin_b == "all",
                left=False,
                right=True,
                labelleft=False,
                labelright=tomo_bin_a == 1 or tomo_bin_a == "all",
            )
            if x_scale is not None:
                ax_minus.set_xscale(x_scale)
            ax_minus.text(
                tomo_bin_label_position[0],
                tomo_bin_label_position[1],
                f"{tomo_bin_a}-{tomo_bin_b}",
                transform=ax_minus.transAxes,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="square",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.8,
                ),
            )
            ax_minus.axhline(0, color="k", linestyle="--")
            if y_scale is not None:
                ax_minus.set_yscale(y_scale)
            if tomo_bin_b == n_tomo_bins_plot or tomo_bin_b == "all":
                ax_minus.set_xlabel(x_label)
            if tomo_bin_a == 1 or tomo_bin_a == "all":
                text_offset = (
                    ax_minus.yaxis.get_offset_text().get_text()
                    if extract_text_offset
                    else ""
                )
                ax_minus.set_ylabel(y_label_minus + text_offset)
            ax_minus.yaxis.get_offset_text().set_visible(
                False
            )  # Hide the offset text for the minus ax

        # Build the legend
        handles = []
        for ver, color in zip(versions, colors):
            label = self.cc[ver]["label"] if "label" in self.cc[ver] else ver
            handles.append(plt.Line2D([0], [0], color=color, lw=2, label=label))
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )

        if savefig is not None:
            plt.savefig(savefig, dpi=300, bbox_inches="tight")
            self.print_done(f"Plot saved to {os.path.abspath(savefig)}")

        if show:
            plt.show()

        if close:
            plt.close()

    def _xiplus_ximinus_sample_x_y_plot_function(
        self,
        ax_plus,
        ax_minus,
        version,
        tomo_bin_a,
        tomo_bin_b,
        idx,
        versions,
        color,
        offset,
        times_theta,
        alpha,
    ):
        """Plot the measured ξ± 2PCF for one version/tomographic-bin pair.

        Uses the jackknife covariance from TreeCorr to plot the error bars. This function is fed into :meth:`plot_2pcf_tomography` as the ``x_y_plot_function`` argument.

        Parameters
        ----------
        ax_plus, ax_minus : matplotlib.axes.Axes
            Axes for the ξ+ and ξ- components.
        version : str
            Catalog version to plot.
        tomo_bin_a, tomo_bin_b : int or str
            Tomographic bin pair (``"all"`` for the non-tomographic case).
        idx : int
            Index of ``version`` within ``versions`` (used for the x-jitter).
        versions : list
            Full list of versions being plotted.
        color : str
            Colour for this version.
        offset : float
            Fractional jitter applied to θ for readability.
        times_theta : bool
            If True, plot θ·ξ± rather than ξ±.
        alpha : float
            Opacity of the plotted points/error bars.
        """
        # Get the measured 2PCF for this version and tomographic-bin pair.
        gg = self.cat_ggs[version][f"tomo_bin_{tomo_bin_a}_tomo_bin_{tomo_bin_b}"]

        # Angular scales of the measurement.
        theta = gg.meanr

        # Add the offset to the theta values for better visualisation.
        jittered_theta = self._get_jittered_theta(theta, idx, len(versions), offset)

        scale = theta if times_theta else 1

        y_plus = gg.xip * scale
        y_minus = gg.xim * scale
        yerr_plus = np.sqrt(gg.varxip) * scale
        yerr_minus = np.sqrt(gg.varxim) * scale

        ax_plus.errorbar(
            jittered_theta,
            y_plus,
            yerr=yerr_plus,
            color=color,
            alpha=alpha,
            fmt="o",
            markersize=3,
            capsize=2,
        )

        ax_minus.errorbar(
            jittered_theta,
            y_minus,
            yerr=yerr_minus,
            color=color,
            alpha=alpha,
            fmt="o",
            markersize=3,
            capsize=2,
        )

    def _mapsq_mxsq_sample_x_y_plot_function(
        self,
        ax_plus,
        ax_minus,
        version,
        tomo_bin_a,
        tomo_bin_b,
        idx,
        versions,
        color,
        offset,
        times_theta,
        alpha,
    ):
        """Plot the aperture-mass dispersion ⟨M_ap²⟩ / ⟨M_×²⟩ for one bin pair.

        Uses the jackknife covariance from TreeCorr to plot the error bars. This function is fed into :meth:`plot_2pcf_tomography` as the ``x_y_plot_function`` argument. The E-mode ⟨M_ap²⟩ is placed on the ``plus`` axis and the B-mode ⟨M_×²⟩ on the ``minus`` axis.

        Parameters
        ----------
        ax_plus, ax_minus : matplotlib.axes.Axes
            Axes for the E-mode (⟨M_ap²⟩) and B-mode (⟨M_×²⟩) components.
        version : str
            Catalog version to plot.
        tomo_bin_a, tomo_bin_b : int or str
            Tomographic bin pair (``"all"`` for the non-tomographic case).
        idx : int
            Index of ``version`` within ``versions`` (used for the x-jitter).
        versions : list
            Full list of versions being plotted.
        color : str
            Colour for this version.
        offset : float
            Fractional jitter applied to θ for readability.
        times_theta : bool
            If True, plot θ·⟨M²⟩ rather than ⟨M²⟩.
        alpha : float
            Opacity of the plotted points/error bars.
        """
        # Get the aperture-mass dispersion for this version and bin pair.
        map2 = self.map2[version][f"tomo_bin_{tomo_bin_a}_tomo_bin_{tomo_bin_b}"]

        # Angular scales of the measurement.
        theta = self.map2["theta_map"]

        # Add the offset to the theta values for better visualisation.
        jittered_theta = self._get_jittered_theta(theta, idx, len(versions), offset)

        scale = theta if times_theta else 1

        y_plus = map2["mapsq"] * scale
        y_minus = map2["mxsq"] * scale

        ax_plus.errorbar(
            jittered_theta,
            y_plus,
            yerr=np.sqrt(map2["varmapsq"]) * scale,
            color=color,
            alpha=alpha,
            fmt="o",
            markersize=3,
            capsize=2,
        )

        ax_minus.errorbar(
            jittered_theta,
            y_minus,
            yerr=np.sqrt(map2["varmxsq"]) * scale,
            color=color,
            alpha=alpha,
            fmt="o",
            markersize=3,
            capsize=2,
        )

    def _ratio_xi_sys_xi_x_y_plot_function(
        self,
        ax_plus,
        ax_minus,
        version,
        tomo_bin_a,
        tomo_bin_b,
        idx,
        versions,
        color,
        offset,
        threshold,
    ):
        """Plot ξ^{PSF, sys}_± / ξ± for one version/tomographic-bin pair.

        Fed into :meth:`plot_2pcf_tomography` as the ``x_y_plot_function``
        argument. The error bar propagates the variances of both ξ^{PSF, sys}
        and ξ±; the first version also draws the ``±threshold`` band.
        """
        key = f"tomo_bin_{tomo_bin_a}_tomo_bin_{tomo_bin_b}"
        gg = self.cat_ggs[version][key]
        xi_psf_sys = self.xi_psf_sys[version][key]

        theta = self._get_jittered_theta(gg.meanr, idx, len(versions), offset)

        for ax, xi, var_xi, component in (
            (ax_plus, gg.xip, gg.varxip, "plus"),
            (ax_minus, gg.xim, gg.varxim, "minus"),
        ):
            mean = xi_psf_sys[f"mean_{component}"]
            var = xi_psf_sys[f"var_{component}"]
            ratio = mean / xi
            ratio_err = np.sqrt(var / xi**2 + mean**2 * var_xi / xi**4)
            ax.errorbar(
                theta,
                ratio,
                yerr=ratio_err,
                color=color,
                fmt=self.cc[version].get("marker", "o"),
                markersize=3,
                capsize=2,
            )
            if idx == 0:
                ax.axhspan(-threshold, threshold, color="black", alpha=0.1)


def _map2_transform(radii, theta, bin_size):
    """The matrix taking [ξ+, ξ−] on a log grid to [⟨M_ap²⟩, ⟨M_×²⟩] at ``radii``.

    TreeCorr's ``calculateMapSq`` sum with the Schneider et al. (2002) filter:
    ⟨M_ap²⟩, ⟨M_×²⟩ = Σ s² (T+ ξ+ ± T− ξ−) dlnθ / 2, s = θ/R, T± zero for s ≥ 2.
    """
    s = np.minimum(np.outer(1.0 / radii, theta), 2.0)
    ssq = s * s
    tp = 12.0 / (5.0 * np.pi) * (2.0 - 15.0 * ssq) * np.arccos(s / 2.0)
    tp += (
        s
        * np.sqrt(4.0 - ssq)
        * (120.0 + ssq * (2320.0 + ssq * (-754.0 + ssq * (132.0 - 9.0 * ssq))))
        / (100.0 * np.pi)
    )
    tm = 3.0 / (70.0 * np.pi) * s * ssq * (4.0 - ssq) ** 3.5
    tp, tm = (x * ssq * 0.5 * bin_size for x in (tp, tm))
    return np.block([[tp, tm], [tp, -tm]])
