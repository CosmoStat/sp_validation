"""Real-space two-point diagnostics for cosmology validation.

This mixin holds the real-space machinery: the TreeCorr two-point correlation
function (2PCF) ξ± measurement, the aperture-mass dispersion ⟨M_ap²⟩
measurement, and the per-bin-pair plots of both. It depends on TreeCorr.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import treecorr

from sp_validation.angular_binning import validate_nested_grids
from sp_validation.correlation import (
    measurement_matches,
    process_gg,
    rebin_gg_means,
    write_measurement_metadata,
)
from sp_validation.statistics import jackknife_patch_centers


class RealSpaceMixin:
    def calculate_2pcf_version(
        self,
        ver,
        npatch=None,
        compute_tomography=False,
        read_cached=False,
        fine_correlations=None,
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

            read_cached (bool, optional): Reuse columns-only dumps for plotting,
            including patched runs. They carry the published means and variances,
            but no patch results for dense or derived jackknife covariance.

            fine_correlations (dict, optional): Unpatched fine-grid GG objects
            keyed by bin pair, measured from the same catalogue selections.
            With a configured integration grid these supply the reporting means;
            otherwise the fine grid is measured once by this method.

            **treecorr_config: Additional TreeCorr configuration parameters that
            will override the instance's default `treecorr_config`. For example,
            `min_sep=1`.

        Returns:
            dict: Mapping of ``"tomo_bin_{b1}_tomo_bin_{b2}"`` to the corresponding
            treecorr.GGCorrelation object. For the non-tomographic case the single
            key is ``"tomo_bin_all_tomo_bin_all"``.

        Notes:
            - The non-tomographic pair is written to a columns-only TreeCorr
              dump with a configuration/mean-source JSON sidecar. Unpatched
              runs can reuse it; patched measurements remeasure to retain
              covariance. Plotting can opt into reading the saved columns.
            - With ``integration`` configured, reporting means are pair-weighted
              fine-grid means on exactly nested edges. Other measurements use
              the configured unpatched means pass.
            - ``b_target`` applies only to means. The patched covariance pass
              omits ``bin_slop`` so TreeCorr uses its default.
            - Seeded patch centres are computed once from the full catalogue and
              shared by every tomographic bin pair.
        """

        npatch = npatch or self.npatch
        treecorr_config = {
            **self._binning(**treecorr_config),
            "var_method": "jackknife" if int(npatch) > 1 else "shot",
        }

        integration = self.integration
        use_fine = integration is not None and any(
            treecorr_config[key] != integration[key]
            for key in ("min_sep", "max_sep", "nbins")
        )
        cache_config = dict(treecorr_config)
        if use_fine:
            validate_nested_grids(treecorr_config, integration)
            cache_config["integration"] = integration
        elif fine_correlations is not None:
            raise ValueError(
                "fine_correlations needs a configured reporting/integration grid pair"
            )

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
        to_compute = []
        for bin1, bin2 in tomo_bin_pairs:
            if (bin1, bin2) == ("all", "all"):
                out_fname = self._xi_txt_path(ver, treecorr_config, npatch)
                if (
                    (int(npatch) == 1 or read_cached)
                    and not self.force_run
                    and fine_correlations is None
                    and measurement_matches(out_fname, cache_config)
                ):
                    self.print_done(f"Skipping 2PCF calculation, {out_fname} exists")
                    gg = treecorr.GGCorrelation(treecorr_config)
                    gg.read(out_fname)
                    ggs["tomo_bin_all_tomo_bin_all"] = gg
                    continue
            to_compute.append((bin1, bin2))

        if to_compute:
            if use_fine and fine_correlations is None:
                fine_correlations = self.calculate_2pcf_version(
                    ver, npatch=1, compute_tomography=compute_tomography, **integration
                )
            cols = self._shear_columns(ver, compute_tomography)
            patch_centers = self._patch_centers(cols, npatch)

            for bin1, bin2 in to_compute:
                cat_gal1 = self._bin_catalog(cols, bin1, npatch, patch_centers)
                cat_gal2 = (
                    self._bin_catalog(cols, bin2, npatch, patch_centers)
                    if bin1 != bin2
                    else None
                )

                means = None
                if use_fine:
                    pair = f"tomo_bin_{bin1}_tomo_bin_{bin2}"
                    if pair not in fine_correlations:
                        raise ValueError(f"missing fine-grid correlation for {pair}")
                    fine = fine_correlations[pair]
                    if fine.npatch1 != 1 or fine.npatch2 != 1:
                        raise ValueError(
                            "fine-grid means must come from unpatched catalogues"
                        )
                    actual = dict(
                        min_sep=fine.min_sep, max_sep=fine.max_sep, nbins=fine.nbins
                    )
                    if actual != integration:
                        raise ValueError(
                            "fine correlation does not match the configured integration grid"
                        )
                    means = rebin_gg_means(fine, treecorr_config)
                gg = process_gg(treecorr_config, cat_gal1, cat_gal2, means=means)

                if (bin1, bin2) == ("all", "all"):
                    # Columns only. The covariance matrix lives in the SACC part;
                    # a per-patch ξ± realisation is an unblinded data vector
                    # nothing reads; and TreeCorr cannot read back a text file
                    # carrying the matrix without the per-patch results.
                    gg.write(
                        self._xi_txt_path(ver, treecorr_config, npatch),
                        write_patch_results=False,
                        write_cov=False,
                        precision=17,
                    )
                    write_measurement_metadata(
                        self._xi_txt_path(ver, treecorr_config, npatch), cache_config
                    )

                ggs[f"tomo_bin_{bin1}_tomo_bin_{bin2}"] = gg

        self.print_done(f"Done 2PCF for {ver}.")

        return ggs

    def _xi_txt_path(self, ver, treecorr_config, npatch):
        """Path of the non-tomographic ξ± TreeCorr dump for a version."""
        return self._output_path(
            f"xi_{self.basename(ver, treecorr_config=treecorr_config, npatch=npatch)}.txt"
        )

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
        read_cached=False,
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

            read_cached (bool, optional): Reuse columns-only dumps for plotting;
            defaults to False so patched measurements retain resampling state.

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
                read_cached=read_cached,
                **treecorr_config,
            )

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

        treecorr_config = self._binning(
            theta_min,
            theta_max,
            nbins,
            var_method="jackknife" if int(npatch) > 1 else "shot",
        )

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
            cols = self._shear_columns(ver, compute_tomography)
            patch_centers = self._patch_centers(cols, npatch)

            for bin1, bin2 in tomo_bin_pairs:
                cat_gal1 = self._bin_catalog(cols, bin1, npatch, patch_centers)
                cat_gal2 = (
                    self._bin_catalog(cols, bin2, npatch, patch_centers)
                    if bin1 != bin2
                    else None
                )

                gg = process_gg(treecorr_config, cat_gal1, cat_gal2)

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

    def plot_2pcf(
        self, tomography=False, offset=0.02, alpha=1.0, show=True, close=True
    ):
        """Plot ξ± of every version, one panel per bin pair.

        Measures, or reads back, the 2PCF with :meth:`calculate_2pcf` and draws
        it with :meth:`plot_2pcf_tomography`, as ξ± (log-log) and as θ·ξ±.
        Writes ``xi_pm_tomography_{tomography}.png`` and
        ``xi_pm_theta_tomography_{tomography}.png`` under the output directory.
        """
        self.calculate_2pcf(compute_tomography=tomography, read_cached=True)

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
        self.calculate_2pcf(compute_tomography=tomography, read_cached=True)

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
        # Both E- and B-mode share the same variance estimate.
        yerr = np.sqrt(map2["varmapsq"]) * scale

        ax_plus.errorbar(
            jittered_theta,
            y_plus,
            yerr=yerr,
            color=color,
            alpha=alpha,
            fmt="o",
            markersize=3,
            capsize=2,
        )

        ax_minus.errorbar(
            jittered_theta,
            y_minus,
            yerr=yerr,
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
