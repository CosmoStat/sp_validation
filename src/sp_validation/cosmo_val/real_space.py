"""Real-space two-point diagnostics for cosmology validation.

This mixin holds the real-space machinery: the TreeCorr two-point correlation
function (2PCF) ξ± measurement and its plots, the ratio of PSF systematics to
the cosmic-shear signal, and the aperture-mass dispersion ⟨M_ap²⟩ measurement
and plots. It depends on TreeCorr.
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import treecorr
from astropy.io import fits

from .. import sacc_io

# calculate_2pcf_version keys its results "tomo_bin_{b1}_tomo_bin_{b2}"; this
# reads the bin pair back out ("all" for the non-tomographic single pair).
_PAIR_KEY = re.compile(r"tomo_bin_(.+?)_tomo_bin_(.+)")


def _sacc_bin_index(ggs):
    """Map each catalog tomographic bin id to its 0-based SACC bin index.

    The catalog numbers its bins however the ``tomo_bin_col`` column does
    (1-based in practice, and the single id ``"all"`` for a non-tomographic
    run), while SACC tracers are ``source_{i}`` counted from zero. The map is
    built from the ids actually present, in ascending order, so it never
    assumes the catalog starts at 1 or numbers its bins without gaps.
    """
    ids = {b for key in ggs for b in _PAIR_KEY.fullmatch(key).groups()}
    order = sorted(ids, key=lambda b: 0 if b == "all" else int(b))
    return {b: i for i, b in enumerate(order)}


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
        compute_tomography=False,
        npatch=None,
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
                compute_tomography=compute_tomography,
                npatch=npatch,
                **treecorr_config,
            )

        return self.cat_ggs

    def save_2pcf_sacc(
        self,
        ver,
        sacc_path,
        ggs=None,
        *,
        type,
        grid="reporting",
        metadata=None,
    ):
        """Write the measured ξ± for ``ver`` to ``sacc_path`` as a SACC file.

        Serialises the TreeCorr output of :meth:`calculate_2pcf_version` (or of
        :meth:`calculate_2pcf`, via ``self.cat_ggs``) into the standard layout
        of :mod:`sp_validation.sacc_io`: the ``source_{i}`` n(z) tracers, one
        ξ+/ξ− block per tomographic bin pair, and the covariance.

        Insertion order is load-bearing. ``sacc_io.add_xi`` writes one bin pair
        as ``[ξ+; ξ−]``, so the data vector is *pair-major*, and the covariance
        is tied to it by position alone. Both the ξ insertion and the
        covariance therefore iterate the same ``pairs`` list, sorted by SACC
        bin index — never recomputed independently.

        The covariance follows what the measurement can support, read off the
        correlations themselves (``var_method``) rather than off ``npatch``,
        which the caller may have overridden per call:

        - jackknife (npatch > 1): ``treecorr.estimate_multi_cov`` over the
          correlations in ``pairs`` order. TreeCorr concatenates each one as
          ``[ξ+; ξ−]``, which is exactly the pair-major insertion order, so
          the result is one contiguous block spanning every ξ point —
          including the cross-pair covariance, which a per-pair ``gg.cov``
          would drop.
        - shot noise (npatch == 1): the ``varxip``/``varxim`` diagonal, stored
          as a SACC ``DiagonalCovariance``.

        Parameters
        ----------
        ver : str
            Catalog version, used for the n(z) lookup and stamped as metadata.
        sacc_path : str
            Path to the SACC product.
        ggs : dict, optional
            ``{"tomo_bin_{b1}_tomo_bin_{b2}": treecorr.GGCorrelation}`` as
            returned by :meth:`calculate_2pcf_version`. Defaults to
            ``self.cat_ggs[ver]``, i.e. the last :meth:`calculate_2pcf` run.
        type : {'data', 'mock'}
            Provenance of the input catalog, required by ``sacc_io.save``.
            No default: only the caller knows whether it ran on a mock, and
            ``sacc_io.load`` refuses unblinded ``type='data'`` files.
        grid : str, optional
            The ``grid`` tag on every point (default ``'reporting'``). Use
            ``'integration'`` for the fine grid COSEBIs / pure-EB integrate
            over, which shares this data type and tracer pair and is told
            apart by nothing else.
        metadata : dict, optional
            Extra key/value pairs stored on the file, merged over the
            version/npatch/blind stamped here.

        Returns
        -------
        sacc.Sacc
            The written data set.
        """
        ggs = self.cat_ggs[ver] if ggs is None else ggs
        if not ggs:
            raise ValueError(f"{ver}: no ξ± measurements to write")

        index = _sacc_bin_index(ggs)
        pairs = sorted(
            ggs, key=lambda key: [index[b] for b in _PAIR_KEY.fullmatch(key).groups()]
        )

        z, *nz_cols = self.get_redshift(ver)
        if len(nz_cols) != len(index):
            raise ValueError(
                f"{ver}: the n(z) file has {len(nz_cols)} distribution "
                f"column(s) but the measurement covers {len(index)} "
                f"tomographic bin(s) — every source bin needs its own n(z)"
            )

        s = sacc_io.new_sacc(
            [(z, nz) for nz in nz_cols],
            metadata={
                "version": ver,
                "npatch": int(ggs[pairs[0]].npatch1),
                **({"blind": self.blind} if self.blind is not None else {}),
                **(metadata or {}),
            },
        )

        for key in pairs:
            gg = ggs[key]
            bins = tuple(index[b] for b in _PAIR_KEY.fullmatch(key).groups())
            sacc_io.add_xi(
                s,
                bins,
                gg.meanr,
                gg.xip,
                gg.xim,
                grid=grid,
                theta_nom=gg.rnom,
                npairs=gg.npairs,
                weight=gg.weight,
            )

        sacc_io.save(s, sacc_path, type=type)
        self.print_done(f"Wrote ξ± SACC for {ver} to {sacc_path}.")
        return s

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
