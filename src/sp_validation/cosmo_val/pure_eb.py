# %%
"""Pure E/B-mode decomposition diagnostic.

Provides :class:`PureEBMixin`, which computes and plots the pure E/B-mode
correlation functions (xi+/xi- pure-mode decomposition) for catalog versions.
"""

import numpy as np

from .. import sacc_io
from ..b_modes import (
    calculate_eb_statistics,
    calculate_pure_eb_correlation,
    plot_eb_covariance_matrix,
    plot_integration_vs_reporting,
    plot_pte_2d_heatmaps,
    plot_pure_eb_correlations,
    save_pure_eb_results,
)
from .sacc_writers import pure_eb_to_sacc


class PureEBMixin:
    def calculate_pure_eb(
        self,
        version,
        min_sep=None,
        max_sep=None,
        nbins=None,
        min_sep_int=0.08,
        max_sep_int=300,
        nbins_int=1000,
        npatch=256,
        var_method="jackknife",
        cov_path_int=None,
        cosmo_cov=None,
        n_samples=1000,
    ):
        """
        Calculate the pure E/B modes for the given catalog version.
        The class instance's treecorr_config will be used for the "reporting" binning
        by default, but any kwargs passed to this function will overwrite the defaults.

        Parameters
        ----------
        version : str
            The catalog version to compute the pure E/B modes for.
        min_sep : float, optional
            Minimum separation for the reporting binning. Defaults to the value in
            self.treecorr_config if not provided.
        max_sep : float, optional
            Maximum separation for the reporting binning. Defaults to the value in
            self.treecorr_config if not provided.
        nbins : int, optional
            Number of bins for the reporting binning. Defaults to the value in
            self.treecorr_config if not provided.
        min_sep_int : float, optional
            Minimum separation for the integration binning. Defaults to 0.08.
        max_sep_int : float, optional
            Maximum separation for the integration binning. Defaults to 300.
        nbins_int : int, optional
            Number of bins for the integration binning. Defaults to 1000.
        npatch : int, optional
            Number of patches for the jackknife or bootstrap resampling. Defaults to
            the value in self.npatch if not provided.
        var_method : str, optional
            Variance estimation method. Defaults to "jackknife".
        cov_path_int : str, optional
            Path to the covariance matrix for the reporting binning. Replaces the
            treecorr covariance matrix if provided, meaning that var_method has no
            effect on the results although it is still passed to
            CosmologyValidation.calculate_2pcf.
        cosmo_cov : pyccl.Cosmology, optional
            Cosmology object to use for theoretical xi+/xi- predictions in the
            semi-analytical covariance calculation. Defaults to self.cosmo if not
            provided.
        n_samples : int, optional
            Number of Monte Carlo samples for semi-analytical covariance propagation.
            Defaults to 1000.

        Returns
        -------
        dict
            A dictionary containing the following keys:

            - "xip_E": Pure E-mode correlation function for xi+.
            - "xim_E": Pure E-mode correlation function for xi-.
            - "xip_B": Pure B-mode correlation function for xi+.
            - "xim_B": Pure B-mode correlation function for xi-.
            - "xip_amb": Ambiguity mode for xi+.
            - "xim_amb": Ambiguity mode for xi-.
            - "cov": Covariance matrix for the pure E/B modes.
            - "gg": The two-point correlation function object for the reporting binning.
            - "gg_int": The two-point correlation function object for the
              integration binning.
            - "eb_samples": (only when using semi-analytical covariance) Semi-analytic
              EB samples used for covariance calculation. Shape: (n_samples, 6*nbins)

        Notes
        -----
        - A shared patch file is used for the reporting and integration binning,
          and is created if it does not exist.
        """
        self.print_start(f"Computing {version} pure E/B")

        # Set up parameters with defaults
        npatch = npatch or self.npatch

        # Create TreeCorr configurations
        treecorr_config = self._binning(min_sep, max_sep, nbins)
        treecorr_config_int = self._binning(min_sep_int, max_sep_int, nbins_int)

        # Calculate correlation functions
        gg = self.calculate_2pcf(version, npatch=npatch, **treecorr_config)
        gg_int = self.calculate_2pcf(version, npatch=npatch, **treecorr_config_int)

        # Get redshift distribution if using analytic covariance
        z_dist = (
            np.column_stack(self.get_redshift(version))
            if cov_path_int is not None
            else None
        )

        # Delegate to b_modes module
        results = calculate_pure_eb_correlation(
            gg=gg,
            gg_int=gg_int,
            var_method=var_method,
            cov_path_int=cov_path_int,
            cosmo_cov=cosmo_cov,
            n_samples=n_samples,
            z_dist=z_dist,
        )

        return results

    def pure_eb_to_sacc_part(self, version, out_path, results, eb_override=None):
        """Write the pure-E/B SACC part (six ``PURE_KEYS`` blocks + covariance).

        ``results`` is the dict ``calculate_pure_eb`` returned: the six pure-mode
        arrays under ``sacc_io.PURE_KEYS``, the ``"cov"`` block (in ``PURE_KEYS``
        order), and the reporting-grid TreeCorr object ``"gg"`` whose ``meanr``
        is the shared ``theta``. The covariance must cover every stored point.

        ``eb_override`` replaces the six arrays with ones derived from the
        reporting + integration ξ± parts; the covariance stays from ``results``.
        """
        theta = results["gg"].meanr
        source = eb_override if eb_override is not None else results
        eb = {key: source[key] for key in sacc_io.PURE_KEYS}
        s = pure_eb_to_sacc(
            self.sacc_nz(version),
            self.sacc_metadata(version),
            theta,
            eb,
            covariance=results["cov"],
        )
        sacc_io.save(s, out_path, type="data")

    def plot_pure_eb(
        self,
        versions=None,
        output_dir=None,
        fiducial_xip_scale_cut=None,
        fiducial_xim_scale_cut=None,
        min_sep=None,
        max_sep=None,
        nbins=None,
        min_sep_int=0.08,
        max_sep_int=300,
        nbins_int=1000,
        npatch=None,
        var_method="jackknife",
        cov_path_int=None,
        cosmo_cov=None,
        n_samples=1000,
        results=None,
        **kwargs,
    ):
        """
        Generate comprehensive pure E/B mode analysis plots.

        Creates four types of plots for each version:
        1. Integration vs Reporting comparison
        2. E/B/Ambiguous correlation functions
        3. 2D PTE heatmaps
        4. Covariance matrix visualization

        Parameters
        ----------
        versions : list, optional
            List of catalog versions to process. Uses self.versions if None.
        output_dir : str, optional
            Output directory for plots. Uses configured output path if None.
        fiducial_xip_scale_cut : tuple, optional
            (min_scale, max_scale) for xi+ fiducial analysis, shown as gray regions
        fiducial_xim_scale_cut : tuple, optional
            (min_scale, max_scale) for xi- fiducial analysis, shown as gray regions
        min_sep, max_sep, nbins : float, float, int, optional
            Binning parameters for reporting scale. Uses treecorr_config if None.
        min_sep_int, max_sep_int, nbins_int : float, float, int
            Binning parameters for integration scale
            (default: 0.08-300 arcmin, 1000 bins)
        npatch : int, optional
            Number of patches for jackknife covariance. Uses self.npatch if None.
        var_method : str
            Variance method ("jackknife" or "semi-analytic").
            Automatically set to "semi-analytic" when cov_path_int is provided.
        cov_path_int : str, optional
            Path to integration covariance matrix for semi-analytical calculation
        cosmo_cov : pyccl.Cosmology, optional
            Cosmology for theoretical predictions in semi-analytical covariance
        n_samples : int
            Number of Monte Carlo samples for semi-analytical covariance (default: 1000)
        results : dict or list, optional
            Precalculated results to avoid recomputation. Can be a single results dict
            for one version, or a list of results dicts for multiple versions.
            If None (default), results will be calculated using calculate_pure_eb.
        **kwargs : dict
            Additional arguments passed to calculate_eb_statistics

        Notes
        -----
        This function orchestrates the full E/B mode analysis workflow:

        - Uses instance configuration as defaults for unspecified parameters
        - Automatically switches to analytical variance when theoretical
          covariance provided
        - Generates standardized output file naming based on all analysis
          parameters
        - Delegates individual plot generation to specialized functions in
          b_modes module
        """
        # Use instance defaults for unspecified parameters
        versions = versions or self.versions
        output_dir = output_dir or self.cc["paths"]["output"]
        npatch = npatch or self.npatch

        # Override var_method to analytic when cov_path_int is provided
        if cov_path_int is not None:
            var_method = "semi-analytic"

        # Use treecorr_config defaults for reporting scale binning
        min_sep = min_sep or self.treecorr_config["min_sep"]
        max_sep = max_sep or self.treecorr_config["max_sep"]
        nbins = nbins or self.treecorr_config["nbins"]

        # Handle results parameter - convert to list format for consistent processing
        if results is not None:
            if isinstance(results, dict):
                # Single results dict provided - should match single version
                if len(versions) != 1:
                    raise ValueError(
                        "Single results dict provided but multiple versions specified. "
                        "Provide results list matching versions length."
                    )
                results_list = [results]
            elif isinstance(results, list):
                # List of results provided
                if len(results) != len(versions):
                    raise ValueError(
                        f"Results list length ({len(results)}) does not match versions "
                        f"length ({len(versions)})"
                    )
                results_list = results
            else:
                raise TypeError("Results must be dict, list, or None")
        else:
            results_list = [None] * len(versions)

        for idx, version in enumerate(versions):
            # Generate standardized output filename stub
            out_stub = (
                f"{output_dir}/{version}_eb_minsep={min_sep}_"
                f"maxsep={max_sep}_nbins={nbins}_minsepint={min_sep_int}_"
                f"maxsepint={max_sep_int}_nbinsint={nbins_int}_npatch={npatch}_"
                f"varmethod={var_method}"
            )

            # Get or calculate results for this version
            version_results = results_list[idx] or self.calculate_pure_eb(
                version,
                min_sep=min_sep,
                max_sep=max_sep,
                nbins=nbins,
                min_sep_int=min_sep_int,
                max_sep_int=max_sep_int,
                nbins_int=nbins_int,
                npatch=npatch,
                var_method=var_method,
                cov_path_int=cov_path_int,
                cosmo_cov=cosmo_cov,
                n_samples=n_samples,
            )

            # Calculate E/B statistics for all bin combinations
            version_results = calculate_eb_statistics(
                version_results,
                cov_path_int=cov_path_int,
                n_samples=n_samples,
                **kwargs,
            )

            # Generate all plots using specialized plotting functions
            gg, gg_int = version_results["gg"], version_results["gg_int"]

            # Integration vs Reporting comparison plot
            plot_integration_vs_reporting(
                gg, gg_int, out_stub + "_integration_vs_reporting.png", version
            )

            # E/B/Ambiguous correlation functions plot
            plot_pure_eb_correlations(
                version_results,
                out_stub + "_xis.png",
                version,
                fiducial_xip_scale_cut=fiducial_xip_scale_cut,
                fiducial_xim_scale_cut=fiducial_xim_scale_cut,
            )

            # 2D PTE heatmaps plot
            plot_pte_2d_heatmaps(
                version_results,
                version,
                out_stub + "_ptes.png",
                fiducial_xip_scale_cut=fiducial_xip_scale_cut,
                fiducial_xim_scale_cut=fiducial_xim_scale_cut,
            )

            # Covariance matrix plot
            plot_eb_covariance_matrix(
                version_results["cov"],
                var_method,
                out_stub + "_covariance.png",
                version,
            )

            # Save data products and store on instance
            save_pure_eb_results(version_results, out_stub + "_data.npz")
            self._pure_eb_results[version] = version_results
