# %%
"""Pure E/B-mode decomposition diagnostic.

Provides :class:`PureEBMixin`, which computes and plots the pure E/B-mode
correlation functions (xi+/xi- pure-mode decomposition) for catalog versions.
"""

import numpy as np

from ..angular_binning import validate_nested_grids
from ..b_modes import (
    calculate_eb_statistics,
    calculate_pure_eb_correlation,
    covariance_label,
    plot_eb_covariance_matrix,
    plot_integration_vs_reporting,
    plot_pte_2d_heatmaps,
    plot_pure_eb_correlations,
    save_pure_eb_results,
)


class PureEBMixin:
    def calculate_pure_eb(
        self,
        version,
        min_sep=None,
        max_sep=None,
        nbins=None,
        min_sep_int=None,
        max_sep_int=None,
        nbins_int=None,
        npatch=None,
        compute_tomography=False,
        cov_path_int=None,
    ):
        """
        Calculate the pure E/B modes for the given catalog version.

        ξ± is measured on the fine integration grid only; the reporting
        binning (the instance's treecorr_config unless overridden) enters as
        bin edges, snapped onto the fine edges, into which
        :func:`~sp_validation.b_modes.pure_eb_operator` averages the modes.

        Parameters
        ----------
        version : str
            The catalog version to compute the pure E/B modes for.
        min_sep, max_sep, nbins : float, float, int, optional
            Reporting binning. Default to the values in self.treecorr_config.
        min_sep_int, max_sep_int, nbins_int : float, float, int, optional
            Integration binning (the configured grid, otherwise 0.08–300 arcmin,
            1000 bins). It must
            extend beyond the reporting range on both sides.
        npatch : int, optional
            Jackknife patch count. Defaults to self.npatch.
        compute_tomography : bool, optional
            Whether to compute the pure E/B modes for every tomographic bin pair
            instead of the non-tomographic pair. Defaults to False.
        cov_path_int : str, optional
            Analytic ξ± covariance on the integration grid. Without it the
            covariance is the jackknife of the integration-grid ξ±. One file
            holds one pair's covariance, so it cannot be combined with
            ``compute_tomography``.

        Returns
        -------
        dict
            Mapping of ``"tomo_bin_{b1}_tomo_bin_{b2}"`` (the single key
            ``"tomo_bin_all_tomo_bin_all"`` without tomography) to the results of
            :func:`~sp_validation.b_modes.calculate_pure_eb_correlation`: the six
            pure-mode arrays, their covariance ``cov`` and its ``npatch`` record
            (``None`` for an analytic covariance), the reporting grid and the
            integration-grid ξ±.
        """
        self.print_start(f"Computing {version} pure E/B")

        if cov_path_int is not None and compute_tomography:
            raise ValueError(
                "cov_path_int holds a single ξ± covariance; it cannot serve every "
                "tomographic bin pair."
            )

        reporting = self._binning(min_sep, max_sep, nbins)
        reporting_edges = np.geomspace(
            reporting["min_sep"], reporting["max_sep"], reporting["nbins"] + 1
        )
        integration = self._integration_binning(
            min_sep_int, max_sep_int, nbins_int, default=(0.08, 300, 1000)
        )
        if self.integration is not None:
            validate_nested_grids(reporting, integration)
        ggs_int = self.calculate_2pcf_version(
            version,
            npatch=npatch,
            compute_tomography=compute_tomography,
            **integration,
        )

        results = {}
        for bin_key, gg_int in ggs_int.items():
            if cov_path_int is not None:
                cov_xi, cov_npatch = np.loadtxt(cov_path_int), None
            else:
                cov_xi = gg_int.estimate_cov("jackknife", cross_patch_weight="match")
                cov_npatch = gg_int.npatch1
            results[bin_key] = calculate_pure_eb_correlation(
                gg_int.meanr,
                gg_int.xip,
                gg_int.xim,
                gg_int.weight,
                np.append(gg_int.left_edges, gg_int.right_edges[-1]),
                cov_xi,
                reporting_edges,
                npatch=cov_npatch,
            )

        return results

    def plot_pure_eb(
        self,
        versions=None,
        output_dir=None,
        fiducial_xip_scale_cut=None,
        fiducial_xim_scale_cut=None,
        min_sep=None,
        max_sep=None,
        nbins=None,
        min_sep_int=None,
        max_sep_int=None,
        nbins_int=None,
        npatch=None,
        compute_tomography=False,
        cov_path_int=None,
        results=None,
    ):
        """
        Generate comprehensive pure E/B mode analysis plots.

        Creates four types of plots for each version and bin pair:
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
            (the configured grid, otherwise 0.08–300 arcmin, 1000 bins)
        npatch : int, optional
            Number of patches for jackknife covariance. Uses self.npatch if None.
        compute_tomography : bool, optional
            Whether to compute and plot every tomographic bin pair instead of the
            non-tomographic pair. Defaults to False.
        cov_path_int : str, optional
            Analytic ξ± covariance on the integration grid; the jackknife is
            used without it.
        results : dict or list, optional
            Precalculated :meth:`calculate_pure_eb` results (``{bin_key: result}``)
            to avoid recomputation: a single such dict for one version, or a list
            of them for multiple versions. If None (default), results will be
            calculated using calculate_pure_eb.

        Notes
        -----
        This function orchestrates the full E/B mode analysis workflow:

        - Uses instance configuration as defaults for unspecified parameters
        - Uses the analytic covariance when cov_path_int is given, the
          jackknife otherwise
        - Generates standardized output file naming based on all analysis
          parameters
        - Delegates individual plot generation to specialized functions in
          b_modes module
        """
        # Use instance defaults for unspecified parameters
        versions = versions or self.versions
        output_dir = output_dir or self.cc["paths"]["output"]
        npatch = npatch or self.npatch

        integration = self._integration_binning(
            min_sep_int, max_sep_int, nbins_int, default=(0.08, 300, 1000)
        )
        min_sep_int, max_sep_int, nbins_int = (
            integration[k] for k in ("min_sep", "max_sep", "nbins")
        )
        var_method = "jackknife" if cov_path_int is None else "analytic"

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
            # Get or calculate results for this version
            bin_results = results_list[idx] or self.calculate_pure_eb(
                version,
                min_sep=min_sep,
                max_sep=max_sep,
                nbins=nbins,
                min_sep_int=min_sep_int,
                max_sep_int=max_sep_int,
                nbins_int=nbins_int,
                npatch=npatch,
                compute_tomography=compute_tomography,
                cov_path_int=cov_path_int,
            )
            self._pure_eb_results[version] = {}

            for bin_key, version_results in bin_results.items():
                # Generate standardized output filename stub
                out_stub = (
                    f"{output_dir}/{version}_{bin_key}_eb_minsep={min_sep}_"
                    f"maxsep={max_sep}_nbins={nbins}_minsepint={min_sep_int}_"
                    f"maxsepint={max_sep_int}_nbinsint={nbins_int}_npatch={npatch}_"
                    f"varmethod={var_method}"
                )
                label = (
                    version
                    if bin_key == "tomo_bin_all_tomo_bin_all"
                    else f"{version} {bin_key}"
                )

                # Calculate E/B statistics for all bin combinations
                version_results = calculate_eb_statistics(version_results)

                # Integration vs Reporting comparison plot
                plot_integration_vs_reporting(
                    version_results,
                    out_stub + "_integration_vs_reporting.png",
                    label,
                )

                # E/B/Ambiguous correlation functions plot
                plot_pure_eb_correlations(
                    version_results,
                    out_stub + "_xis.png",
                    label,
                    fiducial_xip_scale_cut=fiducial_xip_scale_cut,
                    fiducial_xim_scale_cut=fiducial_xim_scale_cut,
                )

                # 2D PTE heatmaps plot
                plot_pte_2d_heatmaps(
                    version_results,
                    label,
                    out_stub + "_ptes.png",
                    fiducial_xip_scale_cut=fiducial_xip_scale_cut,
                    fiducial_xim_scale_cut=fiducial_xim_scale_cut,
                )

                # Covariance matrix plot
                plot_eb_covariance_matrix(
                    version_results["cov"],
                    covariance_label(version_results["npatch"]),
                    out_stub + "_covariance.png",
                    label,
                )

                # Save data products and store on instance
                save_pure_eb_results(version_results, out_stub + "_data.npz")
                self._pure_eb_results[version][bin_key] = version_results
