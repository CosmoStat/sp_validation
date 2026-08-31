# %%
"""COSEBIs diagnostic: complete orthogonal E/B integrals.

Mixin providing COSEBIs calculation and plotting for the
:class:`~sp_validation.cosmo_val.CosmologyValidation` class.
"""

import numpy as np

from .. import sacc_io
from ..b_modes import (
    calculate_cosebis,
    find_conservative_scale_cut_key,
    plot_cosebis_covariance_matrix,
    plot_cosebis_modes,
    plot_cosebis_scale_cut_heatmap,
    save_cosebis_results,
)
from .sacc_writers import cosebis_to_sacc


class CosebisMixin:
    def calculate_cosebis(
        self,
        version,
        min_sep_int=0.5,
        max_sep_int=500,
        nbins_int=1000,
        npatch=None,
        nmodes=10,
        cov_path=None,
        scale_cuts=None,
        evaluate_all_scale_cuts=False,
        min_sep=None,
        max_sep=None,
        nbins=None,
    ):
        """
        Calculate COSEBIs from a finely-binned correlation function.

        COSEBIs fundamentally require fine binning for accurate transformations.
        This function computes a single, finely-binned correlation function using
        integration binning parameters and can evaluate either a single scale cut
        (full range) or multiple scale cuts systematically.

        Parameters
        ----------
        version : str
            The catalog version to compute the COSEBIs for.
        min_sep_int : float, optional
            Minimum separation for integration binning (fine binning for COSEBIs).
            Defaults to 0.5 arcmin.
        max_sep_int : float, optional
            Maximum separation for integration binning (fine binning for COSEBIs).
            Defaults to 500 arcmin.
        nbins_int : int, optional
            Number of bins for integration binning (fine binning for COSEBIs).
            Defaults to 1000.
        npatch : int, optional
            Number of patches for the jackknife resampling. Defaults to self.npatch.
        nmodes : int, optional
            Number of COSEBIs modes to compute. Defaults to 10.
        cov_path : str, optional
            Path to theoretical covariance matrix. When provided, enables analytic
            covariance calculation.
        scale_cuts : list of tuples, optional
            Explicit list of (min_theta, max_theta) scale cuts to evaluate.
            Overrides evaluate_all_scale_cuts when provided.
        evaluate_all_scale_cuts : bool, optional
            If True, evaluates COSEBIs for all possible scale cut combinations
            using the reporting binning parameters. Ignored when scale_cuts is
            provided. Defaults to False.
        min_sep : float, optional
            Minimum separation for reporting binning (only used when
            evaluate_all_scale_cuts=True). Defaults to self.treecorr_config["min_sep"].
        max_sep : float, optional
            Maximum separation for reporting binning (only used when
            evaluate_all_scale_cuts=True). Defaults to self.treecorr_config["max_sep"].
        nbins : int, optional
            Number of bins for reporting binning (only used when
            evaluate_all_scale_cuts=True). Defaults to self.treecorr_config["nbins"].

        Returns
        -------
        dict
            When a single scale cut: Dictionary containing COSEBIs results
            with E/B modes, covariances, and statistics.
            When multiple scale cuts: Dictionary with scale cut tuples as
            keys and results dictionaries as values.
        """
        self.print_start(f"Computing {version} COSEBIs")

        # Set up parameters with defaults
        npatch = npatch or self.npatch

        # Always use integration binning for COSEBIs calculation (fine binning)
        treecorr_config = self._binning(min_sep_int, max_sep_int, nbins_int)

        # Calculate single fine-binned correlation function for COSEBIs
        print(
            f"Computing fine-binned 2PCF with {nbins_int} bins from {min_sep_int} to "
            f"{max_sep_int} arcmin"
        )
        gg = self.calculate_2pcf(version, npatch=npatch, **treecorr_config)

        if scale_cuts is not None:
            # Explicit scale cuts provided
            print(f"Evaluating {len(scale_cuts)} explicit scale cuts")
            results = calculate_cosebis(
                gg=gg, nmodes=nmodes, scale_cuts=scale_cuts, cov_path=cov_path
            )
        elif evaluate_all_scale_cuts:
            # Use reporting binning parameters or inherit from class config
            binning = self._binning(min_sep, max_sep, nbins)
            min_sep, max_sep, nbins = (
                binning["min_sep"],
                binning["max_sep"],
                binning["nbins"],
            )

            # Generate scale cuts using np.geomspace (no TreeCorr needed)
            bin_edges = np.geomspace(min_sep, max_sep, nbins + 1)
            generated_cuts = [
                (bin_edges[start], bin_edges[stop])
                for start in range(nbins)
                for stop in range(start + 1, nbins + 1)
            ]

            print(f"Evaluating {len(generated_cuts)} scale cut combinations")

            # Call b_modes function with scale cuts list
            results = calculate_cosebis(
                gg=gg, nmodes=nmodes, scale_cuts=generated_cuts, cov_path=cov_path
            )
        else:
            # Single scale cut behavior: use full range
            results = calculate_cosebis(
                gg=gg, nmodes=nmodes, scale_cuts=None, cov_path=cov_path
            )
            # Extract single results dict from scale_cuts dictionary
            results = next(iter(results.values()))

        return results

    @staticmethod
    def _fiducial_cosebis_result(results, fiducial_scale_cut):
        """Select the fiducial scale cut's result dict + its ``(min, max)`` cut.

        ``results`` is a single result dict (full range) or a multi-cut mapping
        keyed by ``(theta_min, theta_max)`` tuples; picks via
        ``find_conservative_scale_cut_key`` when ``fiducial_scale_cut`` is given,
        else the widest cut.
        """
        multi_cut = isinstance(results, dict) and all(
            isinstance(k, tuple) for k in results
        )
        if not multi_cut:
            return results, tuple(results["scale_cut"])
        key = (
            find_conservative_scale_cut_key(results, fiducial_scale_cut)
            if fiducial_scale_cut is not None
            else max(results, key=lambda x: x[1] - x[0])
        )
        return results[key], tuple(key)

    def cosebis_to_sacc_part(
        self,
        version,
        out_path,
        results,
        fiducial_scale_cut=None,
        en_override=None,
        bn_override=None,
    ):
        """Write the COSEBIs SACC part at the fiducial scale cut.

        ``results`` is what ``calculate_cosebis`` returned (single dict or
        multi-cut mapping); only the fiducial cut's ``{En, Bn, cov}`` becomes
        the part.

        ``en_override``/``bn_override`` replace ``result["En"]``/``result["Bn"]``;
        the covariance stays from ``result`` (patches exist only in the raw
        patched measurement). The part is stamped under the version's blind.
        """
        result, scale_cut = self._fiducial_cosebis_result(results, fiducial_scale_cut)
        overrides = {
            key: np.asarray(value)
            for key, value in (("En", en_override), ("Bn", bn_override))
            if value is not None
        }
        if overrides:
            result = {**result, **overrides}
        s = cosebis_to_sacc(
            self.sacc_nz(version),
            self.sacc_metadata(version),
            result,
            scale_cut,
        )
        sacc_io.save(
            s, out_path, type=self.run_type, commitment=self.commitment_path(version)
        )

    def plot_cosebis(
        self,
        version=None,
        output_dir=None,
        min_sep_int=0.5,
        max_sep_int=500,
        nbins_int=1000,  # Integration binning
        npatch=None,
        nmodes=10,
        cov_path=None,
        scale_cuts=None,  # Explicit scale cuts
        evaluate_all_scale_cuts=False,  # Grid-based scale cuts
        min_sep=None,
        max_sep=None,
        nbins=None,  # Reporting binning
        fiducial_scale_cut=None,  # For plotting reference
        results=None,
    ):
        """
        Generate comprehensive COSEBIs analysis plots for a single version.

        Creates two types of plots:
        1. COSEBIs E/B mode correlation functions
        2. COSEBIs covariance matrix

        Parameters
        ----------
        version : str, optional
            Version string to process. Defaults to first version in self.versions.
        output_dir : str, optional
            Output directory for plots. Defaults to self.cc['paths']['output'].
        min_sep_int, max_sep_int, nbins_int : float, float, int
            Integration binning parameters for correlation function
            (default: 0.5, 500, 1000)
        npatch : int, optional
            Number of patches for jackknife covariance. Defaults to instance value.
        nmodes : int
            Number of COSEBIs modes to compute (default: 10)
        cov_path : str, optional
            Path to theoretical covariance matrix. When provided, analytic
            covariance is used.
        scale_cuts : list of tuples, optional
            Explicit list of (min_theta, max_theta) scale cuts to evaluate.
            Overrides evaluate_all_scale_cuts when provided.
        evaluate_all_scale_cuts : bool
            Whether to evaluate all scale cuts from reporting binning grid
            (default: False). Ignored when scale_cuts is provided.
        min_sep, max_sep, nbins : float, float, int, optional
            Reporting binning parameters. Only used when evaluate_all_scale_cuts=True.
        fiducial_scale_cut : tuple, optional
            (min_scale, max_scale) reference scale cut for plotting
        results : dict, optional
            Precalculated results to avoid recomputation. If None (default),
            results will be calculated using calculate_cosebis.
        """

        # Use instance defaults if not specified
        version = version or self.versions[0]
        output_dir = output_dir or self.cc["paths"]["output"]
        npatch = npatch or self.treecorr_config.get("npatch", 256)

        # Determine variance method based on whether theoretical covariance is used
        var_method = "analytic" if cov_path is not None else "jackknife"

        # Create output filename with integration parameters to match Snakemake
        out_stub = (
            f"{output_dir}/{version}_cosebis_minsep={min_sep_int}_"
            f"maxsep={max_sep_int}_nbins={nbins_int}_npatch={npatch}_"
            f"varmethod={var_method}_nmodes={nmodes}"
        )

        # Add scale cut info if provided
        if fiducial_scale_cut is not None:
            out_stub += f"_scalecut={fiducial_scale_cut[0]}-{fiducial_scale_cut[1]}"

        # Get or calculate results for this version
        if results is None:
            # Calculate COSEBIs using instance method
            results = self.calculate_cosebis(
                version,
                min_sep_int=min_sep_int,
                max_sep_int=max_sep_int,
                nbins_int=nbins_int,
                npatch=npatch,
                nmodes=nmodes,
                cov_path=cov_path,
                scale_cuts=scale_cuts,
                evaluate_all_scale_cuts=evaluate_all_scale_cuts,
                min_sep=min_sep,
                max_sep=max_sep,
                nbins=nbins,
            )

        # Generate plots using specialized plotting functions
        # Extract single result for plotting if multiple scale cuts were evaluated
        multiple_scale_cuts = isinstance(results, dict) and all(
            isinstance(k, tuple) for k in results
        )
        if multiple_scale_cuts:
            # Multiple scale cuts: use fiducial_scale_cut if provided, otherwise use
            # full range (largest scale cut)
            plot_results = results[
                find_conservative_scale_cut_key(results, fiducial_scale_cut)
                if fiducial_scale_cut is not None
                else max(results, key=lambda x: x[1] - x[0])
            ]
        else:
            # Single result
            plot_results = results

        plot_cosebis_modes(
            plot_results,
            version,
            out_stub + "_cosebis.png",
            fiducial_scale_cut=fiducial_scale_cut,
        )

        plot_cosebis_covariance_matrix(
            plot_results, version, var_method, out_stub + "_covariance.png"
        )

        # Generate scale cut heatmap if we have multiple scale cuts
        if multiple_scale_cuts and len(results) > 1:
            # Create temporary gg object with correct binning for mapping
            treecorr_config_temp = self._binning(min_sep, max_sep, nbins)
            gg_temp = self.calculate_2pcf(
                version, npatch=npatch, **treecorr_config_temp
            )

            plot_cosebis_scale_cut_heatmap(
                results,
                gg_temp,
                version,
                out_stub + "_scalecut_ptes.png",
                fiducial_scale_cut=fiducial_scale_cut,
            )

        # Save data products and store on instance
        save_cosebis_results(results, out_stub + "_data.npz", fiducial_scale_cut)
        self._cosebis_results[version] = results
