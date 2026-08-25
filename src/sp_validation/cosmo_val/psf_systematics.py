# %%
"""PSF systematics diagnostics.

Rho/tau statistics, their fits, and PSF leakage tests (scale-dependent and
object-wise) for catalogue validation.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import treecorr
from astropy.io import fits
from cs_util import plots as cs_plots
from shear_psf_leakage import leakage
from shear_psf_leakage import plots as psfleak_plots
from shear_psf_leakage.rho_tau_stat import PSFErrorFit
from uncertainties import ufloat

from ..rho_tau import (
    get_rho_tau_w_cov,
    get_samples,
)


# TODO: Reorganise the order of functions so it is more readable
class PSFSystematicsMixin:
    # --- property definitions ---
    @property
    def rho_stat_handler(self):
        if not hasattr(self, "_rho_stat_handler"):
            self.calculate_rho_tau_stats(tomography=False)
            if self.compute_tomography:
                self.calculate_rho_tau_stats(tomography=True)
        return self._rho_stat_handler

    @property
    def tau_stat_handler(self):
        if not hasattr(self, "_tau_stat_handler"):
            self.calculate_rho_tau_stats(tomography=False)
            if self.compute_tomography:
                self.calculate_rho_tau_stats(tomography=True)
        return self._tau_stat_handler

    @property
    def psf_fitter(self):
        if not hasattr(self, "_psf_fitter"):
            self._psf_fitter = PSFErrorFit(
                self.rho_stat_handler,
                self.tau_stat_handler,
                self.rho_stat_handler.catalogs._output,
            )
        return self._psf_fitter

    @property
    def rho_tau_fits(self):
        if not hasattr(self, "_rho_tau_fits"):
            self.calculate_rho_tau_fits(tomography=False)
            if self.compute_tomography:
                self.calculate_rho_tau_fits(tomography=True)
        return self._rho_tau_fits

    @property
    def xi_psf_sys(self):
        if not hasattr(self, "_xi_psf_sys"):
            self.calculate_rho_tau_fits()
        return self._xi_psf_sys

    # --- calculate functions ---
    def calculate_rho_tau_stats(self, tomography=True):
        out_dir = f"{self.cc['paths']['output']}/rho_tau_stats"
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        self.print_start("Rho stats")
        for ver in self.versions:
            # Get the tomographic bins
            if tomography:
                tomo_bin_ids, tomo_bin_pairs = self._get_tomo_bins(ver)

                if tomo_bin_ids is None or tomo_bin_pairs is None:
                    raise ValueError(
                        f"Version {ver} does not have tomography information."
                    )

            else:
                tomo_bin_ids, tomo_bin_pairs = ["all"], [("all", "all")]

            # Get the selection for stars
            mask_star = self._get_star_mask(ver)

            for tomo_bin_id in tomo_bin_ids:
                self.print_cyan(f"Computing for the tomographic bin: {tomo_bin_id}")
                base_rho = self.basename(ver)
                base_tau = self.basename(ver, tomo_bin_a=tomo_bin_id)

                # Get the selection for galaxies
                mask_gal = self._get_galaxy_mask(ver, tomo_bin_id)

                # Compute rho and tau statistics
                rho_stat_handler, tau_stat_handler = get_rho_tau_w_cov(
                    self.cc,
                    ver,
                    self.treecorr_config,
                    out_dir,
                    base_rho,
                    base_tau,
                    mask_star=mask_star,
                    mask_gal=mask_gal,
                    method=self.cov_estimate_method,
                    cov_rho=self.compute_cov_rho,
                    npatch=self.npatch,
                )
        self.print_done("Rho stats finished")

        self._rho_stat_handler = rho_stat_handler
        self._tau_stat_handler = tau_stat_handler

    def calculate_rho_tau_fits(self, tomography=True, track_result=True):
        assert self.rho_tau_method != "none"

        # this initializes the rho_tau_fits attribute
        if not hasattr(self, "_rho_tau_fits"):
            self._rho_tau_fits = {}
        quantiles = [1 - self.quantile, self.quantile]

        if not hasattr(self, "_xi_psf_sys"):
            self._xi_psf_sys = {}
        for ver in self.versions:
            params = self.set_params_rho_tau(
                ver, self.results[ver]._params, self.cc[ver]["psf"]
            )

            # Get the tomographic bins
            if tomography:
                tomo_bin_ids, tomo_bin_pairs = self._get_tomo_bins(ver)

                if tomo_bin_ids is None or tomo_bin_pairs is None:
                    raise ValueError(
                        f"Version {ver} does not have tomography information."
                    )

            else:
                tomo_bin_ids, tomo_bin_pairs = ["all"], [("all", "all")]

            # Get the samples for each tomographic bin
            for tomo_bin_id in tomo_bin_ids:
                self.print_cyan(
                    f"Sample PSF error parameters for tomographic bin {tomo_bin_id}"
                )
                _ = self.get_samples(
                    ver, params, tomo_bin_id, track_result=track_result
                )

            # Get the xi_psf_sys for each tomographic bin pairs
            for tomo_bin_a, tomo_bin_b in tomo_bin_pairs:
                xi_psf_sys_samples_plus, xi_psf_sys_samples_minus = (
                    self.get_xi_psf_sys_samples(ver, params, tomo_bin_a, tomo_bin_b)
                )

                if ver not in self._xi_psf_sys.keys():
                    self._xi_psf_sys[ver] = {}

                self._xi_psf_sys[ver][
                    f"tomo_bin_{tomo_bin_a}_tomo_bin_{tomo_bin_b}"
                ] = {
                    "mean_plus": np.mean(xi_psf_sys_samples_plus, axis=0),
                    "var_plus": np.var(xi_psf_sys_samples_plus, axis=0),
                    "quantiles_plus": np.quantile(
                        xi_psf_sys_samples_plus, quantiles, axis=0
                    ),
                    "mean_minus": np.mean(xi_psf_sys_samples_minus, axis=0),
                    "var_minus": np.var(xi_psf_sys_samples_minus, axis=0),
                    "quantiles_minus": np.quantile(
                        xi_psf_sys_samples_minus, quantiles, axis=0
                    ),
                }

    def calculate_scale_dependent_leakage(self):
        # TODO: Upgrade for tomography
        self.print_start("Calculating scale-dependent leakage:")
        for ver in self.versions:
            self.print_magenta(ver)
            results = self.results[ver]

            output_base_path = self._output_path(f"leakage_{ver}/xi_for_leak_scale")
            output_path_ab = f"{output_base_path}_a_b.txt"
            output_path_aa = f"{output_base_path}_a_a.txt"
            with self.results[ver].temporarily_read_data():
                if os.path.exists(output_path_ab) and os.path.exists(output_path_aa):
                    self.print_green(
                        f"Skipping computation, reading {output_path_ab} and "
                        f"{output_path_aa} instead"
                    )

                    results.r_corr_gp = treecorr.GGCorrelation(self.treecorr_config)
                    results.r_corr_gp.read(output_path_ab)

                    results.r_corr_pp = treecorr.GGCorrelation(self.treecorr_config)
                    results.r_corr_pp.read(output_path_aa)

                else:
                    results.compute_corr_gp_pp_alpha(output_base_path=output_base_path)

                results.do_alpha(fast=True)
                results.do_xi_sys()

        self.print_done("Finished scale-dependent leakage calculation.")

    def calculate_objectwise_leakage(self, tomography=False):
        # TODO: Upgrade for tomography

        # TODO: remove and save the results of the scale-dependent leakage independently
        if not hasattr(self.results[self.versions[0]], "alpha_leak_mean"):
            self.calculate_scale_dependent_leakage()

        self.print_start("Object-wise leakage:")
        mix = True
        order = "lin"
        for ver in self.versions:
            self.print_magenta(ver)

            results_obj = self.results_objectwise[ver]
            results_obj.check_params()
            results_obj.update_params()
            results_obj.prepare_output()

            # Skip read_data() and copy catalogue from scale leakage instance instead
            # results_obj._dat = self.results[ver].dat_shear

            out_base = results_obj.get_out_base(mix, order)
            out_path = f"{out_base}.pkl"
            if os.path.exists(out_path):
                self.print_green(
                    f"Skipping object-wise leakage, file {out_path} exists"
                )
                results_obj.par_best_fit = leakage.read_from_file(out_path)
            else:
                self.print_cyan("Computing object-wise leakage regression")

            # Run
            with results_obj.temporarily_read_data():
                try:
                    results_obj.PSF_leakage()
                except KeyError as e:
                    print(f"{e}\nExpected key is missing from catalog.")
                    # remove the results object for this version
                    self.results_objectwise.pop(ver)

        # Gather coefficients
        leakage_coeff = {}
        for ver in self.results_objectwise:
            results = self.results[ver]
            par_best_fit = self.results_objectwise[ver].par_best_fit

            # Object-wise leakage
            a11 = ufloat(par_best_fit["a11"].value, par_best_fit["a11"].stderr)
            a22 = ufloat(par_best_fit["a22"].value, par_best_fit["a22"].stderr)
            leakage_coeff[ver] = {
                "a11": a11,
                "a22": a22,
                "aii_mean": 0.5 * (a11 + a22),
                # Scale-dependent leakage: mean
                "alpha_mean": ufloat(results.alpha_leak_mean, results.alpha_leak_std),
                # Scale-dependent leakage: value at smallest scale
                "alpha_1": ufloat(results.alpha_leak[0], results.sig_alpha_leak[0]),
                # Scale-dependent leakage: value extrapolated to 0 using affine model
                "alpha_0": ufloat(
                    results.alpha_affine_best_fit["c"].value,
                    results.alpha_affine_best_fit["c"].stderr,
                ),
            }

        self.leakage_coeff = leakage_coeff

    # --- utility functions ---
    def _get_galaxy_mask(self, ver, tomo_bin_id):
        cat_gal = fits.getdata(self.cc[ver]["shear"]["path"])
        if tomo_bin_id != "all":
            gal_mask = cat_gal[self.cc[ver]["shear"]["tomo_bin_col"]] == tomo_bin_id
        else:
            gal_mask = np.ones(len(cat_gal), dtype=bool)
        return gal_mask

    def _get_star_mask(self, ver):
        cat_star = fits.getdata(
            self.cc[ver]["psf"]["path"], hdu=self.cc[ver]["psf"]["hdu"]
        )
        PSF_flag = self.cc[ver]["psf"].get("PSF_flag")
        star_flag = self.cc[ver]["psf"].get("star_flag")
        if PSF_flag is not None:
            if star_flag is not None:
                star_mask = (cat_star[PSF_flag] == 0) & (cat_star[star_flag] == 0)
            else:
                star_mask = cat_star[PSF_flag] == 0
        else:
            star_mask = np.ones(len(cat_star), dtype=bool)
        return star_mask

    def set_params_rho_tau(self, ver, params, params_psf):
        params = {**params}

        params["ra_PSF_col"] = params_psf["ra_col"]
        params["dec_PSF_col"] = params_psf["dec_col"]
        params["e1_PSF_col"] = params_psf["e1_PSF_col"]
        params["e2_PSF_col"] = params_psf["e2_PSF_col"]
        params["e1_star_col"] = params_psf["e1_star_col"]
        params["e2_star_col"] = params_psf["e2_star_col"]
        params["PSF_size"] = params_psf["PSF_size"]
        params["star_size"] = params_psf["star_size"]
        params["PSF_flag"] = params_psf.get("PSF_flag")
        params["star_flag"] = params_psf.get("star_flag")
        params["ra_units"] = "deg"
        params["dec_units"] = "deg"

        params["w_col"] = self.cc[ver]["shear"]["w_col"]
        params["patch_number"] = self.cc[ver].get("patch_number", 100)

        return params

    def set_params_leakage_scale(self, ver):
        params_in = {}

        # Set parameters
        params_in["input_path_shear"] = self.cc[ver]["shear"]["path"]
        params_in["input_path_PSF"] = self.cc[ver]["star"]["path"]
        params_in["dndz_path"] = (
            f"{self.cc['nz']['dndz']['path']}_{self.cc[ver]['pipeline']}_{self.cc['nz']['dndz']['blind']}.txt"
        )
        params_in["output_dir"] = f"{self.cc['paths']['output']}/leakage_{ver}"

        # Note: for SP these are calibrated shear estimates
        params_in["e1_col"] = self.cc[ver]["shear"]["e1_col"]
        params_in["e2_col"] = self.cc[ver]["shear"]["e2_col"]
        params_in["w_col"] = self.cc[ver]["shear"]["w_col"]
        params_in["R11"] = None if ver != "DES" else self.cc[ver]["shear"]["R11"]
        params_in["R22"] = None if ver != "DES" else self.cc[ver]["shear"]["R22"]

        params_in["ra_star_col"] = self.cc[ver]["star"]["ra_col"]
        params_in["dec_star_col"] = self.cc[ver]["star"]["dec_col"]
        params_in["e1_PSF_star_col"] = self.cc[ver]["star"]["e1_col"]
        params_in["e2_PSF_star_col"] = self.cc[ver]["star"]["e2_col"]

        params_in["theta_min_amin"] = self.theta_min
        params_in["theta_max_amin"] = self.theta_max
        params_in["n_theta"] = self.nbins

        params_in["verbose"] = False

        return params_in

    def set_params_leakage_object(self, ver):
        params_in = {}

        # Set parameters
        params_in["input_path_shear"] = self.cc[ver]["shear"]["path"]
        params_in["output_dir"] = f"{self.cc['paths']['output']}/leakage_{ver}"

        # Note: for SP these are calibrated shear estimates
        params_in["e1_col"] = self.cc[ver]["shear"]["e1_col"]
        params_in["e2_col"] = self.cc[ver]["shear"]["e2_col"]
        params_in["w_col"] = self.cc[ver]["shear"]["w_col"]

        if (
            "e1_PSF_col" in self.cc[ver]["shear"]
            and "e2_PSF_col" in self.cc[ver]["shear"]
        ):
            params_in["e1_PSF_col"] = self.cc[ver]["shear"]["e1_PSF_col"]
            params_in["e2_PSF_col"] = self.cc[ver]["shear"]["e2_PSF_col"]
        else:
            raise KeyError(
                "Keys 'e1_PSF_col' and 'e2_PSF_col' not found in"
                + f" shear yaml entry for version {ver}"
            )

        params_in["verbose"] = False

        return params_in

    def set_psf_parameter_sampling_method(self, rho_tau_method):
        if rho_tau_method not in ["emcee", "lsq"]:
            raise ValueError("Invalid PSF parameter sampling method.")
        self.rho_tau_method = rho_tau_method

    def set_psf_parameter_nsamples(self, nsamples):
        self.psf_error_nsamples = nsamples

    def set_psf_parameter_nwalkers(self, nwalkers):
        self.psf_error_nwalkers = nwalkers

    def get_samples(self, version, params, tomo_bin_id, track_result=False):
        npatch = params["patch_number"] if self.cov_estimate_method == "jk" else None

        base_rho = self.basename(version)
        base_tau = self.basename(version, tomo_bin_a=tomo_bin_id)

        # Set the number of samples and walkers and fallback to defaults if not attributed.
        n_samples = (
            self.psf_error_nsamples if hasattr(self, "psf_error_nsamples") else 10_000
        )
        n_walkers = (
            self.psf_error_nwalkers if hasattr(self, "psf_error_nwalkers") else 124
        )

        flat_samples, result, q = get_samples(
            self.psf_fitter,
            base_rho,
            base_tau,
            cov_type=self.cov_estimate_method,
            apply_debias=npatch,
            sampler=self.rho_tau_method,
            nsamples=n_samples,
            nwalkers=n_walkers,
        )

        if track_result:
            if version not in self.rho_tau_fits.keys():
                self.rho_tau_fits[version] = {
                    "flat_samples": {},
                    "result": {},
                    "quantile": {},
                }
            self.rho_tau_fits[version]["flat_samples"][f"tomo_bin_{tomo_bin_id}"] = (
                flat_samples
            )
            self.rho_tau_fits[version]["result"][f"tomo_bin_{tomo_bin_id}"] = result
            self.rho_tau_fits[version]["quantile"][f"tomo_bin_{tomo_bin_id}"] = q

        return flat_samples

    def get_xi_psf_sys_samples(self, ver, params, tomo_bin_a, tomo_bin_b):
        base_rho = self.basename(ver)
        self.psf_fitter.load_rho_stat(f"rho_stats_{base_rho}.fits")
        nbins = self.psf_fitter.rho_stat_handler._treecorr_config["nbins"]

        # Get the samples for the given tomographic bins
        flat_samples_a = self.get_samples(ver, params, tomo_bin_a, track_result=False)
        flat_samples_b = self.get_samples(ver, params, tomo_bin_b, track_result=False)

        xi_psf_sys_samples_plus = np.array(
            [
                self.psf_fitter.compute_xi_psf_sys(sample_a, sample_b, p_or_m="p")
                for (sample_a, sample_b) in zip(flat_samples_a, flat_samples_b)
            ]
        ).reshape(-1, nbins)

        xi_psf_sys_samples_minus = np.array(
            [
                self.psf_fitter.compute_xi_psf_sys(sample_a, sample_b, p_or_m="m")
                for (sample_a, sample_b) in zip(flat_samples_a, flat_samples_b)
            ]
        ).reshape(-1, nbins)

        return xi_psf_sys_samples_plus, xi_psf_sys_samples_minus

    def _get_alpha_leakage(
        self,
        rho_stat_handler,
        tau_stat_handler,
        cov_rho=None,
        cov_tau=None,
        n_samples=10_000,
    ):
        """
        Compute the alpha leakage parameter from the rho and tau statistics.

        Parameters
        ----------
        rho_stat_handler : RhoStatHandler
            The handler for the rho statistics.
        tau_stat_handler : TauStatHandler
            The handler for the tau statistics.
        cov_rho : np.ndarray, optional
            The covariance matrix for the rho statistics. If None, it will be computed.
        cov_tau : np.ndarray, optional
            The covariance matrix for the tau statistics. If None, it will be computed.

        Returns
        -------
        alpha_leak : float
            The estimated alpha leakage parameter.
        """
        if cov_rho is None:
            cov_rho = np.diag(rho_stat_handler.rho_stats["varrho_0_p"])
        if cov_tau is None:
            cov_tau = np.diag(tau_stat_handler.tau_stats["vartau_0_p"])

        theta = rho_stat_handler.rho_stats["theta"]
        n_bins = len(theta)
        alpha = (
            tau_stat_handler.tau_stats["tau_0_p"]
            / rho_stat_handler.rho_stats["rho_0_p"]
        )

        # Derive alpha_err by sampling from the covariance matrices of rho and tau statistics
        rho_samples = np.random.multivariate_normal(
            mean=rho_stat_handler.rho_stats["rho_0_p"],
            cov=cov_rho[:n_bins, :n_bins],
            size=n_samples,
        )
        tau_samples = np.random.multivariate_normal(
            mean=tau_stat_handler.tau_stats["tau_0_p"],
            cov=cov_tau[:n_bins, :n_bins],
            size=n_samples,
        )

        alpha_samples = tau_samples / rho_samples
        alpha_err = np.std(alpha_samples, axis=0)

        return theta, alpha, alpha_err

    def _compute_scale_dependent_xi_psf_sys(
        self, rho_0, tau_0_a, tau_0_b, cov_rho, cov_tau_a, cov_tau_b, n_samples=10_000
    ):
        """
        Compute the scale-dependent xi_psf_sys from the rho and tau statistics.

        Parameters
        ----------
        rho_0 : np.ndarray
            The rho_0 statistics.
        tau_0_a : np.ndarray
            The tau_0 statistics for tomographic bin a.
        tau_0_b : np.ndarray
            The tau_0 statistics for tomographic bin b.
        cov_rho : np.ndarray
            The covariance matrix for the rho statistics.
        cov_tau_a : np.ndarray
            The covariance matrix for the tau statistics for tomographic bin a.
        cov_tau_b : np.ndarray
            The covariance matrix for the tau statistics for tomographic bin b.
        """
        # Compute xi_psf_sys for each scale using the formula:
        xi_psf_sys = (tau_0_a * tau_0_b) / rho_0

        # Derive the error bars by sampling the statistics
        rho_samples = np.random.multivariate_normal(
            mean=rho_0,
            cov=cov_rho,
            size=n_samples,
        )
        tau_samples_a = np.random.multivariate_normal(
            mean=tau_0_a,
            cov=cov_tau_a,
            size=n_samples,
        )
        tau_samples_b = np.random.multivariate_normal(
            mean=tau_0_b,
            cov=cov_tau_b,
            size=n_samples,
        )
        xi_psf_sys_samples = (tau_samples_a * tau_samples_b) / rho_samples
        xi_psf_sys_err = np.std(xi_psf_sys_samples, axis=0)

        return xi_psf_sys, xi_psf_sys_err

    # --- plotting functions ---
    def plot_rho_stats(
        self,
        versions=None,
        colors=None,
        abs=False,
        offset=0,
        savefig=None,
        show=True,
        close=True,
    ):
        """
        Plot the Rho statistics.

        Parameters
        ----------
        versions : list, optional
            List of versions to plot. If None, all versions are plotted.
        abs : bool, optional
            If True, plot the absolute values of the Rho statistics.
        offset : float, optional
            Offset to apply to the versions for better visualisation.
        savefig : str, optional
            If provided, save the figure to this file.
        show : bool, optional
            If True, show the figure.
        close : bool, optional
            If True, close the figure after saving or showing.
        """
        if versions is None:
            versions = self.versions

        filenames = [f"rho_stats_{self.basename(ver)}.fits" for ver in versions]

        if colors is None:
            colors = [self.cc[ver]["colour"] for ver in versions]

        if len(colors) != len(versions):
            raise ValueError("Colors and versions must have the same length.")

        self.rho_stat_handler.plot_rho_stats(
            filenames,
            colors,
            versions,
            offset=offset,
            savefig=savefig,
            legend="outside",
            abs=abs,
            show=show,
            close=close,
        )

        if savefig is not None:
            self.print_done(
                "Rho stats plot saved to "
                + f"{os.path.abspath(self.rho_stat_handler.catalogs._output)}/{savefig}",
            )

    def plot_tau_stats(
        self,
        tomography=False,
        cov_type=None,
        versions=None,
        colors=None,
        offset=0,
        savefig=None,
        show=True,
        close=True,
        plot_tau_m=False,
        plot_theta_times_tau=False,
        fmt="",
        capsize=2,
    ):
        if versions is None:
            versions = self.versions

        if colors is None:
            colors = [self.cc[ver]["colour"] for ver in versions]

        if len(colors) != len(versions):
            raise ValueError("Colors and versions must have the same length.")

        if cov_type is None:
            self.print_cyan("Using the error bars from the tau-statistics files")
        else:
            self.print_cyan(
                f"Using the error bars from the covariance files of type: {cov_type}"
            )

        out_dir = f"{self.cc['paths']['output']}/rho_tau_stats"

        if tomography:
            # Write the whole script for the tomography. It does not exist in shear_psf_leakage
            e_obs = r"e^\mathrm{obs}"
            e_psf = r"e^\mathrm{PSF}"
            delta_e_psf = r"\delta e^\mathrm{PSF}"
            delta_T_psf = r"\delta T^\mathrm{PSF}"

            factor_theta_label = r"\theta" if plot_theta_times_tau else r""

            titles = [
                rf"$\tau_0 = \langle {e_obs} {e_psf} \rangle$",
                rf"${factor_theta_label} \tau_2 = {factor_theta_label} \langle {e_obs} {delta_e_psf} \rangle$",
                rf"${factor_theta_label} \tau_5 = {factor_theta_label} \langle {e_obs} {delta_T_psf} \rangle$",
            ]

            dict_index_tau = {
                0: "0",
                1: "2",
                2: "5",
            }

            # From all the versions, get the maximum number of tomo_bin_ids
            tomo_bins = self._get_tomo_bins_for_versions(
                versions, tomography=tomography
            )

            n_tomo_bins_plot = max(len(bins["ids"]) for bins in tomo_bins.values())

            n_rows = n_tomo_bins_plot * (1 + plot_tau_m)

            fig = plt.figure(figsize=(20 * (1 + plot_tau_m), 10 * (1 + plot_tau_m)))
            gs = fig.add_gridspec(n_rows, 3, wspace=0.1, hspace=0)
            all_axs = gs.subplots(sharex="col")

            for k in range(n_rows):
                axs = all_axs[k]

                tomo_bin_id = k // 2 + 1 if plot_tau_m else k + 1
                is_m_component = k % 2 if plot_tau_m else 0

                for file_idx, (ver, color) in enumerate(zip(versions, colors)):
                    # Check if the tomo bin is valid for this version
                    if tomo_bin_id not in tomo_bins[ver]["ids"]:
                        continue

                    # Load the tau-stats in the tau_stat_handler for easier read
                    base_tau = self.basename(ver, tomo_bin_a=tomo_bin_id)

                    self.tau_stat_handler.load_tau_stats(f"tau_stats_{base_tau}.fits")

                    if cov_type is not None:
                        cov_tau_path = (
                            Path(out_dir) / f"cov_tau_{base_tau}_{cov_type}.npy"
                        )
                        cov_tau = np.load(cov_tau_path)

                    # Plot the different tau-stats per row
                    for i in range(3):
                        p_or_m = "m" if is_m_component else "p"
                        p_or_m_label = "-" if is_m_component else "+"

                        # Get the jittered angular scale for the x-axis
                        theta = self.tau_stat_handler.tau_stats["theta"]
                        num_theta_bins = theta.shape[0]

                        jittered_theta = self._get_jittered_theta(
                            theta, file_idx, len(versions), offset
                        )

                        factor_theta = (
                            np.ones_like(jittered_theta)
                            if (i == 0) or not plot_theta_times_tau
                            else theta
                        )

                        y_plot = (
                            self.tau_stat_handler.tau_stats[
                                f"tau_{dict_index_tau[i]}_{p_or_m}"
                            ]
                            * factor_theta
                        )

                        if cov_type is None or p_or_m == "m":
                            cov_diag = self.tau_stat_handler.tau_stats[
                                "vartau_" + dict_index_tau[i] + "_" + p_or_m
                            ]
                        else:
                            cov_diag = np.diag(
                                cov_tau[
                                    i * num_theta_bins : (i + 1) * num_theta_bins,
                                    i * num_theta_bins : (i + 1) * num_theta_bins,
                                ]
                            )

                        yerr_plot = np.sqrt(cov_diag) * factor_theta

                        ver_label = (
                            self.cc[ver]["label"] if "label" in self.cc[ver] else ver
                        )
                        axs[i].errorbar(
                            jittered_theta,
                            y_plot,
                            yerr=yerr_plot,
                            fmt=fmt,
                            label=ver_label,
                            capsize=capsize,
                            color=color,
                        )

                # Set the style of the plot
                for i in range(3):
                    axs[i].set_xscale("log")
                    axs[i].set_xlim(theta.min() * 0.9, theta.max() * 1.1)
                    if i == 0:
                        axs[i].set_ylabel(f"Bin {tomo_bin_id}\n `{p_or_m_label}' comp.")
                    if k == n_rows - 1:
                        axs[i].set_xlabel(r"$\theta$ [arcmin]")
                    if k == 0:
                        axs[i].set_title(titles[i])

                    # --- Force scientific notation and scaling ---
                    axs[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                    axs[i].yaxis.offsetText.set_visible(True)

                    # --- Make it look clean ---
                    axs[i].yaxis.set_major_locator(
                        mticker.MaxNLocator(nbins=5)
                    )  # Fewer, rounded ticks
                    # axs[i].yaxis.get_offset_text().set_fontsize(10)        # Smaller ×10⁻⁴ label
                    axs[i].yaxis.get_offset_text().set_position(
                        (0, 1.02)
                    )  # Move scale factor slightly above axis
                    axs[i].yaxis.get_offset_text().set_ha("left")

                if k == n_rows - 1:
                    axs[1].legend(
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.5),  # (x, y) relative to the axes
                        ncol=3,  # number of columns
                        frameon=False,
                    )

            plt.tight_layout()

            if savefig is not None:
                plt.savefig(savefig, dpi=300, bbox_inches="tight")

            if show:
                plt.show()

            if close:
                plt.close()

        else:
            filenames = [f"tau_stats_{self.basename(ver)}.fits" for ver in versions]
            cov_paths = [
                f"cov_tau_{self.basename(ver)}_{cov_type}.npy" for ver in versions
            ]
            self.tau_stat_handler.plot_tau_stats(
                filenames,
                colors,
                versions,
                cov_paths=cov_paths,
                offset=offset,
                savefig=savefig,
                legend="outside",
                plot_tau_m=plot_tau_m,
                plot_theta_times_tau=plot_theta_times_tau,
                show=show,
                close=close,
                fmt=fmt,
                capsize=capsize,
            )

        if savefig is not None:
            self.print_done(
                "Tau stats plot saved to "
                + f"{os.path.abspath(self.tau_stat_handler.catalogs._output)}/{savefig}",
            )

    def plot_rho_tau_fits(
        self,
        tomography=False,
        versions=None,
        colors=None,
        savefig_contours=None,
        savefig_xi_psf_sys=None,
        savefig_format="png",
        tomo_bin_label_position=None,
        nsamples_to_plot=100,
        offset=0,
        alpha=0.3,
        times_theta=False,
        show=True,
        close=True,
    ):
        """
        Plot the Rho/Tau fits and the xi_psf_sys samples.

        Parameters
        ----------
        versions : list, optional
            List of versions to plot. If None, all versions are plotted.
        colors : list, optional
            List of colors for each version. If None, default colors are used.
        savefig_contours : str, optional
            If provided, save the contour plot to this file.
        savefig_xi_psf_sys : str, optional
            If provided, save the xi_psf_sys samples plot to this file.
        nsamples_to_plot : int, optional
            Number of xi_psf_sys samples to plot. Default is 100.
        offset : float, optional
            Offset to apply to the versions for better visualisation.
        alpha : float, optional
            Alpha value for the xi_psf_sys samples plot. Default is 0.3.
        times_theta : bool, optional
            If True, plot the xi_psf_sys multiplied by theta.
        show : bool, optional
            If True, show the figure.
        close : bool, optional
            If True, close the figure after saving or showing.
        """
        if savefig_format not in ["png", "pdf", "jpg", "jpeg", "svg"]:
            raise ValueError(
                "Invalid savefig_format. Must be one of: 'png', 'pdf', 'jpg', 'jpeg', 'svg'."
            )

        out_dir = self.rho_stat_handler.catalogs._output

        versions = versions if versions is not None else self.versions
        colors = (
            colors
            if colors is not None
            else [self.cc[ver]["colour"] for ver in versions]
        )

        # Print the tau-statistics constraints on the PSF error model parameters.
        savefig = (
            f"{out_dir}/{savefig_contours}" if savefig_contours is not None else None
        )
        self.print_cyan(
            "Only plot contours for the non-tomographic case. For tomography, the contours are not plotted."
        )
        sample_list = [
            self.rho_tau_fits[ver]["flat_samples"]["tomo_bin_all"] for ver in versions
        ]
        psfleak_plots.plot_contours(
            sample_list,
            names=["x0", "x1", "x2"],
            labels=[r"\alpha", r"\beta", r"\eta"],
            savefig=savefig_contours,
            legend_labels=versions,
            legend_loc="upper right",
            contour_colors=colors,
            markers={"x0": 0, "x1": 1, "x2": 1},
            show=show,
            close=close,
        )
        if savefig_contours is not None:
            self.print_done(f"Tau contours plot saved to {os.path.abspath(savefig)}")

        # Plot the xi_psf_sys samples and quantiles.
        if tomography:
            self.print_cyan("Plot the xi_psf_sys for the tomographic case.")
        else:
            self.print_cyan("Plot the xi_psf_sys for the non-tomographic case.")

        x_label = r"$\theta$ [arcmin]"
        y_label_plus = (
            r"$\theta$" if times_theta else ""
        ) + r"$\xi^{\rm PSF, sys}_+(\theta)$"
        y_label_minus = (
            r"$\theta$" if times_theta else ""
        ) + r"$\xi^{\rm PSF, sys}_-(\theta)$"

        if tomo_bin_label_position is None:
            tomo_bin_label_position = (0.05, 0.9) if times_theta else (0.8, 0.95)

        y_scale = "linear" if times_theta else "log"

        out_path = (
            f"{out_dir}/{savefig_xi_psf_sys}_tomography_{tomography}_sample.{savefig_format}"
            if savefig_xi_psf_sys is not None
            else None
        )

        kwargs_x_y_plot_function = {
            "nsamples_to_plot": nsamples_to_plot,
            "alpha": alpha,
            "offset": offset,
            "times_theta": times_theta,
        }

        self.plot_2pcf_tomography(
            self._xi_psf_sys_sample_x_y_plot_function,
            x_label,
            y_label_plus,
            y_label_minus,
            tomo_bin_label_position,
            extract_text_offset=times_theta,
            add_index_version_to_kwargs=True,
            x_scale="log",
            y_scale=y_scale,
            tomography=tomography,
            versions=versions,
            colors=colors,
            savefig=out_path,
            show=show,
            close=close,
            **kwargs_x_y_plot_function,
        )

        # Plot the xi_psf_sys mean and std.
        x_label = r"$\theta$ [arcmin]"
        y_label_plus = (
            r"$\theta$" if times_theta else ""
        ) + r"$\xi^{\rm PSF, sys}_+(\theta)$"
        y_label_minus = (
            r"$\theta$" if times_theta else ""
        ) + r"$\xi^{\rm PSF, sys}_-(\theta)$"

        if tomo_bin_label_position is None:
            tomo_bin_label_position = (0.05, 0.9) if times_theta else (0.8, 0.95)

        y_scale = "linear" if times_theta else "log"

        out_path = (
            f"{out_dir}/{savefig_xi_psf_sys}_tomography_{tomography}_mean_and_std.{savefig_format}"
            if savefig_xi_psf_sys is not None
            else None
        )

        kwargs_x_y_plot_function = {
            "offset": offset,
            "times_theta": times_theta,
            "alpha": alpha,
        }

        self.plot_2pcf_tomography(
            self._xi_psf_sys_mean_and_std_x_y_plot_function,
            x_label,
            y_label_plus,
            y_label_minus,
            tomo_bin_label_position,
            extract_text_offset=times_theta,
            add_index_version_to_kwargs=True,
            x_scale="log",
            y_scale=y_scale,
            tomography=tomography,
            versions=versions,
            colors=colors,
            savefig=out_path,
            show=show,
            close=close,
            **kwargs_x_y_plot_function,
        )

    def plot_scale_dependent_leakage(
        self,
        tomography=False,
        cov_type=None,
        versions=None,
        colors=None,
        offset=0,
        savefig=None,
        show=True,
        close=True,
        plot_theta_times_tau=False,
        ylim_alpha=False,
        fmt="",
        capsize=2,
    ):
        # First plot alpha leakage
        self.plot_scale_dependent_alpha(
            tomography=tomography,
            cov_type=cov_type,
            versions=versions,
            colors=colors,
            offset=offset,
            savefig=savefig,
            show=show,
            close=close,
            fmt=fmt,
            capsize=capsize,
            ylim_alpha=ylim_alpha,
        )

        # Second plot xi_sys
        self.plot_2pcf_tomography(
            self._scale_dependent_xi_psf_sys_x_y_plot_function,
            x_label=r"$\theta$ [arcmin]",
            y_label_plus=(r"$\theta$" if plot_theta_times_tau else "")
            + r"$\xi^{\rm PSF, sys}_+(\theta)$",
            y_label_minus=(r"$\theta$" if plot_theta_times_tau else "")
            + r"$\xi^{\rm PSF, sys}_-(\theta)$",
            tomo_bin_label_position=(0.05, 0.9)
            if not plot_theta_times_tau
            else (0.8, 0.95),
            extract_text_offset=plot_theta_times_tau,
            add_index_version_to_kwargs=True,
            x_scale="log",
            y_scale="linear" if plot_theta_times_tau else "log",
            tomography=tomography,
            versions=versions,
            colors=colors,
            savefig=savefig.replace(".png", "_xi_psf_sys.png")
            if savefig is not None
            else None,
            show=show,
            close=close,
            offset=offset,
            cov_type=cov_type,
            times_theta=plot_theta_times_tau,
            fmt=fmt,
            capsize=capsize,
        )

    def plot_scale_dependent_alpha(
        self,
        tomography=False,
        cov_type=None,
        versions=None,
        colors=None,
        offset=0,
        savefig=None,
        show=True,
        close=True,
        fmt="",
        capsize=2,
        ylim_alpha=False,
    ):
        if versions is None:
            versions = self.versions

        if colors is None:
            colors = [self.cc[ver]["colour"] for ver in versions]

        if len(colors) != len(versions):
            raise ValueError("Colors and versions must have the same length.")

        if cov_type is None:
            self.print_cyan("Using the error bars from the tau-statistics files")
        else:
            self.print_cyan(
                f"Using the error bars from the covariance files of type: {cov_type}"
            )

        tomo_bins = self._get_tomo_bins_for_versions(versions, tomography=tomography)

        n_tomo_bins_plot = max(len(bins["ids"]) for bins in tomo_bins.values())

        out_dir = f"{self.cc['paths']['output']}/rho_tau_stats"

        fig, axs = plt.subplots(
            n_tomo_bins_plot, 1, figsize=(8, 3 * n_tomo_bins_plot), sharex=True
        )

        # Iterate upon each version
        for ver, color in zip(versions, colors):
            label = self.cc[ver]["label"] if "label" in self.cc[ver] else ver
            # Iterate upon each tomographic bin
            for tomo_bin_id in tomo_bins[ver]["ids"]:
                base_rho = self.basename(ver)
                base_tau = self.basename(ver, tomo_bin_a=tomo_bin_id)
                self.rho_stat_handler.load_rho_stats(f"rho_stats_{base_rho}.fits")
                self.tau_stat_handler.load_tau_stats(f"tau_stats_{base_tau}.fits")

                if cov_type is not None:
                    cov_tau_path = Path(out_dir) / f"cov_tau_{base_tau}_{cov_type}.npy"
                    cov_tau = np.load(cov_tau_path)
                    cov_rho_path = Path(out_dir) / f"cov_rho_{base_rho}_jk.npy"
                    cov_rho = np.load(cov_rho_path)
                else:
                    cov_tau = None
                    cov_rho = None

                # Get the error bar sampling from the covariance matrices
                theta, alpha, alpha_err = self._get_alpha_leakage(
                    self.rho_stat_handler, self.tau_stat_handler, cov_rho, cov_tau
                )

                jittered_theta = self._get_jittered_theta(
                    theta, versions.index(ver), len(versions), offset
                )

                if tomo_bin_id == "all":
                    ax = axs
                else:
                    ax = axs[tomo_bin_id - 1]

                ax.errorbar(
                    jittered_theta,
                    alpha,
                    yerr=alpha_err,
                    fmt=fmt,
                    label=f"{label}",
                    capsize=capsize,
                    color=color,
                )

        if tomography:
            for i, ax in enumerate(axs):
                ax.set_xscale("log")
                ax.set_xlim(self.theta_min_plot, self.theta_max_plot)
                if ylim_alpha:
                    ax.set_ylim(self.ylim_alpha)
                ax.set_ylabel(rf"$\alpha_{i + 1}(\theta)$")
                if i == len(axs) - 1:
                    ax.set_xlabel(r"$\theta$ [arcmin]")
                ax.text(0.05, 0.9, f"Tomo bin {i + 1}", transform=ax.transAxes)
        else:
            axs.set_xscale("log")
            axs.set_xlim(self.theta_min_plot, self.theta_max_plot)
            if ylim_alpha:
                axs.set_ylim(self.ylim_alpha)
            axs.set_ylabel(r"$\alpha_{\rm all}(\theta)$")
            axs.set_xlabel(r"$\theta$ [arcmin]")
            axs.text(0.05, 0.9, "All tomographic bins", transform=axs.transAxes)

        if tomography:
            handles, labels = axs[0].get_legend_handles_labels()
        else:
            handles, labels = axs.get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=3,
            frameon=False,
        )

        if savefig is not None:
            plt.savefig(savefig, dpi=300, bbox_inches="tight")
            self.print_done(f"Scale-dependent alpha leakage plot saved to {savefig}")

        if show:
            plt.show()

        if close:
            plt.close()

    def plot_objectwise_leakage(self):
        if not hasattr(self, "leakage_coeff"):
            self.calculate_objectwise_leakage()

        self.print_start("Plotting object-wise leakage:")
        cs_plots.figure(figsize=(15, 15))

        linestyles = ["-", "--", ":"]
        fillstyles = ["full", "none", "left", "right", "bottom", "top"]

        for ver in self.results_objectwise:
            label = ver
            for key, ls, fs in zip(
                ["alpha_mean", "alpha_1", "alpha_0"], linestyles, fillstyles
            ):
                x = self.leakage_coeff[ver]["aii_mean"].nominal_value
                dx = self.leakage_coeff[ver]["aii_mean"].std_dev
                y = self.leakage_coeff[ver][key].nominal_value
                dy = self.leakage_coeff[ver][key].std_dev

                eb = plt.errorbar(
                    x,
                    y,
                    xerr=dx,
                    yerr=dy,
                    fmt=self.cc[ver]["marker"],
                    color=self.cc[ver]["colour"],
                    fillstyle=fs,
                    label=label,
                )
                label = None
                eb[-1][0].set_linestyle(ls)

        # y=x line
        xlim = 0.02
        x = [-xlim, xlim]
        y = x
        plt.plot(x, y, "k:", linewidth=0.5)

        plt.legend()
        plt.xlabel(r"tr $a$ (object-wise)")
        plt.ylabel(r"$\alpha$ (scale-dependent)")
        out_path = self._output_path("leakage_coefficients.png")
        cs_plots.savefig(out_path, close_fig=False)
        cs_plots.show()
        self.print_done(f"Object-wise leakage coefficients plot saved to {out_path}")

    # --- utlility functions for plotting ---
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

    def _xi_psf_sys_sample_x_y_plot_function(
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
        nsamples_to_plot,
        times_theta,
        alpha,
    ):
        # Load the rho-stats to compute the xi_psf_sys samples
        base_rho = self.basename(version)
        self.psf_fitter.load_rho_stat(f"rho_stats_{base_rho}.fits")

        # Get the angular scales for the xi_psf_sys plots
        theta = self.psf_fitter.rho_stat_handler.rho_stats["theta"]

        # Add the offset to the theta values for better visualisation
        jittered_theta = self._get_jittered_theta(theta, idx, len(versions), offset)

        # Get the parameters for the rho-tau fit
        params = self.set_params_rho_tau(
            version, self.results[version]._params, self.cc[version]["psf"]
        )

        # Get the xi_psf_sys samples
        xi_psf_sys_samples_plus, xi_psf_sys_samples_minus = self.get_xi_psf_sys_samples(
            version, params, tomo_bin_a, tomo_bin_b
        )

        y_plus = xi_psf_sys_samples_plus[-nsamples_to_plot:] * (
            theta if times_theta else 1
        )
        y_minus = xi_psf_sys_samples_minus[-nsamples_to_plot:] * (
            theta if times_theta else 1
        )

        ax_plus.plot(
            jittered_theta,
            y_plus.T,
            color=color,
            alpha=alpha,
        )

        ax_minus.plot(jittered_theta, y_minus.T, color=color, alpha=alpha)

    def _xi_psf_sys_mean_and_std_x_y_plot_function(
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
        # Load the rho-stats to compute the xi_psf_sys mean and std.
        base_rho = self.basename(version)
        self.psf_fitter.load_rho_stat(f"rho_stats_{base_rho}.fits")

        # Get the angular scales for the xi_psf_sys plots
        theta = self.psf_fitter.rho_stat_handler.rho_stats["theta"]

        # Add the offset to the theta values for better visualisation
        jittered_theta = self._get_jittered_theta(theta, idx, len(versions), offset)

        xi_psf_sys = self.xi_psf_sys[version][
            f"tomo_bin_{tomo_bin_a}_tomo_bin_{tomo_bin_b}"
        ]

        def plot_axis(ax, ax_type):
            """Plot depending on the axis type ('plus' or 'minus')."""
            ax.plot(
                jittered_theta,
                xi_psf_sys[f"mean_{ax_type}"] * (theta if times_theta else 1),
                color=color,
            )
            ax.plot(
                jittered_theta,
                xi_psf_sys[f"quantiles_{ax_type}"][0] * (theta if times_theta else 1),
                color=color,
            )
            ax.plot(
                jittered_theta,
                xi_psf_sys[f"quantiles_{ax_type}"][1] * (theta if times_theta else 1),
                color=color,
            )
            ax.fill_between(
                jittered_theta,
                xi_psf_sys[f"quantiles_{ax_type}"][0] * (theta if times_theta else 1),
                xi_psf_sys[f"quantiles_{ax_type}"][1] * (theta if times_theta else 1),
                color=color,
                alpha=alpha,
            )

        # Plot the plus axis
        plot_axis(ax_plus, "plus")

        # Plot the minus axis
        plot_axis(ax_minus, "minus")

    def _scale_dependent_xi_psf_sys_x_y_plot_function(
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
        cov_type,
        times_theta,
        fmt,
        capsize,
    ):
        # Load the rho-stats and the tau-stats
        base_rho = self.basename(version)
        base_tau_a = self.basename(version, tomo_bin_a=tomo_bin_a)
        base_tau_b = self.basename(version, tomo_bin_a=tomo_bin_b)
        self.rho_stat_handler.load_rho_stats(f"rho_stats_{base_rho}.fits")

        theta = self.rho_stat_handler.rho_stats["theta"]
        n_bins = len(theta)
        rho_0_p = self.rho_stat_handler.rho_stats["rho_0_p"]
        rho_0_m = self.rho_stat_handler.rho_stats["rho_0_m"]

        self.tau_stat_handler.load_tau_stats(f"tau_stats_{base_tau_a}.fits")

        tau_0_p_a = self.tau_stat_handler.tau_stats["tau_0_p"]
        tau_0_m_a = self.tau_stat_handler.tau_stats["tau_0_m"]

        self.tau_stat_handler.load_tau_stats(f"tau_stats_{base_tau_b}.fits")

        tau_0_p_b = self.tau_stat_handler.tau_stats["tau_0_p"]
        tau_0_m_b = self.tau_stat_handler.tau_stats["tau_0_m"]

        if cov_type is not None:
            outdir = f"{self.cc['paths']['output']}/rho_tau_stats"
            cov_tau_path = Path(outdir) / f"cov_tau_{base_tau_a}_{cov_type}.npy"
            cov_tau_p_a = np.load(cov_tau_path)[:n_bins, :n_bins]
            cov_tau_path = Path(outdir) / f"cov_tau_{base_tau_b}_{cov_type}.npy"
            cov_tau_p_b = np.load(cov_tau_path)[:n_bins, :n_bins]
            cov_rho_path = Path(outdir) / f"cov_rho_{base_rho}_jk.npy"
            cov_rho_p = np.load(cov_rho_path)[:n_bins, :n_bins]
        else:
            cov_tau_p_a = np.diag(self.tau_stat_handler.tau_stats["vartau_0_p"])
            cov_tau_p_b = np.diag(self.tau_stat_handler.tau_stats["vartau_0_p"])
            cov_rho_p = np.diag(self.rho_stat_handler.rho_stats["varrho_0_p"])

        cov_tau_m_a = np.diag(self.tau_stat_handler.tau_stats["vartau_0_m"])
        cov_tau_m_b = np.diag(self.tau_stat_handler.tau_stats["vartau_0_m"])
        cov_rho_m = np.diag(self.rho_stat_handler.rho_stats["varrho_0_m"])

        # Compute the scale-dependent xi_psf_sys and its error bars
        xi_psf_sys_plus, xi_psf_sys_plus_err = self._compute_scale_dependent_xi_psf_sys(
            rho_0_p, tau_0_p_a, tau_0_p_b, cov_rho_p, cov_tau_p_a, cov_tau_p_b
        )
        xi_psf_sys_minus, xi_psf_sys_minus_err = (
            self._compute_scale_dependent_xi_psf_sys(
                rho_0_m, tau_0_m_a, tau_0_m_b, cov_rho_m, cov_tau_m_a, cov_tau_m_b
            )
        )

        jittered_theta = self._get_jittered_theta(theta, idx, len(versions), offset)

        y_plus = xi_psf_sys_plus * (theta if times_theta else 1)
        y_plus_err = xi_psf_sys_plus_err * (theta if times_theta else 1)
        y_minus = xi_psf_sys_minus * (theta if times_theta else 1)
        y_minus_err = xi_psf_sys_minus_err * (theta if times_theta else 1)

        ax_plus.errorbar(
            jittered_theta, y_plus, yerr=y_plus_err, fmt=fmt, capsize=capsize
        )

        ax_minus.errorbar(
            jittered_theta, y_minus, yerr=y_minus_err, fmt=fmt, capsize=capsize
        )

    def _get_jittered_theta(self, theta, idx, n_versions, offset):
        """Get the jittered theta values for better visualisation."""
        theta_widths = np.diff(theta)
        theta_widths = np.append(theta_widths, theta_widths[-1])
        jitter_fraction = (idx - (n_versions - 1) / 2) * offset
        jittered_theta = theta + jitter_fraction * theta_widths
        return jittered_theta

    def _get_ax_plus(self, axs, tomo_bin_a, tomo_bin_b):
        if (tomo_bin_a == "all") ^ (tomo_bin_b == "all"):
            raise ValueError(
                "Invalid combination of tomographic bins: 'all' and a specific bin."
            )

        if tomo_bin_a == "all" and tomo_bin_b == "all":
            return axs[0]

        else:
            nrows = axs.shape[0]
            return axs[nrows - tomo_bin_b, tomo_bin_a - 1]

    def _get_ax_minus(self, axs, tomo_bin_a, tomo_bin_b):
        if (tomo_bin_a == "all") ^ (tomo_bin_b == "all"):
            raise ValueError(
                "Invalid combination of tomographic bins: 'all' and a specific bin."
            )

        if tomo_bin_a == "all" and tomo_bin_b == "all":
            return axs[2]

        else:
            ncols = axs.shape[1]
            return axs[tomo_bin_b - 1, ncols - tomo_bin_a]

    def _set_ax_visibility_to_false(self, axs, n_tomo_bins_plot):
        """Set the visibility of empty axes to False for better visualization."""
        if n_tomo_bins_plot == 1:
            axs[1].set_visible(False)
        else:
            for i in range(n_tomo_bins_plot):
                axs[i, n_tomo_bins_plot - i].set_visible(False)


# %%
