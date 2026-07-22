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
            self.calculate_rho_tau_fits()
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

    def calculate_rho_tau_fits(self, tomography=True):
        assert self.rho_tau_method != "none"

        # this initializes the rho_tau_fits attribute
        self._rho_tau_fits = {"flat_sample_list": [], "result_list": [], "q_list": []}
        quantiles = [1 - self.quantile, self.quantile]

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

            for tomo_bin_id in tomo_bin_ids:
                self.print_cyan(
                    f"Sample PSF error parameters for tomographic bin {tomo_bin_id}"
                )
                xi_psf_sys_samples = self.get_samples(ver, params, tomo_bin_id)

                if ver not in self._xi_psf_sys.keys():
                    self._xi_psf_sys[ver] = {}

                self._xi_psf_sys[ver][f"tomo_bin_{tomo_bin_id}"] = {
                    "mean": np.mean(xi_psf_sys_samples, axis=0),
                    "var": np.var(xi_psf_sys_samples, axis=0),
                    "quantiles": np.quantile(xi_psf_sys_samples, quantiles, axis=0),
                }

    def calculate_scale_dependent_leakage(self):
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

    def calculate_objectwise_leakage(self):
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
            self.rho_tau_fits["flat_sample_list"].append(flat_samples)
            self.rho_tau_fits["result_list"].append(result)
            self.rho_tau_fits["q_list"].append(q)

        self.psf_fitter.load_rho_stat(f"rho_stats_{base_rho}.fits")
        nbins = self.psf_fitter.rho_stat_handler._treecorr_config["nbins"]
        xi_psf_sys_samples = np.array(
            [self.psf_fitter.compute_xi_psf_sys(sample) for sample in flat_samples]
        ).reshape(-1, nbins)

        return xi_psf_sys_samples

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
            tomo_bins = {}
            for ver in versions:
                tomo_bin_ids, tomo_bin_pairs = self._get_tomo_bins(ver)

                tomo_bins[ver] = {"ids": tomo_bin_ids, "pairs": tomo_bin_pairs}

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

                        theta_widths = np.diff(theta)
                        theta_widths = np.append(theta_widths, theta_widths[-1])

                        jitter_fraction = (file_idx - (len(versions) - 1) / 2) * offset
                        jittered_theta = theta + jitter_fraction * theta_widths

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

    def plot_rho_tau_fits(self):
        out_dir = self.rho_stat_handler.catalogs._output

        savefig = f"{out_dir}/contours_tau_stat.png"
        psfleak_plots.plot_contours(
            self.rho_tau_fits["flat_sample_list"],
            names=["x0", "x1", "x2"],
            labels=[r"\alpha", r"\beta", r"\eta"],
            savefig=savefig,
            legend_labels=self.versions,
            legend_loc="upper right",
            contour_colors=self.colors,
            markers={"x0": 0, "x1": 1, "x2": 1},
            show=True,
            close=True,
        )
        self.print_done(f"Tau contours plot saved to {os.path.abspath(savefig)}")

        plt.figure(figsize=(15, 6))
        for mcmc_result, ver, color, flat_sample in zip(
            self.rho_tau_fits["result_list"],
            self.versions,
            self.colors,
            self.rho_tau_fits["flat_sample_list"],
        ):
            self.psf_fitter.load_rho_stat(f"rho_stats_{self.basename(ver)}.fits")
            for i in range(100):
                self.psf_fitter.plot_xi_psf_sys(
                    flat_sample[-i + 1], ver, color, alpha=0.1
                )
            self.psf_fitter.plot_xi_psf_sys(mcmc_result[1], ver, color)
        plt.legend()
        out_path = os.path.abspath(f"{out_dir}/xi_psf_sys_samples.png")
        cs_plots.savefig(out_path, close_fig=False)
        cs_plots.show()
        self.print_done(f"xi_psf_sys samples plot saved to {out_path}")

        plt.figure(figsize=(15, 6))
        for mcmc_result, ver, color, flat_sample in zip(
            self.rho_tau_fits["result_list"],
            self.versions,
            self.colors,
            self.rho_tau_fits["flat_sample_list"],
        ):
            ls = self.cc[ver]["ls"]
            theta = self.psf_fitter.rho_stat_handler.rho_stats["theta"]
            xi_psf_sys = self.xi_psf_sys[ver]
            plt.plot(theta, xi_psf_sys["mean"], linestyle=ls, color=color)
            plt.plot(theta, xi_psf_sys["quantiles"][0], linestyle=ls, color=color)
            plt.plot(theta, xi_psf_sys["quantiles"][1], linestyle=ls, color=color)
            plt.fill_between(
                theta,
                xi_psf_sys["quantiles"][0],
                xi_psf_sys["quantiles"][1],
                color=color,
                alpha=0.25,
                label=ver,
            )

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"$\theta$ [arcmin]")
        plt.ylabel(r"$\xi^{\rm PSF}_{\rm sys}$")
        plt.title(f"{1 - self.quantile:.1%}, {self.quantile:.1%} quantiles")
        plt.legend()
        out_path = os.path.abspath(f"{out_dir}/xi_psf_sys_quantiles.png")
        cs_plots.savefig(out_path, close_fig=False)
        cs_plots.show()
        self.print_done(f"xi_psf_sys quantiles plot saved to {out_path}")

        for mcmc_result, ver, flat_sample in zip(
            self.rho_tau_fits["result_list"],
            self.versions,
            self.rho_tau_fits["flat_sample_list"],
        ):
            self.psf_fitter.load_rho_stat(f"rho_stats_{self.basename(ver)}.fits")
            for yscale in ("linear", "log"):
                out_path = os.path.abspath(
                    f"{out_dir}/xi_psf_sys_terms_{yscale}_{ver}.png"
                )
                self.psf_fitter.plot_xi_psf_sys_terms(
                    ver, mcmc_result[1], out_path, yscale=yscale, show=True
                )
                self.print_done(
                    f"{yscale}-scale xi_psf_sys terms plot saved to {out_path}"
                )

    def plot_scale_dependent_leakage(self):
        if not hasattr(self.results[self.versions[0]], "r_corr_gp"):
            self.calculate_scale_dependent_leakage()

        theta = []
        y = []
        yerr = []
        labels = []
        colors = []
        linestyles = []
        markers = []

        for ver in self.versions:
            if hasattr(self.results[ver], "r_corr_gp"):
                theta.append(self.results[ver].r_corr_gp.meanr)
                y.append(self.results[ver].alpha_leak)
                yerr.append(self.results[ver].sig_alpha_leak)
                labels.append(ver)
                colors.append(self.cc[ver]["colour"])
                linestyles.append(self.cc[ver]["ls"])
                markers.append(self.cc[ver]["marker"])

        if len(theta) > 0:
            # Log x
            out_path = self._output_path("alpha_leak_log.png")

            title = r"$\alpha$ leakage"
            xlabel = r"$\theta$ [arcmin]"
            ylabel = r"$\alpha(\theta)$"
            cs_plots.plot_data_1d(
                theta,
                y,
                yerr,
                title,
                xlabel,
                ylabel,
                out_path=None,
                xlog=True,
                xlim=[self.theta_min_plot, self.theta_max_plot],
                ylim=self.ylim_alpha,
                labels=labels,
                colors=colors,
                linestyles=linestyles,
                shift_x=True,
            )
            cs_plots.savefig(out_path, close_fig=False)
            cs_plots.show()
            self.print_done(f"Log-scale alpha leakage plot saved to {out_path}")

            # Lin x
            out_path = self._output_path("alpha_leak_lin.png")

            title = r"$\alpha$ leakage"
            xlabel = r"$\theta$ [arcmin]"
            ylabel = r"$\alpha(\theta)$"
            cs_plots.plot_data_1d(
                theta,
                y,
                yerr,
                title,
                xlabel,
                ylabel,
                out_path=None,
                xlog=False,
                xlim=[-10, self.theta_max_plot],
                ylim=self.ylim_alpha,
                labels=labels,
                colors=colors,
                linestyles=linestyles,
                shift_x=False,
            )
            cs_plots.savefig(out_path, close_fig=False)
            cs_plots.show()
            self.print_done(f"Lin-scale alpha leakage plot saved to {out_path}")

        # Plot xi_sys
        y = []
        yerr = []
        colors = []
        linestyles = []

        for ver in self.versions:
            if hasattr(self.results[ver], "C_sys_p"):
                y.append(self.results[ver].C_sys_p)
                yerr.append(self.results[ver].C_sys_std_p)
                labels.append(ver)
                colors.append(self.cc[ver]["colour"])
                linestyles.append(self.cc[ver]["ls"])

        if len(y) > 0:
            xlabel = r"$\theta$ [arcmin]"
            ylabel = r"$\xi^{\rm sys}_+(\theta)$"
            title = "Cross-correlation leakage"
            out_path = self._output_path("xi_sys_p.png")
            cs_plots.plot_data_1d(
                theta,
                y,
                yerr,
                title,
                xlabel,
                ylabel,
                out_path=None,
                labels=labels,
                xlog=True,
                xlim=[self.theta_min_plot, self.theta_max_plot],
                colors=colors,
                linestyles=linestyles,
                # shift_x=True,
            )
            cs_plots.savefig(out_path, close_fig=False)
            cs_plots.show()
            self.print_done(f"xi_sys_plus plot saved to {out_path}")

        y = []
        yerr = []
        for ver in self.versions:
            if hasattr(self.results[ver], "C_sys_m"):
                y.append(self.results[ver].C_sys_m)
                yerr.append(self.results[ver].C_sys_std_m)

        if len(y) > 0:
            xlabel = r"$\theta$ [arcmin]"
            ylabel = r"$\xi^{\rm sys}_-(\theta)$"
            title = "Cross-correlation leakage"
            out_path = self._output_path("xi_sys_m.png")
            cs_plots.plot_data_1d(
                theta,
                y,
                yerr,
                title,
                xlabel,
                ylabel,
                out_path=None,
                labels=labels,
                xlog=True,
                xlim=[self.theta_min_plot, self.theta_max_plot],
                ylim=[-1e-7, 1e-6],
                colors=colors,
                linestyles=linestyles,
                # shift_x=True,
            )
            cs_plots.savefig(out_path, close_fig=False)
            cs_plots.show()
            self.print_done(f"xi_sys_minus plot saved to {out_path}")

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
