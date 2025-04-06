# %%
import os
from contextlib import contextmanager
from functools import partial

import colorama
import matplotlib.pyplot as plt
import numpy as np

try:
    import pymaster as nmt
except:
    print("Could not import pymaster, continuing without")

import healpy as hp
import healsparse as hsp
import treecorr
import camb
import yaml
from collections import Counter

import skyproj

from astropy.io import fits
from astropy.cosmology import Planck18
from astropy import units as u
from astropy.coordinates import SkyCoord        

try:
    from cosmo_numba.B_modes.schneider2022 import get_pure_EB_modes
except:
    print("Could not import cosmo_numba, continuing without")

from . import utils_cosmo_val
from cs_util import plots as cs_plots
from shear_psf_leakage import leakage
from shear_psf_leakage import plots as psfleak_plots
from shear_psf_leakage import run_object, run_scale
from shear_psf_leakage.rho_tau_stat import PSFErrorFit
from uncertainties import ufloat
import matplotlib.scale as mscale

mscale.register_scale(utils_cosmo_val.SquareRootScale)


# %%
class CosmologyValidation:

    def __init__(
        self,
        versions,
        data_base_dir,
        catalog_config="./cat_config.yaml",
        rho_tau_method="lsq",
        cov_estimate_method="th",
        compute_cov_rho=True,
        n_cov=100,
        theta_min=0.1,
        theta_max=250,
        nbins=20,
        var_method="jackknife",
        npatch=20,
        quantile=0.683,
        theta_min_plot=0.08,
        theta_max_plot=250,
        ylim_alpha=[-0.005, 0.05],
        ylim_xi_sys_ratio=[-0.02, 0.5],
        nside=1024,
        binning='powspace',
        power=1/2,
        n_ell_bins=32,
        pol_factor=True,
        nrandom_cell=10
    ):

        self.versions = versions
        self.data_base_dir = data_base_dir
        self.rho_tau_method = rho_tau_method
        self.cov_estimate_method = cov_estimate_method
        self.compute_cov_rho = compute_cov_rho
        self.n_cov = n_cov
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.npatch = npatch
        self.nbins = nbins
        self.quantile = quantile
        self.theta_min_plot = theta_min_plot
        self.theta_max_plot = theta_max_plot
        self.ylim_alpha = ylim_alpha
        self.ylim_xi_sys_ratio = ylim_xi_sys_ratio
        #For pseudo-Cls
        self.nside = nside
        self.binning = binning
        self.power = power
        self.n_ell_bins = n_ell_bins
        self.pol_factor = pol_factor
        self.nrandom_cell = nrandom_cell

        self.treecorr_config = {
            "ra_units": "degrees",
            "dec_units": "degrees",
            "min_sep": theta_min,
            "max_sep": theta_max,
            "sep_units": "arcmin",
            "nbins": nbins,
            "var_method": var_method,
        }

        with open(catalog_config, "r") as file:
            self.cc = cc = yaml.load(file.read(), Loader=yaml.FullLoader)

        for ver in ["nz", *versions]:

            if ver not in cc:
                raise KeyError(f"Version string {ver} not found in config file{catalog_config}")
            version_base = f"{data_base_dir}/{cc[ver]['subdir']}"
            for key in cc[ver]:
                if "path" in cc[ver][key]:
                    cc[ver][key]["path"] = f"{version_base}/{cc[ver][key]['path']}"

        if not os.path.exists(cc["paths"]["output"]):
            os.mkdir(cc["paths"]["output"])

    def color_reset(self):
        print(colorama.Fore.BLACK, end="")

    def print_blue(self, msg, end="\n"):
        print(colorama.Fore.BLUE + msg, end=end)
        self.color_reset()

    def print_start(self, msg, end="\n"):
        print()
        self.print_blue(msg, end=end)

    def print_done(self, msg):
        self.print_blue(msg)

    def print_magenta(self, msg):
        print(colorama.Fore.MAGENTA + msg)
        self.color_reset()

    def print_green(self, msg):
        print(colorama.Fore.GREEN + msg)
        self.color_reset()

    def print_cyan(self, msg):
        print(colorama.Fore.CYAN + msg)
        self.color_reset()

    def set_params_leakage_scale(self, ver):
        params_in = {}

        # Set parameters
        params_in["input_path_shear"] = self.cc[ver]["shear"]["path"]
        params_in["input_path_PSF"] = self.cc[ver]["star"]["path"]
        params_in["dndz_path"] = (
            f"{self.cc['nz']['dndz']['path']}_{self.cc[ver]['pipeline']}_{self.cc['nz']['dndz']['blind']}.txt"
        )
        params_in["output_dir"] = f'{self.cc["paths"]["output"]}/leakage_{ver}'

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
        params_in["output_dir"] = f'{self.cc["paths"]["output"]}/leakage_{ver}'

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

        if "cols" in self.cc[ver]["shear"]:
            params_in["cols"] = self.cc[ver]["shear"]["cols"]

        params_in["verbose"] = False

        return params_in

    def init_results(self, objectwise=False):

        results = {}
        for ver in self.versions:

            # Set parameters depending on the type of leakage
            if objectwise:
                results[ver] = run_object.LeakageObject()
                results[ver]._params.update(self.set_params_leakage_object(ver))
            else:
                results[ver] = run_scale.LeakageScale()
                results[ver]._params.update(self.set_params_leakage_scale(ver))

            results[ver].check_params()
            results[ver].prepare_output()

            @contextmanager
            def temporarily_load_data(results_instance):
                try:
                    self.print_start(f"Loading catalog {results_instance._params['input_path_shear']} ...", end="")
                    results_instance.read_data()
                    self.print_done(f"done")
                    yield
                finally:
                    self.print_done(f"Freeing {results_instance._params['input_path_shear']} from memory")
                    del results_instance.dat_shear
                    del results_instance.dat_PSF

            results[ver].temporarily_load_data = partial(temporarily_load_data, results[ver])

        return results

    @property
    def results(self):
        if not hasattr(self, "_results"):
            self._results = self.init_results(objectwise=False)
        return self._results

    @property
    def results_objectwise(self):
        if not hasattr(self, "_results_objectwise"):
            self._results_objectwise = self.init_results(objectwise=True)
        return self._results_objectwise

    def calculate_rho_tau_stats(self):

        out_dir = f"{self.cc['paths']['output']}/rho_tau_stats"
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        self.print_start("Rho stats")
        for ver in self.versions:
            rho_stat_handler, tau_stat_handler = utils_cosmo_val.get_rho_tau_w_cov(
                self.cc,
                ver,
                self.treecorr_config,
                out_dir,
                method=self.cov_estimate_method,
                cov_rho=self.compute_cov_rho,
            )
        self.print_done("Rho stats finished")

        self._rho_stat_handler = rho_stat_handler
        self._tau_stat_handler = tau_stat_handler

    @property
    def rho_stat_handler(self):
        if not hasattr(self, "_rho_stat_handler"):
            self.calculate_rho_tau_stats()
        return self._rho_stat_handler

    @property
    def tau_stat_handler(self):
        if not hasattr(self, "_tau_stat_handler"):
            self.calculate_rho_tau_stats()
        return self._tau_stat_handler

    @property
    def colors(self):
        return [self.cc[ver]["colour"] for ver in self.versions]

    def plot_rho_stats(self, abs=False):

        filenames = [f"rho_stats_{ver}.fits" for ver in self.versions]

        savefig = "rho_stats.png"
        self.rho_stat_handler.plot_rho_stats(
            filenames,
            self.colors,
            self.versions,
            savefig=savefig,
            legend="outside",
            abs=abs,
        )
        plt.close()

        self.print_done(
            f"Rho stats plot saved to "
            + f"{os.path.abspath(self.rho_stat_handler.catalogs._output)}/{savefig}",
        )

    def plot_tau_stats(self, plot_tau_m=False):
        filenames = [f"tau_stats_{ver}.fits" for ver in self.versions]

        savefig = "tau_stats.png"
        self.tau_stat_handler.plot_tau_stats(
            filenames,
            self.colors,
            self.versions,
            savefig=savefig,
            legend="outside",
            plot_tau_m=plot_tau_m,
        )
        plt.close()

        self.print_done(
            f"Tau stats plot saved to "
            + f"{os.path.abspath(self.tau_stat_handler.catalogs._output)}/{savefig}",
        )

    def set_params_rho_tau(self, params, params_psf, survey="other"):
        if survey in ("DES", "SP_axel_v0.0", "SP_axel_v0.0_repr"):
            params["patch_number"] = 120
            print("DES, jackknife patch number = 120")
        elif survey == "SP_axel_v0.0":
            params["patch_number"] = 120
            print("SP_Axel_v0.0, jackknife patch number =120")
        elif survey == "SP_v1.4-P3" or survey == "SP_v1.4-P3_LFmask":
            params["patch_number"] = 120
            print("SP_v1.4, jackknife patch number =120")
        else:
            params["patch_number"] = 150

        params["ra_col"] = params_psf["ra_col"]
        params["dec_col"] = params_psf["dec_col"]
        params["e1_PSF_col"] = params_psf["e1_PSF_col"]
        params["e2_PSF_col"] = params_psf["e2_PSF_col"]
        params["e1_star_col"] = params_psf["e1_star_col"]
        params["e2_star_col"] = params_psf["e2_star_col"]
        params["PSF_size"] = params_psf["PSF_size"]
        params["star_size"] = params_psf["star_size"]
        if survey != "DES":
            params["PSF_flag"] = params_psf["PSF_flag"]
            params["star_flag"] = params_psf["star_flag"]
        params["ra_units"] = "deg"
        params["dec_units"] = "deg"

        return params

    @property
    def psf_fitter(self):
        if not hasattr(self, "_psf_fitter"):
            self._psf_fitter = PSFErrorFit(
                self.rho_stat_handler,
                self.tau_stat_handler,
                self.rho_stat_handler.catalogs._output,
            )
        return self._psf_fitter

    def calculate_rho_tau_fits(self):
        assert self.rho_tau_method != "none"

        # this initializes the rho_tau_fits attribute
        self._rho_tau_fits = {"flat_sample_list": [], "result_list": [], "q_list": []}
        quantiles = [1 - self.quantile, self.quantile]

        self._xi_psf_sys = {}
        for ver in self.versions:
            params = self.set_params_rho_tau(
                self.results[ver]._params,
                self.cc[ver]["psf"],
                survey=ver,
            )

            npatch = {"sim": 300, "jk": params["patch_number"]}.get(
                self.cov_estimate_method, None
            )

            flat_samples, result, q = utils_cosmo_val.get_samples(
                self.psf_fitter,
                ver,
                cov_type=self.cov_estimate_method,
                apply_debias=npatch,
                sampler=self.rho_tau_method,
            )

            self.rho_tau_fits["flat_sample_list"].append(flat_samples)
            self.rho_tau_fits["result_list"].append(result)
            self.rho_tau_fits["q_list"].append(q)

            self.psf_fitter.load_rho_stat("rho_stats_" + ver + ".fits")
            nbins = self.psf_fitter.rho_stat_handler._treecorr_config["nbins"]
            xi_psf_sys_samples = np.array([]).reshape(0, nbins)

            for i in range(len(flat_samples)):
                xi_psf_sys = self.psf_fitter.compute_xi_psf_sys(flat_samples[i])
                xi_psf_sys_samples = np.vstack([xi_psf_sys_samples, xi_psf_sys])

            self._xi_psf_sys[ver] = {
                "mean": np.mean(xi_psf_sys_samples, axis=0),
                "var": np.var(xi_psf_sys_samples, axis=0),
                "quantiles": np.quantile(xi_psf_sys_samples, quantiles, axis=0),
            }

    @property
    def rho_tau_fits(self):
        if not hasattr(self, "_rho_tau_fits"):
            self.calculate_rho_tau_fits()
        return self._rho_tau_fits

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
        )
        plt.close()
        self.print_done(f"Tau contours plot saved to {os.path.abspath(savefig)}")

        plt.figure(figsize=(15, 6))
        for mcmc_result, ver, color, flat_sample in zip(
            self.rho_tau_fits["result_list"],
            self.versions,
            self.colors,
            self.rho_tau_fits["flat_sample_list"],
        ):
            self.psf_fitter.load_rho_stat("rho_stats_" + ver + ".fits")
            for i in range(100):
                self.psf_fitter.plot_xi_psf_sys(
                    flat_sample[-i + 1], ver, color, alpha=0.1
                )
            self.psf_fitter.plot_xi_psf_sys(mcmc_result[1], ver, color)
        plt.legend()
        savefig = os.path.abspath(f"{out_dir}/xi_psf_sys_samples.png")
        cs_plots.savefig(savefig)
        self.print_done(f"xi_psf_sys samples plot saved to {savefig}")

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
        plt.show()
        savefig = os.path.abspath(f"{out_dir}/xi_psf_sys_quantiles.png")
        cs_plots.savefig(savefig)
        self.print_done(f"xi_psf_sys quantiles plot saved to {savefig}")

        for mcmc_result, ver, flat_sample in zip(
            self.rho_tau_fits["result_list"],
            self.versions,
            self.rho_tau_fits["flat_sample_list"],
        ):
            self.psf_fitter.load_rho_stat("rho_stats_" + ver + ".fits")
            for yscale in ("linear", "log"):
                out_path = os.path.abspath(
                    f"{out_dir}/xi_psf_sys_terms_{yscale}_{ver}.png"
                )
                self.psf_fitter.plot_xi_psf_sys_terms(
                    ver, mcmc_result[1], out_path, yscale=yscale
                )
                self.print_done(
                    f"{yscale}-scale xi_psf_sys terms plot saved to {out_path}"
                )

    @property
    def xi_psf_sys(self):
        if not hasattr(self, "_xi_psf_sys"):
            self.calculate_rho_tau_fits()
        return self._xi_psf_sys

    def plot_footprints(self):
        self.print_start("Plotting footprints:")
        for ver in self.versions:
            self.print_magenta(ver)
            results = self.results[ver]
            
            fp = FootprintPlotter()
                
            for region in fp._regions: 
                out_path = os.path.abspath(
                    f"{self.cc['paths']['output']}/footprint_{ver}_{region}.png"
                )
            if os.path.exists(out_path):
                self.print_done(
                    f"Skipping footprint plot, {out_path} exists"
                )
            else:
                with self.results[ver].temporarily_load_data():
                    hsp_map = fp.create_hsp_map(
                        self.results[ver].dat_shear["RA"],
                        self.results[ver].dat_shear["Dec"],
                    )
                fp.plot_region(hsp_map, fp._regions[region], outpath=out_path)
                self.print_done("Footprint plot saved to " + out_path)

    def calculate_scale_dependent_leakage(self):
        self.print_start("Calculating scale-dependent leakage:")
        for ver in self.versions:
            self.print_magenta(ver)
            results = self.results[ver]

            output_base_path = os.path.abspath(
                f'{self.cc["paths"]["output"]}/leakage_{ver}/xi_for_leak_scale'
            )
            output_path_ab = f"{output_base_path}_a_b.txt"
            output_path_aa = f"{output_base_path}_a_a.txt"

            with results.temporarily_load_data():
                if os.path.exists(output_path_ab) and os.path.exists(output_path_aa):
                    self.print_green(
                        f"Skipping computation, reading {output_path_ab} and {output_path_aa} instead"
                    )

                    # MKDEBUG the following lines do not need the data catalogue
                    results.r_corr_gp = treecorr.GGCorrelation(self.treecorr_config)
                    results.r_corr_gp.read(output_path_ab)

                    results.r_corr_pp = treecorr.GGCorrelation(self.treecorr_config)
                    results.r_corr_pp.read(output_path_aa)

                else:
                    results.compute_corr_gp_pp_alpha(output_base_path=output_base_path)

                results.do_alpha(fast=True)
                results.do_xi_sys()

        self.print_done("Finished scale-dependent leakage calculation.")

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
            out_path = os.path.abspath(
                f"{self.cc['paths']['output']}/alpha_leak_log.png"
            )

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
            cs_plots.savefig(out_path)
            self.print_done(f"Log-scale alpha leakage plot saved to {out_path}")

            # Lin x
            out_path = os.path.abspath(
                f"{self.cc['paths']['output']}/alpha_leak_lin.png"
            )

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
            cs_plots.savefig(out_path)
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
            out_path = os.path.abspath(f"{self.cc['paths']['output']}/xi_sys_p.png")
            fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
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
            cs_plots.savefig(out_path)
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
            out_path = os.path.abspath(f"{self.cc['paths']['output']}/xi_sys_m.png")
            fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
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
            cs_plots.savefig(out_path)
            self.print_done(f"xi_sys_minus plot saved to {out_path}")

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

            with self.results[ver].temporarily_load_data():
                results_obj._dat = self.results[ver].dat_shear

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
                    results_obj.PSF_leakage()

        # Gather coefficients
        leakage_coeff = {}
        for ver in self.versions:
            leakage_coeff[ver] = {}
            results = self.results[ver]
            results_obj = self.results_objectwise[ver]
            # Object-wise leakage
            leakage_coeff[ver]["a11"] = ufloat(
                results_obj.par_best_fit["a11"].value,
                results_obj.par_best_fit["a11"].stderr,
            )
            leakage_coeff[ver]["a22"] = ufloat(
                results_obj.par_best_fit["a22"].value,
                results_obj.par_best_fit["a22"].stderr,
            )
            leakage_coeff[ver]["aii_mean"] = 0.5 * (
                leakage_coeff[ver]["a11"] + leakage_coeff[ver]["a22"]
            )

            # Scale-dependent leakage: mean
            leakage_coeff[ver]["alpha_mean"] = ufloat(
                results.alpha_leak_mean, results.alpha_leak_std
            )
            # Scale-dependent leakage: value at smallest scale
            leakage_coeff[ver]["alpha_1"] = ufloat(
                results.alpha_leak[0], results.sig_alpha_leak[0]
            )
            # Scale-dependent leakage: value extrapolated to 0 using affine model
            leakage_coeff[ver]["alpha_0"] = ufloat(
                results.alpha_affine_best_fit["c"].value,
                results.alpha_affine_best_fit["c"].stderr,
            )

        self.leakage_coeff = leakage_coeff

    def plot_objectwise_leakage(self):
        if not hasattr(self, "leakage_coeff"):
            self.calculate_objectwise_leakage()

        self.print_start("Plotting object-wise leakage:")
        fig = cs_plots.figure(figsize=(15, 15))

        linestyles = ["-", "--", ":"]
        fillstyles = ["full", "none", "left", "right", "bottom", "top"]

        for ver in self.versions:
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
        out_path = os.path.abspath(
            f"{self.cc['paths']['output']}/leakage_coefficients.png"
        )
        cs_plots.savefig(out_path)
        self.print_done(f"Object-wise leakage coefficients plot saved to {out_path}")

    def calculate_objectwise_leakage_aux(self):

        self.print_start("Object-wise leakage auxiliary quantities:")
        for ver in self.versions:
            self.print_magenta(ver)

            results_obj = self.results_objectwise[ver]
            results_obj.check_params()
            results_obj.update_params()
            results_obj.prepare_output()

            with self.results[ver].temporarily_load_data():
                results_obj._dat = self.results[ver].dat_shear

                #out_base = results_obj.get_out_base(mix, order)
                #out_path = f"{out_base}.pkl"
                #if os.path.exists(out_path):
                    #self.print_green(
                        #f"Skipping object-wise leakage, file {out_path} exists"
                    #)
                    #results_obj.par_best_fit = leakage.read_from_file(out_path)
                if not "cols" in results_obj._params:
                    self.print_green("Skipping object-wise leakage (aux quantities), no input columns for regression found")
                else:
                    self.print_cyan(f"Computing object-wise leakage regression with aux quantities: {results_obj._params['cols']}")

                    # Run
                    results_obj.obs_leakage()

    def plot_objectwise_leakage_aux(self):
            self.calculate_objectwise_leakage_aux()

    def plot_ellipticity(self, nbins=200):
        out_path = os.path.abspath(f"{self.cc['paths']['output']}/ell_hist.png")
        if os.path.exists(out_path):
            self.print_done(f"Skipping ellipticity histograms, {out_path} exists")
        else:
            self.print_start("Computing ellipticity histograms:")

            fig, axs = plt.subplots(1, 2, figsize=(22, 7))
            for ver in self.versions:
                with self.results[ver].temporarily_load_data():
                    self.print_magenta(ver)
                    R = self.cc[ver]["shear"]["R"]
                    e1 = self.results[ver].dat_shear[self.cc[ver]["shear"]["e1_col"]] / R
                    e2 = self.results[ver].dat_shear[self.cc[ver]["shear"]["e2_col"]] / R
                    w_key = self.cc[ver]["shear"]["w_col"]
                    w = self.results[ver].dat_shear[w_key]

                    axs[0].hist(
                        e1,
                        bins=nbins,
                        density=False,
                        histtype="step",
                        weights=w,
                        label=ver,
                        color=self.cc[ver]["colour"],
                    )
                    axs[1].hist(
                        e2,
                        bins=nbins,
                        density=False,
                        histtype="step",
                        weights=w,
                        label=ver,
                        color=self.cc[ver]["colour"],
                    )

                for idx in (0, 1):
                    axs[idx].set_xlabel(f"$e_{idx}$")
                    axs[idx].set_ylabel("frequency")
                    axs[idx].legend()
                    axs[idx].set_xlim([-1.5, 1.5])
                cs_plots.savefig(out_path)
                self.print_done("Ellipticity histograms saved to " + out_path)

    def plot_separation(self, nbins=200):
        self.print_start("Separation histograms")
        if "SP_matched_MP_v1.0" in self.versions:
            fig, axs = plt.subplots(1, 1, figsize=(10, 7))
            sep = self.results["SP_matched_MP_v1.0"].dat_shear["Separation"]
            axs.hist(
                sep,
                bins=nbins,
                density=False,
                histtype="step",
                label="SP_matched_MP_v1.0",
                color=self.cc["SP_matched_MP_v1.0"]["colour"],
            )
            print("Max separation: %s arcsec" % max(sep))
            axs.set_xlabel(r"Separation $\theta$ [arcsec]")
            axs.legend()
        else:
            self.print_done("SP_matched_MP_v1.0 not in versions")

    def calculate_additive_bias(self):
        self.print_start("Calculating additive bias:")
        self._c1 = {}
        self._c2 = {}
        for ver in self.versions:
            self.print_magenta(ver)
            with self.results[ver].temporarily_load_data():
                R = self.cc[ver]["shear"]["R"]
                self._c1[ver] = np.average(
                    self.results[ver].dat_shear[self.cc[ver]["shear"]["e1_col"]] / R,
                    weights=self.results[ver].dat_shear[self.cc[ver]["shear"]["w_col"]],
                )
                self._c2[ver] = np.average(
                    self.results[ver].dat_shear[self.cc[ver]["shear"]["e2_col"]] / R,
                    weights=self.results[ver].dat_shear[self.cc[ver]["shear"]["w_col"]],
                )
            self.print_done("Finished additive bias calculation.")

    @property
    def c1(self):
        if not hasattr(self, "_c1"):
            print("MKDEBUG calc c1")
            self.calculate_additive_bias()
        return self._c1

    @property
    def c2(self):
        if not hasattr(self, "_c2"):
            print("MKDEBUG calc c2")
            self.calculate_additive_bias()
        return self._c2

    def calculate_2pcf(self):
        self.print_start(f"Computing 2PCF")

        self._cat_ggs = {}
        for ver in self.versions:
            self.print_magenta(ver)
            gg = self._cat_ggs[ver] = treecorr.GGCorrelation(self.treecorr_config)

            out_fname = os.path.abspath(f"{self.cc['paths']['output']}/xi_pm_{ver}.txt")
            if os.path.exists(out_fname):
                self.print_done(f"Skipping 2PCF calculation, {out_fname} exists")
                gg.read(out_fname)
            else:
                # Run TreeCorr
                with self.results[ver].temporarily_load_data():
                    e1 = self.results[ver].dat_shear[self.cc[ver]["shear"]["e1_col"]]
                    e2 = self.results[ver].dat_shear[self.cc[ver]["shear"]["e2_col"]]
                    if ver != "DES":
                        R = self.cc[ver]["shear"]["R"]
                        g1 = (e1 - self.c1[ver]) / R
                        g2 = (e2 - self.c2[ver]) / R
                    else:
                        R11 = self.cc[ver]["shear"]["R11"]
                        R22 = self.cc[ver]["shear"]["R22"]
                        g1 = (e1 - self.c1[ver]) / np.average(
                            self.results[ver].dat_shear[R11]
                        )
                        g2 = (
                            self.results[ver].dat_shear[self.cc[ver]["shear"]["e2_col"]]
                            - self.c2[ver]
                        ) / np.average(self.results[ver].dat_shear[R22])
                    cat_gal = treecorr.Catalog(
                        ra=self.results[ver].dat_shear["RA"],
                        dec=self.results[ver].dat_shear["Dec"],
                        g1=g1,
                        g2=g2,
                        w=self.results[ver].dat_shear[self.cc[ver]["shear"]["w_col"]],
                        ra_units=self.treecorr_config["ra_units"],
                        dec_units=self.treecorr_config["dec_units"],
                        npatch=self.npatch,
                    )
                    gg.process(cat_gal)
                    gg.write(out_fname)

                    # Save xi_p and xi_m results to fits file
                    lst = np.arange(1, self.treecorr_config["nbins"] + 1)

                    col1 = fits.Column(name="BIN1", format="K", array=np.ones(len(lst)))
                    col2 = fits.Column(name="BIN2", format="K", array=np.ones(len(lst)))
                    col3 = fits.Column(name="ANGBIN", format="K", array=lst)
                    col4 = fits.Column(name="VALUE", format="D", array=gg.xip)
                    col5 = fits.Column(
                        name="ANG", format="D", unit="arcmin", array=gg.meanr
                    )
                    coldefs = fits.ColDefs([col1, col2, col3, col4, col5])
                    xiplus_hdu = fits.BinTableHDU.from_columns(coldefs, name="XI_PLUS")

                    col4 = fits.Column(name="VALUE", format="D", array=gg.xim)
                    coldefs = fits.ColDefs([col1, col2, col3, col4, col5])
                    ximinus_hdu = fits.BinTableHDU.from_columns(coldefs, name="XI_MINUS")

                    # append xi_plus header info
                    xiplus_dict = {
                        "2PTDATA": "T",
                        "QUANT1": "G+R",
                        "QUANT2": "G+R",
                        "KERNEL_1": "NZ_SOURCE",
                        "KERNEL_2": "NZ_SOURCE",
                        "WINDOWS": "SAMPLE",
                    }
                    for key in xiplus_dict:
                        xiplus_hdu.header[key] = xiplus_dict[key]
                    xiplus_hdu.writeto(
                        f"{self.cc['paths']['output']}/xi_plus_{ver}.fits", overwrite=True
                    )

                    # append xi_minus header info
                    ximinus_dict = {**xiplus_dict, "QUANT1": "G-R", "QUANT2": "G-R"}
                    for key in ximinus_dict:
                        ximinus_hdu.header[key] = ximinus_dict[key]
                    ximinus_hdu.writeto(
                        f"{self.cc['paths']['output']}/xi_minus_{ver}.fits", overwrite=True
                    )

            self.print_done("Done 2PCF")

    @property
    def cat_ggs(self):
        if not hasattr(self, "_cat_ggs"):
            self.calculate_2pcf()
        return self._cat_ggs

    def plot_2pcf(self):
        # Plot of n_pairs
        fig, ax = plt.subplots(ncols=1, nrows=1)
        for ver in self.versions:
            plt.plot(
                self.cat_ggs[ver].meanr,
                self.cat_ggs[ver].npairs,
                label=ver,
                ls=self.cc[ver]["ls"],
                color=self.cc[ver]["colour"],
            )
        plt.xlabel(rf"$\theta$ [{self.treecorr_config['sep_units']}]")
        plt.ylabel(r"$n_{\rm pair}$")
        plt.legend()
        out_path = os.path.abspath(f"{self.cc['paths']['output']}/n_pair.png")
        cs_plots.savefig(out_path)
        self.print_done(f"n_pair plot saved to {out_path}")

        # Plot of xi_+
        fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
        for idx, ver in enumerate(self.versions):
            plt.errorbar(
                self.cat_ggs[ver].meanr * cs_plots.dx(idx, len(ver)),
                self.cat_ggs[ver].xip,
                yerr=np.sqrt(self.cat_ggs[ver].varxip),
                label=ver,
                ls=self.cc[ver]["ls"],
                color=self.cc[ver]["colour"],
            )
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(fontsize=20, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.ticklabel_format(axis="y")
        plt.xlabel(rf"$\theta$ [{self.treecorr_config['sep_units']}]")
        plt.xlim([self.theta_min_plot, self.theta_max_plot])
        plt.ylabel(r"$\xi_+(\theta)$")
        out_path = os.path.abspath(f"{self.cc['paths']['output']}/xi_p.png")
        cs_plots.savefig(out_path)
        self.print_done(f"xi_plus plot saved to {out_path}")

        # Plot of xi_-
        fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
        for idx, ver in enumerate(self.versions):
            plt.errorbar(
                self.cat_ggs[ver].meanr * cs_plots.dx(idx, len(ver)),
                self.cat_ggs[ver].xim,
                yerr=np.sqrt(self.cat_ggs[ver].varxim),
                label=ver,
                ls=self.cc[ver]["ls"],
                color=self.cc[ver]["colour"],
            )
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(fontsize=20, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.ticklabel_format(axis="y")
        plt.xlabel(rf"$\theta$ [{self.treecorr_config['sep_units']}]")
        plt.xlim([self.theta_min_plot, self.theta_max_plot])
        plt.ylabel(r"$\xi_-(\theta)$")
        out_path = os.path.abspath(f"{self.cc['paths']['output']}/xi_m.png")
        cs_plots.savefig(out_path)
        self.print_done(f"xi_minus plot saved to {out_path}")

        # Plot of xi_+(theta) * theta
        fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
        for idx, ver in enumerate(self.versions):
            plt.errorbar(
                self.cat_ggs[ver].meanr,
                self.cat_ggs[ver].xip * self.cat_ggs[ver].meanr,
                yerr=np.sqrt(self.cat_ggs[ver].varxip) * self.cat_ggs[ver].meanr,
                label=ver,
                ls=self.cc[ver]["ls"],
                color=self.cc[ver]["colour"],
            )
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(fontsize=20, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.ticklabel_format(axis="y")
        plt.xlabel(rf"$\theta$ [{self.treecorr_config['sep_units']}]")
        plt.xlim([self.theta_min_plot, self.theta_max_plot])
        plt.ylabel(r"$\theta \xi_+(\theta)$")
        out_path = os.path.abspath(f"{self.cc['paths']['output']}/xi_p_theta.png")
        cs_plots.savefig(out_path)
        self.print_done(f"xi_plus_theta plot saved to {out_path}")

        # Plot of xi_- * theta
        fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
        for idx, ver in enumerate(self.versions):
            plt.errorbar(
                self.cat_ggs[ver].meanr * cs_plots.dx(idx, len(ver)),
                self.cat_ggs[ver].xim * self.cat_ggs[ver].meanr,
                yerr=np.sqrt(self.cat_ggs[ver].varxim) * self.cat_ggs[ver].meanr,
                label=ver,
                ls=self.cc[ver]["ls"],
                color=self.cc[ver]["colour"],
            )
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(fontsize=20, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.ticklabel_format(axis="y")
        plt.xlabel(rf"$\theta$ [{self.treecorr_config['sep_units']}]")
        plt.xlim([self.theta_min_plot, self.theta_max_plot])
        plt.ylabel(r"$\theta \xi_-(\theta)$")
        out_path = os.path.abspath(f"{self.cc['paths']['output']}/xi_m_theta.png")
        cs_plots.savefig(out_path)
        self.print_done(f"xi_minus_theta plot saved to {out_path}")

        # Plot of xi_+ with and without xi_psf_sys
        for idx, ver in enumerate(self.versions):
            fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
            plt.errorbar(
                self.cat_ggs[ver].meanr * cs_plots.dx(idx, len(ver)),
                self.cat_ggs[ver].xip,
                yerr=np.sqrt(self.cat_ggs[ver].varxim),
                label=r"$\xi_+$",
                ls="solid",
                color="green",
            )
            plt.errorbar(
                self.cat_ggs[ver].meanr * cs_plots.dx(idx, len(ver)),
                self.xi_psf_sys[ver]["mean"],
                yerr=np.sqrt(self.xi_psf_sys[ver]["var"]),
                label=r"$\xi^{\rm psf}_{+, {\rm sys}}$",
                ls="dotted",
                color="red",
            )
            plt.errorbar(
                self.cat_ggs[ver].meanr * cs_plots.dx(idx, len(ver)),
                self.cat_ggs[ver].xip + self.xi_psf_sys[ver]["mean"],
                yerr=np.sqrt(self.cat_ggs[ver].varxip + self.xi_psf_sys[ver]["var"]),
                label=r"$\xi_+ + \xi^{\rm psf}_{+, {\rm sys}}$",
                ls="dashdot",
                color="magenta",
            )

            plt.xscale("log")
            plt.yscale("log")
            plt.legend(fontsize=20, bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.ticklabel_format(axis="y")
            plt.xlabel(rf"$\theta$ [{self.treecorr_config['sep_units']}]")
            plt.xlim([self.theta_min_plot, self.theta_max_plot])
            plt.ylim(1e-8, 5e-4)
            plt.ylabel(r"$\xi_+(\theta)$")
            out_path = os.path.abspath(
                f"{self.cc['paths']['output']}/xi_p_xi_psf_sys_{ver}.png"
            )
            cs_plots.savefig(out_path)
            self.print_done(f"xi_plus_xi_psf_sys {ver} plot saved to {out_path}")

    def calculate_aperture_mass_dispersion(
        self, theta_min=0.3, theta_max=200, nbins=500, nbins_map=15, npatch=25
    ):
        self.print_start("Computing aperture-mass dispersion")

        self._map2 = {}
        theta_map = np.geomspace(theta_min * 5, theta_max / 2, nbins_map)
        self._map2["theta_map"] = theta_map

        treecorr_config = {
            **self.treecorr_config,
            "min_sep": theta_min,
            "max_sep": theta_max,
            "nbins": nbins,
        }

        for ver in self.versions:
            self.print_magenta(ver)

            gg = treecorr.GGCorrelation(treecorr_config)

            out_fname = os.path.abspath(
                f"{self.cc['paths']['output']}/xi_for_map2_{ver}.txt"
            )
            if os.path.exists(out_fname):
                self.print_green(f"Skipping xi for Map2, {out_fname} exists")
                gg.read(out_fname)
            else:
                with self.results[ver].temporarily_load_data():
                    R = self.cc[ver]["shear"]["R"]
                    g1 = (
                        self.results[ver].dat_shear[self.cc[ver]["shear"]["e1_col"]]
                        - self.c1[ver]
                    ) / R
                    g2 = (
                        self.results[ver].dat_shear[self.cc[ver]["shear"]["e2_col"]]
                        - self.c2[ver]
                    ) / R
                    cat_gal = treecorr.Catalog(
                        ra=self.results[ver].dat_shear["RA"],
                        dec=self.results[ver].dat_shear["Dec"],
                        g1=g1,
                        g2=g2,
                        w=self.results[ver].dat_shear[self.cc[ver]["shear"]["w_col"]],
                        ra_units=self.treecorr_config["ra_units"],
                        dec_units=self.treecorr_config["dec_units"],
                        npatch=npatch,
                    )

                    gg.process(cat_gal)
                    gg.write(out_fname)
                    del cat_gal
                    del g1
                    del g2

            mapsq, mapsq_im, mxsq, mxsq_im, varmapsq = gg.calculateMapSq(
                R=theta_map,
                m2_uform="Schneider",
            )
            out_fname_map2 = os.path.abspath(
                f"{self.cc['paths']['output']}/map2_{ver}.txt"
            )
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
            x = []
            y = []
            yerr = []
            labels = []
            colors = []
            linestyles = []
            for ver in self.versions:
                x.append(self.map2["theta_map"])
                y.append(self.map2[ver][mode])
                yerr.append(np.sqrt(self.map2[ver]["varmapsq"]))
                labels.append(ver)
                colors.append(self.cc[ver]["colour"])
                linestyles.append(self.cc[ver]["ls"])

            xlabel = r"$\theta$ [arcmin]"
            ylabel = "dispersion"
            title = f"Aperture-mass dispersion {mode}"
            out_path = os.path.abspath(f"{self.cc['paths']['output']}/{mode}.png")
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
                ylim=[-1e-6, 2e-5],
                colors=colors,
                linestyles=linestyles,
                shift_x=True,
            )
            cs_plots.savefig(out_path)
            self.print_done(f"linear-scale {mode} plot saved to {out_path}")

        for mode in ["mapsq", "mapsq_im", "mxsq", "mxsq_im"]:
            x = []
            y = []
            yerr = []
            for ver in self.versions:
                x.append(self.map2["theta_map"])
                y.append(np.abs(self.map2[ver][mode]))
                yerr.append(np.sqrt(self.map2[ver]["varmapsq"]))
            xlabel = r"$\theta$ [arcmin]"
            ylabel = "dispersion"
            title = f"Aperture-mass dispersion mode {mode}"
            out_path = os.path.abspath(f"{self.cc['paths']['output']}/{mode}_log.png")
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
                ylim=[1e-9, 3e-5],
                colors=colors,
                linestyles=linestyles,
                shift_x=True,
            )
            cs_plots.savefig(out_path)
            self.print_done(f"log-scale {mode} plot saved to {out_path}")

    def calculate_pure_eb(
        self,
        theta_min=0.1,
        theta_max=250,
        nbins=20,
        theta_min_int=0.04,
        theta_max_int=500,
        nbins_int=1000,
    ):
        self.print_start("Computing pure E/B")

        treecorr_config = {
            **self.treecorr_config,
            "min_sep": theta_min,
            "max_sep": theta_max,
            "nbins": nbins,
        }

        # nbins_int = (nbins - 1) * integration_oversample + 1
        treecorr_config_int = {
            **treecorr_config,
            "min_sep": theta_min_int,
            "max_sep": theta_max_int,
            "nbins": nbins_int,
        }

        print("Integration correlation function parameters:")
        print(treecorr_config_int)

        print("Output correlation function parameters:")
        print(treecorr_config)

        for ver in self.versions:
            self.print_magenta(ver)

            gg_int = treecorr.GGCorrelation(treecorr_config_int)
            gg = treecorr.GGCorrelation(treecorr_config)

            out_fname = os.path.abspath(
                # f"{self.cc['paths']['output']}/xi_for_pure_eb_{ver}.txt"
                f"{self.cc['paths']['output']}/xi_for_pure_eb_thetamin={theta_min}_thetamax={theta_max}_nbins={nbins}_npatch={self.npatch}_{ver}.txt"
            )
            out_fname_int = os.path.abspath(
                f"{self.cc['paths']['output']}/xi_for_pure_eb_int_thetaminint={theta_min_int}_thetamaxint_{theta_max_int}_nbinsint={nbins_int}_npatchint={self.npatch}_{ver}.txt"
                # f"{self.cc['paths']['output']}/xi_for_pure_eb_{ver}_int.txt"
            )
            if os.path.exists(out_fname) and os.path.exists(out_fname_int):
                self.print_green(
                    f"Skipping xi for COSEBIs:\n{out_fname}\n{out_fname_int}\nexist."
                )
                gg.read(out_fname)
                gg_int.read(out_fname_int)
            else:
                with self.results[ver].temporarily_load_data():
                    R = self.cc[ver]["shear"]["R"]
                    g1 = (
                        self.results[ver].dat_shear[self.cc[ver]["shear"]["e1_col"]]
                        - self.c1[ver]
                    ) / R
                    g2 = (
                        self.results[ver].dat_shear[self.cc[ver]["shear"]["e2_col"]]
                        - self.c2[ver]
                    ) / R

                    cat = treecorr.Catalog(
                        ra=self.results[ver].dat_shear["RA"],
                        dec=self.results[ver].dat_shear["Dec"],
                        g1=g1,
                        g2=g2,
                        w=self.results[ver].dat_shear[self.cc[ver]["shear"]["w_col"]],
                        ra_units=self.treecorr_config["ra_units"],
                        dec_units=self.treecorr_config["dec_units"],
                        npatch=self.npatch,
                    )
                    del g1, g2, R

                self.print_cyan("Integration binning")
                gg_int.process(cat)
                gg_int.write(out_fname_int, write_patch_results=True, write_cov=True)

                self.print_cyan("Output binning")
                gg.process(cat)
                gg.write(out_fname, write_patch_results=True, write_cov=True)

            def pure_EB(corrs):
                gg, gg_int = corrs
                return get_pure_EB_modes(
                    theta=gg.meanr,
                    xip=gg.xip,
                    xim=gg.xim,
                    theta_int=gg_int.meanr,
                    xip_int=gg_int.xip,
                    xim_int=gg_int.xim,
                    tmin=theta_min,
                    tmax=theta_max,
                )

            xip_E, xim_E, xip_B, xim_B, xip_amb, xim_amb = pure_EB([gg, gg_int])
            cov = treecorr.estimate_multi_cov(
                [gg, gg_int], "jackknife", func=lambda x: np.hstack(pure_EB(x))
            )

            results = {
                "xip_E": xip_E,
                "xim_E": xim_E,
                "xip_B": xip_B,
                "xim_B": xim_B,
                "xip_amb": xip_amb,
                "xim_amb": xim_amb,
                "cov": cov,
            }

            return results

class FootprintPlotter:
    """Class to create footprint plots.
    
    Parameters
    -----------
    nside_coverage: int, optional
        basic resolution of map; default is 32
    nside_map:
        fine resolution for plotting; default is 2048

    """
    
    # Dictionary storing region parameters
    _regions = {
        "NGC": {"ra_0": 180, "extend": [120, 270, 20, 70], "vmax": 60},
        "SGC": {"ra_0": 15, "extend": [-20, 45, 20, 45], "vmax": 60},
        "fullsky": {"ra_0": 150, "extend": [0, 360, -90, 90], "vmax": 60},
    }
    
    def __init__(self, nside_coverage=32, nside_map=2048):
        
        self._nside_coverage = nside_coverage
        self._nside_map = nside_map
    
    def create_hsp_map(self, ra, dec):
        """Create Hsp Map.
        
        Create healsparse map.
        
        Parameters
        ----------
        ra : numpy.ndarray
            right ascension values
        dec : numpy.ndarray
            declination values
            
        Returns
        -------
        hsp.HealSparseMap
            map
            
        """
        # Create empty map
        hsp_map = hsp.HealSparseMap.make_empty(
            self._nside_coverage,
            self._nside_map,
            dtype=np.float32,
            sentinel=np.nan
        )

        # Get pixel list corresponding to coordinates
        hpix = hp.ang2pix(self._nside_map, ra, dec, nest=True, lonlat=True)

        # Get count of objects per pixel
        pixel_counts = Counter(hpix)

        # List of unique pixels
        unique_hpix = np.array(list(pixel_counts.keys()))

        # Number of objects
        values = np.array(list(pixel_counts.values()), dtype=np.float32)

        # Create maps with numbers per pixel
        hsp_map[unique_hpix] = values
    
        return hsp_map
    
    def plot_area(
        self,
        hsp_map,
        ra_0=0,
        extend=[120, 270, 29, 70],
        vmax=60,
        projection=None,
        outpath=None,
        title=None,
    ):
        """Plot Area.
        
        Plot catalogue in an area on the sky.
        
        Parameters
        ----------
        hsp_map : hsp_HealSparseMap
            input map
        ra_0 : float, optional
            anchor point in R.A.; default is 0
        extend : list, optional
            sky region, extend=[ra_low, ra_high, dec_low, dec_high];
            default is [120, 270, 29, 70]
        vmax : float, optional
            maximum pixel value to plot with color; default is 60
        projection : skyproj.McBrydeSkyproj
            if ``None`` (default), a new plot is created
        outpath : str, optional
            output path, default is ``None``
        title : str, optional
            print title if not ``None`` (default)
            
        Returns
        --------
        skyproj.McBrydeSkyproj
            projection instance
        plt.axes.Axes
            axes instance
            
        Raises
        ------
        ValueError
            if no object found in region
        
        """
        if not projection:
            
            # Create new figure and axes
            fig, ax = plt.subplots(figsize=(10, 10))

            # Create new projection
            projection = skyproj.McBrydeSkyproj(
                ax=ax,
                lon_0=ra_0,
                extent=extend,
                autorescale=True,
                vmax=vmax
            )
        else:
            ax = None

        try:
            _ = projection.draw_hspmap(
                hsp_map, lon_range=extend[0:2],
                lat_range=extend[2:]
            )
        except ValueError:
            msg = "No object found in region to draw"
            print(f"{msg}, continuing...")
            #raise ValueError(msg)
            
        projection.draw_milky_way(width=25, linewidth=1.5, color='black', linestyle='-')
            
        if title:
            plt.title(title, pad=5)

        if outpath:
            plt.savefig(outpath)
            
        return projection, ax
        
    def plot_region(self, hsp_map, region, projection=None, outpath=None, title=None):
        
        return self.plot_area(
            hsp_map,
            region["ra_0"],
            region["extend"],
            region["vmax"],
            projection=projection,
            outpath=outpath,
            title=title,
        )

    def plot_all_regions(self, hsp_map, outbase=None):  

        for region in self._regions:
            if outbase:
                outpath = f"{outbase}_{region}.png"
            else:
                outpath = None
            self.plot_region(hsp_map, self._regions[region], outpath=outpath)
            

    @classmethod            
    def hp_pixel_centers(cls, nside, nest=False):
        
        # Get number of pixels for given nside        
        npix = hp.nside2npix(nside)
        
        # Get pixel indices
        pix_indices = np.arange(npix)
    
        # Get coordinates of pixel centers
        ra, dec = hp.pix2ang(nside, pix_indices, nest=nest, lonlat=True)

        return ra, dec, npix

    @classmethod
    def plot_footprint_as_hp(cls, hsp_map, nside, outpath=None, title=None):

        ra, dec, npix = cls.hp_pixel_centers(nside)

        # Create an empty HEALPix map
        m = np.full(npix, np.nan)  

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the HEALPix grid
        hp.mollview(m, title=title, coord="C", notext=True, rot=(180, 0, 0))
        
        # Define the Galactic Plane: l = [0, 360], b = 0°
        for l0, ls in zip((-5, 0, 5), (":", "-", ":")):
            l_values = np.linspace(0, 360, 500)  # 500 points along the plane
            b_values = np.zeros_like(l_values)   # Galactic latitude is 0 (the plane)

            # Convert (l, b) to (λ, β) - Ecliptic coordinates
            coords = SkyCoord(l=l_values*u.degree, b=b_values*u.degree, frame='galactic')
            ecl_coords = coords.transform_to('barycentrictrueecliptic')  # Ecliptic frame

            # Extract Ecliptic longitude (λ) and latitude (β)
            lambda_ecl = ecl_coords.lon.deg  # Ecliptic longitude
            beta_ecl = ecl_coords.lat.deg    # Ecliptic latitude

            # Convert to HEALPix projection coordinates (colatitude, longitude)
            theta = np.radians(90 - beta_ecl)  # HEALPix uses colatitude
            phi = np.radians(lambda_ecl)  # HEALPix uses longitude

            # Create a healpy Mollweide projection in Ecliptic coordinates
            hp.projplot(theta, phi, linestyle=ls, color='black', linewidth=1)  # Plot the outline

        # Apply mask
        mask_values = hsp_map.get_values_pos(ra, dec, valid_mask=True, lonlat=True)

        ok = np.where(mask_values == False)[0]
        #nok = np.where(mask_values == False)[0]

        hp.projscatter(ra[ok], dec[ok], lonlat=True, color="green", s=1, marker=".")        
        #hp.projscatter(ra[nok], dec[nok], lonlat=True, color="red", s=1, marker=".")
        
        plt.tight_layout()
        
        if outpath:
            plt.savefig(outpath)
            
        plt.show()

def hsp_map_logical_or(maps, verbose=False):
    """
    Hsp Map Logical Or.
    
    Logical AND of HealSparseMaps.
    
    """
    if verbose:
        print("Combine all maps...")
    
    # Ensure consistency in coverage and data type
    nside_coverage = maps[0].nside_coverage
    nside_sparse = maps[0].nside_sparse
    dtype = maps[0].dtype

    for m in maps:
        # MKDEBUG TODO: Change nside if possible
        if m.nside_coverage != nside_coverage:
            raise ValueError(
                f"Coverage nside={m.nside_coverage} does not match {nside_coverage}"
            )
        if m.dtype != dtype:
            raise ValueError(
                f"Data type {m.dtype} does not match {dtype}"
            )

    # Create an empty HealSparse map
    map_comb = hsp.HealSparseMap.make_empty(nside_coverage, nside_sparse, dtype=dtype)
    for idx, m in enumerate(maps):
        map_comb |= m
        
        if verbose:
            valid_pixels = map_comb.valid_pixels
            n_tot = np.sum(valid_pixels)
            n_true = np.count_nonzero(valid_pixels)
            n_false = n_tot - n_true
            print(f"after map {idx}: frac_true={n_true / n_tot:g}, frac_false={n_false / n_tot:g}")

    return map_comb

    def calculate_pseudo_cl_eb_cov(self):
        """
        Compute a theoretical Gaussian covariance of the Pseudo-Cl for EE, EB and BB.
        """
        self.print_start("Computing Pseudo-Cl covariance")

        nside = self.nside

        try:
            self._pseudo_cls
        except AttributeError:
            self._pseudo_cls = {}
        for ver in self.versions:
            self.print_magenta(ver)

            if not ver in self._pseudo_cls.keys():
                self._pseudo_cls[ver] = {}

            out_path = os.path.abspath(f"{self.cc['paths']['output']}/pseudo_cl_cov_{ver}.fits")
            if os.path.exists(out_path):
                self.print_done(f"Skipping Pseudo-Cl covariance calculation, {out_path} exists")
                self._pseudo_cls[ver]['cov'] = fits.open(out_path)
            else:

                params = utils_cosmo_val.get_params_rho_tau(self.cc[ver], survey=ver)

                self.print_cyan(f"Extracting the fiducial power spectrum for {ver}")

                lmax = 2*self.nside
                path_redshift_distr = self.data_base_dir + self.cc[ver]["shear"]["redshift_distr"]
                pw = hp.pixwin(nside, lmax=lmax)
                fiducial_cl = self.get_fiducial(lmax, path_redshift_distr)*pw**2

                self.print_cyan("Getting a sample of the fiducial Cl's with noise")

                lmin = 8
                lmax = 2*self.nside
                b_lmax = lmax - 1

                if self.binning == 'linear':
                    step = 10
                    b = nmt.NmtBin.from_nside_linear(self.nside, step)
                elif self.binning == 'powspace':
                    ells = np.arange(lmin, lmax+1)

                    start = np.power(lmin, self.power)
                    end = np.power(lmax, self.power)
                    bins_ell = np.power(np.linspace(start, end, self.n_ell_bins+1), 1/self.power)

                    #Get bandpowers
                    bpws = np.digitize(ells.astype(float), bins_ell) - 1
                    bpws[0] = 0
                    bpws[-1] = self.n_ell_bins-1

                    b = nmt.NmtBin(ells=ells, bpws=bpws, lmax=b_lmax)

                #Load data and create shear and noise maps
                cat_gal = fits.getdata(self.cc[ver]["shear"]["path"])

                n_gal, unique_pix, idx, idx_rep = self.get_n_gal_map(params, nside, cat_gal)
                mask = n_gal != 0
                
                cl_noise, f, wsp = self.get_sample(params, self.nside, b_lmax, b, cat_gal, n_gal, mask, unique_pix, idx_rep)
                
                fiducial_cl = np.array([fiducial_cl, 0.*fiducial_cl, 0.*fiducial_cl, 0.*fiducial_cl])+ np.mean(cl_noise, axis=1, keepdims=True)
                
                self.print_cyan("Computing the Pseudo-Cl covariance")

                cw = nmt.NmtCovarianceWorkspace.from_fields(f, f, f, f)

                covar_22_22 = nmt.gaussian_covariance(cw, 2, 2, 2, 2,
                                                        fiducial_cl,
                                                        fiducial_cl,
                                                        fiducial_cl,
                                                        fiducial_cl,
                                                        wsp, wb=wsp).reshape([self.n_ell_bins, 4, self.n_ell_bins, 4])

                covar_EE_EE = covar_22_22[:, 0, :, 0]
                covar_EE_EB = covar_22_22[:, 0, :, 1]
                covar_EE_BE = covar_22_22[:, 0, :, 2]
                covar_EE_BB = covar_22_22[:, 0, :, 3]
                covar_EB_EE = covar_22_22[:, 1, :, 0]
                covar_EB_EB = covar_22_22[:, 1, :, 1]
                covar_EB_BE = covar_22_22[:, 1, :, 2]
                covar_EB_BB = covar_22_22[:, 1, :, 3]
                covar_BE_EE = covar_22_22[:, 2, :, 0]
                covar_BE_EB = covar_22_22[:, 2, :, 1]
                covar_BE_BE = covar_22_22[:, 2, :, 2]
                covar_BE_BB = covar_22_22[:, 2, :, 3]
                covar_BB_EE = covar_22_22[:, 3, :, 0]
                covar_BB_EB = covar_22_22[:, 3, :, 1]
                covar_BB_BE = covar_22_22[:, 3, :, 2]
                covar_BB_BB = covar_22_22[:, 3, :, 3]

                self.print_cyan("Saving Pseudo-Cl covariance")

                hdu = fits.HDUList()

                hdu.append(fits.ImageHDU(covar_EE_EE, name="COVAR_EE_EE"))
                hdu.append(fits.ImageHDU(covar_EE_EB, name="COVAR_EE_EB"))
                hdu.append(fits.ImageHDU(covar_EE_BE, name="COVAR_EE_BE"))
                hdu.append(fits.ImageHDU(covar_EE_BB, name="COVAR_EE_BB"))
                hdu.append(fits.ImageHDU(covar_EB_EE, name="COVAR_EB_EE"))
                hdu.append(fits.ImageHDU(covar_EB_EB, name="COVAR_EB_EB"))
                hdu.append(fits.ImageHDU(covar_EB_BE, name="COVAR_EB_BE"))
                hdu.append(fits.ImageHDU(covar_EB_BB, name="COVAR_EB_BB"))
                hdu.append(fits.ImageHDU(covar_BE_EE, name="COVAR_BE_EE"))
                hdu.append(fits.ImageHDU(covar_BE_EB, name="COVAR_BE_EB"))
                hdu.append(fits.ImageHDU(covar_BE_BE, name="COVAR_BE_BE"))
                hdu.append(fits.ImageHDU(covar_BE_BB, name="COVAR_BE_BB"))
                hdu.append(fits.ImageHDU(covar_BB_EE, name="COVAR_BB_EE"))
                hdu.append(fits.ImageHDU(covar_BB_EB, name="COVAR_BB_EB"))
                hdu.append(fits.ImageHDU(covar_BB_BE, name="COVAR_BB_BE"))
                hdu.append(fits.ImageHDU(covar_BB_BB, name="COVAR_BB_BB"))

                hdu.writeto(out_path, overwrite=True)

                self._pseudo_cls[ver]['cov'] = hdu

        self.print_done("Done Pseudo-Cl covariance")

    def calculate_pseudo_cl(self):
        """
        Compute the pseudo-Cl of given catalogs.
        """
        self.print_start("Computing pseudo-Cl's")

        nside = self.nside

        try:
            self._pseudo_cls
        except AttributeError:
            self._pseudo_cls = {}
        for ver in self.versions:
            self.print_magenta(ver)

            self._pseudo_cls[ver] = {}

            out_path = os.path.abspath(f"{self.cc['paths']['output']}/pseudo_cl_{ver}.fits")
            if os.path.exists(out_path):
                self.print_done(f"Skipping Pseudo-Cl's calculation, {out_path} exists")
                cl_shear = fits.getdata(out_path)
                self._pseudo_cls[ver]['pseudo_cl'] = cl_shear
            else:
                params = utils_cosmo_val.get_params_rho_tau(self.cc[ver], survey=ver)

                #Load data and create shear and noise maps
                cat_gal = fits.getdata(self.cc[ver]["shear"]["path"])

                w = cat_gal[params['w_col']]
                self.print_cyan("Creating maps and computing Cl's...")
                n_gal_map, unique_pix, idx, idx_rep = self.get_n_gal_map(params, nside, cat_gal)
                mask = n_gal_map != 0

                shear_map_e1 = np.zeros(hp.nside2npix(nside))
                shear_map_e2 = np.zeros(hp.nside2npix(nside))

                e1 = cat_gal[params['e1_col']]
                e2 = cat_gal[params['e2_col']]

                del cat_gal
                
                shear_map_e1[unique_pix] += np.bincount(idx_rep, weights=e1*w)
                shear_map_e2[unique_pix] += np.bincount(idx_rep, weights=e2*w)
                shear_map_e1[mask] /= n_gal_map[mask]
                shear_map_e2[mask] /= n_gal_map[mask]

                shear_map = shear_map_e1 + 1j*shear_map_e2

                del shear_map_e1, shear_map_e2

                ell_eff, cl_shear, wsp = self.get_pseudo_cls(shear_map)

                cl_noise = np.zeros((4, self.n_ell_bins))
                
                for i in range(self.nrandom_cell):

                    noise_map_e1 = np.zeros(hp.nside2npix(nside))
                    noise_map_e2 = np.zeros(hp.nside2npix(nside))

                    e1_rot, e2_rot = self.apply_random_rotation(e1, e2)

                    
                    noise_map_e1[unique_pix] += np.bincount(idx_rep, weights=e1_rot*w)
                    noise_map_e2[unique_pix] += np.bincount(idx_rep, weights=e2_rot*w)

                    noise_map_e1[mask] /= n_gal_map[mask]
                    noise_map_e2[mask] /= n_gal_map[mask]

                    noise_map = noise_map_e1 + 1j*noise_map_e2
                    del noise_map_e1, noise_map_e2

                    _, cl_noise_, _ = self.get_pseudo_cls(noise_map, wsp)
                    cl_noise += cl_noise_
                
                cl_noise /= self.nrandom_cell
                del e1, e2, e1_rot, e2_rot, w
                del n_gal_map              

                #This is a problem because the measurement depends on the seed. To be fixed.
                #cl_shear = cl_shear - np.mean(cl_noise, axis=1, keepdims=True)
                cl_shear = cl_shear - cl_noise

                self.print_cyan("Saving pseudo-Cl's...")
                self.save_pseudo_cl(ell_eff, cl_shear, out_path)

                cl_shear = fits.getdata(out_path)
                self._pseudo_cls[ver]['pseudo_cl'] = cl_shear

        self.print_done("Done pseudo-Cl's")

    def get_fiducial(self, lmax, redshift_distr):
        """
        Get the power spectrum at Planck18 cosmology using CAMB.
        """
        planck = Planck18

        h = planck.H0.value/100
        Om = planck.Om0
        Ob = planck.Ob0
        Oc = Om - Ob
        ns = 0.965
        As = 2.1e-9
        m_nu = 0.06
        w = -1
        
        pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, ns=ns, mnu=m_nu, w=w, As=As, WantTransfer=True, NonLinear=camb.model.NonLinear_both)
        Onu = pars.omeganu
        Oc = Om - Ob - Onu
        pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, ns=ns, mnu=m_nu, w=w, As=As, WantTransfer=True, NonLinear=camb.model.NonLinear_both)

        z, dndz = np.loadtxt(redshift_distr, unpack=True)

        #getthe expected cl's from CAMB
        pars.min_l = 1
        pars.set_for_lmax(lmax)
        pars.SourceWindows = [
            camb.sources.SplinedSourceWindow(z=z, W=dndz, source_type='lensing')
        ]
        theory_cls = camb.get_results(pars).get_source_cls_dict(lmax=lmax, raw_cl=True)
        return theory_cls['W1xW1']

    def get_n_gal_map(self, params, nside, cat_gal):
        """
        Compute the galaxy number density map.
        """
        ra = cat_gal[params['ra_col']]
        dec = cat_gal[params['dec_col']]
        w = cat_gal[params['w_col']]

        theta = (90. - dec) * np.pi / 180.
        phi = ra * np.pi / 180.
        pix = hp.ang2pix(nside, theta, phi)

        unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
        n_gal = np.zeros(hp.nside2npix(nside))
        n_gal[unique_pix] = np.bincount(idx_rep, weights=w)
        return n_gal, unique_pix, idx, idx_rep

    def get_gaussian_real(self, params, nside, lmax, cat_gal, n_gal, mask, unique_pix, idx_rep):

        e1_rot, e2_rot = self.apply_random_rotation(cat_gal[params['e1_col']], cat_gal[params['e2_col']])
        noise_map_e1 = np.zeros(hp.nside2npix(nside))
        noise_map_e2 = np.zeros(hp.nside2npix(nside))

        w = cat_gal[params['w_col']]
        noise_map_e1[unique_pix] += np.bincount(idx_rep, weights=e1_rot*w)
        noise_map_e2[unique_pix] += np.bincount(idx_rep, weights=e2_rot*w)
        noise_map_e1[mask] /= n_gal[mask]
        noise_map_e2[mask] /= n_gal[mask]

        return noise_map_e1 + 1j*noise_map_e2

    def get_sample(self, params, nside, lmax, b, cat_gal, n_gal, mask, unique_pix, idx_rep):
        noise_map = self.get_gaussian_real(params, nside, lmax, cat_gal, n_gal, mask, unique_pix, idx_rep)

        f = nmt.NmtField(mask=mask, maps=[noise_map.real, noise_map.imag], lmax=lmax)

        wsp = nmt.NmtWorkspace.from_fields(f, f, b)

        cl_noise = nmt.compute_coupled_cell(f, f)
        cl_noise = wsp.decouple_cell(cl_noise)

        return cl_noise, f, wsp
    
    def get_pseudo_cls(self, map, wsp=None):
        """
        Compute the pseudo-cl for a given map.
        """

        lmin = 8
        lmax = 2*self.nside
        b_lmax = lmax - 1

        if self.binning == 'linear':
            step = 10
            b = nmt.NmtBin.from_nside_linear(self.nside, step)
        elif self.binning == 'powspace':
            ells = np.arange(lmin, lmax+1)

            start = np.power(lmin, self.power)
            end = np.power(lmax, self.power)
            bins_ell = np.power(np.linspace(start, end, self.n_ell_bins+1), 1/self.power)

            #Get bandpowers
            bpws = np.digitize(ells.astype(float), bins_ell) - 1
            bpws[0] = 0
            bpws[-1] = self.n_ell_bins-1

            b = nmt.NmtBin(ells=ells, bpws=bpws, lmax=b_lmax)

        ell_eff = b.get_effective_ells()

        factor = -1 if self.pol_factor else 1

        f_all = nmt.NmtField(mask=(map!=0), maps=[map.real, factor*map.imag], lmax=b_lmax)
        
        if wsp is None:
            wsp = nmt.NmtWorkspace.from_fields(f_all, f_all, b)
        
        cl_coupled = nmt.compute_coupled_cell(f_all, f_all)
        cl_all = wsp.decouple_cell(cl_coupled)

        return ell_eff, cl_all, wsp

    def apply_random_rotation(self, e1, e2):
        """
        Apply a random rotation to the ellipticity components e1 and e2.

        Parameters
        ----------
        e1 : np.array
            First component of the ellipticity.
        e2 : np.array
            Second component of the ellipticity.

        Returns
        -------
        np.array
            First component of the rotated ellipticity.
        np.array
            Second component of the rotated ellipticity.
        """
        np.random.seed()
        rot_angle = np.random.rand(len(e1))*2*np.pi
        e1_out = e1*np.cos(rot_angle) + e2*np.sin(rot_angle)
        e2_out = -e1*np.sin(rot_angle) + e2*np.cos(rot_angle)
        return e1_out, e2_out

    def save_pseudo_cl(self, ell_eff, pseudo_cl, out_path):
        """
        Save pseudo-Cl's to a FITS file.

        Parameters
        ----------
        pseudo_cl : np.array
            Pseudo-Cl's to save.
        out_path : str
            Path to save the pseudo-Cl's to.
        """
        #Create columns of the fits file
        col1 = fits.Column(name="ELL", format="D", array=ell_eff)
        col2 = fits.Column(name="EE", format="D", array=pseudo_cl[0])
        col3 = fits.Column(name="EB", format="D", array=pseudo_cl[1])
        col4 = fits.Column(name="BB", format="D", array=pseudo_cl[3])
        coldefs = fits.ColDefs([col1, col2, col3, col4])
        cell_hdu = fits.BinTableHDU.from_columns(coldefs, name="PSEUDO_CELL")

        cell_hdu.writeto(out_path, overwrite=True)

    @property
    def pseudo_cls(self):
        if not hasattr(self, "_pseudo_cls"):
            self.calculate_pseudo_cl()
            self.calculate_pseudo_cl_eb_cov()
        return self._pseudo_cls

    def plot_pseudo_cl(self):
        """
        Plot pseudo-Cl's for given catalogs.
        """
        self.print_cyan("Plotting pseudo-Cl's")

        #Plotting EE
        out_path = os.path.abspath(f"{self.cc['paths']['output']}/cell_ee.png")
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

        for ver in self.versions:
            ell = self.pseudo_cls[ver]['pseudo_cl']["ELL"]
            cov = self.pseudo_cls[ver]['cov']["COVAR_EE_EE"].data
            ax[0].errorbar(ell, ell*self.pseudo_cls[ver]['pseudo_cl']["EE"], yerr=ell*np.sqrt(np.diag(cov)),  fmt=self.cc[ver]["marker"], label=ver+" EE", color=self.cc[ver]["colour"], capsize=2)

        ax[0].set_ylabel('$\ell C_\ell$')

        ax[0].set_xlim(ell.min()-10, ell.max()+100)
        ax[0].set_xscale('squareroot')
        ax[0].set_xticks(np.array([100, 400, 900, 1600]))
        ax[0].minorticks_on()
        ax[0].tick_params(axis='x', which='minor', length=2, width=0.8)
        minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
        ax[0].xaxis.set_ticks(minor_ticks, minor=True)

        for ver in self.versions:
            ell = self.pseudo_cls[ver]['pseudo_cl']["ELL"]
            cov = self.pseudo_cls[ver]['cov']["COVAR_EE_EE"].data
            ax[1].errorbar(ell, self.pseudo_cls[ver]['pseudo_cl']["EE"], yerr=np.sqrt(np.diag(cov)),  fmt=self.cc[ver]["marker"], label=ver+" EE", color=self.cc[ver]["colour"])

        ax[1].set_xlabel('$\ell$')
        ax[1].set_ylabel('$C_\ell$')

        ax[1].set_xlim(ell.min()-10, ell.max()+100)
        ax[1].set_xscale('squareroot')
        ax[1].set_yscale('log')
        ax[1].set_xticks(np.array([100, 400, 900, 1600]))
        ax[1].minorticks_on()
        ax[1].tick_params(axis='x', which='minor', length=2, width=0.8)
        minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
        ax[1].xaxis.set_ticks(minor_ticks, minor=True)

        plt.suptitle('Pseudo-Cl EE (Gaussian covariance)')
        plt.legend()
        plt.savefig(out_path)

        #Plotting EB
        out_path = os.path.abspath(f"{self.cc['paths']['output']}/cell_eb.png")

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

        for ver in self.versions:
            ell = self.pseudo_cls[ver]['pseudo_cl']["ELL"]
            cov = self.pseudo_cls[ver]['cov']["COVAR_EB_EB"].data
            ax[0].errorbar(ell, ell*self.pseudo_cls[ver]['pseudo_cl']["EB"], yerr=ell*np.sqrt(np.diag(cov)),  fmt=self.cc[ver]["marker"], label=ver+" EB", color=self.cc[ver]["colour"], capsize=2)

        ax[0].axhline(0, color='black', linestyle='--')
        ax[0].set_ylabel('$\ell C_\ell$')

        ax[0].set_xlim(ell.min()-10, ell.max()+100)
        ax[0].set_xscale('squareroot')
        ax[0].set_xticks(np.array([100, 400, 900, 1600]))
        ax[0].minorticks_on()
        ax[0].tick_params(axis='x', which='minor', length=2, width=0.8)
        minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
        ax[0].xaxis.set_ticks(minor_ticks, minor=True)

        for ver in self.versions:
            ell = self.pseudo_cls[ver]['pseudo_cl']["ELL"]
            cov = self.pseudo_cls[ver]['cov']["COVAR_EB_EB"].data
            ax[1].errorbar(ell, self.pseudo_cls[ver]['pseudo_cl']["EB"], yerr=np.sqrt(np.diag(cov)),  fmt=self.cc[ver]["marker"], label=ver+" EB", color=self.cc[ver]["colour"])

        ax[1].set_xlabel('$\ell$')
        ax[1].set_ylabel('$C_\ell$')

        ax[1].set_xlim(ell.min()-10, ell.max()+100)
        ax[1].set_xscale('squareroot')
        ax[1].set_yscale('log')
        ax[1].set_xticks(np.array([100, 400, 900, 1600]))
        ax[1].minorticks_on()
        ax[1].tick_params(axis='x', which='minor', length=2, width=0.8)
        minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
        ax[1].xaxis.set_ticks(minor_ticks, minor=True)

        plt.suptitle('Pseudo-Cl EB (Gaussian covariance)')
        plt.legend()
        plt.savefig(out_path)

        #Plotting BB
        out_path = os.path.abspath(f"{self.cc['paths']['output']}/cell_bb.png")

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

        for ver in self.versions:
            ell = self.pseudo_cls[ver]['pseudo_cl']["ELL"]
            cov = self.pseudo_cls[ver]['cov']["COVAR_BB_BB"].data
            ax[0].errorbar(ell, ell*self.pseudo_cls[ver]['pseudo_cl']["BB"], yerr=ell*np.sqrt(np.diag(cov)),  fmt=self.cc[ver]["marker"], label=ver+" BB", color=self.cc[ver]["colour"], capsize=2)

        ax[0].axhline(0, color='black', linestyle='--')
        ax[0].set_ylabel('$\ell C_\ell$')

        ax[0].set_xlim(ell.min()-10, ell.max()+100)
        ax[0].set_xscale('squareroot')
        ax[0].set_xticks(np.array([100, 400, 900, 1600]))
        ax[0].minorticks_on()
        ax[0].tick_params(axis='x', which='minor', length=2, width=0.8)
        minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
        ax[0].xaxis.set_ticks(minor_ticks, minor=True)

        for ver in self.versions:
            ell = self.pseudo_cls[ver]['pseudo_cl']["ELL"]
            cov = self.pseudo_cls[ver]['cov']["COVAR_BB_BB"].data
            ax[1].errorbar(ell, self.pseudo_cls[ver]['pseudo_cl']["BB"], yerr=np.sqrt(np.diag(cov)),  fmt=self.cc[ver]["marker"], label=ver+" BB", color=self.cc[ver]["colour"])

        ax[1].set_xlabel('$\ell$')
        ax[1].set_ylabel('$C_\ell$')

        ax[1].set_xlim(ell.min()-10, ell.max()+100)
        ax[1].set_xscale('squareroot')
        ax[1].set_yscale('log')
        ax[1].set_xticks(np.array([100, 400, 900, 1600]))
        ax[1].minorticks_on()
        ax[1].tick_params(axis='x', which='minor', length=2, width=0.8)
        minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
        ax[1].xaxis.set_ticks(minor_ticks, minor=True)

        plt.suptitle('Pseudo-Cl BB (Gaussian covariance)')
        plt.legend()
        plt.savefig(out_path)
# %%
