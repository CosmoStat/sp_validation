# %%
import copy
import os
from pathlib import Path

import colorama
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import treecorr
import yaml
from astropy.io import fits
from cs_util import plots as cs_plots
from shear_psf_leakage import run_object, run_scale

from .b_modes import (
    _get_pte_from_scale_cut,
    calculate_cosebis,
    calculate_eb_statistics,
    calculate_pure_eb_correlation,
    find_conservative_scale_cut_key,
    plot_cosebis_covariance_matrix,
    plot_cosebis_modes,
    plot_cosebis_scale_cut_heatmap,
    plot_eb_covariance_matrix,
    plot_integration_vs_reporting,
    plot_pte_2d_heatmaps,
    plot_pure_eb_correlations,
    save_cosebis_results,
    save_pure_eb_results,
)
from .catalog_characterization import CatalogCharacterizationMixin
from .cosmology import get_cosmo
from .pseudo_cl import PseudoClMixin
from .psf_systematics import PSFSystematicsMixin


# %%
class CosmologyValidation(
    PSFSystematicsMixin, CatalogCharacterizationMixin, PseudoClMixin
):
    """Framework for cosmic shear validation and systematics analysis.

    Handles two-point correlation function measurements, PSF systematics (rho/tau),
    pseudo-C_ell analysis, and covariance estimation for weak lensing surveys.
    Supports multiple catalog versions with automatic leakage-corrected variants.

    Parameters
    ----------
    versions : list of str
        Catalog version identifiers to analyze. Appending '_leak_corr' to a base
        version creates a virtual catalog using leakage-corrected ellipticity columns
        (e1_col_corrected/e2_col_corrected) from the base version configuration.
    catalog_config : str, default './cat_config.yaml'
        Path to catalog configuration YAML defining survey metadata, file paths,
        and analysis settings for each version.
    output_dir : str, optional
        Override for output directory. If None, uses catalog config's paths.output.
    rho_tau_method : {'lsq', 'mcmc'}, default 'lsq'
        Fitting method for PSF leakage systematics parameters.
    cov_estimate_method : {'th', 'jk'}, default 'th'
        Covariance estimation: 'th' for semi-analytic theory, 'jk' for jackknife.
    compute_cov_rho : bool, default True
        Whether to compute covariance for rho statistics during PSF analysis.
    n_cov : int, default 100
        Number of realizations for covariance estimation when using theory method.
    theta_min : float, default 0.1
        Minimum angular separation in arcminutes for correlation function binning.
    theta_max : float, default 250
        Maximum angular separation in arcminutes for correlation function binning.
    nbins : int, default 20
        Number of angular bins for TreeCorr real-space correlation functions.
    var_method : {'jackknife', 'sample', 'bootstrap', 'marked_bootstrap'}, default 'jackknife'
        TreeCorr variance estimation method.
    npatch : int, default 20
        Number of spatial patches for jackknife variance estimation.
    quantile : float, default 0.1587
        Quantile for uncertainty bands in plots (default: 1-sigma ≈ 0.159).
    theta_min_plot : float, default 0.08
        Minimum angular scale for plotting (may differ from analysis cut).
    theta_max_plot : float, default 250
        Maximum angular scale for plotting.
    ylim_alpha : list of float, default [-0.005, 0.05]
        Y-axis limits for alpha systematic parameter plots.
    ylim_xi_sys_ratio : list of float, default [-0.02, 0.5]
        Y-axis limits for xi systematics ratio plots.
    nside : int, default 1024
        HEALPix resolution for pseudo-C_ell analysis and area computation.
    binning : {'powspace', 'linspace', 'logspace'}, default 'powspace'
        Ell binning scheme for pseudo-C_ell (powspace = ell^power spacing).
    power : float, default 0.5
        Exponent for power-law binning when binning='powspace'.
    n_ell_bins : int, default 32
        Number of ell bins for pseudo-C_ell analysis (used with binning='powspace').
    ell_step : int, default 10
        Bin width in ell for linear binning (used with binning='linear').
    pol_factor : bool, default True
        Apply polarization correction factor in pseudo-C_ell calculations.
    nrandom_cell : int, default 10
        Number of random realizations for C_ell error estimation.
    cosmo_params : dict, optional
        Cosmological parameters to pass to get_cosmo(). If None, uses Planck 2018.

    Attributes
    ----------
    versions : list of str
        Validated catalog versions after processing _leak_corr variants.
    cc : dict
        Loaded catalog configuration with resolved absolute paths.
    catalog_config_path : Path
        Resolved path to the catalog configuration file.
    treecorr_config : dict
        Configuration dictionary passed to TreeCorr correlation objects.
    cosmo : pyccl.Cosmology
        Cosmology object for theory predictions.

    Notes
    -----
    - Path resolution: Relative paths in catalog config are resolved using each
      version's 'subdir' field as the base directory.
    - Virtual _leak_corr versions: These create deep copies of the base version
      config, swapping e1_col/e2_col with e1_col_corrected/e2_col_corrected.
    - TreeCorr cross_patch_weight: Automatically set to 'match' for jackknife,
      'simple' otherwise, following TreeCorr best practices.
    """

    @staticmethod
    def _split_seed_variant(version):
        """Return the base version and seed label if version encodes a seed."""
        if "_seed" not in version:
            return None, None
        base, seed_label = version.rsplit("_seed", 1)
        if not base or not seed_label.isdigit():
            return None, None
        return base, seed_label

    @staticmethod
    def _materialize_seed_path(
        base_cfg, seed_label, version, base_version, catalog_config
    ):
        """Render the seed-specific shear path using Python string formatting."""
        shear_cfg = base_cfg["shear"]
        template = shear_cfg.get("path_template")

        try:
            seed_value = int(seed_label)
        except ValueError as error:
            raise ValueError(
                f"Seed suffix for '{version}' is not numeric; cannot materialize path."
            ) from error

        format_context = {"seed": seed_value, "seed_label": seed_label}

        if template:
            try:
                return template.format(**format_context)
            except KeyError as error:
                raise KeyError(
                    f"Missing placeholder '{error.args[0]}' in path_template for "
                    f"'{base_version}' while materializing '{version}'. Update "
                    f"{catalog_config}."
                ) from error
            except ValueError as error:
                raise ValueError(
                    f"Invalid format specification in path_template for '{base_version}' "
                    f"while materializing '{version}'."
                ) from error

        path = shear_cfg.get("path", "")
        token_start = path.rfind("seed")
        if token_start == -1:
            raise ValueError(
                f"Cannot materialize '{version}': '{base_version}' lacks a shear "
                f"path_template and its shear path '{path}' does not contain a 'seed' "
                f"token. Update {catalog_config}."
            )
        cursor = token_start + 4  # len("seed")
        if cursor < len(path) and not path[cursor].isdigit():
            cursor += 1
        digit_start = cursor
        while cursor < len(path) and path[cursor].isdigit():
            cursor += 1
        digit_end = cursor
        digits = path[digit_start:digit_end]
        if not digits:
            raise ValueError(
                f"Cannot materialize '{version}': shear path '{path}' for base version "
                f"'{base_version}' lacks digits after the seed token. Update "
                f"{catalog_config}."
            )

        template = f"{path[:digit_start]}{{seed_label}}{path[digit_end:]}"
        return template.format(**format_context)

    def __init__(
        self,
        versions,
        catalog_config="./cat_config.yaml",
        output_dir=None,
        rho_tau_method="lsq",
        cov_estimate_method="th",
        compute_cov_rho=True,
        n_cov=100,
        theta_min=0.1,
        theta_max=250,
        nbins=20,
        var_method="jackknife",
        npatch=20,
        quantile=0.1587,
        theta_min_plot=0.08,
        theta_max_plot=250,
        ylim_alpha=[-0.005, 0.05],
        ylim_xi_sys_ratio=[-0.02, 0.5],
        nside=1024,
        nside_mask=2**13,
        binning="powspace",
        power=1 / 2,
        n_ell_bins=32,
        ell_step=10,
        pol_factor=True,
        cell_method="map",
        noise_bias_method="analytic",
        fiducial_input_inka="coupled",
        nrandom_cell=10,
        path_onecovariance=None,
        cosmo_params=None,
        blind=None,
    ):
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
        # For pseudo-Cls
        self.nside = nside
        self.binning = binning
        self.power = power
        self.n_ell_bins = n_ell_bins
        self.ell_step = ell_step
        self.pol_factor = pol_factor
        self.nrandom_cell = nrandom_cell
        self.cell_method = cell_method
        self.noise_bias_method = noise_bias_method
        self.fiducial_input_inka = fiducial_input_inka
        self.nside_mask = nside_mask
        self.path_onecovariance = path_onecovariance
        self.blind = blind

        assert self.cell_method in ["map", "catalog"], (
            "cell_method must be 'map' or 'catalog'"
        )
        assert self.noise_bias_method in ["analytic", "randoms"], (
            "noise_bias_method must be 'analytical' or 'randoms'"
        )
        assert self.fiducial_input_inka in ["coupled", "decoupled"], (
            "fiducial_input_inka must be 'coupled' or 'decoupled'"
        )

        # For theory calculations:
        # Create cosmology object using new functionality
        if cosmo_params is not None:
            self.cosmo = get_cosmo(**cosmo_params)
        else:
            # Use Planck 2018 defaults
            self.cosmo = get_cosmo()

        self.treecorr_config = {
            "ra_units": "degrees",
            "dec_units": "degrees",
            "min_sep": theta_min,
            "max_sep": theta_max,
            "sep_units": "arcmin",
            "nbins": nbins,
            "var_method": var_method,
            "cross_patch_weight": "match" if var_method == "jackknife" else "simple",
        }

        self.catalog_config_path = Path(catalog_config)
        with self.catalog_config_path.open("r") as file:
            self.cc = cc = yaml.load(file.read(), Loader=yaml.FullLoader)

        def resolve_paths_for_version(ver):
            """Resolve relative paths for a version using its subdir."""
            subdir = Path(cc[ver]["subdir"])
            for key in cc[ver]:
                if "path" in cc[ver][key]:
                    path = Path(cc[ver][key]["path"])
                    cc[ver][key]["path"] = (
                        str(path) if path.is_absolute() else str(subdir / path)
                    )

        resolve_paths_for_version("nz")
        processed = {"nz"}
        final_versions = []
        leak_suffix = "_leak_corr"

        def ensure_version_exists(ver):
            if ver in processed:
                return

            if ver in cc:
                resolve_paths_for_version(ver)
                processed.add(ver)
                return

            seed_base, seed_label = self._split_seed_variant(ver)

            if ver.endswith(leak_suffix):
                base_ver = ver[: -len(leak_suffix)]
                ensure_version_exists(base_ver)
                shear_cfg = cc[base_ver]["shear"]
                if (
                    "e1_col_corrected" not in shear_cfg
                    or "e2_col_corrected" not in shear_cfg
                ):
                    raise ValueError(
                        f"{base_ver} does not have e1_col_corrected/e2_col_corrected "
                        f"fields; cannot create {ver}"
                    )
                if ver not in cc:
                    cc[ver] = copy.deepcopy(cc[base_ver])
                    cc[ver]["shear"]["e1_col"] = shear_cfg["e1_col_corrected"]
                    cc[ver]["shear"]["e2_col"] = shear_cfg["e2_col_corrected"]
                resolve_paths_for_version(ver)
                processed.add(ver)
                return

            if seed_base is not None:
                ensure_version_exists(seed_base)
                if ver not in cc:
                    cc[ver] = copy.deepcopy(cc[seed_base])
                    seed_path = self._materialize_seed_path(
                        cc[seed_base],
                        seed_label,
                        ver,
                        seed_base,
                        catalog_config,
                    )
                    cc[ver]["shear"]["path"] = seed_path
                resolve_paths_for_version(ver)
                processed.add(ver)
                return

            raise KeyError(
                f"Version string {ver} not found in config file {catalog_config}"
            )

        for ver in versions:
            ensure_version_exists(ver)
            final_versions.append(ver)

        self.versions = final_versions

        # Override output directory if provided
        if output_dir is not None:
            cc["paths"]["output"] = output_dir

        os.makedirs(cc["paths"]["output"], exist_ok=True)

        # B-mode results storage for summarize_bmodes()
        self._pure_eb_results = {}
        self._cosebis_results = {}

    def get_redshift(self, version):
        """Load redshift distribution for a catalog version.

        Parameters
        ----------
        version : str
            Catalog version identifier

        Returns
        -------
        z : ndarray
            Redshift values
        nz : ndarray
            n(z) probability density

        Notes
        -----
        If self.blind is set, the redshift path is modified to use the
        specified blind (A, B, or C) by replacing the blind suffix in the
        configured path.
        """
        import re

        redshift_path = self.cc[version]["shear"]["redshift_path"]

        # Override blind if specified
        if self.blind is not None:
            redshift_path = re.sub(r"_[ABC]\.txt$", f"_{self.blind}.txt", redshift_path)

        return np.loadtxt(redshift_path, unpack=True)

    def _write_catalog_config(self):
        with self.catalog_config_path.open("w") as file:
            yaml.dump(self.cc, file, sort_keys=False)

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

    def basename(self, version, treecorr_config=None, npatch=None):
        cfg = treecorr_config or self.treecorr_config
        patches = npatch or self.npatch
        return (
            f"{version}_minsep={cfg['min_sep']}"
            f"_maxsep={cfg['max_sep']}"
            f"_nbins={cfg['nbins']}"
            f"_npatch={patches}"
        )

    @property
    def colors(self):
        return [self.cc[ver]["colour"] for ver in self.versions]

    def calculate_2pcf(self, ver, npatch=None, save_fits=False, **treecorr_config):
        """
        Calculate the two-point correlation function (2PCF) ξ± for a given catalog
        version with TreeCorr.

        By default the class instance's `npatch` and `treecorr_config` entries are
        used to
        initialize the TreeCorr Catalog and GGCorrelation objects, but may be
        overridden
        by passing keyword arguments.

        Parameters:
            ver (str): The catalog version to process.

            npatch (int, optional): The number of patches to use for the calculation.
            Defaults to the instance's `npatch` attribute.

            save_fits (bool, optional): Whether to save the ξ± results to FITS files.
            Defaults to False.

            **treecorr_config: Additional TreeCorr configuration parameters that will
            override the instance's default `treecorr_config`. For example, `min_sep=1`.

        Returns:
            treecorr.GGCorrelation: The TreeCorr GGCorrelation object containing the
            computed 2PCF results.

        Notes:
            - If the output file for the given configuration already exists, the
              calculation is skipped, and the results are loaded from the file.
            - If a patch file for the given configuration does not exist, it is
              created during the process.
            - FITS files for ξ+ and ξ− are saved with additional metadata in their
              headers if `save_fits` is True.
        """

        self.print_magenta(f"Computing {ver} ξ±")

        npatch = npatch or self.npatch
        treecorr_config = {
            **self.treecorr_config,
            **treecorr_config,
            "var_method": "jackknife" if int(npatch) > 1 else "shot",
        }

        gg = treecorr.GGCorrelation(treecorr_config)

        # If the output file already exists, skip the calculation
        out_fname = os.path.abspath(
            f"{self.cc['paths']['output']}/{ver}_xi_minsep={treecorr_config['min_sep']}_maxsep={treecorr_config['max_sep']}_nbins={treecorr_config['nbins']}_npatch={npatch}.txt"
        )

        if os.path.exists(out_fname):
            self.print_done(f"Skipping 2PCF calculation, {out_fname} exists")
            gg.read(out_fname)

        else:
            # Load data and create a catalog
            with self.results[ver].temporarily_read_data():
                e1 = self.results[ver].dat_shear[self.cc[ver]["shear"]["e1_col"]]
                e2 = self.results[ver].dat_shear[self.cc[ver]["shear"]["e2_col"]]
                w = self.results[ver].dat_shear[self.cc[ver]["shear"]["w_col"]]
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
                    g2 = (e2 - self.c2[ver]) / np.average(
                        self.results[ver].dat_shear[R22]
                    )

                # Use patch file if it exists
                patch_file = os.path.abspath(
                    f"{self.cc['paths']['output']}/{ver}_patches_npatch={npatch}.dat"
                )

                cat_gal = treecorr.Catalog(
                    ra=self.results[ver].dat_shear["RA"],
                    dec=self.results[ver].dat_shear["Dec"],
                    g1=g1,
                    g2=g2,
                    w=w,
                    ra_units=self.treecorr_config["ra_units"],
                    dec_units=self.treecorr_config["dec_units"],
                    npatch=npatch,
                    patch_centers=patch_file if os.path.exists(patch_file) else None,
                )

                # If no patch file exists, save the current patches
                if not os.path.exists(patch_file):
                    cat_gal.write_patch_centers(patch_file)

            # Process the catalog & write the correlation functions
            gg.process(cat_gal)
            gg.write(out_fname, write_patch_results=True, write_cov=True)

        # Save xi_p and xi_m results to fits file
        # (moved outside so it runs even if txt exists)
        if save_fits:
            lst = np.arange(1, treecorr_config["nbins"] + 1)

            col1 = fits.Column(name="BIN1", format="K", array=np.ones(len(lst)))
            col2 = fits.Column(name="BIN2", format="K", array=np.ones(len(lst)))
            col3 = fits.Column(name="ANGBIN", format="K", array=lst)
            col4 = fits.Column(name="VALUE", format="D", array=gg.xip)
            col5 = fits.Column(name="ANG", format="D", unit="arcmin", array=gg.meanr)
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

                col1 = fits.Column(name="BIN1", format="K", array=np.ones(len(lst)))
                col2 = fits.Column(name="BIN2", format="K", array=np.ones(len(lst)))
                col3 = fits.Column(name="ANGBIN", format="K", array=lst)
                col4 = fits.Column(name="VALUE", format="D", array=gg.xip)
                col5 = fits.Column(name="ANG", format="D", unit="arcmin", array=gg.rnom)
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
            # Use same naming format as txt output
            fits_base = out_fname.replace(".txt", "").replace("_xi_", "_")
            xiplus_hdu.writeto(
                f"{fits_base.replace(ver, f'xi_plus_{ver}')}.fits",
                overwrite=True,
            )

            # append xi_minus header info
            ximinus_dict = {**xiplus_dict, "QUANT1": "G-R", "QUANT2": "G-R"}
            for key in ximinus_dict:
                ximinus_hdu.header[key] = ximinus_dict[key]
            ximinus_hdu.writeto(
                f"{fits_base.replace(ver, f'xi_minus_{ver}')}.fits",
                overwrite=True,
            )

        # Add correlation object to class
        if not hasattr(self, "cat_ggs"):
            self.cat_ggs = {}
        self.cat_ggs[ver] = gg

        self.print_done("Done 2PCF")

        return gg

    def plot_2pcf(self):
        # Plot of n_pairs
        fig, ax = plt.subplots(ncols=1, nrows=1)
        for ver in self.versions:
            self.calculate_2pcf(ver)
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
        cs_plots.savefig(out_path, close_fig=False)
        cs_plots.show()
        self.print_done(f"n_pair plot saved to {out_path}")

        # Plot of xi_+
        fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
        for idx, ver in enumerate(self.versions):
            plt.errorbar(
                self.cat_ggs[ver].meanr * cs_plots.dx(idx, fx=1.05, nx=len(ver)),
                self.cat_ggs[ver].xip,
                yerr=np.sqrt(self.cat_ggs[ver].varxip),
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
        out_path = os.path.abspath(f"{self.cc['paths']['output']}/xi_p.png")
        cs_plots.savefig(out_path, close_fig=False)
        cs_plots.show()
        self.print_done(f"xi_plus plot saved to {out_path}")

        # Plot of xi_-
        fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
        for idx, ver in enumerate(self.versions):
            plt.errorbar(
                self.cat_ggs[ver].meanr * cs_plots.dx(idx, fx=1.05, nx=len(ver)),
                self.cat_ggs[ver].xim,
                yerr=np.sqrt(self.cat_ggs[ver].varxim),
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
        out_path = os.path.abspath(f"{self.cc['paths']['output']}/xi_m.png")
        cs_plots.savefig(out_path, close_fig=False)
        cs_plots.show()
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
        plt.legend()
        plt.ticklabel_format(axis="y")
        plt.xlabel(rf"$\theta$ [{self.treecorr_config['sep_units']}]")
        plt.xlim([self.theta_min_plot, self.theta_max_plot])
        plt.ylabel(r"$\theta \xi_+(\theta)$")
        out_path = os.path.abspath(f"{self.cc['paths']['output']}/xi_p_theta.png")
        cs_plots.savefig(out_path, close_fig=False)
        cs_plots.show()
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
        plt.legend()
        plt.ticklabel_format(axis="y")
        plt.xlabel(rf"$\theta$ [{self.treecorr_config['sep_units']}]")
        plt.xlim([self.theta_min_plot, self.theta_max_plot])
        plt.ylabel(r"$\theta \xi_-(\theta)$")
        out_path = os.path.abspath(f"{self.cc['paths']['output']}/xi_m_theta.png")
        cs_plots.savefig(out_path, close_fig=False)
        cs_plots.show()
        self.print_done(f"xi_minus_theta plot saved to {out_path}")

        # Plot of xi_+ with and without xi_psf_sys
        # but skip if xi_psf_sys is not calculated since that takes forever
        if hasattr(self, "_xi_psf_sys"):
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
                    yerr=np.sqrt(
                        self.cat_ggs[ver].varxip + self.xi_psf_sys[ver]["var"]
                    ),
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
                out_path = os.path.abspath(
                    f"{self.cc['paths']['output']}/xi_p_xi_psf_sys_{ver}.png"
                )
                cs_plots.savefig(out_path, close_fig=False)
                cs_plots.show()
                self.print_done(f"xi_plus_xi_psf_sys {ver} plot saved to {out_path}")

    def plot_ratio_xi_sys_xi(self, threshold=0.1, offset=0.02):

        fig, _ = plt.subplots(ncols=1, nrows=1, figsize=(10, 7))

        for idx, ver in enumerate(self.versions):
            self.calculate_2pcf(ver)
            xi_psf_sys = self.xi_psf_sys[ver]
            gg = self.cat_ggs[ver]

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
        out_path = os.path.abspath(f"{self.cc['paths']['output']}/ratio_xi_sys_xi.png")
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
                with self.results[ver].temporarily_read_data():
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
                ylim=[-2e-6, 5e-6],
                colors=colors,
                linestyles=linestyles,
                shift_x=True,
            )
            cs_plots.savefig(out_path, close_fig=False)
            cs_plots.show()
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
                ylim=[1e-8, 1e-5],
                colors=colors,
                linestyles=linestyles,
                shift_x=True,
            )
            cs_plots.savefig(out_path, close_fig=False)
            cs_plots.show()
            self.print_done(f"log-scale {mode} plot saved to {out_path}")

    def calculate_pure_eb(
        self,
        version,
        min_sep=None,
        max_sep=None,
        nbins=None,
        min_sep_int=0.08,
        max_sep_int=300,
        nbins_int=100,
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
            Number of bins for the integration binning. Defaults to 100.
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
        min_sep = min_sep or self.treecorr_config["min_sep"]
        max_sep = max_sep or self.treecorr_config["max_sep"]
        nbins = nbins or self.treecorr_config["nbins"]

        # Create TreeCorr configurations
        treecorr_config = {
            **self.treecorr_config,
            "min_sep": min_sep,
            "max_sep": max_sep,
            "nbins": nbins,
        }

        treecorr_config_int = {
            **treecorr_config,
            "min_sep": min_sep_int,
            "max_sep": max_sep_int,
            "nbins": nbins_int,
        }

        # Calculate correlation functions
        gg = self.calculate_2pcf(version, npatch=npatch, **treecorr_config)
        gg_int = self.calculate_2pcf(version, npatch=npatch, **treecorr_config_int)

        # Get redshift distribution if using analytic covariance
        if cov_path_int is not None:
            z, nz = self.get_redshift(version)
            z_dist = np.column_stack([z, nz])
        else:
            z_dist = None

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
        nbins_int=100,
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
            (default: 0.08-300 arcmin, 100 bins)
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

            # Calculate E/B statistics for all bin combinations (only if not provided)
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
        treecorr_config = {
            **self.treecorr_config,
            "min_sep": min_sep_int,
            "max_sep": max_sep_int,
            "nbins": nbins_int,
        }

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
            min_sep = min_sep or self.treecorr_config["min_sep"]
            max_sep = max_sep or self.treecorr_config["max_sep"]
            nbins = nbins or self.treecorr_config["nbins"]

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
            results = list(results.values())[0]

        return results

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
        if isinstance(results, dict) and all(
            isinstance(k, tuple) for k in results.keys()
        ):
            # Multiple scale cuts: use fiducial_scale_cut if provided, otherwise use
            # full range
            if fiducial_scale_cut is not None:
                plot_results = results[
                    find_conservative_scale_cut_key(results, fiducial_scale_cut)
                ]
            else:
                # Use full range result (largest scale cut)
                max_range_key = max(results.keys(), key=lambda x: x[1] - x[0])
                plot_results = results[max_range_key]
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
        if (
            isinstance(results, dict)
            and all(isinstance(k, tuple) for k in results.keys())
            and len(results) > 1
        ):
            # Create temporary gg object with correct binning for mapping
            treecorr_config_temp = {
                **self.treecorr_config,
                "min_sep": min_sep or self.treecorr_config["min_sep"],
                "max_sep": max_sep or self.treecorr_config["max_sep"],
                "nbins": nbins or self.treecorr_config["nbins"],
            }
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

    def summarize_bmodes(self, fiducial_scale_cut=(12, 83), versions=None):
        """Print and return B-mode PTE summary across all statistics.

        Collects PTEs from pure E/B, COSEBIs, and pseudo-Cl at the specified
        fiducial scale cut. Statistics that haven't been computed show '--'.

        Parameters
        ----------
        fiducial_scale_cut : tuple, optional
            (min_theta, max_theta) for extracting PTEs (default: (12, 83)).
        versions : list, optional
            Versions to summarize. Uses self.versions if None.

        Returns
        -------
        dict
            ``{version: {statistic: pte_value, ...}, ...}``
        """
        from scipy import stats as sp_stats

        versions = versions or self.versions
        summary = {}
        cov_methods = set()

        for ver in versions:
            row = {}

            # Pure E/B PTEs from stored results
            if ver in self._pure_eb_results:
                res = self._pure_eb_results[ver]
                gg = res["gg"]
                try:
                    for stat in ("xip_B", "xim_B", "combined"):
                        row[stat] = _get_pte_from_scale_cut(
                            res["pte_matrices"][stat], gg, fiducial_scale_cut
                        )
                except (KeyError, RuntimeError):
                    pass
                if "eb_samples" in res:
                    cov_methods.add("semi-analytic")
                else:
                    cov_methods.add(f"jackknife ({gg.npatch1} patches)")

            # COSEBIs PTE from stored results
            if ver in self._cosebis_results:
                cosebis_res = self._cosebis_results[ver]
                has_multi_scale_cuts = all(isinstance(k, tuple) for k in cosebis_res)
                if has_multi_scale_cuts:
                    key = find_conservative_scale_cut_key(
                        cosebis_res, fiducial_scale_cut
                    )
                    row["COSEBIS"] = cosebis_res[key]["pte_B"]
                elif "pte_B" in cosebis_res:
                    row["COSEBIS"] = cosebis_res["pte_B"]

            # Pseudo-Cl BB PTE (_pseudo_cls is lazy; check existence without
            # triggering computation)
            if hasattr(self, "_pseudo_cls") and ver in self._pseudo_cls:
                try:
                    cl_bb = self.pseudo_cls[ver]["pseudo_cl"]["BB"]
                    cov_bb = self.pseudo_cls[ver]["cov"]["COVAR_BB_BB"].data
                    chi2_bb = float(cl_bb @ np.linalg.solve(cov_bb, cl_bb))
                    row["C_l_BB"] = sp_stats.chi2.sf(chi2_bb, len(cl_bb))
                    cov_methods.add("Gaussian (NaMaster)")
                except (KeyError, AttributeError):
                    pass

            summary[ver] = row

        # Print summary table
        col_labels = {
            "xip_B": r"xi+B",
            "xim_B": r"xi-B",
            "combined": "Combined",
            "COSEBIS": "COSEBIS",
            "C_l_BB": "C_l^BB",
        }
        stats_order = list(col_labels)

        sc_label = f"[{fiducial_scale_cut[0]}-{fiducial_scale_cut[1]} arcmin]"
        sep = "\u2500" * 70
        header = f"{'Version':<28s}" + "".join(
            f"{label:>10s}" for label in col_labels.values()
        )

        print(f"\nB-mode summary {sc_label}")
        print(sep)
        print(header)
        print(sep)

        for ver in versions:
            row = summary[ver]
            cells = "".join(
                f"{row[s]:>10.4f}" if s in row else f"{'--':>10s}" for s in stats_order
            )
            print(f"{ver:<28s}{cells}")

        print(sep)
        if cov_methods:
            print(f"Covariance: {', '.join(sorted(cov_methods))}")
        print()

        return summary
