# %%
import copy
import os
from pathlib import Path
import configparser

import colorama
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pymaster as nmt
import treecorr
import yaml
from astropy.io import fits
from cs_util import plots as cs_plots
from shear_psf_leakage import leakage, run_object, run_scale
from shear_psf_leakage import plots as psfleak_plots
from shear_psf_leakage.rho_tau_stat import PSFErrorFit
from uncertainties import ufloat

from .cosmology import get_cosmo, get_theo_c_ell
from .plots import FootprintPlotter
from .rho_tau import (
    get_params_rho_tau,
    get_rho_tau_w_cov,
    get_samples,
)


# %%
class CosmologyValidation:
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
        nside_mask = 2**12,
        binning="powspace",
        power=1 / 2,
        n_ell_bins=32,
        ell_step=10,
        pol_factor=True,
        cell_method='map',
        noise_bias_method='analytic',
        fiducial_input_inka='coupled',
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

        assert self.cell_method in ["map", "catalog"], "cell_method must be 'map' or 'catalog'"
        assert self.noise_bias_method in ["analytic", "randoms"], "noise_bias_method must be 'analytical' or 'randoms'"
        assert self.fiducial_input_inka in ["coupled", "decoupled"], "fiducial_input_inka must be 'coupled' or 'decoupled'"

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
                if "e1_col_corrected" not in shear_cfg or "e2_col_corrected" not in shear_cfg:
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
            redshift_path = re.sub(r'_[ABC]\.txt$', f'_{self.blind}.txt', redshift_path)

        return np.loadtxt(redshift_path, unpack=True)

    def compute_survey_stats(
        self,
        ver,
        weights_key_override=None,
        mask_path=None,
        nside=None,
        overwrite_config=False,
    ):
        """Compute effective survey statistics for a catalog version.

        Parameters
        ----------
        ver : str
            Version string registered in the catalog config.
        weights_key_override : str, optional
            Override the weight column key (defaults to the configured `w_col`).
        mask_path : str, optional
            Explicit mask path to use when measuring survey area.
        nside : int, optional
            If provided, compute survey area from the catalog using this NSIDE when no
            mask path is available.
        overwrite_config : bool, optional
            If True, persist the derived statistics back to the catalog configuration.

        Returns
        -------
        dict
            Dictionary containing:
            - area_deg2: Survey area in square degrees.
            - n_eff: Effective number density per arcmin^2.
            - sigma_e: Per-component shape noise.
            - sum_w: Sum of weights.
            - sum_w2: Sum of squared weights.
            - catalog_size: Number of galaxies processed.
        """
        if ver not in self.cc:
            raise KeyError(f"Version {ver} not found in catalog configuration")

        shear_cfg = self.cc[ver]["shear"]
        cov_th = self.cc[ver].get("cov_th", {})

        if "path" not in shear_cfg:
            raise KeyError(f"No shear catalog path defined for version {ver}")

        catalog_path = shear_cfg["path"]
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Shear catalog not found: {catalog_path}")

        data = fits.getdata(catalog_path, memmap=True)
        n_rows = len(data)

        e1 = np.asarray(data[shear_cfg["e1_col"]], dtype=float)
        e2 = np.asarray(data[shear_cfg["e2_col"]], dtype=float)

        weight_column = weights_key_override or shear_cfg["w_col"]
        if weight_column not in data.columns.names:
            raise KeyError(f"Weight column '{weight_column}' missing in {catalog_path}")

        w = np.asarray(data[weight_column], dtype=float)

        sum_w = float(np.sum(w))
        sum_w2 = float(np.sum(w**2))
        sum_w2_e2 = float(np.sum((w**2) * (e1**2 + e2**2)))

        if mask_path is not None:
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask path not found: {mask_path}")
            mask_candidate = mask_path
        else:
            mask_candidate = self.cc[ver].get("mask")
            if isinstance(mask_candidate, str) and not os.path.isabs(mask_candidate):
                mask_candidate = str(Path(self.cc[ver]["subdir"]) / mask_candidate)
            if mask_candidate is not None and not os.path.exists(mask_candidate):
                mask_candidate = None

        area_deg2 = None
        if mask_candidate is not None and os.path.exists(mask_candidate):
            area_deg2 = self._area_from_mask(mask_candidate)
        elif cov_th.get("A") is not None:
            area_deg2 = float(cov_th["A"])
        elif nside is not None:
            area_deg2 = self._area_from_catalog(catalog_path, nside)
        else:
            raise ValueError(
                f"Unable to determine survey area for {ver}. Provide mask_path or nside."
            )

        area_arcmin2 = area_deg2 * 3600.0

        n_eff = (sum_w**2) / (area_arcmin2 * sum_w2) if sum_w2 > 0 else 0.0
        sigma_e = np.sqrt(sum_w2_e2 / sum_w2) if sum_w2 > 0 else 0.0

        results = {
            "area_deg2": area_deg2,
            "n_eff": n_eff,
            "sigma_e": sigma_e,
            "sum_w": sum_w,
            "sum_w2": sum_w2,
            "catalog_size": n_rows,
        }

        if overwrite_config:
            if "cov_th" not in self.cc[ver]:
                self.cc[ver]["cov_th"] = {}
            self.cc[ver]["cov_th"]["A"] = float(area_deg2)
            self.cc[ver]["cov_th"]["n_e"] = float(n_eff)
            self.cc[ver]["cov_th"]["sigma_e"] = float(sigma_e)
            self._write_catalog_config()

        return results

    def _area_from_catalog(self, catalog_path, nside):
        data = fits.getdata(catalog_path, memmap=True)
        ra = np.asarray(data["RA"], dtype=float)
        dec = np.asarray(data["Dec"], dtype=float)
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        pix = hp.ang2pix(nside, theta, phi, lonlat=False)
        unique_pix = np.unique(pix)
        return float(unique_pix.size * hp.nside2pixarea(nside, degrees=True))

    def _area_from_mask(self, mask_map_path):
        mask = hp.read_map(mask_map_path, dtype=np.float64)
        return float(mask.sum() * hp.nside2pixarea(hp.get_nside(mask), degrees=True))

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


    def calculate_rho_tau_stats(self):
        out_dir = f"{self.cc['paths']['output']}/rho_tau_stats"
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        self.print_start("Rho stats")
        for ver in self.versions:
            base = self.basename(ver)
            rho_stat_handler, tau_stat_handler = get_rho_tau_w_cov(
                self.cc,
                ver,
                self.treecorr_config,
                out_dir,
                base,
                method=self.cov_estimate_method,
                cov_rho=self.compute_cov_rho,
                npatch=self.npatch,
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
    
    @property
    def area(self):
        if not hasattr(self, "_area"):
            self.calculate_area()
        return self._area
    
    @property
    def n_eff_gal(self):
        if not hasattr(self, "_n_eff_gal"):
            self.calculate_n_eff_gal()
        return self._n_eff_gal

    @property
    def ellipticity_dispersion(self):
        if not hasattr(self, "_ellipticity_dispersion"):
            self.calculate_ellipticity_dispersion()
        return self._ellipticity_dispersion

    @property
    def pseudo_cls(self):
        if not hasattr(self, "_pseudo_cls"):
            self.calculate_pseudo_cl()
            self.calculate_pseudo_cl_eb_cov()
        return self._pseudo_cls
    
    @property
    def pseudo_cls_onecov(self):
        if not hasattr(self, "_pseudo_cls_onecov"):
            self.calculate_pseudo_cl_onecovariance()
        return self._pseudo_cls_onecov

    def calculate_area(self):
        self.print_start("Calculating area")
        area = {}
        for ver in self.versions:
            self.print_magenta(ver)

            if 'mask' not in self.cc[ver]:
                print("Mask not found in config file, calculating area from binned catalog")
                area[ver] = self.calculate_area_from_binned_catalog(ver)
            else:
                mask = hp.read_map(self.cc[ver]['mask'], verbose=False)
                nside_mask = hp.get_nside(mask)
                print(f"nside_mask = {nside_mask}")
                area[ver] = np.sum(mask) * hp.nside2pixarea(nside_mask, degrees=True)
            print(f"Area = {area[ver]:.2f} deg^2")

        self._area = area
        self.print_done("Area calculation finished")
    
    def calculate_area_from_binned_catalog(self, ver):
        print(f"nside_mask = {self.nside_mask}")
        with self.results[ver].temporarily_read_data():
            ra = self.results[ver].dat_shear["RA"]
            dec = self.results[ver].dat_shear["Dec"]
            hsp_map = hp.ang2pix(
            self.nside_mask,
                np.radians(90 - dec),
                np.radians(ra),
                lonlat=False,
            )
            mask = np.bincount(hsp_map, minlength=hp.nside2npix(self.nside_mask)) > 0

            area = np.sum(mask) * hp.nside2pixarea(self.nside_mask, degrees=True)
            print(f"Area = {area:.2f} deg^2")

        return area

    def calculate_n_eff_gal(self):
        self.print_start("Calculating effective number of galaxy")
        n_eff_gal = {}
        for ver in self.versions:
            self.print_magenta(ver)
            with self.results[ver].temporarily_read_data():
                w = self.results[ver].dat_shear[self.cc[ver]["shear"]["w_col"]]
                n_eff_gal[ver] = 1/(self.area[ver]*60*60)* np.sum(w)**2/np.sum(w**2)
                print(f"n_eff_gal = {n_eff_gal[ver]:.2f} gal./arcmin^-2")
            
        self._n_eff_gal = n_eff_gal
        self.print_done("Effective number of galaxy calculation finished")

    def calculate_ellipticity_dispersion(self):
        self.print_start("Calculating ellipticity dispersion")
        ellipticity_dispersion = {}
        for ver in self.versions:
            self.print_magenta(ver)
            with self.results[ver].temporarily_read_data():
                e1 = self.results[ver].dat_shear[self.cc[ver]["shear"]["e1_col"]]
                e2 = self.results[ver].dat_shear[self.cc[ver]["shear"]["e2_col"]]
                w = self.results[ver].dat_shear[self.cc[ver]["shear"]["w_col"]]
                ellipticity_dispersion[ver] = np.sqrt(
                    0.5*(np.average(e1**2, weights=w**2) + np.average(e2**2, weights=w**2))
                )
                print(f"Ellipticity dispersion = {ellipticity_dispersion[ver]:.4f}")
        self._ellipticity_dispersion = ellipticity_dispersion

    def plot_rho_stats(self, abs=False):
        filenames = [
            f"rho_stats_{self.basename(ver)}.fits" for ver in self.versions
        ]

        savefig = "rho_stats.png"
        self.rho_stat_handler.plot_rho_stats(
            filenames,
            self.colors,
            self.versions,
            savefig=savefig,
            legend="outside",
            abs=abs,
            show=True,
            close=True,
        )

        self.print_done(
            "Rho stats plot saved to "
            + f"{os.path.abspath(self.rho_stat_handler.catalogs._output)}/{savefig}",
        )

    def plot_tau_stats(self, plot_tau_m=False):
        filenames = [
            f"tau_stats_{self.basename(ver)}.fits" for ver in self.versions
        ]

        savefig = "tau_stats.png"
        self.tau_stat_handler.plot_tau_stats(
            filenames,
            self.colors,
            self.versions,
            savefig=savefig,
            legend="outside",
            plot_tau_m=plot_tau_m,
            show=True,
            close=True,
        )

        self.print_done(
            "Tau stats plot saved to "
            + f"{os.path.abspath(self.tau_stat_handler.catalogs._output)}/{savefig}",
        )

    def set_params_rho_tau(self, params, params_psf, survey="other"):
        params = {**params}
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

        params["ra_PSF_col"] = params_psf["ra_col"]
        params["dec_PSF_col"] = params_psf["dec_col"]
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

        params["w_col"] = self.cc[survey]["shear"]["w_col"]

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

            base = self.basename(ver)

            flat_samples, result, q = get_samples(
                self.psf_fitter,
                ver,
                base,
                cov_type=self.cov_estimate_method,
                apply_debias=npatch,
                sampler=self.rho_tau_method,
            )

            self.rho_tau_fits["flat_sample_list"].append(flat_samples)
            self.rho_tau_fits["result_list"].append(result)
            self.rho_tau_fits["q_list"].append(q)

            self.psf_fitter.load_rho_stat(f"rho_stats_{self.basename(ver)}.fits")
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

    @property
    def xi_psf_sys(self):
        if not hasattr(self, "_xi_psf_sys"):
            self.calculate_rho_tau_fits()
        return self._xi_psf_sys

    def plot_footprints(self):
        self.print_start("Plotting footprints:")
        for ver in self.versions:
            self.print_magenta(ver)

            fp = FootprintPlotter()

            for region in fp._regions:
                out_path = os.path.abspath(
                    f"{self.cc['paths']['output']}/footprint_{ver}_{region}.png"
                )
            if os.path.exists(out_path):
                self.print_done(f"Skipping footprint plot, {out_path} exists")
            else:
                with self.results[ver].temporarily_read_data():
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
                f"{self.cc['paths']['output']}/leakage_{ver}/xi_for_leak_scale"
            )
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
            cs_plots.savefig(out_path, close_fig=False)
            cs_plots.show()
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
            out_path = os.path.abspath(f"{self.cc['paths']['output']}/xi_sys_p.png")
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
            out_path = os.path.abspath(f"{self.cc['paths']['output']}/xi_sys_m.png")
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
        out_path = os.path.abspath(
            f"{self.cc['paths']['output']}/leakage_coefficients.png"
        )
        cs_plots.savefig(out_path, close_fig=False)
        cs_plots.show()
        self.print_done(f"Object-wise leakage coefficients plot saved to {out_path}")

    def plot_ellipticity(self, nbins=200):
        out_path = os.path.abspath(f"{self.cc['paths']['output']}/ell_hist.png")
        if os.path.exists(out_path):
            self.print_done(f"Skipping ellipticity histograms, {out_path} exists")
        else:
            self.print_start("Computing ellipticity histograms:")

            fig, axs = plt.subplots(1, 2, figsize=(22, 7))
            bins = np.linspace(-1.1, 1.1, nbins + 1)
            for ver in self.versions:
                self.print_magenta(ver)
                R = self.cc[ver]["shear"]["R"]
                with self.results[ver].temporarily_read_data():
                    e1 = (
                        self.results[ver].dat_shear[self.cc[ver]["shear"]["e1_col"]] / R
                    )
                    e2 = (
                        self.results[ver].dat_shear[self.cc[ver]["shear"]["e2_col"]] / R
                    )
                    w = self.results[ver].dat_shear[self.cc[ver]["shear"]["w_col"]]

                    axs[0].hist(
                        e1,
                        bins=bins,
                        density=True,
                        histtype="step",
                        weights=w,
                        label=ver,
                        color=self.cc[ver]["colour"],
                    )
                    axs[1].hist(
                        e2,
                        bins=bins,
                        density=True,
                        histtype="step",
                        weights=w,
                        label=ver,
                        color=self.cc[ver]["colour"],
                    )

            for idx in (0, 1):
                axs[idx].set_xlabel(f"$e_{idx}$")
                axs[idx].set_ylabel("normalised count")
                axs[idx].legend()
                axs[idx].set_xlim([-1.5, 1.5])
            cs_plots.savefig(out_path, close_fig=False)
            cs_plots.show()
            self.print_done("Ellipticity histograms saved to " + out_path)

    def plot_weights(self, nbins=200):
        out_path = os.path.abspath(f"{self.cc['paths']['output']}/weight_hist.png")
        if os.path.exists(out_path):
            self.print_done(f"Skipping weight histograms, {out_path} exists")
        else:
            self.print_start("Computing weight histograms:")

            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            for ver in self.versions:
                self.print_magenta(ver)
                with self.results[ver].temporarily_read_data():
                    w = self.results[ver].dat_shear[self.cc[ver]["shear"]["w_col"]]

                    plt.hist(
                        w,
                        bins=nbins,
                        density=False,
                        histtype="step",
                        weights=w,
                        label=ver,
                        color=self.cc[ver]["colour"],
                    )

            plt.xlabel("$w$")
            plt.ylabel("frequency")
            plt.yscale("log")
            plt.legend()
            # plt.xlim([-0.01, 1.2])
            cs_plots.savefig(out_path, close_fig=False)
            cs_plots.show()
            self.print_done("Weight histograms saved to " + out_path)

    def plot_separation(self, nbins=200):
        self.print_start("Separation histograms")
        if "SP_matched_MP_v1.0" in self.versions:
            fig, axs = plt.subplots(1, 1, figsize=(10, 7))
            with self.results["SP_matched_MP_v1.0"].temporarily_read_data():
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
            R = self.cc[ver]["shear"]["R"]
            e1_col, e2_col, w_col = [
                self.cc[ver]["shear"][k] for k in ["e1_col", "e2_col", "w_col"]
            ]
            with self.results[ver].temporarily_read_data():
                self._c1[ver] = np.average(
                    self.results[ver].dat_shear[e1_col] / R,
                    weights=self.results[ver].dat_shear[w_col],
                )
                self._c2[ver] = np.average(
                    self.results[ver].dat_shear[e2_col] / R,
                    weights=self.results[ver].dat_shear[w_col],
                )
        self.print_done("Finished additive bias calculation.")

    @property
    def c1(self):
        if not hasattr(self, "_c1"):
            self.calculate_additive_bias()
        return self._c1

    @property
    def c2(self):
        if not hasattr(self, "_c2"):
            self.calculate_additive_bias()
        return self._c2

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
                col5 = fits.Column(
                    name="ANG", format="D", unit="arcmin", array=gg.rnom
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
            jittered_theta = theta * (1+idx*offset)

            plt.errorbar(
                jittered_theta,
                ratio,
                yerr=ratio_err,
                label=ver,
                ls=self.cc[ver]["ls"],
                color=self.cc[ver]["colour"],
                fmt=self.cc[ver].get("marker", None),
                capsize=5
            )

        plt.fill_between(
            [self.theta_min_plot, self.theta_max_plot],
            - threshold,
            + threshold,
            color="black",
            alpha=0.1,
            label=f"{threshold:.0%} threshold",
        )
        plt.plot(
            [self.theta_min_plot, self.theta_max_plot],
            [threshold, threshold],
            ls="dashed",
            color="black")
        plt.plot(
            [self.theta_min_plot, self.theta_max_plot],
            [-threshold, -threshold],
            ls="dashed",
            color="black")
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
        from .b_modes import calculate_pure_eb_correlation

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
            z_dist=z_dist
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
        **kwargs
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
        from .b_modes import (
            calculate_eb_statistics,
            plot_eb_covariance_matrix,
            plot_integration_vs_reporting,
            plot_pte_2d_heatmaps,
            plot_pure_eb_correlations,
            save_pure_eb_results,
        )

        # Use instance defaults for unspecified parameters
        versions = versions or self.versions
        output_dir = output_dir or self.cc['paths']['output']
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
                **kwargs
            )

            # Generate all plots using specialized plotting functions
            gg, gg_int = version_results["gg"], version_results["gg_int"]

            # Integration vs Reporting comparison plot
            plot_integration_vs_reporting(
                gg, gg_int,
                out_stub + "_integration_vs_reporting.png",
                version
            )

            # E/B/Ambiguous correlation functions plot
            plot_pure_eb_correlations(
                version_results,
                out_stub + "_xis.png",
                version,
                fiducial_xip_scale_cut=fiducial_xip_scale_cut,
                fiducial_xim_scale_cut=fiducial_xim_scale_cut
            )

            # 2D PTE heatmaps plot
            plot_pte_2d_heatmaps(
                version_results,
                version,
                out_stub + "_ptes.png",
                fiducial_xip_scale_cut=fiducial_xip_scale_cut,
                fiducial_xim_scale_cut=fiducial_xim_scale_cut
            )

            # Covariance matrix plot
            plot_eb_covariance_matrix(
                version_results["cov"],
                var_method,
                out_stub + "_covariance.png",
                version
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
        from .b_modes import calculate_cosebis

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
                for stop in range(start+1, nbins+1)
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
        min_sep_int=0.5, max_sep_int=500, nbins_int=1000,  # Integration binning
        npatch=None, nmodes=10, cov_path=None,
        scale_cuts=None,                                   # Explicit scale cuts
        evaluate_all_scale_cuts=False,                     # Grid-based scale cuts
        min_sep=None, max_sep=None, nbins=None,           # Reporting binning
        fiducial_scale_cut=None,                          # For plotting reference
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
        from .b_modes import (
            find_conservative_scale_cut_key,
            plot_cosebis_covariance_matrix,
            plot_cosebis_modes,
            plot_cosebis_scale_cut_heatmap,
            save_cosebis_results,
        )

        # Use instance defaults if not specified
        version = version or self.versions[0]
        output_dir = output_dir or self.cc['paths']['output']
        npatch = npatch or self.treecorr_config.get('npatch', 256)

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
        if (isinstance(results, dict) and
            all(isinstance(k, tuple) for k in results.keys())):
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
            fiducial_scale_cut=fiducial_scale_cut
        )

        plot_cosebis_covariance_matrix(
            plot_results,
            version,
            var_method,
            out_stub + "_covariance.png"
        )

        # Generate scale cut heatmap if we have multiple scale cuts
        if (isinstance(results, dict) and
            all(isinstance(k, tuple) for k in results.keys()) and
            len(results) > 1):
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
                fiducial_scale_cut=fiducial_scale_cut
            )

        # Save data products and store on instance
        save_cosebis_results(results, out_stub + "_data.npz", fiducial_scale_cut)
        self._cosebis_results[version] = results


    def get_namaster_bin(self, lmin, lmax, b_lmax):
        """Build NaMaster binning object.

        Parameters
        ----------
        lmin, lmax : int
            Multipole range.
        b_lmax : int
            Maximum multipole for the NmtBin object.

        Returns
        -------
        nmt.NmtBin
        """
        ells = np.arange(lmin, lmax + 1)

        if self.binning == 'linear':
            bpws = (ells - lmin) // self.ell_step
            bpws = np.minimum(bpws, bpws[-1])
            b = nmt.NmtBin(ells=ells, bpws=bpws, lmax=b_lmax)
        elif self.binning == 'logspace':
            # Start geomspace at ell_min_log (>= lmin) to avoid
            # sub-multipole bins at low ell that destabilize the MCM.
            # All ell below ell_min_log go into bin 0 as padding.
            ell_min_log = max(lmin, 50)
            bins_ell = np.geomspace(ell_min_log, lmax, self.n_ell_bins + 1)
            bpws = np.digitize(ells.astype(float), bins_ell) - 1
            bpws = np.clip(bpws, 0, self.n_ell_bins - 1)
            b = nmt.NmtBin(ells=ells, bpws=bpws, lmax=b_lmax)
        elif self.binning == 'powspace':
            start = np.power(lmin, self.power)
            end = np.power(lmax, self.power)
            bins_ell = np.power(
                np.linspace(start, end, self.n_ell_bins + 1), 1 / self.power
            )
            bpws = np.digitize(ells.astype(float), bins_ell) - 1
            bpws[0] = 0
            bpws[-1] = self.n_ell_bins - 1
            b = nmt.NmtBin(ells=ells, bpws=bpws, lmax=b_lmax)
        else:
            raise ValueError(
                f"Unknown binning '{self.binning}'. "
                "Choose from 'linear', 'logspace', 'powspace'."
            )

        return b

    def get_variance_map(self, nside, e1, e2, w, unique_pix, idx_rep):
        """
        Create a variance map from the input catalog.
        """
        
        variance_map = np.zeros(hp.nside2npix(nside))

        variance_map[unique_pix] = np.bincount(
            idx_rep, weights=(e1**2 + e2**2)/2*w**2
        )

        return variance_map
    
    def get_field_and_workspace_from_map(self, mask, lmax, b):
        """
        Create a NaMaster field and workspace from the input map.
        """

        nside = hp.npix2nside(len(mask))

        #Create NaMaster field
        f = nmt.NmtField(mask=mask, maps=[np.zeros(hp.nside2npix(nside)), np.zeros(hp.nside2npix(nside))], lmax=lmax)

        #Create NaMaster workspace
        wsp = nmt.NmtWorkspace.from_fields(f, f, b)

        return f, wsp

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

            if ver not in self._pseudo_cls.keys():
                self._pseudo_cls[ver] = {}

            out_path = os.path.abspath(f"{self.cc['paths']['output']}/pseudo_cl_cov_{ver}.fits")
            if os.path.exists(out_path):
                self.print_done(f"Skipping Pseudo-Cl covariance calculation, {out_path} exists")
                self._pseudo_cls[ver]['cov'] = fits.open(out_path)
            else:
                params = get_params_rho_tau(self.cc[ver], survey=ver)

                self.print_cyan(f"Extracting the fiducial power spectrum for {ver}")

                lmax = 2 * self.nside
                z, dndz = self.get_redshift(ver)
                ell = np.arange(1, lmax + 1)
                pw = hp.pixwin(nside, lmax=lmax)
                if pw.shape[0] != len(ell) + 1:
                    raise ValueError(
                        "Unexpected pixwin length for lmax="
                        f"{lmax}: got {pw.shape[0]}, expected {len(ell)+1}"
                    )
                pw = pw[1:len(ell)+1]

                # Calculate theory C_ell with pixel window
                fiducial_cl = (
                    get_theo_c_ell(
                        ell=ell,
                        z=z,
                        nz=dndz,
                        backend="camb",
                        cosmo=self.cosmo,
                    )
                    * pw**2
                )

                self.print_cyan("Getting a binning, n_gal_map, field and workspace.")

                lmin = 8
                lmax = 2*self.nside
                b_lmax = lmax - 1

                b = self.get_namaster_bin(lmin, lmax, b_lmax)

                #Load data and create shear and noise maps
                cat_gal = fits.getdata(self.cc[ver]["shear"]["path"])

                n_gal, unique_pix, idx, idx_rep = self.get_n_gal_map(params, nside, cat_gal)
                mask = n_gal != 0

                f, wsp = self.get_field_and_workspace_from_map(n_gal, b_lmax, b)

                if self.noise_bias_method == 'randoms':

                    self.print_cyan("Getting a sample of Cls with noise bias.")
                    
                    cl_noise, f, wsp = self.get_sample(params, self.nside, b_lmax, b, cat_gal, n_gal, n_gal, unique_pix, idx_rep)

                    cl_noise = np.mean(cl_noise, axis=0)
                    noise_bias_cl = cl_noise
                    noise_bias_cl = b.unbin_cell(noise_bias_cl) #Unbin
                    lowest_ell = b.get_ell_list(0)[0]
                    for i in range(4):
                        noise_bias_cl[i, :lowest_ell] = noise_bias_cl[i, lowest_ell] #Fill the data vector below lmin
                
                elif self.noise_bias_method == 'analytic':

                    self.print_cyan("Getting analytic noise bias.")

                    e1, e2, w = cat_gal[self.cc[ver]["shear"]["e1_col"]], cat_gal[self.cc[ver]["shear"]["e2_col"]], cat_gal[self.cc[ver]["shear"]["w_col"]]
                    variance_map = self.get_variance_map(self.nside, e1, e2, w, unique_pix, idx_rep)

                    noise_bias = hp.nside2pixarea(self.nside)*np.mean(variance_map)

                    noise_bias_cl = np.zeros((4, lmax))
                    noise_bias_cl[0, :] = noise_bias
                    noise_bias_cl[3, :] = noise_bias

                    noise_bias_cl = wsp.decouple_cell(noise_bias_cl) #Decouple
                    noise_bias_cl = b.unbin_cell(noise_bias_cl) #Unbin
                    lowest_ell = b.get_ell_list(0)[0]
                    for i in range(4):
                        noise_bias_cl[i, :lowest_ell] = noise_bias_cl[i, lowest_ell] #Fill the data vector below lmin
                
                else:
                    raise ValueError(f"Noise bias method {self.noise_bias_method} not recognized. It should be 'randoms' or 'analytic'.")
                
                self.print_cyan("Adding noise bias to the fiducial Cls.")
                
                fiducial_cl = np.array([fiducial_cl, 0.*fiducial_cl, 0.*fiducial_cl, 0.*fiducial_cl])+ noise_bias_cl

                if self.fiducial_input_inka == 'coupled':
                    self.print_cyan("Coupling the fiducial Cls.")

                    coupling_mat = wsp.get_coupling_matrix()
                    coupling_mat_re = np.reshape(coupling_mat, (4, lmax, 4, lmax), order='F')
                    fiducial_cl = np.tensordot(
                        coupling_mat_re, fiducial_cl
                    ) / np.mean(n_gal**2) #couple and divide by the mean of the mask squared

                
                self.print_cyan("Computing the Pseudo-Cl covariance")

                cw = nmt.NmtCovarianceWorkspace.from_fields(f, f, f, f)

                # Get actual number of ell bins from binning scheme
                n_ell_actual = b.get_n_bands()

                covar_22_22 = nmt.gaussian_covariance(cw, 2, 2, 2, 2,
                                                        fiducial_cl,
                                                        fiducial_cl,
                                                        fiducial_cl,
                                                        fiducial_cl,
                                                        wsp, wb=wsp).reshape([n_ell_actual, 4, n_ell_actual, 4])

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

    def calculate_pseudo_cl_onecovariance(self):
        """
        Compute the pseudo-Cl covariance using OneCovariance.
        """
        self.print_start("Computing Pseudo-Cl covariance with OneCovariance")

        if self.path_onecovariance is None:
            raise ValueError("path_onecovariance must be provided to use OneCovariance")
        
        if not os.path.exists(self.path_onecovariance):
            raise ValueError(f"OneCovariance path {self.path_onecovariance} does not exist")
        
        template_config = os.path.join(self.path_onecovariance, "config_files", "config_3x2pt_pure_Cell_UNIONS.ini")
        if not os.path.exists(template_config):
            raise ValueError(f"Template config file {template_config} does not exist")

        self._pseudo_cls_onecov = {}
        for ver in self.versions:
            self.print_magenta(ver)

            out_dir = os.path.abspath(f"{self.cc['paths']['output']}/pseudo_cl_cov_onecov_{ver}/")
            os.makedirs(out_dir, exist_ok=True)

            if os.path.exists(os.path.join(out_dir, "covariance_list_3x2pt_pure_Cell.dat")):
                self.print_done(f"Skipping OneCovariance calculation, {out_dir} exists")
                self._load_onecovariance_cov(out_dir, ver)
            else:

                mask_path = self.cc[ver]['mask']
                redshift_distr_path = self.cc[ver]['shear']['redshift_path']

                config_path = os.path.join(out_dir, f"config_onecov_{ver}.ini")

                self.print_cyan(f"Modifying OneCovariance config file and saving it to {config_path}")
                self._modify_onecov_config(template_config, config_path, out_dir, mask_path, redshift_distr_path, ver)

                self.print_cyan("Running OneCovariance...")
                cmd = f"python {os.path.join(self.path_onecovariance, 'covariance.py')} {config_path}"
                self.print_cyan(f"Command: {cmd}")
                ret = os.system(cmd)
                if ret != 0:
                    raise RuntimeError(f"OneCovariance command failed with return code {ret}")
                self.print_cyan("OneCovariance completed successfully.")
                self._load_onecovariance_cov(out_dir, ver)
            
        self.print_done("Done Pseudo-Cl covariance with OneCovariance")

    def _modify_onecov_config(self, template_config, config_path, out_dir, mask_path, redshift_distr_path, ver):
        """
        Modify OneCovariance configuration file with correct mask, redshift distribution, 
        and ellipticity dispersion parameters.
        
        Parameters
        ----------
        template_config : str
            Path to the template configuration file
        config_path : str
            Path where the modified configuration will be saved
        mask_path : str
            Path to the mask file
        redshift_distr_path : str
            Path to the redshift distribution file
        """
        config = configparser.ConfigParser()        
        # Load the template configuration
        config.read(template_config)
        
        # Update mask path
        mask_base = os.path.basename(os.path.abspath(mask_path))
        mask_folder = os.path.dirname(os.path.abspath(mask_path))
        config['survey specs']['mask_directory'] = mask_folder
        config['survey specs']['mask_file_lensing'] = mask_base
        config['survey specs']['survey_area_lensing_in_deg2'] = str(self.area[ver])
        config['survey specs']['ellipticity_dispersion'] = str(self.ellipticity_dispersion[ver])
        config['survey specs']['n_eff_lensing'] = str(self.n_eff_gal[ver])

        # Update redshift distribution path
        redshift_distr_base = os.path.basename(os.path.abspath(redshift_distr_path))
        redshift_distr_folder = os.path.dirname(os.path.abspath(redshift_distr_path))
        config['redshift']['z_directory'] = redshift_distr_folder
        config['redshift']['zlens_file'] = redshift_distr_base

        #Update output directory
        config['output settings']['directory'] = out_dir
        
        # Save the modified configuration
        with open(config_path, 'w') as f:
            config.write(f)

    def get_cov_from_onecov(self, cov_one_cov, gaussian=True):
        n_bins = np.sqrt(cov_one_cov.shape[0]).astype(int)
        cov = np.zeros((n_bins, n_bins))

        index_value = 10 if gaussian else 9
        for i in range(n_bins):
            for j in range(n_bins):
                cov[i, j] = cov_one_cov[i * n_bins + j, index_value]
        
        return cov

    def _load_onecovariance_cov(self, out_dir, ver):
        self.print_cyan(f"Loading OneCovariance results from {out_dir}")
        cov_one_cov = np.genfromtxt(os.path.join(out_dir, "covariance_list_3x2pt_pure_Cell.dat"))
        gaussian_one_cov = self.get_cov_from_onecov(cov_one_cov, gaussian=True)
        all_one_cov = self.get_cov_from_onecov(cov_one_cov, gaussian=False)

        self._pseudo_cls_onecov[ver] = {
            'gaussian_cov': gaussian_one_cov,
            'all_cov': all_one_cov
        }

    def calculate_pseudo_cl_g_ng_cov(self, gaussian_part="iNKA"):
        assert gaussian_part in ["iNKA", "OneCovariance"], "gaussian_part must be 'iNKA' or 'OneCovariance'"
        self.print_start(f"Gaussian and Non-Gaussian covariance of the Pseudo-Cl's using {gaussian_part} for the Gaussian part")
    
        self._pseudo_cls_cov_g_ng = {}

        for ver in self.versions:
            self.print_magenta(ver)
            out_file = os.path.abspath(f"{self.cc['paths']['output']}/pseudo_cl_cov_g_ng_{gaussian_part}_{ver}.fits")
            if os.path.exists(out_file):
                self.print_done(f"Skipping Gaussian and Non-Gaussian covariance calculation, {out_file} exists")
                cov_hdu = fits.open(out_file)
                self._pseudo_cls_cov_g_ng[ver] = cov_hdu
                continue
            if gaussian_part == "iNKA":
                gaussian_cov = self.pseudo_cls[ver]['cov']['COVAR_EE_EE'].data
                non_gaussian_cov = self.pseudo_cls_onecov[ver]['all_cov'] - self.pseudo_cls_onecov[ver]['gaussian_cov']
                full_cov = gaussian_cov + non_gaussian_cov
            elif gaussian_part == "OneCovariance":
                gaussian_cov = self.pseudo_cls_onecov[ver]['gaussian_cov']
                non_gaussian_cov = self.pseudo_cls_onecov[ver]['all_cov'] - self.pseudo_cls_onecov[ver]['gaussian_cov']
                full_cov = self.pseudo_cls_onecov[ver]['all_cov']
            else:
                raise ValueError(f"Unknown gaussian_part: {gaussian_part}")
            self.print_cyan("Saving Gaussian and Non-Gaussian covariance...")
            hdu = fits.HDUList()
            hdu.append(fits.ImageHDU(gaussian_cov, name="COVAR_GAUSSIAN"))
            hdu.append(fits.ImageHDU(non_gaussian_cov, name="COVAR_NON_GAUSSIAN"))
            hdu.append(fits.ImageHDU(full_cov, name="COVAR_FULL"))
            hdu.writeto(out_file, overwrite=True)
            self._pseudo_cls_cov_g_ng[ver] = hdu
        self.print_done(f"Done Gaussian and Non-Gaussian covariance of the Pseudo-Cl's using {gaussian_part} for the Gaussian part")                

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
            elif self.cell_method == 'map':
                self.calculate_pseudo_cl_map(ver, nside, out_path)
            elif self.cell_method == 'catalog':
                self.calculate_pseudo_cl_catalog(ver, out_path)
            else:
                raise ValueError(f"Unknown cell method: {self.cell_method}")

        self.print_done("Done pseudo-Cl's")

    def calculate_pseudo_cl_map(self, ver, nside, out_path):
        params = get_params_rho_tau(self.cc[ver], survey=ver)

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

        ell_eff, cl_shear, wsp = self.get_pseudo_cls_map(shear_map, n_gal_map)

        cl_noise = np.zeros_like(cl_shear)
        
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

            _, cl_noise_, _ = self.get_pseudo_cls_map(noise_map, n_gal_map, wsp)
            cl_noise += cl_noise_
        
        cl_noise /= self.nrandom_cell
        del e1, e2, w
        try:
            del e1_rot, e2_rot
        except NameError: #Continue if the random generation has been skipped.
            pass
        del n_gal_map              

        #This is a problem because the measurement depends on the seed. To be fixed.
        #cl_shear = cl_shear - np.mean(cl_noise, axis=1, keepdims=True)
        cl_shear = cl_shear - cl_noise

        self.print_cyan("Saving pseudo-Cl's...")
        self.save_pseudo_cl(ell_eff, cl_shear, out_path)

        cl_shear = fits.getdata(out_path)
        self._pseudo_cls[ver]['pseudo_cl'] = cl_shear

    def calculate_pseudo_cl_catalog(self, ver, out_path):
        params = get_params_rho_tau(self.cc[ver], survey=ver)

        #Load data and create shear and noise maps
        cat_gal = fits.getdata(self.cc[ver]["shear"]["path"])

        ell_eff, cl_shear, wsp = self.get_pseudo_cls_catalog(catalog=cat_gal, params=params)

        self.print_cyan("Saving pseudo-Cl's...")
        self.save_pseudo_cl(ell_eff, cl_shear, out_path)

        cl_shear = fits.getdata(out_path)
        self._pseudo_cls[ver]['pseudo_cl'] = cl_shear

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

    def get_gaussian_real(
        self, params, nside, lmax, cat_gal, n_gal, mask, unique_pix, idx_rep
    ):
        e1_rot, e2_rot = self.apply_random_rotation(
            cat_gal[params["e1_col"]], cat_gal[params["e2_col"]]
        )
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
    
    def get_pseudo_cls_map(self, map, mask, wsp=None):
        """
        Compute the pseudo-cl for a given map.
        """

        lmin = 8
        lmax = 2*self.nside
        b_lmax = lmax - 1

        b = self.get_namaster_bin(lmin, lmax, b_lmax)

        ell_eff = b.get_effective_ells()

        factor = -1 if self.pol_factor else 1

        f_all = nmt.NmtField(mask=mask, maps=[map.real, factor*map.imag], lmax=b_lmax)
        if wsp is None:
            wsp = nmt.NmtWorkspace.from_fields(f_all, f_all, b)
        
        cl_coupled = nmt.compute_coupled_cell(f_all, f_all)
        cl_all = wsp.decouple_cell(cl_coupled)

        return ell_eff, cl_all, wsp

    def get_pseudo_cls_catalog(self, catalog, params, wsp=None):
        """
        Compute the pseudo-cl for a given catalog.
        """

        lmin = 8
        lmax = 2*self.nside
        b_lmax = lmax - 1

        b = self.get_namaster_bin(lmin, lmax, b_lmax)

        ell_eff = b.get_effective_ells()

        factor = -1 if self.pol_factor else 1

        f_all = nmt.NmtFieldCatalog(positions=[catalog[params['ra_col']], catalog[params['dec_col']]],
                        weights=catalog[params['w_col']],
                        field=[catalog[params['e1_col']], factor*catalog[params['e2_col']]],
                        lmax=b_lmax,
                        lmax_mask=b_lmax,
                        spin=2,
                        lonlat=True)
        
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

        ax[0].set_ylabel(r"$\ell C_\ell$")

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

        ax[1].set_xlabel(r"$\ell$")
        ax[1].set_ylabel(r"$C_\ell$")

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

        ax[0].axhline(0, color="black", linestyle="--")
        ax[0].set_ylabel(r"$\ell C_\ell$")

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

        ax[1].set_xlabel(r"$\ell$")
        ax[1].set_ylabel(r"$C_\ell$")

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

        ax[0].axhline(0, color="black", linestyle="--")
        ax[0].set_ylabel(r"$\ell C_\ell$")

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

        ax[1].set_xlabel(r"$\ell$")
        ax[1].set_ylabel(r"$C_\ell$")

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

        # Print C_l^BB PTE for each version and save BB data
        from scipy import stats as sp_stats
        print("\nC_l^BB PTE summary:")
        for ver in self.versions:
            cl_bb = self.pseudo_cls[ver]['pseudo_cl']["BB"]
            cov_bb = self.pseudo_cls[ver]['cov']["COVAR_BB_BB"].data
            chi2_bb = float(cl_bb @ np.linalg.solve(cov_bb, cl_bb))
            pte_bb = sp_stats.chi2.sf(chi2_bb, len(cl_bb))
            print(
                f"  {ver}: C_l^BB PTE = {pte_bb:.4f} "
                f"(chi2/dof = {chi2_bb:.1f}/{len(cl_bb)})"
            )

            # Save BB data + covariance to .npz
            ell = self.pseudo_cls[ver]['pseudo_cl']["ELL"]
            bb_out = os.path.abspath(
                f"{self.cc['paths']['output']}/{ver}_cell_bb_data.npz"
            )
            np.savez(
                bb_out,
                ell=ell, cl_bb=cl_bb, cov_bb=cov_bb,
                chi2_bb=np.array(chi2_bb), pte_bb=np.array(pte_bb),
            )
            print(f"  Saved BB data to {bb_out}")

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
        from .b_modes import _get_pte_from_scale_cut, find_conservative_scale_cut_key

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
                has_multi_scale_cuts = all(
                    isinstance(k, tuple) for k in cosebis_res
                )
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
                    cl_bb = self.pseudo_cls[ver]['pseudo_cl']["BB"]
                    cov_bb = self.pseudo_cls[ver]['cov']["COVAR_BB_BB"].data
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
                f"{row[s]:>10.4f}" if s in row else f"{'--':>10s}"
                for s in stats_order
            )
            print(f"{ver:<28s}{cells}")

        print(sep)
        if cov_methods:
            print(f"Covariance: {', '.join(sorted(cov_methods))}")
        print()

        return summary
