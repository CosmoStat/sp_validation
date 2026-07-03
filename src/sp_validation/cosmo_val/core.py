# %%
import copy
import itertools
import os
import re
from pathlib import Path

import colorama
import numpy as np
import yaml
from astropy.io import fits
from shear_psf_leakage import run_object, run_scale

from ..b_modes import (
    _get_pte_from_scale_cut,
    find_conservative_scale_cut_key,
)
from ..cosmology import get_cosmo
from ..statistics import chi2_and_pte
from .catalog_characterization import CatalogCharacterizationMixin
from .cosebis import CosebisMixin
from .pseudo_cl import PseudoClMixin
from .psf_systematics import PSFSystematicsMixin
from .pure_eb import PureEBMixin
from .real_space import RealSpaceMixin


# %%
class CosmologyValidation(
    CosebisMixin,
    PureEBMixin,
    RealSpaceMixin,
    PSFSystematicsMixin,
    CatalogCharacterizationMixin,
    PseudoClMixin,
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
    pol_factor : int, default -1
        Apply polarization correction factor in pseudo-C_ell calculations.
    nrandom_cell : int, default 10
        Number of random realizations for C_ell error estimation.
    cell_seed : int, default 8192
        Seed for the random-rotation noise realizations in the pseudo-C_ell
        noise debiasing, making those realizations reproducible run-to-run.
    cosmo_params : dict, optional
        Cosmological parameters to pass to get_cosmo(). If None, uses Planck 2018.
    compute_tomography : bool, default False
        Whether to compute tomographic correlation functions and pseudo-C_ell.
    force_run : bool, default False
        If True, forces re-computation of results even if cached outputs exist.

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
        cell_seed=8192,
        path_onecovariance=None,
        cosmo_params=None,
        blind=None,
        compute_tomography=False,
        force_run=False,
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
        self.cell_seed = cell_seed
        self.cell_method = cell_method
        self.noise_bias_method = noise_bias_method
        self.fiducial_input_inka = fiducial_input_inka
        self.nside_mask = nside_mask
        self.path_onecovariance = path_onecovariance
        self.blind = blind
        self.compute_tomography = compute_tomography
        self.force_run = force_run

        assert self.cell_method in ["map", "catalog"], (
            "cell_method must be 'map' or 'catalog'"
        )
        assert self.noise_bias_method in ["analytic", "randoms"], (
            "noise_bias_method must be 'analytic' or 'randoms'"
        )
        assert self.fiducial_input_inka in ["coupled", "decoupled"], (
            "fiducial_input_inka must be 'coupled' or 'decoupled'"
        )

        # Cosmology for theory predictions: caller-supplied params, else the
        # get_cosmo() Planck 2018 defaults.
        self.cosmo = (
            get_cosmo(**cosmo_params) if cosmo_params is not None else get_cosmo()
        )

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
            self.cc = cc = yaml.load(file, Loader=yaml.FullLoader)

        def resolve_paths_for_version(ver):
            """Resolve relative paths for a version using its subdir."""
            subdir = Path(cc[ver]["subdir"])
            for section in cc[ver].values():
                if "path" in section:
                    path = Path(section["path"])
                    section["path"] = (
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

    def _output_path(self, *parts):
        """Absolute path under the catalog config's output directory.

        Joins ``*parts`` onto ``self.cc["paths"]["output"]`` and absolutises
        the result, mirroring the ``os.path.abspath(f"{output}/...")`` pattern
        used throughout the mixins. A single ``parts`` string may contain
        ``/`` separators.
        """
        return os.path.abspath(os.path.join(self.cc["paths"]["output"], *parts))

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

    def _cprint(self, color, msg, end="\n"):
        """Print ``msg`` in ``color``, then restore the default foreground."""
        print(color + msg, end=end)
        self.color_reset()

    def print_blue(self, msg, end="\n"):
        self._cprint(colorama.Fore.BLUE, msg, end=end)

    def print_start(self, msg, end="\n"):
        print()
        self.print_blue(msg, end=end)

    def print_done(self, msg):
        self.print_blue(msg)

    def print_magenta(self, msg):
        self._cprint(colorama.Fore.MAGENTA, msg)

    def print_green(self, msg):
        self._cprint(colorama.Fore.GREEN, msg)

    def print_cyan(self, msg):
        self._cprint(colorama.Fore.CYAN, msg)

    def init_results(self, objectwise=False):
        # Branch is loop-invariant: pick the leakage class and its parameter
        # builder once, then apply per version.
        make_leakage, set_params = (
            (run_object.LeakageObject, self.set_params_leakage_object)
            if objectwise
            else (run_scale.LeakageScale, self.set_params_leakage_scale)
        )

        results = {}
        for ver in self.versions:
            leakage = results[ver] = make_leakage()
            leakage._params.update(set_params(ver))
            leakage.check_params()
            leakage.prepare_output()

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

    def _binning(self, min_sep=None, max_sep=None, nbins=None, **extra):
        """treecorr_config with min_sep/max_sep/nbins overridden.

        None falls back to the instance's treecorr_config value for that key;
        any further keys in `extra` override on top.
        """
        return {
            **self.treecorr_config,
            "min_sep": min_sep or self.treecorr_config["min_sep"],
            "max_sep": max_sep or self.treecorr_config["max_sep"],
            "nbins": nbins or self.treecorr_config["nbins"],
            **extra,
        }

    def _read_shear_cols(self, ver, *keys):
        """Read shear-catalog columns by their config-key names.

        Each key in ``*keys`` (e.g. ``"e1_col"``, ``"w_col"``) is resolved to a
        column name via ``self.cc[ver]["shear"][key]`` and indexed out of
        ``self.results[ver].dat_shear``. Must be called inside a
        ``self.results[ver].temporarily_read_data()`` context, since it touches
        ``dat_shear`` directly.

        Returns one array per key (a bare array, not a 1-tuple, when a single
        key is requested).
        """
        cols = tuple(
            self.results[ver].dat_shear[self.cc[ver]["shear"][key]] for key in keys
        )
        return cols[0] if len(cols) == 1 else cols

    def _calibrated_g(self, ver):
        """Calibrated shear components ``(g1, g2)`` for a catalog version.

        Applies additive-bias subtraction and the multiplicative response:
        ``g = (e − c) / R``. For DES the response is the catalog-averaged
        per-component ``R11``/``R22`` (column names in the config); for every
        other version it is the scalar ``R`` from the config. Used identically
        by :meth:`calculate_2pcf` and :meth:`calculate_aperture_mass_dispersion`.

        Must be called inside a ``self.results[ver].temporarily_read_data()``
        context, since it reads ``dat_shear`` columns.
        """
        e1, e2 = self._read_shear_cols(ver, "e1_col", "e2_col")
        if ver == "DES":
            R1 = np.average(self.results[ver].dat_shear[self.cc[ver]["shear"]["R11"]])
            R2 = np.average(self.results[ver].dat_shear[self.cc[ver]["shear"]["R22"]])
        else:
            R1 = R2 = self.cc[ver]["shear"]["R"]
        return (e1 - self.c1[ver]) / R1, (e2 - self.c2[ver]) / R2

    @property
    def colors(self):
        return [self.cc[ver]["colour"] for ver in self.versions]

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
                cov_methods.add(
                    "semi-analytic"
                    if "eb_samples" in res
                    else f"jackknife ({gg.npatch1} patches)"
                )

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
                    _, _, row["C_l_BB"] = chi2_and_pte(cl_bb, cov_bb)
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

    def _get_tomo_bins(self, version):
        """
        Return the tomo_bin_ids for a given version. If the version does not have tomography, return None.

        Returns
        -------
        tomo_bin_ids : list or None
            List of unique tomographic bin IDs for the version, or None if no tomography is available
        tomo_bin_pairs : list of tuples or None
            List of unique pairs of tomographic bin IDs (including self-pairs) for the version, or None if no tomography is available
        """
        if "tomo_bin_ids" in self.cc[version]["shear"]:
            self.print_cyan(
                f"Extracting tomography information from version {version}."
            )
            cat_gal = fits.getdata(self.cc[version]["shear"]["path"])
            tomo_bin = cat_gal[self.cc[version]["shear"]["tomo_bin_ids"]]
            tomo_bin_ids = np.unique(tomo_bin)
            tomo_bin_ids = tomo_bin_ids[
                tomo_bin_ids > 0
            ]  # Exclude zero or negative bins
            self.print_cyan(
                f"Found {len(tomo_bin_ids)} tomographic bins for version {version}: {tomo_bin_ids}."
            )

            tomo_bin_pairs = list(
                itertools.combinations_with_replacement(tomo_bin_ids, 2)
            )
            return tomo_bin_ids, tomo_bin_pairs
        else:
            self.print_cyan(f"Version {version} does not have tomography information.")
            return None, None
