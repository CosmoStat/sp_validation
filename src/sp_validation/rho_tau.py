import gc
import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import treecorr
from astropy.io import fits
from astropy.table import Table
from shear_psf_leakage.rho_tau_cov import CovTauTh
from shear_psf_leakage.rho_tau_stat import RhoStat, TauStat

from sp_validation import grammar, io, leakage_fit
from sp_validation.correlation import (
    measure_with_patches,
    measurement_matches,
    selection_digest,
    write_measurement_metadata,
)

# SquareRootScale lives in sp_validation.plots; re-exported here so that
# `from sp_validation.rho_tau import SquareRootScale` keeps working.
from sp_validation.plots import SquareRootScale  # noqa: F401
from sp_validation.statistics import jackknife_patch_centers


def _extract_xip(correlations):
    """Return flattened array of xip values from a list of correlations."""
    return np.array([corr.xip for corr in correlations]).flatten()


def _compute_stats_with_patches(
    handler, kind, catalog_id, filename, save_cov=False, means=None, selection=None
):
    """Publish unpatched ρ/τ means with the patched variances and covariance.

    The handlers expose catalogue mappings and FITS tables, rather than their
    internal GG objects. The shared measurement mechanism uses the configured
    bin_slop for means and TreeCorr's default for the patched covariance pass.
    Cached means avoid repeating the full-sample pass across covariance draws.
    """
    catalogs = handler.catalogs.catalogs_dict
    config = handler._treecorr_config
    selected = {
        key: cat for key, cat in catalogs.items() if key.endswith(f"_{catalog_id}")
    }
    attribute = f"{kind}_stats"
    compute = getattr(handler, f"compute_{kind}_stats")

    def measure(cats, measurement_config):
        patched = any(cat.npatch > 1 for cat in cats.values())
        handler.catalogs.catalogs_dict = cats
        handler._treecorr_config = {
            **measurement_config,
            "var_method": measurement_config.get("var_method", "shot")
            if patched
            else "shot",
        }
        try:
            compute(
                catalog_id,
                filename,
                save_cov=save_cov and patched,
                func=_extract_xip if save_cov and patched else None,
                var_method="jackknife" if patched else "shot",
            )
            return getattr(handler, attribute).copy()
        finally:
            handler.catalogs.catalogs_dict = catalogs
            handler._treecorr_config = config

    means, patched = measure_with_patches(measure, selected, config, means=means)
    table = Table(patched)
    for name in table.colnames:
        if not name.startswith("var"):
            table[name] = means[name]
    setattr(handler, attribute, table)
    getattr(handler, f"save_{kind}_stats")(filename)
    getattr(handler, f"load_{kind}_stats")(filename)
    write_measurement_metadata(
        Path(handler.catalogs._output) / filename, config, selection=selection
    )
    return means


class _CatalogueLoader:
    """Supply FITS paths to the file-based rho/tau readers.

    The rho/tau consumers (``shear_psf_leakage``) read FITS HDU 1 from a path.
    A catalogue that already is exactly that, with no column renamed by the
    grammar or the entry's ``column_map``, passes through unchanged. Any other
    (another HDU, HDF5, a v1 or foreign naming convention) is read through
    ``io.Catalogue``, keeping only the configured columns, into a temporary
    FITS file.
    """

    def __init__(self, info, params):
        self._info = info
        self._params = params
        self._cache = {}
        self._temporary_directory = None

    def __enter__(self):
        if self._temporary_directory is not None:
            raise RuntimeError("catalogue loader is already open")
        self._temporary_directory = TemporaryDirectory(prefix="sp-validation-rho-tau-")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._temporary_directory.cleanup()
        self._temporary_directory = None

    def _columns(self, block):
        if block == "psf":
            keys = (
                "ra_PSF_col",
                "dec_PSF_col",
                "e1_PSF_col",
                "e2_PSF_col",
                "e1_star_col",
                "e2_star_col",
                "PSF_size",
                "star_size",
                "PSF_flag",
                "star_flag",
            )
        elif block == "shear":
            keys = ("ra_col", "dec_col", "w_col", "e1_col", "e2_col")
        else:
            raise KeyError(f"unknown rho/tau catalogue block {block!r}")

        columns = [self._params.get(key) for key in keys]
        if block == "psf":
            for psf_key, shear_key in (
                ("ra_PSF_col", "ra_col"),
                ("dec_PSF_col", "dec_col"),
            ):
                if self._params.get(psf_key) is None:
                    columns.append(self._params.get(shear_key))
            columns.extend(
                self._params.get(key)
                for key in (
                    "M_4_1_psf_col",
                    "M_4_2_psf_col",
                    "M_4_1_star_col",
                    "M_4_2_star_col",
                )
            )
        else:
            columns.extend(
                value
                for value in (self._params.get("R11"), self._params.get("R22"))
                if isinstance(value, str)
            )
        return tuple(dict.fromkeys(name for name in columns if name is not None))

    def __call__(self, block):
        if block not in self._cache:
            entry = self._info[block]
            with io.Catalogue(
                entry["path"], hdu=entry.get("hdu"), column_map=entry.get("column_map")
            ) as catalogue:
                if catalogue.hdu == 1 and not isinstance(
                    catalogue.table(), grammar.V2View
                ):
                    self._cache[block] = catalogue.path
                else:
                    if self._temporary_directory is None:
                        raise RuntimeError(
                            "use the catalogue loader inside a with block"
                        )
                    table = catalogue.read(columns=self._columns(block))
                    output = Path(self._temporary_directory.name) / f"{block}.fits"
                    with fits.HDUList(
                        [fits.PrimaryHDU(), fits.BinTableHDU(data=table)]
                    ) as hdus:
                        hdus.writeto(output)
                    self._cache[block] = os.fspath(output)
        return self._cache[block]


def get_params_rho_tau(cat):
    """Rho/tau parameters for one catalogue-config entry ``cat``.

    The jackknife patch count is the entry's ``patch_number``; a missing key
    raises ``KeyError``.
    """
    params = {"patch_number": cat["patch_number"]}
    params["ra_PSF_col"] = cat["psf"]["ra_col"]
    params["dec_PSF_col"] = cat["psf"]["dec_col"]
    params["e1_PSF_col"] = cat["psf"]["e1_PSF_col"]
    params["e2_PSF_col"] = cat["psf"]["e2_PSF_col"]
    params["e1_star_col"] = cat["psf"]["e1_star_col"]
    params["e2_star_col"] = cat["psf"]["e2_star_col"]
    params["PSF_size"] = cat["psf"]["PSF_size"]
    params["star_size"] = cat["psf"]["star_size"]
    params["PSF_flag"] = cat["psf"].get("PSF_flag")
    params["star_flag"] = cat["psf"].get("star_flag")
    params["ra_units"] = "deg"
    params["dec_units"] = "deg"

    params["ra_col"] = cat["shear"].get("ra_col", "RA")
    params["dec_col"] = cat["shear"].get("dec_col", "Dec")
    params["w_col"] = cat["shear"]["w_col"]
    params["e1_col"] = cat["shear"]["e1_col"]
    params["e2_col"] = cat["shear"]["e2_col"]
    params["tomo_bin_col"] = cat["shear"].get("tomo_bin_col")
    params["R11"] = cat["shear"].get("R11")
    params["R22"] = cat["shear"].get("R22")

    return params


def get_rho_tau_w_cov(
    config,
    version,
    treecorr_config,
    outdir,
    base_rho,
    base_tau,
    method,
    mask_star=None,
    mask_gal=None,
    cov_rho=False,
    ncov=100,
    **kwargs,
):
    """Compute rho/tau statistics and, if requested, their covariance."""
    if method == "th":
        nbin_ang, nbin_rad = kwargs.get("nbin_ang", 100), kwargs.get("nbin_rad", 200)
        compute_minus = kwargs.get("compute_minus", True)
        rho_stat_handler, tau_stat_handler = get_rho_tau(
            config,
            version,
            treecorr_config,
            outdir,
            base_rho,
            base_tau,
            cov_rho=cov_rho,
            mask_star=mask_star,
            mask_gal=mask_gal,
        )
        get_theory_cov(
            config,
            version,
            treecorr_config,
            outdir,
            base_tau,
            nbin_ang=nbin_ang,
            nbin_rad=nbin_rad,
            compute_minus=compute_minus,
            mask_star=mask_star,
            mask_gal=mask_gal,
        )
        return rho_stat_handler, tau_stat_handler
    elif method == "jk":
        npatch = kwargs.get("npatch", 100)
        return get_jackknife_cov(
            config,
            version,
            treecorr_config,
            outdir,
            base_rho,
            base_tau,
            npatch=npatch,
            ncov=ncov,
            mask_star=mask_star,
            mask_gal=mask_gal,
        )
    elif method == "sim":
        tau_cov_path = Path(outdir) / f"cov_tau_{base_tau}_th.npy"

        if tau_cov_path.exists():
            print(f"Found existing covariance at {tau_cov_path}")
            print(f"Computing rho/tau statistics for {version}")
            return get_rho_tau(
                config,
                version,
                treecorr_config,
                outdir,
                base_rho,
                base_tau,
                cov_rho=cov_rho,
                mask_star=mask_star,
                mask_gal=mask_gal,
            )
        else:
            raise ValueError(
                "Covariance from simulation not available. Please compute it first."
            )
    else:
        raise ValueError("Method must be either 'jk' or 'th' or 'sim'.")


def get_rho_tau(
    config,
    version,
    treecorr_config,
    outdir,
    base_rho,
    base_tau,
    mask_star=None,
    mask_gal=None,
    cov_rho=False,
    force_run=False,
):
    """
    Compute rho and tau statistics for a given version of the catalogue.

    Parameters
    ----------
    config: dict
        Configuration file.
    version : str
        Version of the catalogue to use.
    treecorr_config : dict
        TreeCorr configuration (must include 'min_sep', 'max_sep', and 'nbins').
    outdir : str
        Output directory.
    base_rho : str
        Base name for the rho output files.
    base_tau : str
        Base name for the tau output files.
    mask_star : array-like, optional
        Boolean selection over the rows of the ``psf`` entry, in the order
        ``io.open_entry`` reads them. If None, no masking is applied.
    mask_gal : array-like, optional
        Boolean selection over the rows of the ``shear`` entry, in the order
        ``io.open_entry`` reads them. If None, no masking is applied.
    cov_rho : bool, optional
        If True, compute the covariance of rho statistics.
    force_run : bool, optional
        If True, force the computation even if output files already exist.
    """

    params = get_params_rho_tau(config[version])

    print("Compute Rho and Tau statistics for the version: ", version)
    start_time = time.time()

    outdir_path = Path(outdir)
    rho_path = outdir_path / f"rho_stats_{base_rho}.fits"
    catalog_id = f"{base_rho}_jk" if cov_rho else base_rho
    cov_rho_path = outdir_path / f"cov_rho_{catalog_id}.npy" if cov_rho else None

    rho_stat_handler = RhoStat(
        output=outdir, treecorr_config=treecorr_config, verbose=True
    )

    with _CatalogueLoader(config[version], params) as load:
        rho_selection = selection_digest(mask_star)
        tau_selection = selection_digest(mask_star, mask_gal)
        rho_stats_exists = measurement_matches(
            rho_path, treecorr_config, selection=rho_selection
        )
        cov_exists = True if not cov_rho else cov_rho_path.exists()
        need_compute = (not rho_stats_exists) or (not cov_exists) or force_run

        if need_compute:
            rho_stat_handler.catalogs.set_params(params, outdir)
            rho_stat_handler.build_cat_to_compute_rho(
                load("psf"), catalog_id=catalog_id, mask=mask_star
            )

            _compute_stats_with_patches(
                rho_stat_handler,
                "rho",
                catalog_id,
                rho_path.name,
                save_cov=cov_rho,
                selection=rho_selection,
            )
            rho_stat_handler.load_rho_stats(rho_path.name)
        else:
            print(
                f"Skipping rho statistics computation, file {rho_path} already exists."
            )
            rho_stat_handler.load_rho_stats(rho_path.name)

        tau_path = outdir_path / f"tau_stats_{base_tau}.fits"
        tau_stat_handler = TauStat(
            catalogs=rho_stat_handler.catalogs,
            output=outdir,
            treecorr_config=treecorr_config,
            verbose=True,
        )

        if (
            measurement_matches(tau_path, treecorr_config, selection=tau_selection)
            and not force_run
        ):
            print(
                f"Skipping tau statistics computation, file {tau_path} already exists."
            )
            tau_stat_handler.load_tau_stats(tau_path.name)
        else:
            tau_stat_handler.catalogs.set_params(params, outdir)

            # Build the different catalogs if necessary
            if f"psf_{version}" not in tau_stat_handler.catalogs.catalogs_dict:
                tau_stat_handler.build_cat_to_compute_tau(
                    load("psf"), cat_type="psf", catalog_id=version, mask=mask_star
                )

            # Build the catalog of galaxies. PSF was computed above
            tau_stat_handler.build_cat_to_compute_tau(
                load("shear"), cat_type="gal", catalog_id=version, mask=mask_gal
            )

            _compute_stats_with_patches(
                tau_stat_handler,
                "tau",
                version,
                tau_path.name,
                selection=tau_selection,
            )

    print(f"Time to compute rho and tau statistics: {time.time() - start_time:.2f} s")
    return rho_stat_handler, tau_stat_handler


# The correlations of the joint leakage measurement, as (first, second)
# catalogue types of shear_psf_leakage's Catalogs: p = "psf",
# de = "psf_error", q = "psf_size_error", galaxies = "gal", and the PSF-size
# slope field u de = "psf_error_size_slope".
LEAKAGE_RHO_PAIRS = (
    ("psf", "psf"),
    ("psf_error", "psf_error"),
    ("psf_error", "psf"),
    ("psf_size_error", "psf_size_error"),
    ("psf_error", "psf_size_error"),
    ("psf", "psf_size_error"),
)
LEAKAGE_TAU_PAIRS = (("gal", "psf"), ("gal", "psf_error"), ("gal", "psf_size_error"))
LEAKAGE_SLOPE_PAIRS = (
    ("psf_error_size_slope", "psf"),
    ("psf_error_size_slope", "psf_error"),
    ("psf_error_size_slope", "psf_size_error"),
)
LEAKAGE_SLOPE_AUTO = ("psf_error_size_slope", "psf_error_size_slope")


def leakage_cache_key(
    treecorr_config,
    npatch_fine,
    patch_counts,
    beta_psf_size,
    mask_star=None,
    mask_gal=None,
):
    """The settings and row selections that identify a joint leakage measurement.

    ``mask_star``/``mask_gal`` are the star and galaxy selections the
    measurement uses, recorded as `selection_digest`s. TreeCorr settings the
    measurement overrides (``var_method``, ``cross_patch_weight``) or that
    only change round-off (``num_threads``) are left out.
    """
    config = {
        key: value
        for key, value in treecorr_config.items()
        if key not in ("num_threads", "var_method", "cross_patch_weight")
    }
    return {
        "treecorr_config": config,
        "npatch_fine": int(npatch_fine),
        "patch_counts": sorted(int(n) for n in patch_counts),
        "beta_psf_size": bool(beta_psf_size),
        "selection": {
            "stars": selection_digest(mask_star),
            "stars_galaxies": selection_digest(mask_star, mask_gal),
        },
    }


def read_leakage_jackknife(path, key):
    """Products of a cached joint leakage measurement made with ``key``, or None."""
    path = Path(path)
    metadata_path = path.with_suffix(".json")
    if not (path.exists() and metadata_path.exists()):
        return None
    if json.loads(metadata_path.read_text()).get("key") != key:
        return None
    with np.load(path) as data:
        return dict(data)


def measure_leakage_jackknife(
    config,
    version,
    treecorr_config,
    path,
    key,
    mask_star=None,
    mask_gal=None,
):
    """Measure the rho/tau vector and its joint delete-one jackknife rows.

    ``key`` (from `leakage_cache_key` with these masks) holds the settings:
    TreeCorr configuration, ``npatch_fine``, ``patch_counts``,
    ``beta_psf_size``, and the selections of ``mask_star``/``mask_gal``.
    Stars and galaxies are built into TreeCorr catalogues exactly as
    `get_rho_tau` builds them (``shear_psf_leakage`` field definitions;
    galaxy ellipticities are centred on the weighted mean of the full,
    unmasked shear catalogue before ``mask_gal`` selects rows), sharing one
    set of ``npatch_fine`` k-means patch centres found on the selected stars. One
    pass of GG correlations, with 'match' cross-patch weights, measures
    rho0..rho5 and tau0, tau2, tau5 and, with ``beta_psf_size``, the
    correlations of u de (u = T_PSF / <T_PSF> - 1 over the selected stars)
    with p, de, q and itself. Delete-one jackknife rows come from the
    per-patch-pair results at ``npatch_fine`` and, for every count in
    ``patch_counts``, at Ward-linkage unions of the fine patches.

    Writes ``path`` (``.npz``) and its ``.json`` metadata, and returns the
    products as a dict: ``theta``, ``vector`` (rho0..rho5, tau0, tau2, tau5
    xi+, flattened), ``rho_xim``, ``rows_<N>`` (one row per patch, the
    vector's order followed, with the slope, by the three u de templates),
    ``centres``, ``groups_<N>``, and with the slope ``template_vector``,
    ``slope_xip`` and ``slope_xim`` (u de with p, de, q, u de).
    """
    path = Path(path)
    metadata_path = path.with_suffix(".json")
    npatch_fine = key["npatch_fine"]
    patch_counts = key["patch_counts"]
    beta_psf_size = key["beta_psf_size"]
    start_time = time.time()
    params = get_params_rho_tau(config[version])
    params["patch_number"] = npatch_fine
    measurement_config = {
        **treecorr_config,
        "var_method": "shot",
        "cross_patch_weight": "match",
    }
    rho_handler = RhoStat(output=str(path.parent), treecorr_config=measurement_config)
    tau_handler = TauStat(
        catalogs=rho_handler.catalogs,
        output=str(path.parent),
        treecorr_config=measurement_config,
    )
    catalogs = rho_handler.catalogs
    catalogs.set_params(params, str(path.parent))
    with _CatalogueLoader(config[version], params) as load:
        rho_handler.build_cat_to_compute_rho(
            load("psf"), catalog_id="leakage", mask=mask_star
        )
        tau_handler.build_cat_to_compute_tau(
            load("shear"), cat_type="gal", catalog_id="leakage", mask=mask_gal
        )
        cats = {
            name.removesuffix("_leakage"): cat
            for name, cat in catalogs.catalogs_dict.items()
        }
        centres = cats["psf"].patch_centers
        mean_psf_size = None
        if beta_psf_size:
            stars = catalogs.read_shear_cat(path_gal=None, path_psf=load("psf"))
            size = np.asarray(stars[params["PSF_size"]], dtype=float)
            if mask_star is not None:
                size = size[mask_star]
            del stars
            mean_psf_size = float(size.mean())
            u = size / mean_psf_size - 1
            de = cats["psf_error"]
            if u.size != de.ntot:
                raise ValueError(
                    f"PSF-size template has {u.size} rows but the de catalogue "
                    f"{de.ntot}; the two must be the same selected stars"
                )
            cats["psf_error_size_slope"] = treecorr.Catalog(
                ra=de.ra,
                dec=de.dec,
                ra_units="rad",
                dec_units="rad",
                g1=u * de.g1,
                g2=u * de.g2,
                w=de.w,
                patch_centers=centres,
            )
            del size, u

    groups = {n: leakage_fit.ward_group_map(centres, n) for n in patch_counts}
    counts = [npatch_fine, *patch_counts]
    pairs = list(LEAKAGE_RHO_PAIRS + LEAKAGE_TAU_PAIRS)
    if beta_psf_size:
        pairs += [*LEAKAGE_SLOPE_PAIRS, LEAKAGE_SLOPE_AUTO]
    xip, xim, rows = [], [], {n: [] for n in counts}
    theta = None
    max_check_error = 0.0
    for first, second in pairs:
        print(f"Joint leakage correlation <{first}, {second}>", flush=True)
        corr = treecorr.GGCorrelation(measurement_config)
        corr.process(cats[first], cats[second])
        xip.append(corr.xip.copy())
        xim.append(corr.xim.copy())
        if (first, second) == LEAKAGE_TAU_PAIRS[0]:
            theta = corr.meanr.copy()
        if (first, second) != LEAKAGE_SLOPE_AUTO:
            fine, error = leakage_fit.treecorr_jackknife_rows([corr])
            max_check_error = max(max_check_error, error)
            rows[npatch_fine].append(fine)
            for n in patch_counts:
                grouped, _ = leakage_fit.treecorr_jackknife_rows(
                    [corr], groups=groups[n], check_rows=False
                )
                rows[n].append(grouped)
        del corr
        gc.collect()

    n_joint = len(LEAKAGE_RHO_PAIRS) + len(LEAKAGE_TAU_PAIRS)
    products = {
        "theta": theta,
        "vector": np.concatenate(xip[:n_joint]),
        "rho_xim": np.array(xim[: len(LEAKAGE_RHO_PAIRS)]),
        "centres": centres,
        **{f"rows_{n}": np.concatenate(rows[n], axis=1) for n in counts},
        **{f"groups_{n}": groups[n] for n in patch_counts},
    }
    if beta_psf_size:
        products["template_vector"] = np.concatenate(xip[n_joint : n_joint + 3])
        products["slope_xip"] = np.array(xip[n_joint:])
        products["slope_xim"] = np.array(xim[n_joint:])

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **products)
    metadata = {
        "key": key,
        "version": version,
        "stars": int(cats["psf"].nobj),
        "galaxies": int(cats["gal"].nobj),
        "mean_psf_size": mean_psf_size,
        "order": [f"<{a}, {b}>" for a, b in pairs],
        "jackknife_row_check_max_abs_error": max_check_error,
        "treecorr_version": treecorr.__version__,
        "seconds": time.time() - start_time,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Joint leakage measurement written to {path} in {metadata['seconds']:.0f} s")
    return products


def fit_leakage(products, npatch, beta_psf_size=False, cross_term=True):
    """Joint EIV fit of `measure_leakage_jackknife` products at ``npatch``.

    Returns the `leakage_fit.fit_joint_eiv` result dict, whose
    ``parameters`` are (alpha, beta, eta), or (alpha, beta0, eta, beta1)
    with the PSF-size slope.
    """
    rows = np.asarray(products[f"rows_{npatch}"])
    n_joint = len(products["vector"])
    template = {}
    if beta_psf_size:
        template = dict(
            template_vector=products["template_vector"],
            template_rows=rows[:, n_joint:],
        )
    return leakage_fit.fit_joint_eiv(
        products["vector"], rows[:, :n_joint], cross_term=cross_term, **template
    )


def get_samples_eiv(
    products, patch_counts, fiducial_npatch, beta_psf_size=False, nsamples=10000, seed=0
):
    """Joint EIV fits at each patch count and Gaussian samples at the fiducial.

    Returns ``(samples, fits, quantiles)``: ``nsamples`` draws (generator
    ``seed``) from N(mu, Sigma_p) of the ``fiducial_npatch`` fit, the fit
    results keyed by patch count, and the 16/50/84 per cent sample quantiles
    per parameter. Raises if the fiducial fit's profile Hessian is not
    positive definite, since it then has no parameter covariance.
    """
    fits = {
        int(n): fit_leakage(products, n, beta_psf_size=beta_psf_size)
        for n in patch_counts
    }
    for n, result in fits.items():
        errors = ", ".join(
            f"{mu:.4g} +/- {sigma:.2g}" if sigma is not None else f"{mu:.4g} +/- ?"
            for mu, sigma in zip(result["parameters"], result["sigma"])
        )
        flags = [
            message
            for ok, message in (
                (result["bfgs_success"], "BFGS did not converge"),
                (result["profile_hessian_positive"], "Hessian not positive definite"),
            )
            if not ok
        ]
        print(
            f"EIV leakage fit, {n} patches: chi2 = {result['chi2']:.2f}/"
            f"{result['dof']} (PTE {result['pte']:.3g}), parameters {errors}"
            + (f" [{'; '.join(flags)}]" if flags else "")
        )
    fiducial = fits[int(fiducial_npatch)]
    if not fiducial["profile_hessian_positive"]:
        raise ValueError(
            f"The {fiducial_npatch}-patch EIV leakage fit has a profile Hessian "
            "that is not positive definite (eigenvalues "
            f"{fiducial['profile_hessian_eigenvalues']}); its parameters are not "
            "constrained, so no samples can be drawn."
        )
    samples = np.random.default_rng(seed).multivariate_normal(
        fiducial["parameters"], fiducial["parameter_covariance"], size=nsamples
    )
    quantiles = np.percentile(samples, [16, 50, 84], axis=0)
    return samples, fits, quantiles


def get_theory_cov(
    config,
    version,
    treecorr_config,
    outdir,
    base,
    nbin_ang=100,
    nbin_rad=100,
    compute_minus=True,
    mask_star=None,
    mask_gal=None,
):
    """
    Compute an analytical estimate of the covariance matrix of rho and tau-statistics.

    ``CovTauTh`` takes the survey area and effective galaxy density from the
    galaxy catalogue after ``mask_gal``. The masks select rows as in
    ``get_rho_tau``; their `selection_digest` is recorded beside the cached
    covariance, which is reused only for the same selection.
    """

    params = get_params_rho_tau(config[version])

    info = config[version]

    target_cov = Path(outdir) / f"cov_tau_{base}_th.npy"
    metadata = Path(str(target_cov) + ".json")
    selection = selection_digest(mask_star, mask_gal)

    if (
        target_cov.exists()
        and metadata.exists()
        and json.loads(metadata.read_text()).get("selection") == selection
    ):
        print(f"Skipping covariance computation, file {target_cov} already exists.")
        return

    print("Computing the covariance matrix for the version: ", version)
    start_time = time.time()

    with _CatalogueLoader(info, params) as load:
        cov_tau_th = CovTauTh(
            path_gal=load("shear"),
            path_psf=load("psf"),
            hdu_psf=1,
            treecorr_config=treecorr_config,
            params=params,
            mask_star=mask_star,
            mask_gal=mask_gal,
        )

        elapsed = time.time() - start_time
        print(f"--- Rho/tau statistics for covariance computed in {elapsed:.2f}s ---")

        cov = cov_tau_th.build_cov(
            nbin_ang=nbin_ang, nbin_rad=nbin_rad, compute_minus=compute_minus
        )
        print(f"--- Covariance matrix assembled in {time.time() - start_time:.2f}s ---")
        target_cov.parent.mkdir(parents=True, exist_ok=True)
        np.save(target_cov, cov)
        metadata.write_text(json.dumps({"selection": selection}) + "\n")
        print("Saved covariance matrix of version: ", version)
        del cov_tau_th
    gc.collect()
    return


def rho_cache_valid(
    rho_path, cov_rho_path, treecorr_config, force_run=False, selection=None
):
    """Whether a cached ρ measurement and its jackknife covariance can be reused.

    The ρ FITS must record the current means, patched configurations and star
    selection (:func:`~sp_validation.correlation.measurement_matches`); the PSF-only
    result is then shared by every source bin of a tomographic run.
    """
    return (
        not force_run
        and Path(cov_rho_path).exists()
        and measurement_matches(rho_path, treecorr_config, selection=selection)
    )


def get_jackknife_cov(
    config,
    version,
    treecorr_config,
    outdir,
    base_rho,
    base_tau,
    npatch,
    ncov=100,
    mask_star=None,
    mask_gal=None,
    force_run=False,
):
    """
    Compute the covariance matrix of rho and tau-statistics using the jackknife method.
    Also compute rho and tau-statistics.
    """
    rho_filename = f"rho_stats_{base_rho}.fits"
    tau_filename = f"tau_stats_{base_tau}.fits"
    tau_cov_path = Path(outdir) / f"cov_tau_{base_tau}_jk.npy"
    rho_selection = selection_digest(mask_star)
    tau_selection = selection_digest(mask_star, mask_gal)

    if (
        tau_cov_path.exists()
        and (Path(outdir) / f"cov_rho_{base_rho}_jk.npy").exists()
        and measurement_matches(
            Path(outdir) / rho_filename, treecorr_config, selection=rho_selection
        )
        and measurement_matches(
            Path(outdir) / tau_filename, treecorr_config, selection=tau_selection
        )
        and not force_run
    ):
        print(f"Skipping covariance computation, file {tau_cov_path} already exists.")
        rho_stat_handler = RhoStat(
            output=outdir, treecorr_config=treecorr_config, verbose=False
        )

        tau_stat_handler = TauStat(
            catalogs=rho_stat_handler.catalogs,
            output=outdir,
            treecorr_config=treecorr_config,
            verbose=True,
        )

        rho_path = Path(outdir) / rho_filename
        tau_path = Path(outdir) / tau_filename
        if rho_path.exists() and not force_run:
            rho_stat_handler.load_rho_stats(rho_path.name)
        if tau_path.exists() and not force_run:
            tau_stat_handler.load_tau_stats(tau_path.name)
        return rho_stat_handler, tau_stat_handler

    params = get_params_rho_tau(config[version])
    params["patch_number"] = npatch

    rho_stat_handler = RhoStat(
        output=outdir, treecorr_config=treecorr_config, verbose=False
    )

    rho_stat_handler.catalogs.set_params(params, outdir)

    tau_stat_handler = TauStat(
        catalogs=rho_stat_handler.catalogs,
        output=outdir,
        treecorr_config=treecorr_config,
        verbose=True,
    )

    tau_stat_handler.catalogs.set_params(params, outdir)

    rho_path = Path(outdir) / rho_filename
    cov_rho_path = Path(outdir) / f"cov_rho_{base_rho}_jk.npy"
    reuse_rho = rho_cache_valid(
        rho_path, cov_rho_path, treecorr_config, force_run, selection=rho_selection
    )
    if reuse_rho:
        rho_stat_handler.load_rho_stats(rho_filename)
        cached_rho_cov = np.load(cov_rho_path)

    # shear_psf_leakage keys its catalogues by catalog_id and writes each
    # draw's covariance to cov_{rho,tau}_{catalog_id}.npy; base_tau names the
    # draws so that the bins of a tomographic run never share them.
    def catalog_id(i):
        return f"{base_tau}{i}"

    def chunks(i):
        return [
            os.path.join(outdir, f"cov_{kind}_{catalog_id(i)}.npy")
            for kind in ("rho", "tau")
        ]

    rho_means = tau_means = None
    with _CatalogueLoader(config[version], params) as load:
        for i in range(ncov):
            if not all(os.path.exists(chunk) for chunk in chunks(i)):
                print(f"Computing rho-statistics for {version} (patch {i + 1}/{ncov})")

                if (
                    f"psf_{catalog_id(i)}"
                    not in rho_stat_handler.catalogs.catalogs_dict
                ):
                    # Build catalogues
                    rho_stat_handler.build_cat_to_compute_rho(
                        load("psf"), catalog_id=catalog_id(i), mask=mask_star
                    )

                    tau_stat_handler.catalogs.catalogs_dict = (
                        rho_stat_handler.catalogs.catalogs_dict
                    )

                catalogs = rho_stat_handler.catalogs
                centers = jackknife_patch_centers(
                    catalogs.catalogs_dict[f"psf_{catalog_id(i)}"],
                    catalogs._params["patch_number"],
                    seed=i,
                    init="kmeans++",
                )
                # Fresh catalogues keep TreeCorr's cached patches consistent with
                # this draw's shared star/galaxy layout.
                stars = catalogs.read_shear_cat(
                    path_gal=None, path_psf=load("psf"), hdu=1
                )
                for cat_type in ("psf", "psf_error", "psf_size_error"):
                    catalogs.build_catalog(
                        cat=stars,
                        cat_type=cat_type,
                        key=f"{cat_type}_{catalog_id(i)}",
                        patch_centers=centers,
                        mask=mask_star,
                    )
                tau_stat_handler.build_cat_to_compute_tau(
                    load("shear"),
                    cat_type="gal",
                    catalog_id=catalog_id(i),
                    mask=mask_gal,
                )

                # The PSF-only result is shared by every source bin.
                if reuse_rho:
                    np.save(chunks(i)[0], cached_rho_cov)
                else:
                    rho_means = _compute_stats_with_patches(
                        rho_stat_handler,
                        "rho",
                        catalog_id(i),
                        rho_filename,
                        save_cov=True,
                        means=rho_means,
                        selection=rho_selection,
                    )
                tau_means = _compute_stats_with_patches(
                    tau_stat_handler,
                    "tau",
                    catalog_id(i),
                    tau_filename,
                    save_cov=True,
                    means=tau_means,
                    selection=tau_selection,
                )

                # Update the keys in the dictionaries
                rho_dict = rho_stat_handler.catalogs.catalogs_dict
                tau_dict = tau_stat_handler.catalogs.catalogs_dict
                for prefix in ("psf", "psf_error", "psf_size_error"):
                    rho_dict[f"{prefix}_{catalog_id(i + 1)}"] = rho_dict.pop(
                        f"{prefix}_{catalog_id(i)}"
                    )
                tau_dict[f"gal_{catalog_id(i + 1)}"] = tau_dict.pop(
                    f"gal_{catalog_id(i)}"
                )

    cov_rho_loc, cov_tau_loc = (np.zeros_like(np.load(c)) for c in chunks(0))
    for i in range(ncov):
        rho_chunk, tau_chunk = chunks(i)
        cov_rho_loc += np.load(rho_chunk)
        cov_tau_loc += np.load(tau_chunk)
        os.remove(rho_chunk)
        os.remove(tau_chunk)

    cov_tau = cov_tau_loc / ncov
    cov_rho = cov_rho_loc / ncov

    tau_cov_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(tau_cov_path, cov_tau)

    cov_rho_path.parent.mkdir(parents=True, exist_ok=True)
    if not reuse_rho:
        np.save(cov_rho_path, cov_rho)

    return rho_stat_handler, tau_stat_handler


def _sample_cache_name(psf_fitter, base_tau, sampler, cov_type):
    """Name of a sample cache, keyed on the tau statistics' row selection.

    The selection is the `selection_digest` recorded beside
    ``tau_stats_<base_tau>.fits`` (stars and galaxies), so samples of a
    different selection are never reused.
    """
    name = f"{base_tau}_sampler_{sampler}_cov_tau_type_{cov_type}"
    metadata = (
        Path(psf_fitter.tau_stat_handler.catalogs._output)
        / f"tau_stats_{base_tau}.fits.json"
    )
    if not metadata.exists():
        return name
    selection = json.loads(metadata.read_text()).get("selection")
    return name if selection is None else f"{name}_selection={selection[:12]}"


def get_samples(
    psf_fitter,
    base_rho,
    base_tau,
    cov_type="jk",
    apply_debias=None,
    sampler="emcee",
    nsamples=10000,
    nwalkers=124,
):
    """Return (alpha, beta, eta) samples using ``emcee`` or least squares.

    Parameters
    ----------
    psf_fitter : PSFFitter
        PSF fitter instance that provides ``load_*`` helpers.
    base_rho : str
        Precomputed basename (e.g. ``SP_v1.4_minsep=…``) used for rho-stat filenames.
    base_tau : str
        Precomputed basename (e.g. ``SP_v1.4_minsep=…``) used for tau-stat filenames.
    cov_type : str, optional
        Covariance label (``'jk'``, ``'th'``, or ``'sim'``). Defaults to ``'jk'``.
    apply_debias : int or None, optional
        Jackknife patch count used to debias samples. Disabled when ``None``.
    sampler : str, optional
        ``'emcee'`` for MCMC sampling, ``'lsq'`` for least squares
        (default ``'emcee'``).
    nsamples : int, optional
        Number of samples to draw (default ``10000``).
    nwalkers : int, optional
        Number of walkers for the MCMC run (default ``124``).
    """
    if sampler == "emcee":
        return get_samples_emcee(
            psf_fitter,
            base_rho,
            base_tau,
            cov_type=cov_type,
            apply_debias=apply_debias,
            nsamples=nsamples,
            nwalkers=nwalkers,
        )
    elif sampler == "lsq":
        return get_samples_lsq(
            psf_fitter,
            base_rho,
            base_tau,
            cov_type=cov_type,
            apply_debias=apply_debias,
            nsamples=nsamples,
        )
    else:
        raise ValueError("Sampler must be either 'emcee' or 'lsq'.")


def get_samples_emcee(
    psf_fitter,
    base_rho,
    base_tau,
    nwalkers=124,
    nsamples=10000,
    cov_type="jk",
    apply_debias=None,
):
    """Draw (alpha, beta, eta) samples using ``emcee`` and the tau covariance.

    Parameters
    ----------
    psf_fitter : PSFFitter
        PSF fitter instance managing rho/tau statistics and covariances.
    base_rho : str
        Precomputed basename for locating rho statistics/covariance files.
    base_tau : str
        Precomputed basename for locating tau statistics/covariance files.
    nwalkers : int, optional
        Number of walkers for the MCMC run (default ``124``).
    nsamples : int, optional
        Number of samples drawn per walker (default ``10000``).
    cov_type : str, optional
        Covariance label (``'jk'``/``'th'``/``'sim'``). Defaults to ``'jk'``.
    apply_debias : int or None, optional
        Jackknife patch count applied during debiasing. Disabled when ``None``.
    """
    # Load rho and tau stats
    psf_fitter.load_rho_stat(f"rho_stats_{base_rho}.fits")
    psf_fitter.load_tau_stat(f"tau_stats_{base_tau}.fits")

    # Check if the path exists (use base for cache key to account for different TreeCorr configs)
    base_sample = _sample_cache_name(psf_fitter, base_tau, "emcee", cov_type)
    sample_file_path = psf_fitter.get_sample_path(base_sample)
    if os.path.exists(sample_file_path):
        print(f"Skipping sampling; {sample_file_path} exists.")
        flat_samples = psf_fitter.load_samples(base_sample)
        mcmc_result, q = psf_fitter.get_mcmc_from_samples(flat_samples)
        print(mcmc_result)
    # Or run MCMC
    else:
        print("MCMC sampling")
        cov_filename = f"cov_tau_{base_tau}_{cov_type}.npy"
        psf_fitter.load_covariance(cov_filename, cov_type="tau")

        debias_npatch = apply_debias if (apply_debias is not None) else None
        flat_samples, mcmc_result, q = psf_fitter.run_chain(
            nwalkers=nwalkers,
            nsamples=nsamples,
            npatch=debias_npatch,
            apply_debias=debias_npatch is not None,
            savefig="mcmc_samples_" + base_tau + ".png",
        )
        psf_fitter.save_samples(flat_samples, base_sample)
    return flat_samples, mcmc_result, q


def get_samples_lsq(
    psf_fitter,
    base_rho,
    base_tau,
    nsamples=10000,
    apply_debias=None,
    cov_type="jk",
):
    """Compute least-squares samples of (alpha, beta, eta) using the tau covariance.

    Parameters
    ----------
    psf_fitter : PSFFitter
        PSF fitter instance managing rho/tau statistics and covariances.
    base_rho : str
        Precomputed basename for locating rho statistics/covariance files.
    base_tau : str
        Precomputed basename for locating tau statistics/covariance files.
    apply_debias : int or None, optional
        Jackknife patch count applied during debiasing. Disabled when ``None``.
    cov_type : str, optional
        Covariance label (defaults to ``'jk'``).
    """
    # Load rho and tau stats
    psf_fitter.load_rho_stat(f"rho_stats_{base_rho}.fits")
    psf_fitter.load_tau_stat(f"tau_stats_{base_tau}.fits")

    base_sample = _sample_cache_name(psf_fitter, base_tau, "lsq", cov_type)
    # Check if the path exists (use base for cache key to account for different TreeCorr configs)
    sample_file_path = psf_fitter.get_sample_path(base_sample)
    if os.path.exists(sample_file_path):
        print(f"Skipping sampling; {sample_file_path} exists.")
        flat_samples = psf_fitter.load_samples(base_sample)
        mcmc_result, q = psf_fitter.get_mcmc_from_samples(flat_samples)
        print(mcmc_result)
    # Or run MCMC
    else:
        print("Least square sampling")
        tau_covariance = f"cov_tau_{base_tau}_{cov_type}.npy"
        rho_covariance = f"cov_rho_{base_rho}_jk.npy"
        psf_fitter.load_covariance(tau_covariance, cov_type="tau")
        psf_fitter.load_covariance(rho_covariance, cov_type="rho")
        debias_npatch = apply_debias if (apply_debias is not None) else None
        flat_samples, mcmc_result, q = psf_fitter.get_least_squares_params_samples(
            npatch=debias_npatch,
            apply_debias=(debias_npatch is not None),
            n_samples=nsamples,
        )
        psf_fitter.save_samples(flat_samples, base_sample)
    return flat_samples, mcmc_result, q
