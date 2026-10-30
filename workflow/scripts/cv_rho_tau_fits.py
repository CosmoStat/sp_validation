"""Fit PSF parameters and persist all non-cosmological plot arrays."""

import numpy as np
from cv_runner import make_cv, prepare_cached_psf, verify_outputs

cv = make_cv(snakemake)
prepare_cached_psf(cv)
for ver, path in zip(cv.versions, snakemake.output.products, strict=True):
    params = cv.set_params_rho_tau(ver, cv.results[ver]._params, cv.cc[ver]["psf"])
    flat_samples = cv.get_samples(ver, params, "all")
    plus, minus = cv.get_xi_psf_sys_samples(ver, params, "all", "all")
    if cv.rho_tau_method == "eiv":
        leakage = cv.leakage_jackknife(ver)
        theta = leakage["theta"]
        fit_summary = cv.leakage_jackknife_path(ver, "all", "leakage_eiv_fit")
        extra = dict(leakage_eiv_fit=fit_summary.read_text())
        if cv.psf_leakage["beta_psf_size"]:
            # <u de, {p, de, q}>: the beta1 term of the tau0, tau2, tau5 model.
            extra["tau_slope_template"] = leakage["slope_xip"][:3]
    else:
        theta = cv.psf_fitter.rho_stat_handler.rho_stats["theta"]
        extra = {}
    data = dict(
        theta=theta,
        flat_samples=flat_samples,
        **extra,
        theta_edges=np.geomspace(
            cv.treecorr_config["min_sep"],
            cv.treecorr_config["max_sep"],
            cv.treecorr_config["nbins"] + 1,
        ),
        theta_unit="arcmin",
    )
    for component, samples in (("plus", plus), ("minus", minus)):
        data[f"samples_{component}"] = samples
        data[f"mean_{component}"] = samples.mean(axis=0)
        data[f"var_{component}"] = samples.var(axis=0)
        data[f"quantiles_{component}"] = np.quantile(
            samples, [1 - cv.quantile, cv.quantile], axis=0
        )
    np.savez_compressed(path, **data)
verify_outputs(snakemake)
