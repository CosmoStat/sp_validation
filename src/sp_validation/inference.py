"""SACC inputs, CosmoSIS configuration and weighted chain summaries.

The workflow owns execution; values, priors and sampler settings are run inputs.
No catalogue or blind record is read here.
"""

import configparser
import json
import os
from pathlib import Path

import numpy as np
from astropy.io import fits

from . import sacc_io as sio


def prepare(
    sacc_path,
    output,
    *,
    n_bins=1,
    statistic="xi",
    covariance=None,
    rho_stats=None,
    tau_stats=None,
):
    """Convert the selected reporting statistic, optionally replacing its covariance.

    A covariance reference contributes only its covariance, never its signal or
    n(z). Selection keys enforce the same point order in the two products.
    A ξ-only file is intentional: CSL's twopoint reader cannot attach separate
    COVMAT and COVMAT_CELL matrices to a mixed ξ/Cℓ file.
    """
    s = sio.load(sacc_path)
    types = {sio.XI_PLUS, sio.XI_MINUS}
    if statistic == "cell":
        types.add(sio.CL_EE)
    if rho_stats is not None:
        types.update(sio.TAU_PLUS.format(k=k) for k in (0, 2))
    rows = [
        i
        for i, p in enumerate(s.data)
        if p.data_type in types and p.tags.get("grid") == "reporting"
    ]
    s.keep_indices(rows)
    if covariance:
        ref = sio.load(covariance)

        def key(p):
            return (
                p.data_type,
                p.tracers,
                p.tags.get("grid"),
                p.tags.get("theta_nom", p.tags.get("theta")),
                p.tags.get("ell"),
            )

        lookup = {key(p): i for i, p in enumerate(ref.data)}
        indices = [lookup[key(p)] for p in s.data]
        s.add_covariance(ref.covariance.get_block(indices), overwrite=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    sidecars = {}
    for name, path in (("rho_stats_hdu", rho_stats), ("tau_stats_hdu", tau_stats)):
        if path:
            with fits.open(path) as hdus:
                sidecars[name] = hdus[1].copy()
    hdus = sio.sacc_to_twopoint_fits(s, output, n_bins=n_bins, **sidecars)
    if statistic == "cell":
        # Separate likelihood files avoid the legacy reader's mixed-covariance
        # limitation; conversion itself retains the historical single-bin bytes.
        del hdus["XI_PLUS"]
        del hdus["XI_MINUS"]
        del hdus["COVMAT"]
        hdus.writeto(output, overwrite=True)
    return {
        "sacc": str(sacc_path),
        "covariance": str(covariance or sacc_path),
        "blind": sio.stamp(s),
        "n_bins": n_bins,
        "statistic": statistic,
    }


def render(
    template,
    output,
    *,
    fits_path,
    values,
    priors,
    chain,
    n_bins=1,
    sampler="nautilus",
    settings=None,
    scale_cuts=None,
    overrides=None,
):
    """Render a small run INI without changing the supplied values or priors."""
    ini = configparser.ConfigParser(interpolation=None)
    ini.read(template)
    ini["DEFAULT"]["CSL_DIR"] = os.environ.get(
        "CSL_DIR", "/opt/cosmosis-standard-library"
    )
    ini["DEFAULT"]["SCRATCH"] = str(Path(chain).resolve().parent)
    ini["DEFAULT"]["FITS_FILE"] = str(Path(fits_path).resolve())
    ini["pipeline"]["values"] = str(Path(values).resolve())
    ini["pipeline"]["priors"] = str(Path(priors).resolve())
    ini["runtime"]["sampler"] = sampler
    ini["output"]["filename"] = str(Path(chain).resolve())
    if sampler not in ini:
        ini[sampler] = {}
    for key, value in (settings or {}).items():
        ini[sampler][key] = str(value)
    # Remove the Paper-III single-bin cuts; every run supplies its own cuts.
    like = ini["2pt_like"]
    for key in list(like):
        if key.startswith("angle_range_"):
            del like[key]
    for kind, limits in (scale_cuts or {}).items():
        for i in range(1, n_bins + 1):
            for j in range(i, n_bins + 1):
                like[f"angle_range_{kind}_{i}_{j}"] = " ".join(
                    str(float(x)) for x in limits
                )
    for section, options in (overrides or {}).items():
        if section not in ini:
            ini[section] = {}
        for key, value in options.items():
            ini[section][key] = str(value)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(chain).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as stream:
        ini.write(stream)


def summarize(chain, output):
    """Weighted means, standard deviations and central 68% intervals.

    Nautilus writes log weights; emcee and other equal-weight chains do not.
    Burn-in for an MCMC run must be handled by its sampler configuration.
    """
    with open(chain) as stream:
        names = stream.readline().lstrip("#").lower().split()
    data = np.loadtxt(chain, ndmin=2)
    columns = dict(zip(names, data.T))
    if "log_weight" in columns:
        logw = columns["log_weight"]
        weights = np.exp(logw - logw.max())
    else:
        weights = columns.get("weight", np.ones(len(data)))
    weights = weights / weights.sum()
    result = {
        "samples": len(data),
        "effective_samples": float(1 / (weights @ weights)),
        "parameters": {},
    }
    for name, x in columns.items():
        if "--" not in name:
            continue
        mean = weights @ x
        order = np.argsort(x)
        q = np.interp([0.15865, 0.5, 0.84135], np.cumsum(weights[order]), x[order])
        result["parameters"][name] = {
            "mean": float(mean),
            "sigma": float(np.sqrt(weights @ (x - mean) ** 2)),
            "q16": float(q[0]),
            "median": float(q[1]),
            "q84": float(q[2]),
        }
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result
