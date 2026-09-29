"""Theory data vectors: a prediction for the rows of a SACC, per data type.

@sc theory-plugin
A theory is a function ``f(params, s, rows) -> values``: the prediction, at the
parameter point ``params``, for rows ``rows`` of SACC ``s``, which share a data
type's theory, a tracer pair and a bandpower window; values are aligned to
``rows``. ``params`` is a plain mapping with the keys of :func:`fiducial`,
never a CCL object, so an emulator can stand in for CCL. :func:`shear_xi` and
:func:`shear_cl` are the defaults :data:`sp_validation.blinding.STANDARD`
shifts ξ± and Cℓ_EE by.

@sc theory-ccl-default
The defaults build one ``pyccl.Cosmology`` per point through
``cs_util.cosmo.get_cosmo``, on the CAMB HMCode2020-feedback route the CosmoSIS
inference runs, with σ8 = S8/√(Ωm/0.3). ``Omega_m`` is ``get_cosmo``'s
argument, from which it subtracts CAMB's Ω_ν for Ω_c.

pyccl and cs_util are imported only when a default theory runs.
"""

import functools

import numpy as np

from .sacc_io import XI_PLUS

# Multipoles the ξ± Hankel transform integrates over: every ℓ below 50,
# then 200 log-spaced up to 6·10⁴.
ELL = np.unique(np.concatenate([np.arange(2, 50), np.geomspace(50, 6e4, 200)]))
_COSMOLOGY = ("S8", "Omega_m", "Omega_b", "h", "n_s", "m_nu", "w0", "wa", "logT_AGN")


def fiducial():
    """cs_util's Planck 2018 point, with feedback and no IA."""
    from cs_util.cosmo import PLANCK18 as p

    return {
        "S8": float(p["sigma_8"] * np.sqrt(p["Omega_m"] / 0.3)),
        **{
            k: float(p[k])
            for k in ("Omega_m", "Omega_b", "h", "n_s", "m_nu", "w0", "wa")
        },
        "logT_AGN": 7.5,
        "A_IA": 0.0,
    }


def cosmology(params):
    """The ``pyccl.Cosmology`` at ``params``, built once per point."""
    return _cosmology(tuple(float(params[k]) for k in _COSMOLOGY))


@functools.lru_cache(maxsize=4)
def _cosmology(point):
    from cs_util.cosmo import get_cosmo

    p = dict(zip(_COSMOLOGY, point))
    cosmo = get_cosmo(
        Omega_m=p["Omega_m"],
        Omega_b=p["Omega_b"],
        h=p["h"],
        sig8=p["S8"] / np.sqrt(p["Omega_m"] / 0.3),
        ns=p["n_s"],
        w0=p["w0"],
        wa=p["wa"],
        mnu=p["m_nu"],
        # get_cosmo's default "halofit" ignores extra_params.
        matter_power_spectrum="camb",
        extra_params={
            "camb": {
                "halofit_version": "mead2020_feedback",
                "HMCode_logT_AGN": p["logT_AGN"],
            }
        },
    )
    # CCL's Hankel transform extrapolates C_ℓ to ELL_MAX_CORR; its default
    # (6·10⁴) rings in ξ− below a few arcmin.
    cosmo.cosmo.spline_params.ELL_MAX_CORR = 10_000_000
    cosmo.cosmo.spline_params.N_ELL_CORR = 5_000
    return cosmo


def tag(s, rows, name):
    """Tag ``name`` (or ``"data_type"``) of each of ``rows``."""
    return np.array(
        [
            s.data[i].data_type if name == "data_type" else s.data[i].tags[name]
            for i in rows
        ]
    )


def nz(s, name):
    """The ``(z, n(z))`` of tracer ``name``."""
    tracer = s.tracers[name]
    if getattr(tracer, "nz", None) is None:
        raise ValueError(f"tracer {name} has no n(z)")
    return np.asarray(tracer.z, float), np.asarray(tracer.nz, float)


def _lensing(cosmo, params, s, name):
    import pyccl as ccl

    z, n = nz(s, name)
    ia = None if params["A_IA"] == 0 else (z, np.full_like(z, params["A_IA"]))
    return ccl.WeakLensingTracer(cosmo, dndz=(z, n), ia_bias=ia)


def _shear_cl(params, s, rows, ell):
    import pyccl as ccl

    cosmo = cosmology(params)
    a, b = (_lensing(cosmo, params, s, t) for t in s.data[rows[0]].tracers)
    return ccl.angular_cl(cosmo, a, b, ell)


def shear_xi(params, s, rows):
    """ξ± of one shear pair: Limber C_ℓ on :data:`ELL`, then CCL's Hankel transform."""
    import pyccl as ccl

    cosmo, cl = cosmology(params), _shear_cl(params, s, rows, ELL)
    theta = tag(s, rows, "theta") / 60.0
    xip, xim = (
        ccl.correlation(cosmo, ell=ELL, C_ell=cl, theta=theta, type=t)
        for t in ("GG+", "GG-")
    )
    return np.where(tag(s, rows, "data_type") == XI_PLUS, xip, xim)


def shear_cl(params, s, rows):
    """Cℓ_EE of one shear pair through its bandpower window."""
    window = s.get_bandpower_windows(rows)
    cl = _shear_cl(params, s, rows, np.asarray(window.values, float))
    return np.asarray(window.weight).T @ cl


def predict(s, params, theory, rows=None):
    """The theory of ``rows`` (default: all) of ``s`` at ``params``.

    ``theory`` maps data type to function; a row without one raises. Rows are
    grouped by function, tracer pair and bandpower window, one call per group,
    so ξ+ and ξ− of a pair share their C_ℓ.
    """
    rows = np.arange(len(s.data)) if rows is None else np.asarray(rows, int)
    groups = {}
    for n, i in enumerate(rows):
        dp = s.data[i]
        if dp.data_type not in theory:
            raise ValueError(f"no theory for {dp.data_type}")
        window = id(dp.tags.get("window"))
        groups.setdefault((theory[dp.data_type], dp.tracers, window), []).append(n)
    out = np.empty(len(rows))
    for (function, *_), at in groups.items():
        out[at] = function(params, s, rows[at])
    return out
