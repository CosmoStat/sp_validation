"""Theory data vectors: a prediction for every row of a SACC.

A theory is a function ``theory(params, s) -> np.ndarray``:

- ``params`` is a plain dict of cosmological parameters by name, with exactly
  the keys of :func:`fiducial`: ``S8``, ``Omega_m``, ``Omega_b``, ``h``,
  ``n_s``, ``m_nu``, ``w0``, ``wa``, ``logT_AGN`` and ``A_IA``. Blinding calls
  the theory twice, at a blind's fiducial point and at the hidden point drawn
  from it; the two differ only in ``S8`` and ``Omega_m``.
- ``s`` is the ``sacc.Sacc`` being blinded. Its rows, ``s.data``, each carry a
  data type, a tracer pair and a θ or ℓ tag (Cℓ rows also a bandpower window,
  ``s.get_bandpower_windows(rows)``); its tracers, ``s.tracers``, carry the
  n(z).
- It returns an array the length of ``s.mean``, in the same order: the
  prediction for each row, zero where the cosmology has no effect.

For a SACC of Cℓ_EE/BB/EB bandpowers, a theory could read::

    def my_theory(params, s):
        out = np.zeros(len(s.mean))
        ee = s.indices(sacc_io.CL_EE)
        window = s.get_bandpower_windows(ee)
        cl = my_cl_ee(params, window.values)  # Cℓ_EE on the window's ℓ
        out[ee] = window.weight.T @ cl
        return out  # Cℓ_BB and Cℓ_EB stay zero

:func:`shear` is the default, :func:`none` the theory of statistics with no
cosmological signal.
"""

import functools

import numpy as np

from .sacc_io import CL_BB, CL_EB, CL_EE, XI_MINUS, XI_PLUS

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


def none(params, s):
    """Zeros: the theory of a statistic with no cosmological signal (ρ/τ)."""
    return np.zeros(len(s.mean))


def shear(params, s):
    """Cosmic-shear ξ± and Cℓ_EE from pyccl; zeros for Cℓ_BB and Cℓ_EB.

    ξ± is evaluated at each row's stored θ (TreeCorr's ``meanr``) and Cℓ_EE
    through each row's bandpower window. Any other data type raises: pass your
    own theory. pyccl and cs_util are imported only when this runs.
    """
    out = np.zeros(len(s.mean))
    groups = {}
    for i, dp in enumerate(s.data):
        if dp.data_type in (XI_PLUS, XI_MINUS):
            key = (_xi, dp.tracers)
        elif dp.data_type == CL_EE:
            key = (_cl, dp.tracers, id(dp.tags.get("window")))
        elif dp.data_type in (CL_BB, CL_EB):
            continue
        else:
            raise ValueError(
                f"theory.shear has no prediction for {dp.data_type}; pass your "
                "own theory"
            )
        groups.setdefault(key, []).append(i)
    # One call per tracer pair (and window), so ξ+ and ξ− share their Cℓ.
    for (function, *_), rows in groups.items():
        out[rows] = function(params, s, np.asarray(rows))
    return out


def cosmology(params):
    """The ``pyccl.Cosmology`` at ``params``, built once per point.

    Built through ``cs_util.cosmo.get_cosmo`` on the CAMB HMCode2020-feedback
    route the CosmoSIS inference runs, with σ8 = S8/√(Ωm/0.3); ``get_cosmo``
    subtracts CAMB's Ω_ν from ``Omega_m`` for Ω_c.
    """
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


def _tag(s, rows, name):
    return np.array([s.data[i].tags[name] for i in rows])


def _lensing(cosmo, params, s, name):
    import pyccl as ccl

    tracer = s.tracers[name]
    if getattr(tracer, "nz", None) is None:
        raise ValueError(f"tracer {name} has no n(z)")
    z, n = np.asarray(tracer.z, float), np.asarray(tracer.nz, float)
    ia = None if params["A_IA"] == 0 else (z, np.full_like(z, params["A_IA"]))
    return ccl.WeakLensingTracer(cosmo, dndz=(z, n), ia_bias=ia)


def _shear_cl(params, s, rows, ell):
    import pyccl as ccl

    cosmo = cosmology(params)
    a, b = (_lensing(cosmo, params, s, t) for t in s.data[rows[0]].tracers)
    return ccl.angular_cl(cosmo, a, b, ell)


def _xi(params, s, rows):
    """ξ± of one shear pair: Limber Cℓ on :data:`ELL`, then CCL's Hankel transform."""
    import pyccl as ccl

    cosmo, cl = cosmology(params), _shear_cl(params, s, rows, ELL)
    theta = _tag(s, rows, "theta") / 60.0
    xip, xim = (
        ccl.correlation(cosmo, ell=ELL, C_ell=cl, theta=theta, type=t)
        for t in ("GG+", "GG-")
    )
    plus = np.array([s.data[i].data_type == XI_PLUS for i in rows])
    return np.where(plus, xip, xim)


def _cl(params, s, rows):
    """Cℓ_EE of one shear pair through its bandpower window."""
    window = s.get_bandpower_windows(rows)
    cl = _shear_cl(params, s, rows, np.asarray(window.values, float))
    return np.asarray(window.weight).T @ cl
