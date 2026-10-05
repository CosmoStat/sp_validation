"""Blinds, the theory a blind shifts by, and the call that conceals a SACC.

A blind is a named secret, ``<paths.blinds>/<name>.blind.json`` holding a
seed, the envelope the hidden point is drawn in and the fiducial point. It is
drawn once, ``python -m sp_validation.blinding init <name>``, and never
committed (``*.blind.json`` is gitignored). Each catalogue entry of the
catalogue config declares ``blind: none`` (public) or ``blind: <name>``
(:func:`blind_of`).

:func:`conceal` adds the blind's shift, t(hidden) − t(fiducial), to every row
of a SACC; Smokescreen draws the hidden point from the seed and evaluates the
theory t at both points.

A theory is a function ``theory(params, s)``:

- ``params`` is a dict of cosmological parameters with the keys of
  :func:`fiducial` (``S8``, ``Omega_m``, ``Omega_b``, ``h``, ``n_s``,
  ``m_nu``, ``w0``, ``wa``, ``logT_AGN``, ``A_IA``). The fiducial and hidden
  points differ only in ``S8`` and ``Omega_m``.
- ``s`` is the ``sacc.Sacc`` being concealed: its rows carry a data type, a
  tracer pair and a θ or ℓ tag, its tracers the n(z).
- It returns an array shaped like the data, ``len(s.mean)`` values in the
  order of ``s.data``: the prediction for each row, zero where the cosmology
  has no effect.

For Cℓ_EE/BB/EB bandpowers, for example::

    def my_theory(params, s):
        out = np.zeros(len(s.mean))
        ee = s.indices(sacc_io.CL_EE)
        window = s.get_bandpower_windows(ee)
        out[ee] = window.weight.T @ my_cl_ee(params, window.values)
        return out  # Cℓ_BB and Cℓ_EB stay zero

:func:`shear` is the default theory; :func:`no_signal` is that of a statistic
with no cosmological signal (ρ/τ).

The host Snakemake loads this module by path, so at import it needs only the
standard library.
"""

import argparse
import dataclasses
import functools
import json
import math
import os
import re
import secrets
from pathlib import Path

ENVELOPE = {"S8": 0.075, "Omega_m": 0.1}
REPO_CAT_CONFIG = Path(__file__).parents[2] / "cosmo_val" / "cat_config.yaml"
_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]*")


class BlindingError(RuntimeError):
    """A blind cannot be opened or applied; the message carries no hidden value."""


@dataclasses.dataclass(frozen=True)
class Blind:
    """A blind's name and the path of its record; ``none`` has no record."""

    name: str
    path: Path | None = None

    def record(self):
        """The blind's ``{seed, envelope, fiducial}``."""
        return json.loads(self.path.read_text())


NONE = Blind("none")


def _shear_file(entry):
    shear = entry.get("shear") if isinstance(entry, dict) else None
    if not isinstance(shear, dict) or "path" not in shear:
        return None
    return os.path.normpath(os.path.join(str(entry.get("subdir", "")), shear["path"]))


def blind_of(catalogues, entry):
    """The ``blind:`` of catalogue-config entry ``entry``.

    Refuses an entry that declares none, and entries reading one shear file
    under different blinds.
    """
    blind = catalogues[entry].get("blind")
    if not isinstance(blind, str):
        raise ValueError(
            f"{entry} declares no `blind:`; declare `blind: none` (public) or "
            "`blind: <name>` in its catalogue config entry"
        )
    path = _shear_file(catalogues[entry])
    readers = {
        name: other.get("blind")
        for name, other in catalogues.items()
        if path and _shear_file(other) == path
    }
    if len(set(readers.values())) > 1:
        raise ValueError(f"{path} is read under different blinds: {readers}")
    return blind


def registry(catalogues):
    """The directory a catalogue config keeps its blinds in (``paths.blinds``)."""
    return Path(os.path.expanduser(catalogues["paths"]["blinds"]))


def open_blind(name, catalogues):
    """Blind ``name`` of the catalogue config ``catalogues``; its record must exist."""
    if name == NONE.name:
        return NONE
    path = registry(catalogues) / f"{name}.blind.json"
    if not os.access(path, os.R_OK):
        raise BlindingError(
            f"cannot read blind {name} at {path}; `python -m sp_validation.blinding "
            f"init {name}` draws it, if it is meant to be new"
        )
    return Blind(name, path)


def conceal(s, blind, theory=None):
    """A copy of SACC ``s`` with ``blind``'s shift added to every row.

    The shift is t(hidden) − t(fiducial), t being ``theory`` (default
    :func:`shear`). A failure raises :class:`BlindingError` naming only the
    exception type, so no message or traceback shows the hidden point.
    """
    import numpy as np
    from smokescreen.datavector import concealing_factor

    theory = theory or shear
    record, failure = blind.record(), None
    try:
        shift = concealing_factor(
            record["fiducial"],
            record["envelope"],
            seed=record["seed"],
            theory_fn=lambda params: theory(params, s),
        )
    except Exception as err:
        failure = type(err).__name__
    if failure:  # raised outside the except, so nothing is chained
        raise BlindingError(f"the theory failed under blind {blind.name} ({failure})")
    shift = np.asarray(shift, float)
    if shift.shape != (len(s.mean),):
        raise BlindingError(
            f"the theory returned {shift.shape[0] if shift.ndim else 0} values for "
            f"{len(s.mean)} rows"
        )
    out = s.copy()
    for dp, d in zip(out.data, shift):
        dp.value += float(d)
    return out


def init(name, catalogues):
    """Draw blind ``name`` into the catalogue config's registry, once."""
    if not _NAME.fullmatch(name) or name == NONE.name:
        raise BlindingError(f"blind name {name!r}: use a-z, 0-9, - _ .")
    where = registry(catalogues)
    where.mkdir(mode=0o700, parents=True, exist_ok=True)
    record = {
        "seed": secrets.token_hex(32),
        "envelope": ENVELOPE,
        "fiducial": fiducial(),
    }
    path = where / f"{name}.blind.json"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    except FileExistsError:
        raise BlindingError(f"blind {name} exists at {path}") from None
    with os.fdopen(fd, "w") as f:
        json.dump(record, f, indent=1)
    print(f"[blinding] drew blind {name} at {path}")
    return Blind(name, path)


# --------------------------------------------------------------------------- #
# Theories
# --------------------------------------------------------------------------- #
_COSMOLOGY = ("S8", "Omega_m", "Omega_b", "h", "n_s", "m_nu", "w0", "wa", "logT_AGN")


def fiducial():
    """cs_util's Planck 2018 point, with feedback and no IA."""
    from cs_util.cosmo import PLANCK18 as p

    return {
        "S8": float(p["sigma_8"] * math.sqrt(p["Omega_m"] / 0.3)),
        **{
            k: float(p[k])
            for k in ("Omega_m", "Omega_b", "h", "n_s", "m_nu", "w0", "wa")
        },
        "logT_AGN": 7.5,
        "A_IA": 0.0,
    }


def no_signal(params, s):
    """Zeros: the theory of a statistic with no cosmological signal (ρ/τ)."""
    import numpy as np

    return np.zeros(len(s.mean))


def shear(params, s):
    """Cosmic-shear ξ± and Cℓ_EE from pyccl; zeros for Cℓ_BB and Cℓ_EB.

    ξ± is evaluated at each row's stored θ (TreeCorr's ``meanr``) and Cℓ_EE
    through each row's bandpower window. Any other data type raises: pass your
    own theory.
    """
    import numpy as np

    from .sacc_io import CL_BB, CL_EB, CL_EE, XI_MINUS, XI_PLUS

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
                f"blinding.shear has no prediction for {dp.data_type}; pass your "
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
        sig8=p["S8"] / math.sqrt(p["Omega_m"] / 0.3),
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


@functools.cache
def _ell():
    """The multipoles the ξ± Hankel transform integrates over: every ℓ below 50,
    then 200 log-spaced up to 6·10⁴."""
    import numpy as np

    return np.unique(np.concatenate([np.arange(2, 50), np.geomspace(50, 6e4, 200)]))


def _lensing(cosmo, params, s, name):
    import numpy as np
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
    """ξ± of one shear pair: Limber Cℓ on :func:`_ell`, then CCL's Hankel transform."""
    import numpy as np
    import pyccl as ccl

    from .sacc_io import XI_PLUS

    ell = _ell()
    cosmo, cl = cosmology(params), _shear_cl(params, s, rows, ell)
    theta = np.array([s.data[i].tags["theta"] for i in rows]) / 60.0
    xip, xim = (
        ccl.correlation(cosmo, ell=ell, C_ell=cl, theta=theta, type=t)
        for t in ("GG+", "GG-")
    )
    plus = np.array([s.data[i].data_type == XI_PLUS for i in rows])
    return np.where(plus, xip, xim)


def _cl(params, s, rows):
    """Cℓ_EE of one shear pair through its bandpower window."""
    import numpy as np

    window = s.get_bandpower_windows(rows)
    cl = _shear_cl(params, s, rows, np.asarray(window.values, float))
    return np.asarray(window.weight).T @ cl


def main(argv=None):
    import yaml

    parser = argparse.ArgumentParser(prog="python -m sp_validation.blinding")
    parser.add_argument("command", choices=("init",))
    parser.add_argument("blind")
    parser.add_argument("--cat-config", default=str(REPO_CAT_CONFIG))
    a = parser.parse_args(argv)
    init(a.blind, yaml.safe_load(Path(a.cat_config).read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
