"""The blind: a secret seed, drawn once, and the shift it conceals signal by.

@sc blind-record
A blind is one read-only file, ``<paths.blinds>/<name>.blind.json``, outside
any git worktree: its seed (unencrypted: registry access is blind access), the
envelope the hidden point is drawn in, the fiducial and the fork's draw scheme.
Nothing here prints the seed or the hidden cosmology: :class:`Blind` holds a
name and a path, :func:`_hidden` returns a mapping whose repr is
``<hidden cosmology>``, and a theory failure at the hidden point is reported
by exception type alone.

@sc hidden-draw-uniform-s8-om
The hidden point is the fiducial (:func:`sp_validation.theory.fiducial`) moved
by the fork's per-key draw from the seed, uniform within the envelope in S8 and
Ωm.

``python -m sp_validation.blinding init <name>`` draws a blind; ``show <name>``
prints its public record (everything but the seed).
"""

import argparse
import dataclasses
import io
import json
import os
import secrets
import warnings
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from . import custody as _custody
from . import sacc_io
from . import theory as _theory

ENVELOPE = {"S8": 0.075, "Omega_m": 0.1}
SECRET = "secret: never print, paste or commit"


class BlindingError(RuntimeError):
    """Concealment failed; the message carries no hidden value."""


@dataclasses.dataclass(frozen=True)
class Blind:
    """An opened blind: its name and the path of its record."""

    name: str
    path: Path


def draw_scheme():
    """The installed fork's shift-draw semantics (``smokescreen.DRAW_SCHEME``)."""
    from smokescreen import DRAW_SCHEME

    return int(DRAW_SCHEME)


def open_blind(custody):
    """The blind a blinded custody names, its record matching the commitment."""
    path = Path(custody.registry or "") / f"{custody.blind}.blind.json"
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        raise _custody.CustodyError(
            f"cannot read blind {custody.blind} at {path}"
        ) from None
    if _custody.commitment(record) != custody.commitment:
        raise _custody.CustodyError(
            f"blind {custody.blind}'s record no longer matches the commitment this "
            "run was launched under; launch again"
        )
    if record["draw_scheme"] != draw_scheme():
        raise _custody.CustodyError(
            f"blind {custody.blind} was drawn under draw scheme "
            f"{record['draw_scheme']}; this smokescreen draws under {draw_scheme()}"
        )
    return Blind(custody.blind, path)


class _Hidden(Mapping):
    def __init__(self, values):
        self._values = values

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return "<hidden cosmology>"

    __str__ = __repr__


def _hidden(blind):
    """The blind's hidden point: its fiducial moved by the seed's draw."""
    from smokescreen.param_shifts import draw_param_shifts

    record = json.loads(blind.path.read_text())
    shift = draw_param_shifts(dict(record["envelope"]), record["seed"])
    fiducial = record["fiducial"]
    return _Hidden({k: v + shift[k] if k in shift else v for k, v in fiducial.items()})


def _at_hidden(s, rows, theory, blind):
    """The theory of ``rows`` at the hidden point; a failure names only its type.

    Warnings and the theory's own prints are silenced, and the error is raised
    outside the ``except``, so no context or frame carries the hidden point.
    """

    def evaluate():
        quiet = io.StringIO()
        with warnings.catch_warnings(), redirect_stdout(quiet), redirect_stderr(quiet):
            warnings.simplefilter("ignore")
            return _theory.predict(s, _hidden(blind), rows, theory)

    try:
        return evaluate()
    except Exception as err:  # the message must carry no value
        failure = type(err).__name__
    raise BlindingError(
        f"the theory failed at blind {blind.name}'s hidden point ({failure})"
    )


def conceal(s, blind, theory=None):
    """A copy of ``s`` with its shiftable rows moved by the blind's shift.

    The shift is t(hidden) − t(fiducial), from ``theory`` (default
    :data:`sp_validation.theory.THEORY`), evaluated at the fiducial first. A
    shiftable type without a theory, and a data type and tracer pair the blind
    leaves unmoved or moves to a non-finite value, are refused.
    """
    theory = _theory.THEORY if theory is None else theory
    rows = sacc_io.shiftable(s)
    types = {s.data[i].data_type for i in rows}
    if types - set(theory):
        raise BlindingError(
            f"no theory for the shiftable {sorted(types - set(theory))}; "
            "give it a function in sp_validation.theory.THEORY"
        )
    fiducial = json.loads(blind.path.read_text())["fiducial"]
    at_fiducial = _theory.predict(s, fiducial, rows, theory)
    delta = _at_hidden(s, rows, theory, blind) - at_fiducial
    groups = {}
    for n, i in enumerate(rows):
        groups.setdefault((s.data[i].data_type, s.data[i].tracers), []).append(n)
    for (data_type, pair), at in groups.items():
        if not np.all(np.isfinite(delta[at])) or not delta[at].any():
            raise BlindingError(
                f"blind {blind.name} leaves {data_type} of {pair} unmoved or "
                "non-finite; its theory must depend on the cosmology"
            )
    out = s.copy()
    for row, d in zip(rows, delta):
        out.data[int(row)].value += float(d)
    return out


def init(name, catalogues):
    """Draw blind ``name`` into the catalogue config's registry, once."""
    if not _custody.BLIND_NAME.fullmatch(name) or name in (
        _custody.NONE,
        _custody.MOCK,
    ):
        raise _custody.CustodyError(f"blind name {name!r}: use a-z, 0-9, - _ .")
    where = _custody.registry(catalogues)
    where.mkdir(mode=0o700, parents=True, exist_ok=True)
    record = {
        "_": SECRET,
        "seed": secrets.token_hex(32),
        "envelope": ENVELOPE,
        "fiducial": _theory.fiducial(),
        "draw_scheme": draw_scheme(),
    }
    path = where / f"{name}.blind.json"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    except FileExistsError:
        raise _custody.CustodyError(
            f"blind {name} exists at {path}; a blind is drawn once"
        ) from None
    with os.fdopen(fd, "w") as f:
        json.dump(record, f, indent=1)
    print(f"[blinding] drew blind {name}, commitment {_custody.commitment(record)}")
    return Blind(name, path)


def show(name, catalogues):
    """Print blind ``name``'s public record: everything but the seed."""
    path = _custody.registry(catalogues) / f"{name}.blind.json"
    record = json.loads(path.read_text())
    print(f"blind {name}, commitment {_custody.commitment(record)}")
    for key in ("envelope", "fiducial", "draw_scheme"):
        print(f"  {key}: {json.dumps(record[key])}")


def main(argv=None):
    import yaml

    parser = argparse.ArgumentParser(prog="python -m sp_validation.blinding")
    parser.add_argument("command", choices=("init", "show"))
    parser.add_argument("blind")
    parser.add_argument("--cat-config", default=str(_custody.REPO_CAT_CONFIG))
    a = parser.parse_args(argv)
    catalogues = yaml.safe_load(Path(a.cat_config).read_text())
    (init if a.command == "init" else show)(a.blind, catalogues)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
