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
The hidden point is the fiducial moved by the fork's per-key draw from the
seed, uniform within the envelope in S8 and Ωm.

``python -m sp_validation.blinding init <name>`` draws a blind; ``show <name>``
prints its public record (everything but the seed).
"""

import argparse
import dataclasses
import json
import os
import secrets
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from . import custody as _custody
from . import sacc_io
from .blinding_theory import TheoryConfig, cl_ee, xi_ccl

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


def _nz(s, name):
    tracer = s.tracers[name]
    if not hasattr(tracer, "nz"):
        raise ValueError(f"shiftable rows name tracer {name}, which has no n(z)")
    return np.asarray(tracer.z, float), np.asarray(tracer.nz, float)


def _blocks(s):
    """Each ξ± tracer pair and Cℓ_EE (pair, window): rows, and theory(params, config)."""
    xi, cl = {}, {}
    for i, dp in enumerate(s.data):
        if dp.data_type in (sacc_io.XI_PLUS, sacc_io.XI_MINUS):
            xi.setdefault(tuple(dp.tracers), []).append(i)
        elif dp.data_type == sacc_io.CL_EE:
            key = (tuple(dp.tracers), id(dp.tags.get("window")))
            cl.setdefault(key, []).append(i)
    blocks = []
    for pair, rows in xi.items():
        theta = np.array([s.data[i].tags["theta"] for i in rows], float)
        grid, at = np.unique(theta, return_inverse=True)
        plus = np.array([s.data[i].data_type == sacc_io.XI_PLUS for i in rows])
        nzs = [_nz(s, t) for t in pair]

        def theory(p, c, nzs=nzs, grid=grid, at=at, plus=plus):
            xip, xim = xi_ccl(p, c, *nzs, grid)
            return np.where(plus, np.asarray(xip)[at], np.asarray(xim)[at])

        blocks.append((np.array(rows), theory))
    for (pair, _), rows in cl.items():
        window = s.get_bandpower_windows(rows)
        nzs = [_nz(s, t) for t in pair]

        def theory(p, c, nzs=nzs, w=window):
            return np.asarray(w.weight).T @ cl_ee(p, c, *nzs, np.asarray(w.values))

        blocks.append((np.array(rows), theory))
    return blocks


def _at_hidden(blocks, fiducial, blind):
    """Each block's theory at the hidden point; a failure names only its type."""

    def evaluate():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            point = dataclasses.replace(fiducial, **_hidden(blind))
            return [theory(point.ccl_params(), fiducial) for _, theory in blocks]

    try:
        return evaluate()
    except Exception as err:  # the message must carry no value
        failure = type(err).__name__
    raise BlindingError(
        f"the theory failed at blind {blind.name}'s hidden point ({failure})"
    )


def conceal(s, blind):
    """A copy of ``s`` with every ξ± and Cℓ_EE row shifted by the blind.

    The shift is t(hidden) − t(fiducial), evaluated at the fiducial first; a
    block the blind leaves unmoved, or moves to a non-finite value, is refused.
    """
    record = json.loads(blind.path.read_text())
    fiducial = TheoryConfig(**record["fiducial"])
    blocks = _blocks(s)
    at_fiducial = [theory(fiducial.ccl_params(), fiducial) for _, theory in blocks]
    out = s.copy()
    for (rows, _), t_fid, t_hid in zip(
        blocks, at_fiducial, _at_hidden(blocks, fiducial, blind)
    ):
        delta = np.asarray(t_hid, float) - np.asarray(t_fid, float)
        if (
            delta.shape != rows.shape
            or not np.all(np.isfinite(delta))
            or not delta.any()
        ):
            raise BlindingError(
                f"blind {blind.name} leaves {len(rows)} shiftable rows unmoved or "
                "non-finite; the theory must depend on the cosmology"
            )
        for row, d in zip(rows, delta):
            out.data[int(row)].value += float(d)
    return out


def init(name, catalogues, *, fiducial=None):
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
        "fiducial": fiducial or dataclasses.asdict(TheoryConfig()),
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
