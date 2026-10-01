"""Blinds, and the one call that conceals a data vector under one.

A blind is a named secret, ``<paths.blinds>/<name>.blind.json`` holding a
seed, the envelope the hidden point is drawn in and the fiducial point. It is
drawn once, ``python -m sp_validation.blinding init <name>``, and never
committed (``*.blind.json`` is gitignored).

:func:`conceal` adds the blind's shift, t(hidden) − t(fiducial), to every row
of a SACC; Smokescreen draws the hidden point from the seed and evaluates the
theory t at both points (:mod:`sp_validation.theory`).
"""

import argparse
import dataclasses
import json
import os
import re
import secrets
from pathlib import Path

import numpy as np

from . import theory as _theory

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
    :func:`sp_validation.theory.shear`) evaluated on ``s``. A failure raises
    :class:`BlindingError` naming only the exception type, so no message or
    traceback shows the hidden point.
    """
    from smokescreen.datavector import concealing_factor

    theory = theory or _theory.shear
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
        "fiducial": _theory.fiducial(),
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
