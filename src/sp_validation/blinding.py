"""The blind: one secret seed, hence one hidden cosmology, and its custody.

:Name: blinding.py

:Description: A blind conceals a catalogue's cosmological signal (Muir et al.
    2019): every ξ± and pseudo-Cℓ_EE row is shifted by t(hidden) − t(fiducial),
    the theory difference between a hidden cosmology and the fiducial, before
    the row is first written (:func:`sp_validation.sacc_io.seal`). The hidden
    point is drawn uniformly in S8 and Ωm about the fiducial from a secret seed,
    with the Smokescreen fork's per-key RNG. COSEBIs and pure-E/B computed from
    concealed ξ± are concealed with it; the shift is pure E-mode, so B-mode null
    tests stay valid.

    A blind is drawn once, by a person, with ``python -m sp_validation.blinding
    init``, into the registry beside the catalogue config
    (``cosmo_val/blinds/<blind>/``): the public ``commitment.json`` (the seed's
    commitment and the full :class:`BlindingConfig`), the seed as Fernet
    ciphertext (``seed.fernet``), its ``key``, and the ``bases`` it covers. The
    seed exists on disk only inside the ciphertext; the encryption keeps anyone
    from reading it by accident. ``share`` adds a catalogue to a blind;
    ``reveal`` publishes the seed and moves the concealed files aside;
    ``audit`` proves blinded − true = shift(seed) once the true files are
    re-measured; ``verify`` checks a file's stamp against the record, seedless.
"""

import argparse
import dataclasses
import functools
import hashlib
import json
import os
import secrets
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from . import custody as _custody
from . import sacc_io
from .blinding_theory import TheoryConfig, cl_ee, xi_ccl

_BLIND_NAME = "abcdefghijklmnopqrstuvwxyz0123456789-_."


# --------------------------------------------------------------------------- #
# The config a blind is drawn under, and its digest
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class BlindingConfig:
    """The envelope of the hidden draw and the theory the shift is computed in.

    ``envelope`` maps a :class:`TheoryConfig` field to the half-width of its
    uniform draw about the fiducial ``theory``.
    """

    envelope: dict = dataclasses.field(
        default_factory=lambda: {"S8": 0.075, "Omega_m": 0.1}
    )
    theory: TheoryConfig = dataclasses.field(default_factory=TheoryConfig)

    def record(self):
        """The config as plain data: every field, numbers as floats."""
        return _canonical(self)

    @classmethod
    def from_record(cls, record):
        """The config a record describes; fields it omits take their defaults."""
        fields = {}
        if "envelope" in record:
            fields["envelope"] = {k: float(v) for k, v in record["envelope"].items()}
        if "theory" in record:
            fields["theory"] = TheoryConfig(**record["theory"])
        return cls(**fields)

    def digest(self):
        """sha256 of the canonical record; int and float literals agree."""
        text = json.dumps(self.record(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(value):
    if dataclasses.is_dataclass(value):
        return {
            f.name: _canonical(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, bool) or isinstance(value, str):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    raise TypeError(f"cannot serialise {value!r} into a blind's record")


def draw_scheme():
    """The installed fork's shift-draw semantics (``smokescreen.DRAW_SCHEME``)."""
    from smokescreen import DRAW_SCHEME

    return int(DRAW_SCHEME)


def hidden_theory(seed, config):
    """The hidden point: the fiducial theory moved by the seed's draw.

    @sc hidden-draw-uniform-s8-om
    The draw is uniform within the envelope in the physical coordinates it
    names (S8 and Ωm), by the fork's per-key RNG; CCL's σ8 and Ω_c follow from
    the drawn point through ``TheoryConfig.ccl_params``, never the reverse.
    """
    from smokescreen.param_shifts import draw_param_shifts

    shift = draw_param_shifts(dict(config.envelope), seed)
    return dataclasses.replace(
        config.theory,
        **{key: getattr(config.theory, key) + delta for key, delta in shift.items()},
    )


# --------------------------------------------------------------------------- #
# Opening a blind
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class Blind:
    """An opened blind. Its ``repr`` hides the seed."""

    name: str
    seed: str = dataclasses.field(repr=False)
    config: BlindingConfig
    hidden: TheoryConfig


def _read_record(registry, name):
    record = Path(registry) / name
    try:
        commitment = json.loads((record / "commitment.json").read_text())
    except (OSError, ValueError) as err:
        raise _custody.CustodyError(
            f"blind {name}: no readable record: {err}"
        ) from None
    return record, commitment


@functools.cache
def _open(registry, name):
    from cryptography.fernet import Fernet, InvalidToken

    record, commitment = _read_record(registry, name)
    try:
        key = (record / "key").read_bytes()
        payload = json.loads(Fernet(key).decrypt((record / "seed.fernet").read_bytes()))
    except (OSError, ValueError, InvalidToken) as err:
        raise _custody.CustodyError(
            f"blind {name}: the seed does not decrypt with its key ({type(err).__name__})"
        ) from None
    config = BlindingConfig.from_record(commitment["config"])
    checks = {
        "blind name": (payload["blind"], commitment["blind"], name),
        "config digest": (
            payload["config_digest"],
            commitment["config_digest"],
            config.digest(),
        ),
        "draw scheme": (
            payload["draw_scheme"],
            commitment["draw_scheme"],
            draw_scheme(),
        ),
        "seed commitment": (
            _custody.seed_commitment(payload["seed"]),
            commitment["seed_commitment"],
        ),
    }
    for what, values in checks.items():
        if len(set(values)) != 1:
            raise _custody.CustodyError(
                f"blind {name}: the {what} disagrees between the sealed seed, the "
                "record and this install; the record was edited or the install "
                "draws differently"
            )
    seed = payload["seed"]
    return Blind(name, seed, config, hidden_theory(seed, config))


def open_blind(custody):
    """The verified blind a blinded custody names (cached per process)."""
    if custody.status != "blinded":
        raise _custody.CustodyError(f"{custody.catalogue} is {custody.status}")
    blind = _open(Path(custody.registry), custody.blind)
    if _custody.seed_commitment(blind.seed) != custody.commitment:
        raise _custody.CustodyError(
            f"blind {custody.blind} is not the one {custody.catalogue} resolved"
        )
    return blind


# --------------------------------------------------------------------------- #
# Blocks, from the content, and the concealment
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class Block:
    """Rows of one statistic and tracer pair, and their theory at a point."""

    name: str
    rows: np.ndarray
    theory: object  # (ccl params, TheoryConfig) -> values aligned to rows


def _pair_nz(s, tracers):
    nzs = []
    for name in tracers:
        tracer = s.tracers[name]
        if not hasattr(tracer, "nz"):
            raise ValueError(f"shiftable rows name tracer {name}, which has no n(z)")
        nzs.append((np.asarray(tracer.z, float), np.asarray(tracer.nz, float)))
    return nzs


def _xi_block(s, tracers, rows):
    nz_i, nz_j = _pair_nz(s, tracers)
    theta = np.array([s.data[i].tags["theta"] for i in rows], float)
    plus = np.array([s.data[i].data_type == sacc_io.XI_PLUS for i in rows])
    grid = np.unique(theta)
    at = np.searchsorted(grid, theta)

    def theory(params, config):
        xip, xim = xi_ccl(params, config, nz_i, nz_j, grid)
        return np.where(plus, np.asarray(xip)[at], np.asarray(xim)[at])

    return Block(f"ξ± {tracers[0]}×{tracers[1]}", np.asarray(rows), theory)


def _cl_block(s, tracers, rows):
    nz_i, nz_j = _pair_nz(s, tracers)
    window = s.get_bandpower_windows(rows)
    ell = np.asarray(window.values, float)
    weight = np.asarray(window.weight, float)

    def theory(params, config):
        return weight.T @ cl_ee(params, config, nz_i, nz_j, ell)

    return Block(f"Cℓ_EE {tracers[0]}×{tracers[1]}", np.asarray(rows), theory)


def shiftable_blocks(s):
    """Every ξ± and Cℓ_EE block of ``s``: by tracer pair (and window, for Cℓ).

    @sc shift-from-content
    Blocks are discovered from the data types and tracers alone, whatever the
    ``grid`` tags or file names; each ξ± row is evaluated at its own ``theta``
    from its pair's two n(z), each Cℓ_EE row through its bandpower window. A
    shiftable row without a ``theta`` tag or a window is refused.
    """
    xi, cl = {}, {}
    for i, dp in enumerate(s.data):
        if dp.data_type in (sacc_io.XI_PLUS, sacc_io.XI_MINUS):
            if "theta" not in dp.tags:
                raise ValueError(f"ξ± row {i} has no theta tag to evaluate theory at")
            xi.setdefault(tuple(dp.tracers), []).append(i)
        elif dp.data_type == sacc_io.CL_EE:
            if dp.tags.get("window") is None:
                raise ValueError(f"Cℓ_EE row {i} has no bandpower window")
            key = (tuple(dp.tracers), id(dp.tags["window"]))
            cl.setdefault(key, []).append(i)
    return [_xi_block(s, t, rows) for t, rows in xi.items()] + [
        _cl_block(s, t, rows) for (t, _), rows in cl.items()
    ]


def _factor(block, fiducial, hidden):
    """t(hidden) − t(fiducial) on ``block``'s rows, by the fork."""
    from smokescreen import factor_from_params

    return np.asarray(
        factor_from_params(
            fiducial.ccl_params(),
            hidden.ccl_params(),
            theory_fn=lambda params: block.theory(params, fiducial),
        ),
        float,
    )


def factors(s, fiducial, hidden):
    """Each shiftable block of ``s`` and its factor; together they cover every row."""
    blocks = shiftable_blocks(s)
    covered = np.sort(np.concatenate([b.rows for b in blocks])) if blocks else []
    shiftable = [i for i, dp in enumerate(s.data) if dp.data_type in sacc_io.SHIFTABLE]
    if not np.array_equal(covered, shiftable):
        raise ValueError("shiftable blocks do not cover every ξ± and Cℓ_EE row once")
    out = []
    for block in blocks:
        factor = _factor(block, fiducial, hidden)
        if factor.shape != block.rows.shape or not np.all(np.isfinite(factor)):
            raise ValueError(
                f"{block.name}: the theory left rows unfilled or non-finite"
            )
        out.append((block, factor))
    return out


def conceal(s, blind):
    """A copy of ``s`` with every ξ± and Cℓ_EE row shifted by the blind.

    Every block's factor is computed before any value changes, and only the
    copy changes.
    """
    shifts = factors(s, blind.config.theory, blind.hidden)
    out = s.copy()
    for block, factor in shifts:
        for row, delta in zip(block.rows, factor):
            out.data[int(row)].value = out.data[int(row)].value + float(delta)
    return out


# --------------------------------------------------------------------------- #
# Custody commands: init, share, reveal, audit, verify
# --------------------------------------------------------------------------- #
def _catalogues(cat_config):
    return yaml.safe_load(Path(cat_config).read_text())


def _theory_stack():
    from importlib.metadata import PackageNotFoundError, version

    stack = {}
    for package in ("pyccl", "camb", "smokescreen"):
        try:
            stack[package] = version(package)
        except PackageNotFoundError:
            stack[package] = None
    return stack


def _declared_blinded(catalogues, registry, base):
    """Refuse a base ``init`` or ``share`` cannot give a blind."""
    if _custody.base_catalogue(catalogues, base) != base:
        raise _custody.CustodyError(
            f"{base} is a variant of {_custody.base_catalogue(catalogues, base)}; "
            "a blind covers base catalogues"
        )
    declared = _custody.declaration(catalogues, base)
    if declared in ("unblinded", "mock"):
        raise _custody.CustodyError(
            f"{base} is declared {declared}; declare it blinded first"
        )
    covering = [r.name for r in _custody.records(registry).values() if base in r.bases]
    if covering:
        raise _custody.CustodyError(f"{base} is already covered by blind {covering[0]}")


def init(name, bases, *, cat_config, config=None):
    """Draw blind ``name`` for ``bases`` and write its record; return its directory.

    @sc blind-drawn-once
    The one place a seed is drawn. Existing state is refused, never replaced,
    so no second draw exists for a catalogue unless a committed record is
    deleted. The seed is held in memory and written only as ciphertext.
    """
    from cryptography.fernet import Fernet

    config = config or BlindingConfig()
    registry = _custody.registry_of(cat_config)
    catalogues = _catalogues(cat_config)
    record, staging = registry / name, registry / f".{name}.tmp"
    if not name or name[0] in "-_." or set(name) - set(_BLIND_NAME):
        raise _custody.CustodyError(f"blind name {name!r}: use a-z, 0-9, - _ .")
    for path in (record, staging):
        if path.exists():
            raise _custody.CustodyError(
                f"{path} exists; a blind is drawn once (delete a staging "
                "directory an interrupted init left)"
            )
    for base in bases:
        _declared_blinded(catalogues, registry, base)

    seed = secrets.token_hex(16)
    key = Fernet.generate_key()
    payload = {
        "blind": name,
        "seed": seed,
        "config_digest": config.digest(),
        "draw_scheme": draw_scheme(),
    }
    ciphertext = Fernet(key).encrypt(json.dumps(payload).encode("utf-8"))
    if json.loads(Fernet(key).decrypt(ciphertext)) != payload:
        raise _custody.CustodyError("the sealed seed does not round-trip")
    commitment = {
        "blind": name,
        "seed_commitment": _custody.seed_commitment(seed),
        "config": config.record(),
        "config_digest": config.digest(),
        "draw_scheme": draw_scheme(),
        "theory_stack": _theory_stack(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    staging.mkdir(parents=True)
    (staging / "commitment.json").write_text(json.dumps(commitment, indent=2) + "\n")
    (staging / "seed.fernet").write_bytes(ciphertext)
    (staging / "key").write_bytes(key)
    (staging / "bases").write_text("".join(f"{b}\n" for b in bases))
    for part in ("commitment.json", "seed.fernet", "key"):
        os.chmod(staging / part, 0o444)
    os.replace(staging, record)
    print(f"[blinding] drew blind {name} for {', '.join(bases)}")
    print(f"[blinding] commit {record.relative_to(registry.parent.parent)}/")
    return record


def share(name, base, *, cat_config):
    """Add ``base`` to blind ``name``'s ``bases``."""
    registry = _custody.registry_of(cat_config)
    records = _custody.records(registry)
    if name not in records:
        raise _custody.CustodyError(f"no blind {name} in {registry}")
    if records[name].revealed is not None:
        raise _custody.CustodyError(f"blind {name} is revealed; draw a new one")
    _declared_blinded(_catalogues(cat_config), registry, base)
    with open(registry / name / "bases", "a") as f:
        f.write(f"{base}\n")
    print(f"[blinding] {base} shares blind {name}; commit {registry / name / 'bases'}")


def _parts_under(root, commitment, skip):
    """SACC files under ``root`` concealed under ``commitment``, found by content."""
    for path in sorted(Path(root).rglob("*.sacc")):
        if skip in path.parents:
            continue
        try:
            stamp = _custody.read_stamp(sacc_io.load(path).metadata)
        except _custody.CustodyError:
            continue  # not born through the door, so not under any blind
        if stamp.commitment == commitment:
            yield path


def reveal(name, *, root, cat_config):
    """Publish blind ``name``'s seed and move its concealed files aside.

    @sc reveal-is-reproduction
    Revealing never subtracts a shift: the seed is published, every file
    concealed under the blind moves to ``<root>/revealed/<blind>/``, the
    declaration is flipped by a reviewed change, and the pipeline re-measures
    true products that :func:`audit` compares against the archive. No file ever
    mixes shifted and true signal.
    """
    registry = _custody.registry_of(cat_config)
    record, commitment = _read_record(registry, name)
    blind = _open(registry, name)
    revealed = record / "revealed.json"
    if revealed.exists():
        if json.loads(revealed.read_text())["seed"] != blind.seed:
            raise _custody.CustodyError(f"{revealed} records another seed")
    else:
        fd = os.open(revealed, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(fd, "w") as f:
            json.dump({"seed": blind.seed}, f)
    archive = Path(root) / "revealed" / name
    moved = 0
    for path in _parts_under(root, commitment["seed_commitment"], archive):
        target = archive / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, target)
        moved += 1
    print(f"[blinding] published the seed of {name}; moved {moved} files to {archive}")
    print(
        f"[blinding] commit {revealed} and declare `blinding: unblinded` on "
        f"{', '.join(_custody.records(registry)[name].bases)}, then re-run"
    )
    return archive


# A re-measurement repeats a measurement to float noise (TreeCorr's threaded
# sums reorder); the audit compares its numbers to this relative tolerance.
_REMEASURED = 1e-10
# How far, in σ, a derived B-mode may move under a blind. COSEBIs B-modes move
# by ~1e-4σ. Pure-mode ξ_B carries the estimator's own E→B leakage of the
# shift: up to 1e-2σ on the production grids at the envelope's edge
# (test_blinding), several times more on coarse grids. A derivation from mixed
# inputs moves B by ~1σ.
_B_SIGMA = {
    sacc_io.COSEBI_BB: 1e-2,
    sacc_io.PURE_TYPES["xip_B"]: 1e-1,
    sacc_io.PURE_TYPES["xim_B"]: 1e-1,
}


def _same(x, y):
    if isinstance(x, str) or isinstance(y, str):
        return x == y
    return bool(np.isclose(x, y, rtol=_REMEASURED, atol=0.0) or x == y)


def _rows_match(a, b):
    if len(a.data) != len(b.data):
        return False
    for x, y in zip(a.data, b.data):
        tx, ty = ({k: v for k, v in d.tags.items() if k != "window"} for d in (x, y))
        if (x.data_type, tuple(x.tracers), set(tx)) != (
            y.data_type,
            tuple(y.tracers),
            set(ty),
        ) or not all(_same(tx[k], ty[k]) for k in tx):
            return False
    return True


def _covariance(s):
    return None if s.covariance is None else np.asarray(s.covariance.dense)


def _same_covariance(a, b):
    if a is None or b is None:
        return a is None and b is None
    scale = np.max(np.abs(b))
    return a.shape == b.shape and np.allclose(a, b, rtol=0.0, atol=_REMEASURED * scale)


def _audit_part(blinded, true, fiducial, hidden, tolerance):
    """Problems with one archived part against its re-measured twin."""
    stamp, true_stamp = (_custody.read_stamp(x.metadata) for x in (blinded, true))
    if true_stamp.token != f"unblinded:{stamp.catalogue}":
        return [f"the live file is stamped {true_stamp.token}"], None
    problems = []
    if not _rows_match(blinded, true):
        problems.append("rows, tags or tracers differ")
        return problems, None
    if set(blinded.tracers) != set(true.tracers):
        problems.append("tracers differ")
    cov_b, cov_t = _covariance(blinded), _covariance(true)
    if not _same_covariance(cov_b, cov_t):
        problems.append("covariances differ")
    centres = [x.metadata.get("patch_centers_sha256") for x in (blinded, true)]
    if centres[0] != centres[1]:
        problems.append(f"patch centres differ: {centres}")

    delta = np.asarray(blinded.mean) - np.asarray(true.mean)
    residual, shifted = 0.0, np.zeros(len(delta), bool)
    for block, factor in factors(true, fiducial, hidden):
        scale = np.max(np.abs(factor))
        residual = max(residual, np.max(np.abs(delta[block.rows] - factor)) / scale)
        shifted[block.rows] = True
    if residual > tolerance:
        problems.append(f"blinded − true ≠ shift(seed): residual {residual:.2e}")

    sigma = None if cov_t is None else np.sqrt(np.diag(cov_t))
    derived = (sacc_io.COSEBI_EE, sacc_io.COSEBI_BB, *sacc_io.PURE_TYPES.values())
    moved = {}
    for i, dp in enumerate(true.data):
        kind = dp.data_type
        if shifted[i]:
            continue
        if kind in derived:
            if sigma is None:
                problems.append(f"row {i} ({kind}) has no σ to judge its shift by")
            elif np.isfinite(delta[i]):
                moved[kind] = max(moved.get(kind, 0.0), abs(delta[i]) / sigma[i])
        elif abs(delta[i]) > 1e-8 * max(abs(dp.value), 1e-300):
            problems.append(
                f"row {i} ({kind}) moved, but the blind leaves it unshifted"
            )
    for kind, bound in _B_SIGMA.items():
        if moved.get(kind, 0.0) > bound:
            problems.append(f"{kind} moved by {moved[kind]:.2e}σ under the blind")
    return problems, {"residual": residual, "shift_over_sigma": moved}


def audit(name, *, archive, true_root, cat_config, out=None):
    """Check every archived file of blind ``name`` against its re-measured twin.

    Needs no key: the seed is the published one, checked against the
    commitment. For each archived file, the live file at the same relative path
    must be stamped unblinded for the same catalogue, with the same rows, tags,
    tracers, covariance and patch centres (numbers to a re-measurement's float
    noise), and blinded − true must equal the seed's shift on every ξ± and
    Cℓ_EE block to 1e-6 of the block's largest shift (1e-3 when the theory
    stack differs from the one the blind was drawn with); COSEBIs and pure-E/B
    B rows may move by at most ``_B_SIGMA``, and rows the blind leaves
    unshifted not at all. Each part reports the largest shift of each derived
    statistic, in σ.
    """
    registry = _custody.registry_of(cat_config)
    record, commitment = _read_record(registry, name)
    report = {"blind": name, "ok": False, "problems": [], "parts": {}}
    revealed = record / "revealed.json"
    seed = json.loads(revealed.read_text())["seed"] if revealed.exists() else None
    if seed is None or _custody.seed_commitment(seed) != commitment["seed_commitment"]:
        report["problems"].append("the published seed does not match the commitment")
        return _write_report(report, out)
    config = BlindingConfig.from_record(commitment["config"])
    if config.digest() != commitment["config_digest"]:
        report["problems"].append("the recorded config does not match its digest")
    if int(commitment["draw_scheme"]) != draw_scheme():
        report["problems"].append("this install draws under another scheme")
    stack = _theory_stack()
    tolerance = 1e-6
    if stack != commitment.get("theory_stack"):
        tolerance = 1e-3
        report["theory_stack"] = {
            "record": commitment.get("theory_stack"),
            "now": stack,
        }
    hidden = hidden_theory(seed, config)
    report["shift"] = {
        key: getattr(hidden, key) - getattr(config.theory, key)
        for key in config.envelope
    }
    archive, true_root = Path(archive), Path(true_root)
    for path in sorted(archive.rglob("*.sacc")):
        relative = str(path.relative_to(archive))
        live = true_root / relative
        if not live.exists():
            report["parts"][relative] = {"problems": ["no re-measured file"]}
            continue
        blinded = sacc_io.load(path)
        stamp = _custody.read_stamp(blinded.metadata)
        if stamp.commitment != commitment["seed_commitment"]:
            report["parts"][relative] = {"problems": ["not concealed under this blind"]}
            continue
        problems, numbers = _audit_part(
            blinded, sacc_io.load(live), config.theory, hidden, tolerance
        )
        report["parts"][relative] = {"problems": problems, **(numbers or {})}
    report["ok"] = (
        bool(report["parts"])
        and not report["problems"]
        and not any(part["problems"] for part in report["parts"].values())
    )
    return _write_report(report, out)


def _write_report(report, out):
    if out is not None:
        Path(out).write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


def verify(path, *, cat_config):
    """Problems with a file's stamp against the registry and declaration (seedless)."""
    stamp = _custody.read_stamp(sacc_io.load(path).metadata)
    registry = _custody.registry_of(cat_config)
    problems = []
    if stamp.status == "blinded":
        record = _custody.records(registry).get(stamp.blind)
        if record is None:
            return [f"no blind {stamp.blind} in {registry}"]
        c = record.commitment
        for what, stamped, recorded in (
            ("seed commitment", stamp.commitment, c["seed_commitment"]),
            ("config digest", stamp.config_digest, c["config_digest"]),
            ("draw scheme", stamp.draw_scheme, int(c["draw_scheme"])),
        ):
            if stamped != recorded:
                problems.append(f"the {what} differs from the record")
        if stamp.draw_scheme != draw_scheme():
            problems.append("this install draws under another scheme")
        if stamp.catalogue not in record.bases:
            problems.append(f"blind {record.name} does not cover {stamp.catalogue}")
    try:
        declared = _custody.custody_of(
            _catalogues(cat_config), stamp.catalogue, registry=registry
        )
        if declared.stamp != stamp.stamp:
            problems.append(f"{stamp.catalogue} is declared {declared.token}")
    except _custody.CustodyError as err:
        problems.append(str(err))
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m sp_validation.blinding", description=__doc__.split("\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command, arguments in {
        "init": ("blind", "bases+"),
        "share": ("blind", "base"),
        "reveal": ("blind",),
        "audit": ("blind",),
        "verify": ("file",),
    }.items():
        sub = commands.add_parser(command)
        for argument in arguments:
            name, plus = argument.rstrip("+"), argument.endswith("+")
            sub.add_argument(name, nargs="+" if plus else None)
        sub.add_argument("--cat-config", required=True)
    commands.choices["init"].add_argument(
        "--config", help="JSON record of BlindingConfig fields (defaults otherwise)"
    )
    commands.choices["reveal"].add_argument("--root", required=True)
    commands.choices["audit"].add_argument("--archive", required=True)
    commands.choices["audit"].add_argument("--true-root", required=True)
    commands.choices["audit"].add_argument("--out")
    a = parser.parse_args(argv)

    if a.command == "init":
        config = (
            BlindingConfig.from_record(json.loads(Path(a.config).read_text()))
            if a.config
            else None
        )
        init(a.blind, a.bases, cat_config=a.cat_config, config=config)
    elif a.command == "share":
        share(a.blind, a.base, cat_config=a.cat_config)
    elif a.command == "reveal":
        reveal(a.blind, root=a.root, cat_config=a.cat_config)
    elif a.command == "audit":
        report = audit(
            a.blind,
            archive=a.archive,
            true_root=a.true_root,
            cat_config=a.cat_config,
            out=a.out,
        )
        print(json.dumps(report, indent=2, default=float))
        return 0 if report["ok"] else 1
    elif a.command == "verify":
        problems = verify(a.file, cat_config=a.cat_config)
        for problem in problems:
            print(f"[verify] {problem}")
        print(f"[verify] {a.file}: {'ok' if not problems else 'FAILED'}")
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
