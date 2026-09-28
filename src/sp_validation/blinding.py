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

    The commands of ``python -m sp_validation.blinding`` are the only writers
    of the blind registry (``cosmo_val/blinds/CONTRACTS``): ``init`` draws a
    blind, once, by a person; ``share`` adds a catalogue to it; ``reveal``
    publishes its seed and moves the concealed files aside. ``audit`` checks
    the re-measured true files against that archive, and ``verify`` checks a
    file's stamp against its catalogue's custody, seedless.
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

    def digest(self):
        """sha256 of the canonical record; int and float literals agree."""
        return record_digest(self.record())


def record_digest(record):
    """sha256 of a config record's canonical JSON, whatever the code's schema."""
    text = json.dumps(record, sort_keys=True, separators=(",", ":"))
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


# A blind's record opens under this code only if it names every TheoryConfig
# field, or omits one listed here with the value under which that field changes
# no shift: a record drawn without it then reproduces its shift.
NEUTRAL = {}


def _recorded_config(name, record):
    """The :class:`BlindingConfig` of blind ``name``, from its config record.

    The one reader of the record format. Refuses, naming the fields, a record
    whose shift this code cannot reproduce: one naming a field the code lacks,
    or lacking a code field that has no value in :data:`NEUTRAL`.
    """
    fields = {f.name for f in dataclasses.fields(TheoryConfig)}
    theory = {**NEUTRAL, **record.get("theory", {})}
    envelope = record.get("envelope", {})
    foreign = sorted(set(record) - {"envelope", "theory"})
    foreign += sorted((set(theory) | set(envelope)) - fields)
    unset = sorted(fields - set(theory)) + ["envelope"] * ("envelope" not in record)
    if foreign or unset:
        raise _custody.CustodyError(
            f"blind {name}'s config does not fit this code. Fields the record "
            f"names and this code lacks: {foreign or 'none'}. Fields this code "
            "has and the record lacks, with no value in blinding.NEUTRAL: "
            f"{unset or 'none'}. A drawn blind opens under the code it was "
            "drawn with, or once each new field has its neutral value."
        )
    return BlindingConfig(
        envelope={k: float(v) for k, v in envelope.items()},
        theory=TheoryConfig(**theory),
    )


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


@functools.cache
def _open(registry, name):
    path = Path(registry) / name
    if not path.is_dir():
        raise _custody.CustodyError(f"no blind {name} in {registry}")
    return _read_blind(path)


def _read_blind(path):
    """The blind stored at directory ``path``, its sealed seed checked against
    its public record."""
    from cryptography.fernet import Fernet, InvalidToken

    name = path.name
    c = _custody.read_record(path).commitment
    try:
        key = (path / "key").read_bytes()
        ciphertext = (path / "seed.fernet").read_bytes()
        payload = json.loads(Fernet(key).decrypt(ciphertext))
    except (OSError, ValueError, InvalidToken) as err:
        raise _custody.CustodyError(
            f"blind {name}: the seed does not decrypt with its key "
            f"({type(err).__name__})"
        ) from None
    for what, values in {
        "blind name": (payload["blind"], c["blind"], name),
        "config digest": (
            payload["config_digest"],
            c["config_digest"],
            record_digest(c["config"]),
        ),
        "draw scheme": (payload["draw_scheme"], c["draw_scheme"]),
        "seed commitment": (
            _custody.seed_commitment(payload["seed"]),
            c["seed_commitment"],
        ),
    }.items():
        if len(set(values)) != 1:
            raise _custody.CustodyError(
                f"blind {name}: the {what} disagrees between the sealed seed and "
                "the record; the record was edited"
            )
    if c["draw_scheme"] != draw_scheme():
        raise _custody.CustodyError(
            f"blind {name} was drawn under draw scheme {c['draw_scheme']}; this "
            f"install's smokescreen draws under {draw_scheme()}"
        )
    config = _recorded_config(name, c["config"])
    return Blind(name, payload["seed"], config, hidden_theory(payload["seed"], config))


def open_blind(custody):
    """The verified blind a blinded custody names (cached per process)."""
    if custody.status != "blinded":
        raise _custody.CustodyError(f"{custody.catalogue} is {custody.status}")
    return _open(Path(custody.registry), custody.blind)


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
    """Each shiftable block of ``s`` and its factor."""
    out = []
    for block in shiftable_blocks(s):
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


def declared_custody(cat_config, version):
    """The custody ``version`` is declared under in the catalogue config at
    ``cat_config``, with the blind registry beside it."""
    return _custody.custody_of(
        _catalogues(cat_config), version, registry=_custody.registry_of(cat_config)
    )


def _theory_stack():
    from importlib.metadata import PackageNotFoundError, version

    stack = {}
    for package in ("pyccl", "camb", "smokescreen"):
        try:
            stack[package] = version(package)
        except PackageNotFoundError:
            stack[package] = None
    return stack


def _may_cover(catalogues, recs, name, bases):
    """Refuse ``bases`` blind ``name`` cannot cover: a variant, a catalogue
    declared unblinded or mock or covered already, or one whose shear file
    another catalogue would go on reading otherwise concealed."""
    for base in bases:
        if _custody.base_catalogue(catalogues, base) != base:
            raise _custody.CustodyError(
                f"{base} is a variant of "
                f"{_custody.base_catalogue(catalogues, base)}; a blind covers base "
                "catalogues"
            )
        declared = _custody.declaration(catalogues, base)
        if declared in ("unblinded", "mock"):
            raise _custody.CustodyError(
                f"{base} is declared {declared}; declare it blinded first"
            )
        covering = [r.name for r in recs.values() if base in r.bases]
        if covering:
            raise _custody.CustodyError(
                f"{base} is already covered by blind {covering[0]}"
            )
    covered = recs[name].bases if name in recs else ()
    after = {**recs, name: _custody.Record(name, {}, (*covered, *bases), None)}
    for base in bases:
        _custody.refuse_twins(catalogues, after, base)


def _conceal_one_row(blind):
    """Conceal one ξ± row under ``blind``, as every blinded birth does; raise
    if it cannot."""
    z = np.linspace(0.0, 2.0, 101)
    s = sacc_io.new_sacc({0: (z, np.exp(-0.5 * ((z - 0.7) / 0.2) ** 2))})
    sacc_io.add_xi(s, (0, 0), [10.0], [0.0], [0.0], grid="probe")
    try:
        factors(s, blind.config.theory, blind.hidden)
    except Exception as err:
        raise _custody.CustodyError(
            f"no row conceals under blind {blind.name}'s config: {type(err).__name__}: {err}"
        ) from err


def init(name, bases, *, cat_config, config=None):
    """Draw blind ``name`` for ``bases`` and write its record; return its directory.

    The one place a seed is drawn. Existing state is refused, never replaced,
    and the seed is held in memory and written only as ciphertext. The record
    is written to a staging directory, opened from there as jobs open it, and
    made to conceal a row before it takes its name, so every blind in the
    registry is one its jobs can conceal under.
    """
    from cryptography.fernet import Fernet

    config = config or BlindingConfig()
    registry = _custody.registry_of(cat_config)
    catalogues = _catalogues(cat_config)
    record, staging = registry / name, registry / f".{name}.tmp" / name
    if not name or name[0] in "-_." or set(name) - set(_BLIND_NAME):
        raise _custody.CustodyError(f"blind name {name!r}: use a-z, 0-9, - _ .")
    for path in (record, staging.parent):
        if path.exists():
            raise _custody.CustodyError(
                f"{path} exists; a blind is drawn once (delete a staging "
                "directory an interrupted init left)"
            )
    _may_cover(catalogues, _custody.records(registry), name, bases)

    seed = secrets.token_hex(16)
    key = Fernet.generate_key()
    payload = {
        "blind": name,
        "seed": seed,
        "config_digest": config.digest(),
        "draw_scheme": draw_scheme(),
    }
    ciphertext = Fernet(key).encrypt(json.dumps(payload).encode("utf-8"))
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
    try:
        _conceal_one_row(_read_blind(staging))
    except _custody.CustodyError:
        shutil.rmtree(staging.parent)
        raise
    os.replace(staging, record)
    staging.parent.rmdir()
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
    _may_cover(_catalogues(cat_config), records, name, [base])
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
    blind = _open(registry, name)
    record = _custody.records(registry)[name]
    revealed = registry / name / "revealed.json"
    if record.revealed is not None and record.revealed != blind.seed:
        raise _custody.CustodyError(f"{revealed} records another seed")
    archive = Path(root) / "revealed" / name
    concealed = list(_parts_under(root, record.commitment["seed_commitment"], archive))
    # Publishing cannot be undone, and a reveal that archived nothing leaves
    # the concealed products unauditable: an empty (mistyped, or unbound in
    # the container) root is refused first. A re-run finds its archive.
    if not concealed and not any(archive.rglob("*.sacc")):
        raise _custody.CustodyError(
            f"nothing concealed under blind {name} in {root}; check --root (and "
            "that the container binds it). The seed stays unpublished."
        )
    if record.revealed is None:
        fd = os.open(revealed, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(fd, "w") as f:
            json.dump({"seed": blind.seed}, f)
    for path in concealed:
        target = archive / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, target)
    print(
        f"[blinding] published the seed of {name}; moved {len(concealed)} files "
        f"to {archive}"
    )
    print(
        f"[blinding] commit {revealed} and declare `blinding: unblinded` on "
        f"{', '.join(record.bases)}, then re-run"
    )
    return archive


# How closely blinded − true must equal the seed's shift, relative to the
# block's largest shift: under the theory stack the blind was drawn with, and
# under another one.
SHIFT_TOLERANCE = 1e-6
SHIFT_TOLERANCE_OTHER_STACK = 1e-3
# A re-measurement repeats a measurement to float noise (TreeCorr's threaded
# sums reorder); the audit compares its numbers to this relative tolerance.
_REMEASURED = 1e-10
# How far, in σ, a derived B-mode may move under a blind: 10× what a shift at
# the envelope's corners induces on the production grids, with shape-noise σ at
# UNIONS depth. There COSEBIs B (20 modes, [12, 83]′) move by ≤ 6e-3σ, and
# pure-mode ξ_B by ≤ 1e-2σ (ξ−_B at θ ≈ 2′: the estimator's own E→B leakage of
# the shift). A derivation from mixed inputs moves B by ~1σ.
B_SIGMA = {
    sacc_io.COSEBI_BB: 5e-2,
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


def _signal(s):
    """``s``'s signal rows and their covariance, or None if it has none."""
    keep = np.array([dp.data_type in sacc_io.SIGNAL for dp in s.data], bool)
    if not keep.any():
        return None
    out = s.copy()
    out.keep_indices(keep)
    return out


def _audit_part(blinded, true, fiducial, hidden, tolerance):
    """Problems with one archived part against its re-measured twin.

    Only signal rows are compared: ρ/τ carries none, and a re-run does not
    reproduce it (TreeCorr draws its jackknife patches afresh), so a part
    without signal is judged by its stamp alone.
    """
    stamp, true_stamp = (_custody.read_stamp(x.metadata) for x in (blinded, true))
    if true_stamp != _custody.Custody("unblinded", stamp.catalogue):
        return [f"the live file is stamped {true_stamp.token}"], None
    blinded, true = _signal(blinded), _signal(true)
    if blinded is None and true is None:
        return [], {"signal_rows": 0}
    if blinded is None or true is None or not _rows_match(blinded, true):
        return ["rows, tags or tracers differ"], None
    problems = []
    cov = _covariance(true)
    if not _same_covariance(_covariance(blinded), cov):
        problems.append("covariances differ")
    centres = [x.metadata.get("patch_centers_sha256") for x in (blinded, true)]
    if centres[0] != centres[1]:
        problems.append(f"patch centres differ: {centres}")

    values = np.asarray(true.mean)
    delta = np.asarray(blinded.mean) - values
    residual, rest = 0.0, np.ones(len(delta), bool)
    for block, factor in factors(true, fiducial, hidden):
        scale = np.max(np.abs(factor))
        residual = max(residual, np.max(np.abs(delta[block.rows] - factor)) / scale)
        rest[block.rows] = False
    if residual > tolerance:
        problems.append(f"blinded − true ≠ shift(seed): residual {residual:.2e}")

    kinds = np.array([dp.data_type for dp in true.data])
    moved = {}
    for kind in dict.fromkeys(kinds[rest]):
        rows = rest & (kinds == kind)
        if kind not in sacc_io.DERIVED:
            if np.max(np.abs(delta[rows])) > _REMEASURED * np.max(np.abs(values[rows])):
                problems.append(f"{kind} moved, but the blind leaves it unshifted")
        elif cov is None:
            problems.append(f"{kind} has no σ to judge its shift by")
        else:
            ratio = np.abs(delta[rows]) / np.sqrt(np.diag(cov)[rows])
            moved[kind] = float(np.max(ratio[np.isfinite(ratio)], initial=0.0))
    for kind, bound in B_SIGMA.items():
        if moved.get(kind, 0.0) > bound:
            problems.append(f"{kind} moved by {moved[kind]:.2e}σ under the blind")
    return problems, {"residual": residual, "shift_over_sigma": moved}


def audit(name, *, archive, true_root, cat_config, out=None):
    """Check every archived file of blind ``name`` against its re-measured twin.

    The published seed must be the committed one. For each archived file, the
    live file at the same relative path must be stamped unblinded for the same
    catalogue. Its signal rows must match the archived ones in tags, tracers,
    covariance and patch centres (numbers to a re-measurement's float noise),
    and blinded − true must equal the seed's shift on every ξ± and Cℓ_EE block
    to :data:`SHIFT_TOLERANCE` of the block's largest shift
    (:data:`SHIFT_TOLERANCE_OTHER_STACK` under another theory stack); Cℓ_BB and Cℓ_EB may not move, and
    a derived statistic's B rows by at most :data:`B_SIGMA`. A derived
    statistic's E rows are reported, in σ, not proven: the archive does not
    record which ξ± parts, through which kernel, they came from.
    """
    registry = _custody.registry_of(cat_config)
    report = {"blind": name, "ok": False, "problems": [], "parts": {}}
    try:
        blind = _open(registry, name)
    except _custody.CustodyError as err:
        report["problems"].append(str(err))
        return _write_report(report, out)
    record = _custody.records(registry)[name]
    if record.revealed != blind.seed:
        report["problems"].append("the published seed is not the committed one")
        return _write_report(report, out)
    stack = _theory_stack()
    tolerance = SHIFT_TOLERANCE
    if stack != record.commitment.get("theory_stack"):
        tolerance = SHIFT_TOLERANCE_OTHER_STACK
        report["theory_stack"] = {
            "record": record.commitment.get("theory_stack"),
            "now": stack,
        }
    fiducial, hidden = blind.config.theory, blind.hidden
    report["shift"] = {
        key: getattr(hidden, key) - getattr(fiducial, key)
        for key in blind.config.envelope
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
        if stamp.commitment != record.commitment["seed_commitment"]:
            report["parts"][relative] = {"problems": ["not concealed under this blind"]}
            continue
        problems, numbers = _audit_part(
            blinded, sacc_io.load(live), fiducial, hidden, tolerance
        )
        report["parts"][relative] = {"problems": problems, **(numbers or {})}
    if not report["parts"]:
        report["problems"].append("the archive holds no parts")
    report["ok"] = not report["problems"] and not any(
        part["problems"] for part in report["parts"].values()
    )
    return _write_report(report, out)


def _write_report(report, out):
    if out is not None:
        Path(out).write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


def verify(path, *, cat_config):
    """Problems with a file's stamp against its catalogue's custody (seedless)."""
    stamp = _custody.read_stamp(sacc_io.load(path).metadata)
    try:
        declared = declared_custody(cat_config, stamp.catalogue)
    except _custody.CustodyError as err:
        return [str(err)]
    problems = []
    if stamp != declared:
        problems.append(
            f"the stamp, {stamp.token}, is not {stamp.catalogue}'s custody, "
            f"{declared.token}"
        )
    if stamp.status == "blinded" and stamp.draw_scheme != draw_scheme():
        problems.append("this install draws under another scheme")
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
        "--config",
        help="JSON of BlindingConfig fields over the defaults: an `envelope` "
        "replaces the default one; `theory` fields replace theirs",
    )
    commands.choices["reveal"].add_argument("--root", required=True)
    commands.choices["audit"].add_argument("--archive", required=True)
    commands.choices["audit"].add_argument("--true-root", required=True)
    commands.choices["audit"].add_argument("--out")
    a = parser.parse_args(argv)

    if a.command == "init":
        config = None
        if a.config:
            given = json.loads(Path(a.config).read_text())
            default = BlindingConfig().record()
            theory = {**default["theory"], **given.get("theory", {})}
            config = _recorded_config(a.blind, {**default, **given, "theory": theory})
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
