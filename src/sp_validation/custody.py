"""Custody: whether a catalogue's signal is blinded, unblinded or a mock.

Custody is declared once per *base* catalogue, as ``blinding:`` on its entry in
``cosmo_val/cat_config.yaml``; an entry that declares nothing is blinded, so a
new catalogue fails closed. Variants share their base's custody and blind:
``_leak_corr`` and ``_seed<N>`` versions, and entries that name their parent
with ``base:``.

A blinded base must be covered by exactly one blind in the registry beside the
catalogue config (``cosmo_val/blinds/<blind>/``: ``commitment.json``, the
``bases`` it covers, and ``revealed.json`` once its seed is published). The
registry is written only by ``python -m sp_validation.blinding``.

Standard library only: the host Snakemake loads this file by path, and
container jobs import it, so both resolve custody from the same declaration.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

STATUSES = ("blinded", "unblinded", "mock")

# Top-level keys of the catalogue config that are not catalogues.
NOT_CATALOGUES = ("nz", "paths")

# The Smokescreen fork's commitment prefix; test_blinding pins it to the fork's.
COMMITMENT_DOMAIN = b"smokescreen-seed-commitment-v1|"

# The stamp every SACC carries: the custody it was born under.
STAMP_KEYS = (
    "blinding",
    "blinding_catalogue",
    "blinding_blind",
    "blinding_commitment",
    "blinding_config",
    "blinding_scheme",
)
_BLIND_KEYS = STAMP_KEYS[2:]

_SEED_SUFFIX = re.compile(r"_seed\d+$")
_LEAK_SUFFIX = "_leak_corr"


class CustodyError(ValueError):
    """A catalogue's custody cannot be resolved, or a stamp is not one."""


@dataclass(frozen=True)
class Custody:
    """The custody of one base catalogue, and of every variant of it."""

    status: str
    catalogue: str
    blind: str | None = None
    commitment: str | None = None
    config_digest: str | None = None
    draw_scheme: int | None = None
    registry: Path | None = None

    @property
    def token(self):
        """One string that changes whenever the custody does."""
        if self.status == "blinded":
            return f"blinded:{self.catalogue}:{self.blind}:{self.commitment}"
        return f"{self.status}:{self.catalogue}"

    @property
    def stamp(self):
        """The ``blinding*`` metadata a SACC born under this custody carries."""
        stamp = {"blinding": self.status, "blinding_catalogue": self.catalogue}
        if self.status == "blinded":
            stamp.update(
                {
                    "blinding_blind": self.blind,
                    "blinding_commitment": self.commitment,
                    "blinding_config": self.config_digest,
                    "blinding_scheme": int(self.draw_scheme),
                }
            )
        return stamp


def seed_commitment(seed):
    """The public commitment to a seed: sha256 of the domain-prefixed seed."""
    return hashlib.sha256(COMMITMENT_DOMAIN + str(seed).encode("utf-8")).hexdigest()


def registry_of(cat_config):
    """The blind registry beside a catalogue config."""
    return Path(cat_config).resolve().parent / "blinds"


def entry_of(version):
    """The catalogue-config entry describing ``version``'s data.

    Strips the variants CosmologyValidation materialises from an entry:
    ``_leak_corr``, then ``_seed<N>``.
    """
    if version.endswith(_LEAK_SUFFIX):
        version = version[: -len(_LEAK_SUFFIX)]
    return _SEED_SUFFIX.sub("", version)


def base_catalogue(catalogues, version):
    """The base catalogue of ``version``: its entry, then its ``base:`` links."""
    name, seen = entry_of(version), []
    while True:
        if name in NOT_CATALOGUES:
            raise CustodyError(f"{name} is not a catalogue")
        if name not in catalogues:
            via = f" (reached from {version} through base:)" if seen else ""
            raise CustodyError(f"no catalogue {name} in the catalogue config{via}")
        if name in seen:
            raise CustodyError(
                f"base: links form a cycle: {' -> '.join(seen + [name])}"
            )
        seen.append(name)
        parent = catalogues[name].get("base")
        if parent is None:
            return name
        if "blinding" in catalogues[name]:
            raise CustodyError(
                f"{name} names its base {parent}; declare custody on {parent}, "
                f"not on {name}"
            )
        name = parent


def declaration(catalogues, base):
    """What base catalogue ``base`` declares: a status, or ``None`` (blinded)."""
    declared = catalogues[base].get("blinding")
    if declared is not None and declared not in STATUSES:
        raise CustodyError(
            f"{base} declares blinding: {declared!r}; declare blinded, unblinded "
            "or mock"
        )
    return declared


@dataclass(frozen=True)
class Record:
    """A blind's public record: its commitment, the bases it covers, and the
    seed once it is revealed."""

    name: str
    commitment: dict
    bases: tuple
    revealed: str | None


def records(registry):
    """Every blind in the registry, by name."""
    found = {}
    if registry is None or not Path(registry).is_dir():
        return found
    for path in sorted(Path(registry).iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        try:
            commitment = json.loads((path / "commitment.json").read_text())
            bases = tuple((path / "bases").read_text().split())
        except (OSError, ValueError) as err:
            raise CustodyError(f"blind record {path} is incomplete: {err}") from None
        revealed = path / "revealed.json"
        seed = json.loads(revealed.read_text())["seed"] if revealed.exists() else None
        found[path.name] = Record(path.name, commitment, bases, seed)
    return found


def _no_blind(version, base, declared, registry):
    why = (
        "declared blinded"
        if declared == "blinded"
        else "cosmo_val/cat_config.yaml declares no custody for it, so it is blinded"
    )
    repo = registry.parent.parent
    cat_config = registry.parent / "cat_config.yaml"
    return (
        f"{version} is blinded ({why}) and no blind covers it.\n"
        "Draw one, once:\n"
        f"  APPTAINERENV_PYTHONPATH={repo}/src spv-container exec "
        f"python -m sp_validation.blinding init <name> {base} "
        f"--cat-config {cat_config}\n"
        "then commit cosmo_val/blinds/<name>/.\n"
        "Re-processing a catalogue whose blind is still concealed? Share it "
        f"instead: python -m sp_validation.blinding share <blind> {base} "
        f"--cat-config {cat_config}\n"
        "Predates blinding: declare `blinding: unblinded`. A mock: "
        "`blinding: mock`. A variant: `base: <parent>`."
    )


def custody_of(catalogues, version, *, registry):
    """The custody of ``version``, from its base's declaration and the registry.

    @sc custody-is-declared
    Custody is read only here, from the parsed catalogue config and the blind
    registry; no workflow config, environment variable or call argument can
    change it. An entry declaring nothing is blinded, and variants share their
    base's custody and blind.

    Raises
    ------
    CustodyError
        With the one thing to do, when the declaration and the registry
        disagree: a blinded catalogue with no blind, a revealed blind still
        declared blinded, an unblinded catalogue under a concealed blind, a
        mock under a blind, or two blinds over one catalogue.
    """
    registry = Path(registry)
    base = base_catalogue(catalogues, version)
    declared = declaration(catalogues, base)
    covering = [r for r in records(registry).values() if base in r.bases]
    if len(covering) > 1:
        names = ", ".join(r.name for r in covering)
        raise CustodyError(f"{base} is covered by several blinds: {names}")
    record = covering[0] if covering else None

    if declared == "mock":
        if record is not None:
            raise CustodyError(
                f"{base} is declared mock but blind {record.name} covers it; a "
                "mock is never blinded"
            )
        return Custody("mock", base)

    if declared == "unblinded":
        if record is None:
            return Custody("unblinded", base)
        if record.revealed is None:
            raise CustodyError(
                f"{base} is declared unblinded, but blind {record.name} still "
                "conceals it. Unblinding is the reveal: python -m "
                f"sp_validation.blinding reveal {record.name} --root <COSMO_VAL> "
                f"--cat-config {registry.parent / 'cat_config.yaml'}"
            )
        if seed_commitment(record.revealed) != record.commitment["seed_commitment"]:
            raise CustodyError(
                f"the seed published for blind {record.name} does not match its "
                "commitment"
            )
        return Custody("unblinded", base)

    if record is None:
        raise CustodyError(_no_blind(version, base, declared, registry))
    if record.revealed is not None:
        raise CustodyError(
            f"blind {record.name} is public; declare `blinding: unblinded` on "
            f"{base} (the reveal PR does this)"
        )
    c = record.commitment
    return Custody(
        "blinded",
        base,
        record.name,
        c["seed_commitment"],
        c["config_digest"],
        int(c["draw_scheme"]),
        registry,
    )


def summary(catalogues, versions, *, registry):
    """One ``[custody]`` line per base catalogue among ``versions``."""
    by_base = {}
    for version in dict.fromkeys(versions):
        by_base.setdefault(base_catalogue(catalogues, version), []).append(version)
    lines = []
    for base, members in by_base.items():
        custody = custody_of(catalogues, base, registry=registry)
        variants = [v for v in members if v != base]
        also = f" (+ {', '.join(variants)})" if variants else ""
        state = (
            f"blinded under {custody.blind}"
            if custody.status == "blinded"
            else custody.status
        )
        lines.append(f"[custody] {base}{also}: {state}")
    return lines


def read_stamp(metadata):
    """The custody a SACC's stamp records; a missing or malformed stamp raises.

    The result carries no registry: it is what the file says, to compare with
    what a catalogue is declared under (``.stamp`` or ``.token``).
    """
    status = metadata.get("blinding")
    if status not in STATUSES:
        raise CustodyError(
            f"no valid custody stamp (blinding={status!r}): every SACC is born "
            "through sacc_io.save"
        )
    if "blinding_catalogue" not in metadata:
        raise CustodyError("custody stamp names no catalogue")
    present = [k for k in _BLIND_KEYS if k in metadata]
    if status == "blinded" and len(present) != len(_BLIND_KEYS):
        missing = sorted(set(_BLIND_KEYS) - set(present))
        raise CustodyError(f"blinded custody stamp lacks {missing}")
    if status != "blinded" and present:
        raise CustodyError(f"a {status} custody stamp carries blind keys {present}")
    if status != "blinded":
        return Custody(status, metadata["blinding_catalogue"])
    return Custody(
        status,
        metadata["blinding_catalogue"],
        metadata["blinding_blind"],
        metadata["blinding_commitment"],
        metadata["blinding_config"],
        int(metadata["blinding_scheme"]),
    )


def confirm(custody, token):
    """``custody``, if its token is the one the launch resolved; else raise."""
    if custody.token != token:
        raise CustodyError(
            f"custody of {custody.catalogue} changed since the launch: the job "
            f"resolves {custody.token}, the launch resolved {token}"
        )
    return custody
