"""Custody: whether a catalogue's signal is blinded, unblinded or a mock.

The custody of every catalogue version (:func:`custody_of`), the stamp a SACC
born under it carries, and the reader of the blind registry beside the
catalogue config (``cosmo_val/blinds/``; its writer is
:mod:`sp_validation.blinding`).

The host Snakemake loads this module by path, with no sp_validation installed,
so it imports nothing but the standard library.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

STATUSES = ("blinded", "unblinded", "mock")

# Top-level keys of the catalogue config that are not catalogues.
NOT_CATALOGUES = ("nz", "paths")

# smokescreen.COMMITMENT_DOMAIN, spelt here so this module needs only the
# standard library.
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
    # Where the blind is opened from; two custodies are equal whatever it is.
    registry: Path | None = field(default=None, compare=False)

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
    """The blind registry beside a catalogue config, spelled as the config is."""
    return Path(cat_config).absolute().parent / "blinds"


def _variant_chain(version):
    """``version``, then each name stripping a trailing ``_leak_corr`` or
    ``_seed<N>`` leaves: the variants CosmologyValidation materialises from an
    entry, in any order."""
    chain = [version]
    while True:
        name = chain[-1]
        if name.endswith(_LEAK_SUFFIX):
            stripped = name[: -len(_LEAK_SUFFIX)]
        else:
            stripped = _SEED_SUFFIX.sub("", name)
        if stripped in ("", name):
            return chain
        chain.append(stripped)


def base_catalogue(catalogues, version):
    """The base catalogue of ``version``: the entry it is a variant of, or its
    own, then that entry's ``base:`` links.

    A config entry named as a variant of another (``<entry>_leak_corr``,
    ``<entry>_seed<N>``) shares that entry's custody, as the variant it is
    named for would; it may repeat that entry's ``blinding`` or ``base``, never
    declare another.
    """
    chain = _variant_chain(version)
    base = _follow_base(catalogues, chain[-1], version)
    for alias in chain[:-1]:
        entry = catalogues.get(alias)
        if not isinstance(entry, dict):
            continue
        status = declaration(catalogues, base) or "blinded"
        if "base" in entry and _follow_base(catalogues, entry["base"], alias) != base:
            claim = f"base: {entry['base']}"
        elif entry.get("blinding", status) != status:
            claim = f"blinding: {entry['blinding']}"
        else:
            continue
        raise CustodyError(
            f"{alias} is a variant of {chain[-1]}, so it shares the custody of "
            f"{base} ({status}), but declares {claim}; declare custody on {base}, "
            f"or rename {alias} to make it a catalogue of its own"
        )
    return base


def _follow_base(catalogues, name, version):
    """The entry the ``base:`` links from ``name`` end at."""
    seen = []
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


def read_record(path):
    """The public record of the blind stored at directory ``path``.

    The one reader of a blind's public files; its sealed seed is
    ``blinding``'s to open.
    """
    path = Path(path)
    try:
        commitment = json.loads((path / "commitment.json").read_text())
        bases = tuple((path / "bases").read_text().split())
    except (OSError, ValueError) as err:
        raise CustodyError(f"blind record {path} is incomplete: {err}") from None
    revealed = path / "revealed.json"
    seed = json.loads(revealed.read_text())["seed"] if revealed.exists() else None
    return Record(path.name, commitment, bases, seed)


def records(registry):
    """Every blind in the registry, by name."""
    if registry is None or not Path(registry).is_dir():
        return {}
    return {
        path.name: read_record(path)
        for path in sorted(Path(registry).iterdir())
        if path.is_dir() and not path.name.startswith(".")
    }


def _reads(entry):
    """The shear catalogue file a catalogue entry reads, if it names one."""
    shear = entry.get("shear") if isinstance(entry, dict) else None
    if not isinstance(shear, dict) or "path" not in shear:
        return None
    return os.path.normpath(os.path.join(str(entry.get("subdir", "")), shear["path"]))


def entry_name(catalogues, version):
    """The catalogue-config entry describing ``version``: its own, or that of
    the first name its variant chain reaches."""
    for name in _variant_chain(version):
        if name in catalogues and name not in NOT_CATALOGUES:
            return name
    raise CustodyError(f"no catalogue {version} in the catalogue config")


def seed_path(shear, seed):
    """The shear path of seed label ``seed``, from an entry's ``shear`` block.

    Its ``path_template`` formatted with ``seed`` (an int) and ``seed_label``;
    without one, its ``path`` with the digits after its last ``seed`` replaced.
    """
    if shear.get("path_template"):
        return shear["path_template"].format(seed=int(seed), seed_label=seed)
    path = shear.get("path", "")
    match = re.match(r"(.*seed\D?)\d+", path)
    if match is None:
        raise CustodyError(
            f"shear path {path!r} has no path_template and no seed<digits> to "
            "put a seed in"
        )
    return match.group(1) + seed + path[match.end() :]


def shear_file(catalogues, version):
    """The shear catalogue file ``version`` reads, as CosmologyValidation reads
    it: its entry's, with the seed of a ``_seed<N>`` variant rendered in."""
    chain = _variant_chain(version)
    name = entry_name(catalogues, version)
    entry = catalogues[name]
    seeds = [_SEED_SUFFIX.search(n) for n in chain[: chain.index(name)]]
    seeds = [m.group()[len("_seed") :] for m in seeds if m]
    if seeds:
        shear = {**entry["shear"], "path": seed_path(entry["shear"], seeds[0])}
        entry = {**entry, "shear": shear}
    return _reads(entry)


def _concealer(catalogues, recs, base):
    """What conceals ``base``'s data: the unrevealed blind covering it; ``""``
    when it is declared blinded and none does; ``None`` when it is public."""
    covering = sorted(
        r.name for r in recs.values() if base in r.bases and not r.revealed
    )
    if covering:
        return ", ".join(covering)
    return None if declaration(catalogues, base) in ("unblinded", "mock") else ""


def refuse_twins(catalogues, recs, version):
    """Refuse when a catalogue reading ``version``'s shear file is concealed
    otherwise than ``version`` is, under the blind records ``recs``.

    A blind conceals data, not a name: every catalogue reading one shear file
    is concealed under one blind, or none is.
    """
    base = base_catalogue(catalogues, version)
    try:
        here = shear_file(catalogues, version)
    except CustodyError:
        # A seed file its entry cannot name, so no entry reads it: whatever
        # would read it (rule xi, CosmologyValidation) refuses it there.
        here = None
    readers = {
        name: base_catalogue(catalogues, name)
        for name, entry in catalogues.items()
        if here is not None and name not in NOT_CATALOGUES and _reads(entry) == here
    }
    hiding = {
        b: _concealer(catalogues, recs, b)
        for b in dict.fromkeys([base, *readers.values()])
    }
    twins = [name for name, b in readers.items() if hiding[b] != hiding[base]]
    if twins:
        states = "; ".join(
            f"{b}: {'public' if c is None else f'blind {c}' if c else 'no blind yet'}"
            for b, c in hiding.items()
        )
        raise CustodyError(
            f"{version} and {', '.join(twins)} read one shear file, {here}, but are "
            f"not concealed alike ({states}). Catalogues reading one file share "
            "one blind: declare them blinded and draw it for them together "
            "(blinding init) or share it (blinding share), or make them variants "
            "of one base with `base:`"
        )


def _no_blind(version, base, declared, registry):
    why = (
        "declared blinded"
        if declared == "blinded"
        else "cosmo_val/cat_config.yaml declares no custody for it, so it is blinded"
    )
    repo = registry.parent.parent
    cat_config = registry.parent / "cat_config.yaml"
    return (
        f"{version} is blinded ({why}) and no blind covers it in this checkout.\n"
        "git pull first: its blind may already be drawn and committed. Never "
        "draw a second one.\n"
        "Only the custodian draws a blind, once:\n"
        f"  APPTAINERENV_PYTHONPATH={repo}/src spv-container exec "
        f"python -m sp_validation.blinding init <name> {base} "
        f"--cat-config {cat_config}\n"
        "then commits cosmo_val/blinds/<name>/.\n"
        "Re-processing a catalogue whose blind is still concealed? Share it "
        f"instead: python -m sp_validation.blinding share <blind> {base} "
        f"--cat-config {cat_config}\n"
        "Predates blinding: declare `blinding: unblinded`. A mock: "
        "`blinding: mock`. A variant: `base: <parent>`."
    )


def custody_of(catalogues, version, *, registry):
    """The custody of ``version``, from its base's declaration and the registry.

    @sc custody-is-declared
    Custody is declared once per *base* catalogue, as ``blinding:`` on its
    entry in ``cosmo_val/cat_config.yaml``, and read only here, with the blind
    registry; no workflow config, environment variable or call argument can
    change it. An entry declaring nothing is blinded, so a new catalogue fails
    closed. Variants share their base's custody and blind: ``_leak_corr`` and
    ``_seed<N>`` versions, and entries naming their parent with ``base:``. A
    blinded base is covered by exactly one blind, which also conceals every
    catalogue reading its shear file (:func:`refuse_twins`).

    Raises
    ------
    CustodyError
        With the one thing to do, when the declaration and the registry
        disagree: a blinded catalogue with no blind, a revealed blind still
        declared blinded, an unblinded catalogue under a concealed blind, a
        mock under a blind, two blinds over one catalogue, or catalogues
        reading one shear file concealed otherwise.
    """
    registry = Path(registry)
    base = base_catalogue(catalogues, version)
    declared = declaration(catalogues, base)
    recs = records(registry)
    refuse_twins(catalogues, recs, version)
    covering = [r for r in recs.values() if base in r.bases]
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
    what a catalogue is declared under (``==``).
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
    """``custody``, if its token is the job's ``params.custody``; else raise.

    ``token`` is resolved by the Snakemake process that runs the job: the
    launch under a local executor, the job step at job start under slurm. The
    job's own resolution, in the container, must agree with it.
    """
    if custody.token != token:
        raise CustodyError(
            f"custody of {custody.catalogue} differs between the job and its "
            f"Snakemake: the job resolves {custody.token}, Snakemake resolved "
            f"{token}; launch again"
        )
    return custody
