"""Custody: whether a catalogue's signal is blinded, and under which blind.

@sc custody-is-declared
Every catalogue entry of the catalogue config declares ``blind: none``,
``blind: mock`` or ``blind: <name>``, and an entry without one is refused.
``_leak_corr`` and ``_seed<N>`` versions take their entry's. Entries reading
one shear file declare one blind, the repository's ``cosmo_val/cat_config.yaml``
included whatever config is passed; and a set of versions shown together may
hold a blinded catalogue only beside mocks and catalogues under the same blind.

A blind is the record ``<paths.blinds>/<name>.blind.json``, drawn by
:mod:`sp_validation.blinding`. Its commitment, the fork's
``seed_commitment`` of the whole canonical record, is public: it names the
blind in custody tokens, and a SACC's stamp is its custody token.

The host Snakemake loads this module by path, so it imports only the standard
library (and PyYAML, which Snakemake carries, for the repository config).
"""

import functools
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

NONE, MOCK = "none", "mock"
NOT_CATALOGUES = ("nz", "paths")
REPO_CAT_CONFIG = Path(__file__).parents[2] / "cosmo_val" / "cat_config.yaml"
# smokescreen.COMMITMENT_DOMAIN, spelt here for the host; test_custody pins it.
COMMITMENT_DOMAIN = b"smokescreen-seed-commitment-v1|"
STAMP_KEY = "custody"
BLIND_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]*")
_SEED_SUFFIX = re.compile(r"_seed\d+$")
_LEAK_SUFFIX = "_leak_corr"


class CustodyError(ValueError):
    """A catalogue's custody cannot be resolved, or a stamp is not one."""


@dataclass(frozen=True)
class Custody:
    """``none``, ``mock`` or a blind's name, and that blind's commitment."""

    blind: str
    commitment: str | None = None
    registry: Path | None = field(default=None, compare=False)

    @property
    def blinded(self):
        return self.blind not in (NONE, MOCK)

    @property
    def token(self):
        """One string that changes whenever the custody does."""
        return f"{self.blind}:{self.commitment}" if self.blinded else self.blind


def parse(token, catalogues=None):
    """The custody a token (:attr:`Custody.token`) names, with the registry of
    ``catalogues`` (a catalogue config) when it is blinded."""
    blind, _, commitment = token.partition(":")
    custody = Custody(blind, commitment or None)
    if custody.blinded and catalogues is not None:
        return Custody(blind, commitment, registry(catalogues))
    return custody


def commitment(record):
    """sha256 of the domain-prefixed canonical JSON of a blind's record."""
    text = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(COMMITMENT_DOMAIN + text.encode("utf-8")).hexdigest()


@functools.cache
def _guard(path):
    for parent in (path, *path.parents):
        if (parent / ".git").exists():
            if parent != path:
                raise CustodyError(
                    f"the blind registry {path} lies inside the git worktree "
                    f"{parent}; point paths.blinds outside it, or at the root of "
                    "a dedicated private blinds repository"
                )
            break
    return path


def registry(catalogues):
    """The blind registry the catalogue config names (``paths.blinds``)."""
    path = (catalogues.get("paths") or {}).get("blinds")
    if not path or not os.path.isabs(os.path.expanduser(path)):
        raise CustodyError(f"paths.blinds, {path!r}, must be an absolute path")
    return _guard(Path(os.path.realpath(os.path.expanduser(path))))


def _variant_chain(version):
    """``version``, then each name that stripping ``_leak_corr``/``_seed<N>`` leaves."""
    chain = [version]
    while True:
        name = chain[-1]
        stripped = (
            name[: -len(_LEAK_SUFFIX)]
            if name.endswith(_LEAK_SUFFIX)
            else _SEED_SUFFIX.sub("", name)
        )
        if stripped in ("", name):
            return chain
        chain.append(stripped)


def entry_name(catalogues, version):
    """The config entry describing ``version``: its own, or its variant chain's."""
    for name in _variant_chain(version):
        if name in catalogues and name not in NOT_CATALOGUES:
            return name
    raise CustodyError(f"no catalogue {version} in the catalogue config")


def seed_path(shear, seed):
    """The shear path of seed label ``seed``: ``path_template`` formatted, or the
    digits after ``path``'s last ``seed`` replaced."""
    if shear.get("path_template"):
        return shear["path_template"].format(seed=int(seed), seed_label=seed)
    path = shear.get("path", "")
    match = re.match(r"(.*seed\D?)\d+", path)
    if match is None:
        raise CustodyError(f"shear path {path!r} has no seed<digits> to put a seed in")
    return match.group(1) + seed + path[match.end() :]


def _reads(entry):
    shear = entry.get("shear") if isinstance(entry, dict) else None
    if not isinstance(shear, dict) or "path" not in shear:
        return None
    return os.path.normpath(os.path.join(str(entry.get("subdir", "")), shear["path"]))


def shear_file(catalogues, version):
    """The shear catalogue file ``version`` reads, a ``_seed<N>``'s seed rendered."""
    name = entry_name(catalogues, version)
    entry, chain = catalogues[name], _variant_chain(version)
    seeds = [
        m.group()[5:]
        for n in chain[: chain.index(name)]
        if (m := _SEED_SUFFIX.search(n))
    ]
    if seeds:
        entry = {
            **entry,
            "shear": {**entry["shear"], "path": seed_path(entry["shear"], seeds[0])},
        }
    return _reads(entry)


def _blinds_by_file(catalogues):
    by_file = {}
    for name, entry in catalogues.items():
        if name not in NOT_CATALOGUES and (path := _reads(entry)):
            by_file.setdefault(os.path.realpath(path), set()).add(entry.get("blind"))
    return by_file


@functools.cache
def _repo_by_file(path, mtime):
    import yaml

    return _blinds_by_file(yaml.safe_load(Path(path).read_text()))


def of_file(path):
    """The blinds the repository config declares for shear file ``path``."""
    if not REPO_CAT_CONFIG.exists():
        return set()
    by_file = _repo_by_file(str(REPO_CAT_CONFIG), REPO_CAT_CONFIG.stat().st_mtime_ns)
    return by_file.get(os.path.realpath(path), set())


def declared(catalogues, version):
    """The blind ``version`` is declared under: ``none``, ``mock`` or a name."""
    name = entry_name(catalogues, version)
    blind = catalogues[name].get("blind")
    if blind is None:
        raise CustodyError(
            f"{name} declares no `blind:`; declare `blind: none` (public), "
            "`blind: mock` or `blind: <name>` in its catalogue config entry"
        )
    if not isinstance(blind, str) or not BLIND_NAME.fullmatch(blind):
        raise CustodyError(f"{name} declares blind: {blind!r}, not a blind name")
    try:
        path = shear_file(catalogues, version)
    except CustodyError:
        return blind  # a seed file no entry names: nothing else reads it
    if path is None:
        return blind
    blinds = {blind} | _blinds_by_file(catalogues).get(os.path.realpath(path), set())
    blinds |= of_file(path)
    if len(blinds) > 1:
        raise CustodyError(
            f"{version} reads {path}, which catalogue entries declare under "
            f"{sorted(map(str, blinds))}; entries reading one file declare one blind "
            f"(and {REPO_CAT_CONFIG} is authoritative for the files it names)"
        )
    return blind


def custody_of(catalogues, version):
    """The custody of ``version``: a blinded one reads its blind's commitment."""
    blind = declared(catalogues, version)
    if blind in (NONE, MOCK):
        return Custody(blind)
    where = registry(catalogues)
    try:
        record = json.loads((where / f"{blind}.blind.json").read_text())
    except (OSError, ValueError):
        fix = (
            f"no blind {blind} in {where}; `python -m sp_validation.blinding init "
            f"{blind}` draws one, only if you mean to create it"
            if where.is_dir() and os.access(where, os.R_OK | os.X_OK)
            else f"get read access to {where}"
        )
        raise CustodyError(
            f"{version} is blinded under {blind}: {fix}. Setting `blind: none` "
            "would unblind it."
        ) from None
    return Custody(blind, commitment(record), where)


def check_mix(blinds):
    """Refuse ``{version: blind}`` showing a blinded catalogue beside another
    real-sky catalogue not under its blind."""
    real = {v: b for v, b in blinds.items() if b != MOCK}
    for version, blind in real.items():
        for other, theirs in real.items():
            if blind not in (NONE, theirs):
                raise CustodyError(
                    f"{version} (blinded under {blind}) and {other} "
                    f"({'public' if theirs == NONE else f'blinded under {theirs}'}) "
                    "cannot be shown together: overlaying them shows the blind's shift"
                )


def summary(catalogues, versions):
    """One ``[custody]`` line per catalogue entry among ``versions``."""
    by_entry = {}
    for version in dict.fromkeys(versions):
        by_entry.setdefault(entry_name(catalogues, version), []).append(version)
    lines = []
    for name, members in by_entry.items():
        custody = custody_of(catalogues, name)
        also = [v for v in members if v != name]
        also = f" (+ {', '.join(also)})" if also else ""
        state = f"blinded under {custody.blind}" if custody.blinded else custody.blind
        lines.append(f"[custody] {name}{also}: {state}")
    return lines


def read_stamp(metadata):
    """The custody a SACC's stamp (its ``custody`` token) records; a missing or
    malformed stamp raises."""
    token = metadata.get(STAMP_KEY)
    custody = parse(token) if isinstance(token, str) else None
    if (
        custody is None
        or not BLIND_NAME.fullmatch(custody.blind)
        or custody.blinded != (custody.commitment is not None)
    ):
        raise CustodyError(
            f"no valid custody stamp ({STAMP_KEY}={token!r}): every SACC is born "
            "through sacc_io.save"
        )
    return custody
