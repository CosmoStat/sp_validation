"""Which blind each catalogue is declared under, and where its shear file is.

Every catalogue entry of the catalogue config declares ``blind: none`` or
``blind: <name>`` (:mod:`sp_validation.blinding`); an entry without one is
refused. ``_leak_corr`` and ``_seed<N>`` versions take their entry's, and
entries reading one shear file declare one blind.

The host Snakemake loads this module by path, so it imports only the standard
library.
"""

import os
import re

NOT_CATALOGUES = ("nz", "paths")
_SEED_SUFFIX = re.compile(r"_seed\d+$")
_LEAK_SUFFIX = "_leak_corr"


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
    raise ValueError(f"no catalogue {version} in the catalogue config")


def seed_path(shear, seed):
    """The shear path of seed label ``seed``: ``path_template`` formatted, or the
    digits after ``path``'s last ``seed`` replaced."""
    if shear.get("path_template"):
        return shear["path_template"].format(seed=int(seed), seed_label=seed)
    path = shear.get("path", "")
    match = re.match(r"(.*seed\D?)\d+", path)
    if match is None:
        raise ValueError(f"shear path {path!r} has no seed<digits> to put a seed in")
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


def declared(catalogues, version):
    """The blind ``version`` is declared under: ``none`` or a blind's name."""
    name = entry_name(catalogues, version)
    blind = catalogues[name].get("blind")
    if not isinstance(blind, str):
        raise ValueError(
            f"{name} declares no `blind:`; declare `blind: none` (public) or "
            "`blind: <name>` in its catalogue config entry"
        )
    path = _reads(catalogues[name])
    others = {
        other: entry.get("blind")
        for other, entry in catalogues.items()
        if other not in NOT_CATALOGUES and path and _reads(entry) == path
    }
    if len(set(others.values())) > 1:
        raise ValueError(
            f"{path} is read by entries declaring different blinds: {others}"
        )
    return blind


def summary(catalogues, versions):
    """One ``[blind]`` line per catalogue entry among ``versions``."""
    by_entry = {}
    for version in dict.fromkeys(versions):
        by_entry.setdefault(entry_name(catalogues, version), []).append(version)
    lines = []
    for name, members in by_entry.items():
        also = [v for v in members if v != name]
        also = f" (+ {', '.join(also)})" if also else ""
        lines.append(f"[blind] {name}{also}: {declared(catalogues, name)}")
    return lines
