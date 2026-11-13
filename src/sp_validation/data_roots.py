"""Named data roots: one config names an input the same way on every machine.

``config/data_roots.yaml`` maps each root name to the directories that hold it
on each machine. A config value writes ``${name}/relative/path``; on load the
root becomes the first of its directories that exists here. The environment
variable ``SPV_ROOT_<NAME>`` (upper case) overrides the file, for a copy the
file doesn't list. A root with no directory on this machine stays written as
``${name}``, so a path into it fails when opened and names the missing root.

Only declared root names are substituted; any other ``${...}`` passes through.

A root may be a CADC VOSpace URI, ``vos:cfis/weak_lensing/ShapePipe/v2.x``, so
a release is read straight from the vault: :func:`local` fetches a ``vos:``
path, file or directory, once into ``~/.cache/sp_validation/vos/`` and returns
the copy. It authenticates with the CADC proxy certificate
``~/.ssl/cadcproxy.pem`` (``cadc-get-cert -u <user>`` writes it).
"""

import os
import re
from functools import cache
from pathlib import Path

import yaml

ROOTS_FILE = Path(__file__).parents[2] / "config" / "data_roots.yaml"
_TOKEN = re.compile(r"\$\{(\w+)\}")
VOS = "vos:"
CACHE = Path.home() / ".cache" / "sp_validation" / "vos"
CERT = Path.home() / ".ssl" / "cadcproxy.pem"


@cache
def roots(roots_file=ROOTS_FILE):
    """Root name -> its directory on this machine, for roots present here."""
    declared = yaml.safe_load(Path(roots_file).read_text()) or {}
    resolved = {}
    for name, candidates in declared.items():
        override = os.environ.get(f"SPV_ROOT_{name.upper()}")
        found = override or next(
            (c for c in candidates if c.startswith(VOS) or Path(c).is_dir()), None
        )
        if found:
            resolved[name] = found
    return {"declared": set(declared), "resolved": resolved}


def expand(value, roots_file=ROOTS_FILE):
    """Return ``value`` with ``{root}`` tokens replaced, recursing into containers."""
    table = roots(roots_file)

    def substitute(match):
        name = match.group(1)
        if name in table["declared"]:
            return table["resolved"].get(name, match.group(0))
        return match.group(0)

    if isinstance(value, str):
        return _TOKEN.sub(substitute, value)
    if isinstance(value, dict):
        return {key: expand(item, roots_file) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, roots_file) for item in value]
    return value


def load_yaml(path, roots_file=ROOTS_FILE):
    """Read a YAML config with its data roots expanded for this machine."""
    return expand(yaml.safe_load(Path(path).read_text()), roots_file)


def local(path):
    """A local path for ``path``: itself, or for a ``vos:`` URI its cached copy.

    A directory is fetched whole, its files in parallel; a copy is complete
    once it exists (files are written beside it and renamed, a directory is
    marked by ``<dir>.complete``), so an interrupted fetch resumes.
    """
    path = str(path)
    if not path.startswith(VOS):
        return path
    import vos

    client = vos.Client(vospace_certfile=str(CERT))
    target = CACHE / path.removeprefix(VOS).strip("/")

    def fetch(uri, dest):
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            client.copy(uri, str(part))
            part.replace(dest)

    if not client.isdir(path):
        fetch(path, target)
        return str(target)
    marker = target.with_name(target.name + ".complete")
    if not marker.exists():
        files, stack = [], [path]
        while stack:
            uri = stack.pop()
            for name in client.listdir(uri, force=True):
                child = f"{uri}/{name}"
                (stack if client.isdir(child) else files).append(child)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(8) as pool:
            list(
                pool.map(
                    lambda uri: fetch(uri, target / uri.removeprefix(path + "/")),
                    files,
                )
            )
        marker.touch()
    return str(target)


def localise(value):
    """``value`` with every ``vos:`` string fetched (:func:`local`), recursing
    into containers."""
    if isinstance(value, str):
        return local(value)
    if isinstance(value, dict):
        return {key: localise(item) for key, item in value.items()}
    if isinstance(value, list):
        return [localise(item) for item in value]
    return value
