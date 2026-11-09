"""Named data roots: one config names an input the same way on every machine.

``config/data_roots.yaml`` maps each root name to the directories that hold it
on each machine. A config value writes ``${name}/relative/path``; on load the
root becomes the first of its directories that exists here. The environment
variable ``SPV_ROOT_<NAME>`` (upper case) overrides the file, for a copy the
file doesn't list. A root with no directory on this machine stays written as
``${name}``, so a path into it fails when opened and names the missing root.

Only declared root names are substituted; any other ``${...}`` passes through.
"""

import os
import re
from functools import cache
from pathlib import Path

import yaml

ROOTS_FILE = Path(__file__).parents[2] / "config" / "data_roots.yaml"
_TOKEN = re.compile(r"\$\{(\w+)\}")


@cache
def roots(roots_file=ROOTS_FILE):
    """Root name -> its directory on this machine, for roots present here."""
    declared = yaml.safe_load(Path(roots_file).read_text()) or {}
    resolved = {}
    for name, candidates in declared.items():
        override = os.environ.get(f"SPV_ROOT_{name.upper()}")
        found = override or next((c for c in candidates if Path(c).is_dir()), None)
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
