"""Architecture on the resolved code tree: each assertion names its contract.

Import boundaries are read from the ``CONTRACTS`` files (the rule lines the
scientific-software-development skill's ``check-imports`` also reads); the
rest are AST queries over every Python file outside the tests.
"""

import ast
import fnmatch
import re
import sys
from pathlib import Path

import pytest

from sp_validation import custody

REPO = Path(__file__).resolve().parents[3]
TESTS = Path(__file__).resolve().parent

HEADER = re.compile(r"^\s*@sc(?:\s+\[[^\]]*\])?\s+([\w][\w.-]*)\s*$")
ONLY = re.compile(r"^\s*only:\s*(\S+)\s+may import\s+(.+?)\s*$")
ONLY_IMPORTER = re.compile(r"^\s*only:\s*(\S+)\s+may be imported by\s+(.+?)\s*$")


def _sources():
    """``{module name: path}`` for every Python file outside the tests."""
    found = {}
    for root, package in ((REPO / "src" / "sp_validation", "sp_validation"),):
        for path in root.rglob("*.py"):
            if TESTS in path.parents:
                continue
            parts = path.relative_to(root).with_suffix("").parts
            parts = parts[:-1] if parts[-1] == "__init__" else parts
            found[".".join((package, *parts))] = path
    for top in ("workflow", "scripts", "papers", "cosmo_inference"):
        for path in (REPO / top).rglob("*.py"):
            if "tests" in path.relative_to(REPO).parts or ".snakemake" in path.parts:
                continue
            found[".".join(path.relative_to(REPO).with_suffix("").parts)] = path
    return found


SOURCES = _sources()


def _imports(module, path):
    """Absolute names of every module ``path`` imports."""
    names = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = module.split(".")
                package = package if path.name == "__init__.py" else package[:-1]
                base = ".".join(package[: len(package) - node.level + 1])
                target = f"{base}.{node.module}" if node.module else base
                names += (
                    [target]
                    if node.module
                    else [f"{target}.{alias.name}" for alias in node.names]
                )
            else:
                names.append(node.module)
    return names


def _matches(name, pattern):
    if pattern == "stdlib":
        return name.split(".")[0] in sys.stdlib_module_names | {"__future__"}
    return fnmatch.fnmatchcase(name, pattern)


def _rules():
    """``(contract, kind, subject, patterns)`` from every CONTRACTS file."""
    rules = []
    for contracts in sorted(REPO.rglob("CONTRACTS")):
        if ".snakemake" in contracts.parts:
            continue
        ident = None
        for line in contracts.read_text().splitlines():
            if header := HEADER.match(line):
                ident = header[1]
            elif rule := ONLY_IMPORTER.match(line):
                rules.append((ident, "importers", rule[1], rule[2].split(", ")))
            elif rule := ONLY.match(line):
                rules.append((ident, "imports", rule[1], rule[2].split(", ")))
    return rules


RULES = _rules()


def test_the_contracts_carry_the_boundaries():
    assert {r[0] for r in RULES} >= {
        "custody-is-stdlib",
        "container-is-stdlib",
        "blinding-owns-smokescreen",
        "blinding-owns-the-seed-cipher",
        "host-importable",
    }


@pytest.mark.parametrize(
    "contract, kind, subject, patterns", RULES, ids=[r[0] for r in RULES]
)
def test_import_boundaries(contract, kind, subject, patterns):
    if kind == "imports":
        assert subject in SOURCES, f"[{contract}] no module {subject}"
        stray = [
            name
            for name in _imports(subject, SOURCES[subject])
            if not any(_matches(name, p) for p in patterns)
        ]
        assert not stray, f"[{contract}] {subject} imports {stray}"
    else:
        stray = sorted(
            module
            for module, path in SOURCES.items()
            if not any(_matches(module, p) for p in patterns)
            and any(fnmatch.fnmatchcase(n, subject) for n in _imports(module, path))
        )
        assert not stray, f"[{contract}] {subject} is imported by {stray}"


def _nodes(kind):
    for module, path in SOURCES.items():
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, kind):
                yield module, node


def test_stamp_keys_are_spelt_only_in_custody():
    """[stamped-or-refused] Stamps are read and minted through custody.stamp and read_stamp."""
    spelt = sorted(
        (module, node.value)
        for module, node in _nodes(ast.Constant)
        if isinstance(node.value, str)
        and node.value in custody.STAMP_KEYS
        and module != "sp_validation.custody"
    )
    assert not spelt, spelt


def test_custody_is_constructed_only_where_it_is_declared():
    """[custody-is-declared] Only custody_of builds a Custody."""
    built = sorted(
        module
        for module, node in _nodes(ast.Call)
        if (
            (isinstance(node.func, ast.Name) and node.func.id == "Custody")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "Custody")
        )
        and module != "sp_validation.custody"
    )
    assert not built, built


@pytest.mark.parametrize("method", ["save_fits", "load_fits"])
def test_sacc_files_pass_one_door(method):
    """[one-door] Only sacc_io writes or reads a SACC file."""
    callers = sorted(
        module
        for module, node in _nodes(ast.Call)
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == method
        and module != "sp_validation.sacc_io"
    )
    assert not callers, callers
