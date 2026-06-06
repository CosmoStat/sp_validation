"""Back-pressure guard #1: every module imports / resolves.

This is the first guard in the restructuring invariant suite (see the
``sp-validation-restructuring`` fiber). A pure file-move must not break
imports, so this is the first thing to go red on a bad ``git mv``.

The suite has two halves because *scripts live outside the package* and a
package-only walk silently misses them — the exact gap that surfaced in the
ngmix review's ``test_scripts_import.py``:

1. **Package modules** (``src/sp_validation/*.py``) are imported for real.
   They are library code and import-safe, so a genuine import resolves their
   own cross-module references; a move that breaks ``from .foo import bar``
   fails here immediately.

2. **Standalone scripts** (``scripts/*.py``) are *not executed* — several do
   real work at module level and one isn't even a valid module name
   (``shear_calibration+leakage_correction_ellipticities.py``). Executing them
   would conflate runtime failures (missing cluster data, argparse) with the
   thing we actually care about: does a move break an intra-repo import?
   Instead each script is parsed with :mod:`ast` (which also asserts it is
   syntactically valid) and every first-party import target it names
   (``sp_validation.*``) is resolved with :func:`importlib.util.find_spec`.
"""

import ast
import importlib
import importlib.util
import pkgutil
from pathlib import Path

import pytest

import sp_validation

# First-party top-level import roots whose resolution a file-move can break.
FIRST_PARTY_ROOTS = ("sp_validation",)


def _repo_root() -> Path:
    """Locate the repository root by walking up to the pyproject.toml marker.

    Marker-based (not a fixed parent count) so the guard survives the
    restructuring itself moving ``tests/`` to the top level.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate repo root (no pyproject.toml above test)")


def _package_modules() -> list[str]:
    """Every importable ``sp_validation`` submodule, excluding the test package."""
    names = []
    for _finder, name, ispkg in pkgutil.iter_modules(sp_validation.__path__):
        if ispkg or name == "tests":
            continue
        names.append(f"sp_validation.{name}")
    return sorted(names)


def _script_paths() -> list[Path]:
    """Every standalone script under ``scripts/`` (excluding ``__init__``)."""
    scripts_dir = _repo_root() / "scripts"
    return sorted(p for p in scripts_dir.glob("*.py") if p.name != "__init__.py")


def _script_param(path: Path):
    """Parametrize entry for a script with a readable test ID."""
    return pytest.param(path, id=path.name)


def _first_party_import_targets(path: Path) -> set[str]:
    """First-party fully-qualified import targets named in ``path`` (no exec)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FIRST_PARTY_ROOTS:
                    targets.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # level == 0 => absolute import; scripts are not a package, so any
            # relative import would itself be a bug worth surfacing.
            if node.level == 0 and node.module:
                if node.module.split(".")[0] in FIRST_PARTY_ROOTS:
                    targets.add(node.module)
    return targets


PACKAGE_MODULES = _package_modules()
SCRIPT_PATHS = _script_paths()


def test_package_module_inventory_nonempty():
    """Guard the guard: a glob that silently finds nothing is a false green."""
    assert PACKAGE_MODULES, "no sp_validation submodules discovered"
    assert SCRIPT_PATHS, "no scripts discovered under scripts/"


@pytest.mark.parametrize("module_name", PACKAGE_MODULES)
def test_package_module_imports(module_name):
    """Every library module imports cleanly (real import)."""
    importlib.import_module(module_name)


@pytest.mark.parametrize("script_path", [_script_param(p) for p in SCRIPT_PATHS])
def test_script_first_party_imports_resolve(script_path):
    """Every first-party import named by a script resolves (AST, no exec)."""
    for target in sorted(_first_party_import_targets(script_path)):
        assert importlib.util.find_spec(target) is not None, (
            f"{script_path.name} imports '{target}', which does not resolve"
        )
