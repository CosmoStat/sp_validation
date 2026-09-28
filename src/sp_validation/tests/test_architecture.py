"""Architecture on the resolved code tree: every SACC file passes one door.

``sacc_io.save`` seals and stamps what it writes and ``sacc_io.load`` refuses
what it did not, so no other module may write or read a SACC file directly.
"""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TESTS = Path(__file__).resolve().parent


def _sources():
    """Every Python file of the package and the workflow, outside the tests."""
    for top in (
        "src/sp_validation",
        "workflow",
        "scripts",
        "papers",
        "cosmo_inference",
    ):
        for path in (REPO / top).rglob("*.py"):
            parts = path.relative_to(REPO).parts
            if (
                TESTS not in path.parents
                and "tests" not in parts
                and ".snakemake" not in parts
            ):
                yield path


@pytest.mark.parametrize("method", ["save_fits", "load_fits"])
def test_sacc_files_pass_one_door(method):
    """[one-door] Only sacc_io writes or reads a SACC file."""
    door = REPO / "src" / "sp_validation" / "sacc_io.py"
    callers = sorted(
        str(path.relative_to(REPO))
        for path in _sources()
        if path != door
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
    )
    assert not callers, callers
