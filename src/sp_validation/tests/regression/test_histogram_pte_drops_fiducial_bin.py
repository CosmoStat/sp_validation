"""Exercise the three paper-script PTE blocks without their data/chain I/O.

These scripts have no callable PTE function and execute expensive, hard-coded
I/O at import time (realspace also rewrites chain configuration).  Compile their
original AST assignment nodes from the histogram through the printed PTE; do not
reimplement the estimator. Source comes from the imported checkout, so the
same test can cross-check the tomography branch without any survey data.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

import sp_validation

ROOT = Path(__file__).resolve().parents[4]
# Use the active checkout's paper scripts when testing another branch's code.
CODE_ROOT = Path(sp_validation.__file__).resolve().parents[2]
if CODE_ROOT != ROOT:
    ROOT = CODE_ROOT

DROPPED_BIN = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#378: histogram PTE drops the fiducial bin's mass",
)

SITES = [
    ("papers/harmonic/2026_01_05_get_p_value_glass_mock.py", "chi2_glass_mocks"),
    ("papers/realspace/get_chi2_glass_mock.py", "chi2_tots"),
    ("papers/realspace/get_chi2_glass_mock.py", "xi_chi2s"),
]


def run_pte_block(relative_path, mock_variable, draws, fiducial):
    """Run original assignment statements, omitting plotting and all earlier I/O."""
    path = ROOT / relative_path
    if not path.is_file():
        pytest.skip(f"{relative_path} missing; may require the tomography branch")
    tree = ast.parse(path.read_text(), filename=str(path))
    starts = [
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "np"
        and node.value.func.attr == "histogram"
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == mock_variable
    ]
    assert len(starts) == 1, f"Cannot isolate histogram for {path}:{mock_variable}"
    start = starts[0]
    # Stop at the first P-value print.  Keep assignments, including any future
    # corrected p_value expression, verbatim; plotting is unrelated to the PTE.
    stop = next(
        index
        for index in range(start + 1, len(tree.body))
        if isinstance(tree.body[index], ast.Expr)
        and isinstance(tree.body[index].value, ast.Call)
        and isinstance(tree.body[index].value.func, ast.Name)
        and tree.body[index].value.func.id == "print"
        and "P-value:" in ast.unparse(tree.body[index])
    )
    nodes = [
        node
        for node in tree.body[start:stop]
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
    ]
    namespace = {"np": np, mock_variable: draws, "chi2_fiducial": fiducial}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return float(namespace["p_value"])


@pytest.mark.parametrize(
    "relative_path,mock_variable",
    SITES,
    ids=["harmonic", "realspace-total", "realspace-xi"],
)
@pytest.mark.parametrize(
    "fiducial,expected",
    [
        pytest.param(12.25, 0.75, marks=DROPPED_BIN),
        pytest.param(24.5, 0.25, marks=DROPPED_BIN),
        (-0.5, 1.0),
    ],
    ids=["occupied-fiducial-bin", "last-bin-tail", "below-support"],
)
def test_mock_pte_keeps_exceedances_in_fiducial_histogram_bin(
    relative_path, mock_variable, fiducial, expected
):
    """Protect reported GLASS-mock PTEs from dropping the fiducial bin's mass.
    Four equally weighted mock chi2 values [0, 12.5, 12.75, 25] have exactly
    three, one, and four exceedances at thresholds 12.25, 24.5, and -0.5,
    respectively, so their empirical PTEs are 3/4, 1/4, and 1.  With 25 bins
    spanning [0, 25], every occupied fiducial bin contains only exceedances;
    there is no within-bin interpolation ambiguity.  Plotting-bin boundaries
    must not remove those mocks from the printed probability.
    The below-support control passes on develop because no occupied bin is
    excluded there; it stays unmarked to catch a tomography-merge regression.
    """
    draws = np.array([0.0, 12.5, 12.75, 25.0])
    reference = sum(value >= fiducial for value in draws.tolist()) / len(draws)
    assert reference == expected
    actual = run_pte_block(relative_path, mock_variable, draws, fiducial)
    assert actual == pytest.approx(expected, abs=1e-14), (
        f"{relative_path} [{mock_variable}]: chi2_fiducial={fiducial:g}; "
        f"reported PTE={actual:.6f}, expected empirical PTE={expected:.6f} "
        f"({int(expected * len(draws))}/{len(draws)} mocks >= fiducial); "
        "the occupied fiducial histogram bin must not be dropped"
    )
