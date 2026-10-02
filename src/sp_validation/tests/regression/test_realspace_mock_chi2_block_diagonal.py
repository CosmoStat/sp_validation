"""Execute the paper script's actual mock loop without its destructive setup.

There is no callable estimator in this script: all calculations are top-level.
AST isolation retains the entire original FITS/model-read/scale-cut/chi2 loop
unchanged, avoiding its preceding CosmoSIS launches and INI-file writes.
Only I/O is replaced; scipy interpolation and numpy calculations are real.
"""

import ast
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.stats import chi2

import sp_validation

DROPPED_COVARIANCE = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#380: mock total chi2 drops cross-covariance",
)


def source_path():
    root = Path(__file__).resolve().parents[4]
    # Cross-check the paper script belonging to the imported checkout.
    code_root = Path(sp_validation.__file__).resolve().parents[2]
    if code_root != root:
        root = code_root
    path = root / "papers/realspace/get_chi2_glass_mock.py"
    if not path.is_file():
        pytest.skip("real-space mock script missing; may require the tomography branch")
    return path


def assigns(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == name for t in node.targets
    )


def run_mock_loop(monkeypatch, correlations="xi_plus_minus"):
    path = source_path()
    tree = ast.parse(path.read_text(), filename=str(path))
    loops = [
        node
        for node in tree.body
        if isinstance(node, ast.For)
        and any(assigns(n, "chi2_tot") for n in ast.walk(node))
    ]
    assert len(loops) == 1, "Locate the production mock chi2 loop"
    theta = np.array([12.0, 20.0, 40.0, 70.0, 83.0])
    n = len(theta)
    covariance = np.eye(4 * n)
    residual = np.zeros(4 * n)
    pairs = []
    if correlations in ("xi_plus_minus", "both", "none"):
        pairs.append((1, n + 1))
    if correlations in ("xi_tau", "both"):
        pairs.append((2, 2 * n + 2))
    for i, j in pairs:
        residual[[i, j]] = 1
        if correlations != "none":
            covariance[i, j] = covariance[j, i] = 0.5
    # Strict boundary cuts must exclude these xi residuals.
    residual[[0, n - 1, n, 2 * n - 1]] = 99
    full_mask = np.r_[
        np.tile((theta > 12) & (theta < 83), 2), np.ones(2 * n, dtype=bool)
    ]
    selected = residual[full_mask]
    selected_cov = covariance[np.ix_(full_mask, full_mask)]
    hand_value = len(pairs) * (2 if correlations == "none" else 4 / 3)
    # Independent reference: each correlated pair contributes (1,1) C^-1 (1,1)^T
    # = 2/(1+rho) = 4/3, whereas separate marginal inverses give 2.
    assert selected @ np.linalg.solve(selected_cov, selected) == pytest.approx(
        hand_value
    )

    roots = [f"synthetic_{i:05d}" for i in range(3)]
    block_values = np.array([60.0, 90.0, 120.0])
    hdus = {}
    for root, block_value in zip(roots, block_values):
        amplitude = np.sqrt(block_value / (2 * len(pairs)))
        vector = amplitude * residual
        extensions = [fits.PrimaryHDU(), fits.ImageHDU(covariance, name="COVMAT")]
        for k, name in enumerate(("XI_PLUS", "XI_MINUS", "TAU_0_PLUS", "TAU_2_PLUS")):
            extensions.append(
                fits.BinTableHDU.from_columns(
                    [
                        fits.Column(name="ANG", format="D", array=theta),
                        fits.Column(
                            name="VALUE", format="D", array=vector[k * n : (k + 1) * n]
                        ),
                    ],
                    name=name,
                )
            )
        hdus[root] = fits.HDUList(extensions)

    def open_mock(filename):
        return next(hdu for root, hdu in hdus.items() if root[-5:] in str(filename))

    def load_model(filename, *args, **kwargs):
        return (
            theta * np.pi / (180 * 60)
            if str(filename).endswith("theta.txt")
            else np.zeros(n)
        )

    monkeypatch.setattr(np, "loadtxt", load_model)
    namespace = dict(
        np=np,
        fits=SimpleNamespace(open=open_mock),
        interp1d=interp1d,
        roots=roots,
        root_glass_dv="synthetic",
        output_folder_chains="synthetic/",
        lower_bound_xi=12,
        upper_bound_xi=83,
    )
    for name in (
        "xi_plus_chi2s",
        "xi_minus_chi2s",
        "xi_chi2s",
        "tau_chi2s",
        "chi2_tots",
    ):
        namespace[name] = np.array([])
    with redirect_stdout(StringIO()):
        exec(
            compile(ast.Module(body=loops, type_ignores=[]), str(path), "exec"),
            namespace,
        )
    # Execute the original total-panel statistic and printed PTE calculation too.
    start = next(
        i for i, node in enumerate(tree.body) if assigns(node, "chi2_fiducial")
    )
    end = next(
        i for i in range(start, len(tree.body)) if assigns(tree.body[i], "p_value")
    )
    namespace.update(
        chi2=chi2, ax1=None, sns=SimpleNamespace(histplot=lambda *a, **k: None)
    )
    panel = [
        node
        for node in tree.body[start : end + 1]
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "print"
        )
    ]
    exec(compile(ast.Module(body=panel, type_ignores=[]), str(path), "exec"), namespace)
    expected = block_values if correlations == "none" else block_values * (2 / 3)
    return namespace, expected


@pytest.mark.parametrize(
    "correlations",
    [
        "none",
        pytest.param("xi_plus_minus", marks=DROPPED_COVARIANCE),
        pytest.param("xi_tau", marks=DROPPED_COVARIANCE),
        pytest.param("both", marks=DROPPED_COVARIANCE),
    ],
)
def test_mock_total_chi2_retains_cross_covariance(monkeypatch, correlations):
    """Protect the full joint mock statistic used against the fiducial likelihood.
    With unit marginal variances and rho=1/2, each residual pair (a,a)
    contributes 4*a**2/3 to the joint quadratic form, not 2*a**2.
    The synthetic FITS vector has xi bins at and within the real 12--83
    arcmin cuts and uncut tau bins; the zero-cross-block control must agree.
    For three mock amplitudes the correct joint totals are [40,60,80],
    so summing marginal statistics [60,90,120] is demonstrably wrong.
    The zero-cross-covariance control passes on develop because marginal and
    joint inverses agree for a diagonal matrix; it stays unmarked to guard
    against a regression introduced by the tomography merge.
    """
    result, expected = run_mock_loop(monkeypatch, correlations)
    actual = result["chi2_tots"]
    actual_tail = np.mean(actual > result["chi2_fiducial"])
    expected_tail = np.mean(expected > result["chi2_fiducial"])
    assert np.allclose(actual, expected), (
        f"{correlations}: mock chi2_tot={actual.tolist()}, expected full-covariance "
        f"chi2={expected.tolist()}; empirical PTE at {result['chi2_fiducial']:.6f}: "
        f"{actual_tail:.6f}, expected {expected_tail:.6f}"
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#380: mock PTE drops joint covariance",
)
def test_mock_pte_uses_same_full_covariance_statistic_as_fiducial(monkeypatch):
    """Protect the reported total-panel PTE from inconsistent quadratic forms.
    Only xi+/xi- are correlated here, with all xi/tau blocks exactly zero,
    as in the current FITS construction; hence no xi/tau assumption is needed.
    At the script's fiducial threshold 75.121834, the known joint totals
    [40,60,80] give a tail of 1/3, while the marginal sum [60,90,120] gives
    2/3; these samples avoid the histogram threshold bin so binning cannot
    explain the difference in the original panel's calculation.
    """
    result, expected = run_mock_loop(monkeypatch)
    correct_pte = np.mean(expected > result["chi2_fiducial"])
    assert result["p_value"] == pytest.approx(correct_pte), (
        f"printed-panel PTE={result['p_value']:.6f}, expected full-covariance "
        f"PTE={correct_pte:.6f}; mock chi2_tot={result['chi2_tots'].tolist()}"
    )
