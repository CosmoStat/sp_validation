"""Posterior modes must preserve density-axis order and parameter identity.

The weighted mean must use sample weights; exact grid selections have zero
statistical false-alarm probability, and the mean allows only floating-point error.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.fast, pytest.mark.decision("inference.posterior_summary")]

SCRIPT = Path.cwd() / "cosmo_inference" / "scripts" / "chain_postprocessing.py"


class AnalyticDensityChain:
    """Small chain adapter exposing the GetDist methods used by the estimators."""

    x_2d = np.array([0.1, 0.3, 0.8])
    y_2d = np.array([0.6, 0.9])
    p_2d = np.array([[1.0, 2.0, 3.0], [9.0, 4.0, 5.0]])
    x_1d = np.array([0.2, 0.5, 0.7])
    p_1d = np.array([1.0, 4.0, 2.0])

    def getParamNames(self):
        return SimpleNamespace(parWithName=lambda name: SimpleNamespace(name=name))

    def get2DDensity(self, par_x, par_y, fine_bins_2D):
        assert (par_x.name, par_y.name) == ("S_8", "OMEGA_M")
        assert fine_bins_2D == 1000
        return SimpleNamespace(x=self.x_2d, y=self.y_2d, P=self.p_2d)

    def get1DDensity(self, par, num_bins):
        assert par.name == "m1"
        assert num_bins == 1000
        return SimpleNamespace(x=self.x_1d, P=self.p_1d)

    def getMargeStats(self):
        return None

    def getLikeStats(self):
        return SimpleNamespace(
            names=[SimpleNamespace(name=name) for name in self.names]
        )

    names = ("S_8", "OMEGA_M", "m1")


def _load_chain_postprocessing():
    spec = importlib.util.spec_from_file_location(
        "chain_postprocessing_under_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    exec(
        compile(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT), "exec"),
        module.__dict__,
    )
    return module


def test_posterior_mode_reads_density_axes_in_xy_order():
    """The deterministic identity has false-alarm probability 0;
    floating-point mean tolerance is rtol=1e-12, atol=1e-14.
    """
    from getdist import MCSamples

    post = _load_chain_postprocessing()
    chain = AnalyticDensityChain()

    mode_2d = post.compute_map_2D(chain, "S_8", "OMEGA_M")
    assert tuple(map(float, mode_2d)) == (0.1, 0.9), (
        f"2D KDE peak should use x[col], y[row]; got {mode_2d}, expected (0.1, 0.9)"
    )
    assert float(post.compute_map_1D(chain, "m1")) == 0.5, (
        "1D KDE peak should use its x coordinate"
    )

    best_fit = post.extract_best_fit_params(chain, best_fit_method="2Dkde")
    assert best_fit == {"S_8": 0.1, "OMEGA_M": 0.9, "m1": 0.5}, (
        f"2D KDE coordinates or the 1D m1 mode were assigned to the wrong key: {best_fit}"
    )

    values = np.repeat([0.1, 0.3, 0.8], 16)
    weights = np.repeat([1.0, 2.0, 1.0], 16)
    order = np.random.default_rng(1729).permutation(values.size)
    weighted_chain = MCSamples(
        samples=values[order, None],
        weights=weights[order],
        names=["m1"],
        ignore_rows=0,
    )
    got_mean = post.compute_average(weighted_chain, "m1")
    np.testing.assert_allclose(
        got_mean,
        0.375,
        rtol=1e-12,
        atol=1e-14,
        err_msg="posterior mean must use GetDist sample weights, not an unweighted mean",
    )
