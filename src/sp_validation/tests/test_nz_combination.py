"""Known answers for the SOM + clustering combination."""

import numpy as np

from sp_validation.nz import combination as cb

GRID = np.round(np.arange(0, 3.001, 0.05), 3)
LO = np.round(np.arange(0.3, 1.0, 0.05), 3)
HI = LO + 0.05


def _template(mean=0.65, sigma=0.12):
    n = np.exp(-0.5 * ((GRID - mean) / sigma) ** 2)
    return n / np.trapezoid(n, GRID)


def test_slice_matrix_averages_a_linear_nz():
    n = 0.2 + 0.1 * GRID
    expected = 0.2 + 0.1 * (LO + HI) / 2
    np.testing.assert_allclose(cb.slice_matrix(GRID, LO, HI) @ n, expected, rtol=1e-4)


def test_combination_recovers_a_shifted_truth_and_its_bias_slope():
    rng = np.random.default_rng(0)
    draws = cb.shift_stretch(_template(), GRID, 0.05, 0.0, 2000, rng)
    truth = draws[np.argmin(np.abs(cb.mean_z(GRID, draws) - 0.68))]
    z = (LO + HI) / 2
    y = 3.0 * cb.slice_matrix(GRID, LO, HI) @ truth * (1 + z) ** 2.0
    precision = np.eye(len(y)) / 0.02**2
    r = cb.combine(GRID, draws, y, precision, LO, HI, (1.0, 2.0))
    assert abs(r["posterior"][0] - cb.mean_z(GRID, truth)) < 0.005
    assert r["posterior"][1] < 0.1 * r["prior"][1]
    assert abs(r["alpha"][0] - 2.0) < 0.3


def test_flat_data_leave_the_prior_unchanged():
    rng = np.random.default_rng(1)
    draws = cb.shift_stretch(_template(), GRID, 0.02, 0.0, 500, rng)
    y = cb.slice_matrix(GRID, LO, HI) @ _template()
    r = cb.combine(GRID, draws, y, np.eye(len(y)) * 1e-8, LO, HI, (1.0, 1.0))
    np.testing.assert_allclose(r["posterior"], r["prior"], atol=1e-4)
    assert r["ess"] > 499
