"""VALUE-DRIFT CHARACTERIZATION TESTS FOR THE B-MODE ESTIMATORS.

This module pins the numeric behavior of the pure E/B-mode helpers in
``sp_validation.b_modes`` against fixed, deterministic, in-memory inputs
(seeded RNG and hand-built arrays — no cluster data, no catalogue files).
Every pinned literal was produced by an actual run of the estimator inside
the container; a future refactor that changes the numbers must fail.

Each test carries *teeth*: alongside the pinned values we assert that a
perturbed input yields a provably different result, so the pins cannot be
satisfied by a trivially-broken implementation.

:Author: cdaley

"""

import types

import numpy as np
import numpy.testing as npt
import pytest

from sp_validation import b_modes

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Shared fixtures: deterministic stubs and inputs
# ---------------------------------------------------------------------------

# A 10-bin log-spaced angular grid (1 -> 100 arcmin), used by the scale-cut
# tests. Edges and bin centres are fully determined by geomspace.
_NBINS_GRID = 10
_EDGES = np.geomspace(1.0, 100.0, _NBINS_GRID + 1)
_LEFT_EDGES = _EDGES[:-1]
_RIGHT_EDGES = _EDGES[1:]
_MEANR = np.sqrt(_LEFT_EDGES * _RIGHT_EDGES)


def _grid_gg():
    """Lightweight stub exposing the three arrays scale_cut_to_bins reads."""
    return types.SimpleNamespace(
        meanr=_MEANR, left_edges=_LEFT_EDGES, right_edges=_RIGHT_EDGES
    )


def _eb_inputs():
    """Fixed seeded input for calculate_eb_statistics.

    nbins=4, npatch=50 so the Hartlap factor (n_eff - nbins_eff - 2)/(n_eff-1)
    is well-defined and strictly positive for every scale-cut combination.
    The covariance is built SPD via A @ A.T + I; the B-mode vectors are O(1)
    so the chi-squared (and hence PTE) lands in a meaningful range rather than
    being saturated at 1.0.
    """
    nbins, npatch = 4, 50
    rng = np.random.default_rng(12345)
    A = rng.standard_normal((6 * nbins, 6 * nbins))
    cov = A @ A.T + np.eye(6 * nbins)
    xip_B = rng.standard_normal(nbins)
    xim_B = rng.standard_normal(nbins)
    gg = types.SimpleNamespace(nbins=nbins, npatch1=npatch)
    return {"gg": gg, "cov": cov, "xip_B": xip_B, "xim_B": xim_B}, nbins


# ---------------------------------------------------------------------------
# 1. correlation_from_covariance
# ---------------------------------------------------------------------------


def test_correlation_from_covariance_exact():
    """Pin the correlation matrix derived from a known SPD covariance.

    For cov with diagonal (4, 9, 16) the standard deviations are (2, 3, 4),
    so the off-diagonals are exactly cov_ij / (s_i s_j): 2/(2*3)=1/3 and
    -3/(3*4)=-1/4. The diagonal is exactly unity. Tight rtol because the
    math is exact rational arithmetic.

    Teeth: a *different* covariance (test below) yields a different
    correlation, and the unit-diagonal / known off-diagonal structure would
    break under any rescaling bug in the normalization.
    """
    cov = np.array(
        [
            [4.0, 2.0, 0.0],
            [2.0, 9.0, -3.0],
            [0.0, -3.0, 16.0],
        ]
    )
    corr = b_modes.correlation_from_covariance(cov)

    expected = np.array(
        [
            [1.0, 1.0 / 3.0, 0.0],
            [1.0 / 3.0, 1.0, -1.0 / 4.0],
            [0.0, -1.0 / 4.0, 1.0],
        ]
    )
    npt.assert_allclose(corr, expected, rtol=1e-12, atol=0)
    # Diagonal is exactly unity, off-diagonal symmetric.
    npt.assert_allclose(np.diag(corr), np.ones(3), rtol=0, atol=1e-14)
    npt.assert_allclose(corr, corr.T, rtol=0, atol=1e-14)


def test_correlation_from_covariance_has_teeth():
    """Teeth for #1: perturbing one covariance entry changes the correlation.

    Flipping the sign of cov[0,1] flips the (0,1) correlation; the pinned
    +1/3 must NOT survive this perturbation.
    """
    cov = np.array(
        [
            [4.0, 2.0, 0.0],
            [2.0, 9.0, -3.0],
            [0.0, -3.0, 16.0],
        ]
    )
    cov_perturbed = cov.copy()
    cov_perturbed[0, 1] = cov_perturbed[1, 0] = -2.0
    corr_perturbed = b_modes.correlation_from_covariance(cov_perturbed)
    npt.assert_allclose(corr_perturbed[0, 1], -1.0 / 3.0, rtol=1e-12)
    assert not np.isclose(corr_perturbed[0, 1], 1.0 / 3.0)


# ---------------------------------------------------------------------------
# 2. scale_cut_to_bins
# ---------------------------------------------------------------------------


def test_scale_cut_to_bins_known_cuts():
    """Pin (start_bin, stop_bin) for known cuts on the log grid.

    The conservative logic excludes any bin whose left edge falls below
    min_scale (searchsorted on left_edges, side='left') or whose right edge
    rises above max_scale (searchsorted on right_edges, side='right').

    With edges = geomspace(1, 100, 11):
      left_edges  ~ [1, 1.585, 2.512, 3.981, 6.310, 10, 15.85, 25.12, 39.81, 63.10]
      right_edges ~ [1.585, 2.512, 3.981, 6.310, 10, 15.85, 25.12, 39.81, 63.10, 100]
    cut (2, 50)  -> start=2 (first left_edge >= 2 is 2.512), stop=8
    cut (5, 30)  -> start=4, stop=7
    None passthrough -> (0, nbins)
    full (1, 100) -> (0, nbins)
    """
    gg = _grid_gg()
    assert b_modes.scale_cut_to_bins(gg, 2.0, 50.0) == (2, 8)
    assert b_modes.scale_cut_to_bins(gg, 5.0, 30.0) == (4, 7)
    assert b_modes.scale_cut_to_bins(gg, None, None) == (0, _NBINS_GRID)
    assert b_modes.scale_cut_to_bins(gg, 1.0, 100.0) == (0, _NBINS_GRID)


def test_scale_cut_to_bins_has_teeth():
    """Teeth for #2: shifting the cut moves the bin boundaries.

    Tightening the lower cut from 2.0 to 4.0 advances start_bin (2 -> 4,
    since the left edge 3.981 < 4 is now excluded); tightening the upper cut
    from 50.0 to 24.0 retracts stop_bin (8 -> 6). The cut must move the bins.
    """
    gg = _grid_gg()
    base = b_modes.scale_cut_to_bins(gg, 2.0, 50.0)
    shifted = b_modes.scale_cut_to_bins(gg, 4.0, 24.0)
    assert base == (2, 8)
    assert shifted == (4, 6)
    assert shifted != base


# ---------------------------------------------------------------------------
# 3. find_conservative_scale_cut_key
# ---------------------------------------------------------------------------

_SCALE_CUT_KEYS = {
    (1.0, 100.0): "full",
    (2.0, 50.0): "a",
    (5.0, 30.0): "b",
    (2.0, 80.0): "c",
    (10.0, 40.0): "d",
}


def test_find_conservative_scale_cut_key_conservative_and_fallthrough():
    """Pin the chosen key for a conservative match and a fallthrough.

    Conservative branch: a request whose interval contains at least one key
    picks the widest such key (max k[1]-k[0]).
      req (2, 80): matches {(2,50),(5,30),(2,80),(10,40)}, widest is (2,80).
      req (1.5, 60): matches {(2,50),(5,30),(10,40)}, widest span is (2,50).
    Fallthrough branch: a request with NO conservative match picks the key
    minimizing |k0-min| + |k1-max|.
      req (4, 20): no key fits inside [4,20]; closest is (5,30)
                   (dist |5-4|+|30-20| = 11, beats all others).
    """
    assert b_modes.find_conservative_scale_cut_key(_SCALE_CUT_KEYS, (2.0, 80.0)) == (
        2.0,
        80.0,
    )
    assert b_modes.find_conservative_scale_cut_key(_SCALE_CUT_KEYS, (1.5, 60.0)) == (
        2.0,
        50.0,
    )
    # No conservative match -> closest-by-distance fallthrough.
    assert b_modes.find_conservative_scale_cut_key(_SCALE_CUT_KEYS, (4.0, 20.0)) == (
        5.0,
        30.0,
    )


def test_find_conservative_scale_cut_key_has_teeth():
    """Teeth for #3: changing the requested range changes the chosen key.

    The conservative request (2, 80) -> (2, 80) (widest span 78), but
    narrowing the upper bound to (2, 45) excludes (2,80) and (2,50), leaving
    {(5,30) span 25, (10,40) span 30} whose widest is (10,40) -> a *different*
    key.
    """
    wide = b_modes.find_conservative_scale_cut_key(_SCALE_CUT_KEYS, (2.0, 80.0))
    narrow = b_modes.find_conservative_scale_cut_key(_SCALE_CUT_KEYS, (2.0, 45.0))
    assert wide == (2.0, 80.0)
    assert narrow == (10.0, 40.0)
    assert wide != narrow


# ---------------------------------------------------------------------------
# 4. calculate_eb_statistics  (headline)
# ---------------------------------------------------------------------------


def test_calculate_eb_statistics_pte_matrices():
    """Pin representative PTE-matrix entries from the full 2D E/B analysis.

    Inputs are fixed (seed 12345, nbins=4, npatch=50, SPD cov = A@A.T + I,
    O(1) B-mode vectors). With cov_path_int=None the Hartlap correction uses
    n_eff = npatch = 50. For each of xip_B, xim_B and combined we pin the
    full-range entry [0, nbins-1] (start=0, stop=nbins) and an interior entry
    [0, 2] (start=0, stop=3). These chi2->sf PTE values are deterministic
    functions of the seeded input.

    rtol 1e-9: scipy's chi2.sf is double-precision deterministic, so the
    pins are tight enough that any change in the chi2/Hartlap/covariance-block
    arithmetic shifts them past tolerance.
    """
    results, nbins = _eb_inputs()
    out = b_modes.calculate_eb_statistics(results, cov_path_int=None)
    pm = out["pte_matrices"]

    # Full-range entries [0, nbins-1].
    npt.assert_allclose(pm["xip_B"][0, nbins - 1], 0.9985059590347458, rtol=1e-9)
    npt.assert_allclose(pm["xim_B"][0, nbins - 1], 0.9979174764123961, rtol=1e-9)
    npt.assert_allclose(pm["combined"][0, nbins - 1], 0.9999930883595443, rtol=1e-9)

    # Interior entries [0, 2] (start_bin=0, stop_bin=3).
    npt.assert_allclose(pm["xip_B"][0, 2], 0.9991074524059739, rtol=1e-9)
    npt.assert_allclose(pm["xim_B"][0, 2], 0.9896253892931961, rtol=1e-9)
    npt.assert_allclose(pm["combined"][0, 2], 0.9999497648089674, rtol=1e-9)

    # Structural pins: off the valid upper triangle the matrices are NaN.
    for key in ("xip_B", "xim_B", "combined"):
        m = pm[key]
        assert np.isnan(m[1, 0])  # start > stop-1 region is invalid
        assert np.isnan(m[2, 0])
        assert np.all(np.isfinite(np.diag(m)))  # single-bin cuts are valid


def test_calculate_eb_statistics_has_teeth():
    """Teeth for #4: a 10x-louder B-mode signal must drop the full-range PTE.

    Scaling xip_B and xim_B by 10 multiplies the chi-squared by ~100, so the
    survival-function PTE must fall sharply. We assert each perturbed
    full-range PTE is strictly (and substantially) smaller than the pinned
    quiet-signal value. Observed: xip 0.9985 -> 0.0251, xim 0.9979 -> 0.0104,
    combined 0.99999 -> 0.0031.
    """
    results, nbins = _eb_inputs()
    out = b_modes.calculate_eb_statistics(results, cov_path_int=None)
    pm = out["pte_matrices"]

    loud, _ = _eb_inputs()
    loud["xip_B"] = loud["xip_B"] * 10.0
    loud["xim_B"] = loud["xim_B"] * 10.0
    out_loud = b_modes.calculate_eb_statistics(loud, cov_path_int=None)
    pm_loud = out_loud["pte_matrices"]

    for key in ("xip_B", "xim_B", "combined"):
        quiet_pte = pm[key][0, nbins - 1]
        loud_pte = pm_loud[key][0, nbins - 1]
        assert loud_pte < quiet_pte
        assert loud_pte < 0.05  # louder B-modes are clearly rejected
