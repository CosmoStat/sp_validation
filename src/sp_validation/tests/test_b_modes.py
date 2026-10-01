"""VALUE-DRIFT CHARACTERIZATION TESTS FOR THE B-MODE ESTIMATORS.

This module pins the numeric behavior of the pure E/B-mode helpers in
``sp_validation.b_modes`` against fixed, deterministic inputs (seeded RNG,
hand-built arrays and one committed fine-grid ξ± fixture — no cluster data, no
catalogue files).
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

    nbins=4, npatch=50 so the Hartlap factor (npatch - p - 2)/(npatch - 1) is
    well-defined and strictly positive for every scale-cut combination.
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
    return {
        "theta": np.geomspace(1.0, 100.0, nbins),
        "npatch": npatch,
        "cov": cov,
        "xip_B": xip_B,
        "xim_B": xim_B,
    }, nbins


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
    O(1) B-mode vectors). The Hartlap correction uses npatch = 50 over the
    length of the inverted vector (2x for combined). For each of xip_B, xim_B
    and combined we pin the
    full-range entry [0, nbins-1] (start=0, stop=nbins) and an interior entry
    [0, 2] (start=0, stop=3). These chi2->sf PTE values are deterministic
    functions of the seeded input.

    rtol 1e-9: scipy's chi2.sf is double-precision deterministic, so the
    pins are tight enough that any change in the chi2/Hartlap/covariance-block
    arithmetic shifts them past tolerance.
    """
    results, nbins = _eb_inputs()
    out = b_modes.calculate_eb_statistics(results)
    pm = out["pte_matrices"]

    # Full-range entries [0, nbins-1].
    npt.assert_allclose(pm["xip_B"][0, nbins - 1], 0.9985059590347458, rtol=1e-9)
    npt.assert_allclose(pm["xim_B"][0, nbins - 1], 0.9979174764123961, rtol=1e-9)
    npt.assert_allclose(pm["combined"][0, nbins - 1], 0.9999952393605003, rtol=1e-9)

    # Interior entries [0, 2] (start_bin=0, stop_bin=3).
    npt.assert_allclose(pm["xip_B"][0, 2], 0.9991074524059739, rtol=1e-9)
    npt.assert_allclose(pm["xim_B"][0, 2], 0.9896253892931961, rtol=1e-9)
    npt.assert_allclose(pm["combined"][0, 2], 0.9999590178816327, rtol=1e-9)

    # Structural pins: off the valid upper triangle the matrices are NaN.
    for key in ("xip_B", "xim_B", "combined"):
        m = pm[key]
        assert np.isnan(m[1, 0])  # start > stop-1 region is invalid
        assert np.isnan(m[2, 0])
        assert np.all(np.isfinite(np.diag(m)))  # single-bin cuts are valid


def test_calculate_eb_statistics_analytic_covariance_skips_hartlap():
    """An analytic covariance (npatch None) gives the plain χ² PTE.

    The full-range χ² is data·C⁻¹·data with no factor, so its PTE is pinned
    directly against scipy; the jackknife PTE on the same input differs.
    """
    from scipy import stats

    results, nbins = _eb_inputs()
    results["npatch"] = None
    pm = b_modes.calculate_eb_statistics(results)["pte_matrices"]

    cov_B = results["cov_xip_B"]
    chi2 = results["xip_B"] @ np.linalg.solve(cov_B, results["xip_B"])
    npt.assert_allclose(pm["xip_B"][0, nbins - 1], stats.chi2.sf(chi2, nbins))
    npt.assert_allclose(pm["combined"][0, nbins - 1], 0.9999894806723678, rtol=1e-9)

    jackknife, _ = _eb_inputs()
    pm_jk = b_modes.calculate_eb_statistics(jackknife)["pte_matrices"]
    assert pm_jk["xip_B"][0, nbins - 1] != pm["xip_B"][0, nbins - 1]


def test_calculate_eb_statistics_has_teeth():
    """Teeth for #4: a 10x-louder B-mode signal must drop the full-range PTE.

    Scaling xip_B and xim_B by 10 multiplies the chi-squared by ~100, so the
    survival-function PTE must fall sharply. We assert each perturbed
    full-range PTE is strictly (and substantially) smaller than the pinned
    quiet-signal value. Observed: xip 0.9985 -> 0.0251, xim 0.9979 -> 0.0104,
    combined 0.99999 -> 0.0031.
    """
    results, nbins = _eb_inputs()
    out = b_modes.calculate_eb_statistics(results)
    pm = out["pte_matrices"]

    loud, _ = _eb_inputs()
    loud["xip_B"] = loud["xip_B"] * 10.0
    loud["xim_B"] = loud["xim_B"] * 10.0
    out_loud = b_modes.calculate_eb_statistics(loud)
    pm_loud = out_loud["pte_matrices"]

    for key in ("xip_B", "xim_B", "combined"):
        quiet_pte = pm[key][0, nbins - 1]
        loud_pte = pm_loud[key][0, nbins - 1]
        assert loud_pte < quiet_pte
        assert loud_pte < 0.05  # louder B-modes are clearly rejected


# ---------------------------------------------------------------------------
# 5. The pure-E/B operator on committed fine-grid ξ±
# ---------------------------------------------------------------------------

# calculate_pure_eb_correlation(**fixture) modes; regenerated only when the
# estimator is meant to move.
_PURE_EB_PINS = {
    "xip_E": [
        0.0001267927403538086,
        0.00011661471866769028,
        0.00011285998097487562,
        0.00011342826992976761,
        9.413729942634515e-05,
        8.504628509885555e-05,
    ],
    "xim_E": [
        7.84636379789262e-06,
        -2.1854362709680197e-08,
        1.2476650697806643e-06,
        5.931628899598654e-06,
        -5.748552740802833e-06,
        5.307121527471898e-06,
    ],
    "xip_B": [
        -5.890048396074994e-05,
        -5.413134881293027e-05,
        -7.033477076572411e-05,
        -6.464207645137761e-05,
        -6.438127715272536e-05,
        -5.781687325365341e-05,
    ],
    "xim_B": [
        -1.2460349217324633e-05,
        1.1641684220790763e-06,
        3.7838279437020602e-06,
        9.134846190559186e-06,
        5.499212937161553e-06,
        4.460746452672484e-06,
    ],
    "xip_amb": [
        8.984494180845635e-05,
        8.879536535671083e-05,
        8.704153035693623e-05,
        8.410960918777171e-05,
        7.923091774259722e-05,
        7.107059682711236e-05,
    ],
    "xim_amb": [
        1.5108576414191464e-06,
        9.059561894981808e-07,
        5.429016867191619e-07,
        3.250730808381593e-07,
        1.9496276632352545e-07,
        1.1676364182463817e-07,
    ],
}


def _spd(n, seed):
    """A seeded SPD matrix standing in for a ξ± covariance."""
    A = np.random.default_rng(seed).standard_normal((n, n))
    return A @ A.T / n + np.eye(n)


def test_pure_eb_reproduces_pins_on_committed_xi(pure_eb_xi):
    """The pure-E/B estimator on the committed ξ± reproduces its pins.

    With ξ± frozen, these pins move only when the estimator does. The operator
    is deterministic linear algebra, so rtol=1e-8 leaves room only for BLAS
    summation order.
    """
    n_fine = len(pure_eb_xi["theta_int"])
    results = b_modes.calculate_pure_eb_correlation(
        **pure_eb_xi, cov_xi=np.eye(2 * n_fine)
    )
    for key in b_modes._EB_KEYS:
        npt.assert_allclose(results[key], _PURE_EB_PINS[key], rtol=1e-8, err_msg=key)

    # Teeth: uniform rather than pair weights leave the pins.
    moved = b_modes.calculate_pure_eb_correlation(
        **{**pure_eb_xi, "weight_int": np.ones(n_fine)}, cov_xi=np.eye(2 * n_fine)
    )
    assert not np.allclose(moved["xip_E"], _PURE_EB_PINS["xip_E"], rtol=1e-6, atol=0)


def test_pure_eb_modes_sum_to_the_averaged_xi(pure_eb_xi):
    """ξ± = E ± B + amb holds in every reporting bin.

    It holds at each fine node by construction of the decomposition, and the
    modes and the reported ξ± are the same pair-weighted average of those nodes.
    """
    n_fine = len(pure_eb_xi["theta_int"])
    r = b_modes.calculate_pure_eb_correlation(**pure_eb_xi, cov_xi=np.eye(2 * n_fine))
    scale = np.abs(r["xip"]).max()
    npt.assert_allclose(
        r["xip"], r["xip_E"] + r["xip_B"] + r["xip_amb"], rtol=0, atol=1e-12 * scale
    )
    npt.assert_allclose(
        r["xim"], r["xim_E"] - r["xim_B"] + r["xim_amb"], rtol=0, atol=1e-12 * scale
    )


def test_pure_eb_covariance_is_the_operator_sandwich(pure_eb_xi):
    """``cov`` is K C Kᵀ for the supplied ξ± covariance, and records npatch.

    The reported ξ± variances are the same pair-weighted average pushed through
    the ξ+ and ξ− blocks of C.
    """
    n_fine = len(pure_eb_xi["theta_int"])
    cov_xi = _spd(2 * n_fine, seed=7)
    r = b_modes.calculate_pure_eb_correlation(**pure_eb_xi, cov_xi=cov_xi, npatch=40)
    K, P = b_modes.pure_eb_operator(
        pure_eb_xi["theta_int"],
        pure_eb_xi["weight_int"],
        pure_eb_xi["left_edges"],
        pure_eb_xi["right_edges"],
    )
    npt.assert_allclose(r["cov"], K @ cov_xi @ K.T, rtol=1e-12)
    npt.assert_allclose(r["cov"], r["cov"].T, rtol=1e-12)
    npt.assert_allclose(r["var_xip"], np.diag(P @ cov_xi[:n_fine, :n_fine] @ P.T))
    npt.assert_allclose(r["var_xim"], np.diag(P @ cov_xi[n_fine:, n_fine:] @ P.T))
    assert r["npatch"] == 40

    with pytest.raises(ValueError, match="npatch > 1"):
        b_modes.calculate_pure_eb_correlation(**pure_eb_xi, cov_xi=cov_xi, npatch=1)


def test_pure_eb_binning_is_a_pair_weighted_average():
    """Each reporting row averages its fine nodes with TreeCorr pair weights.

    Rows sum to one; nodes outside the reporting range or without pairs carry
    no weight; equal weights give the plain mean; an empty bin raises.
    """
    theta = np.geomspace(1.0, 100.0, 40)
    weight = np.arange(1.0, 41.0)
    weight[20] = 0.0
    left, right = b_modes.log_bin_edges(2.0, 50.0, 4)
    P = b_modes._weight_binning_matrix(theta, weight, left, right)

    npt.assert_allclose(P.sum(axis=1), 1.0)
    assert np.all(P[:, (theta < 2.0) | (theta >= 50.0)] == 0)
    assert np.all(P[:, 20] == 0)
    inside = (theta >= left[1]) & (theta < right[1]) & (weight > 0)
    npt.assert_allclose(P[1, inside], weight[inside] / weight[inside].sum())

    flat = b_modes._weight_binning_matrix(theta, np.ones(40), left, right)
    inside = (theta >= left[0]) & (theta < right[0])
    npt.assert_allclose(flat[0, inside], 1.0 / inside.sum())

    with pytest.raises(ValueError, match="hold no integration-grid pairs"):
        b_modes._weight_binning_matrix(theta, np.zeros(40), left, right)


def test_pure_eb_binning_reproduces_the_reporting_measurement():
    """P on nested fine bins gives TreeCorr's reporting-bin ξ± and meanr.

    TreeCorr's ξ± and meanr in a bin are averages over its pairs weighted by
    ``w_i w_j``, so pooling fine bins whose edges nest the reporting edges with
    the fine ``weight`` is the same sum. A weighted catalogue with exact
    binning makes that hold to round-off; weighting by ``npairs`` instead
    does not.
    """
    treecorr = pytest.importorskip("treecorr")

    rng = np.random.default_rng(2024)
    n_gal = 3000
    x, y = rng.uniform(0.0, 300.0, (2, n_gal))
    g1, g2 = 0.02 + 0.05 * rng.standard_normal((2, n_gal))
    cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, w=rng.uniform(0.2, 1.0, n_gal))

    exact = {"bin_slop": 0, "angle_slop": 0}
    reporting = treecorr.GGCorrelation(min_sep=15.0, max_sep=70.0, nbins=6, **exact)
    # Eight fine bins per reporting bin, plus four on either side.
    step = np.log(70.0 / 15.0) / 48
    fine = treecorr.GGCorrelation(
        min_sep=15.0 * np.exp(-4 * step),
        max_sep=70.0 * np.exp(4 * step),
        nbins=56,
        **exact,
    )
    reporting.process(cat)
    fine.process(cat)

    P = b_modes._weight_binning_matrix(
        fine.meanr, fine.weight, reporting.left_edges, reporting.right_edges
    )
    for key in ("xip", "xim", "meanr"):
        npt.assert_allclose(
            P @ getattr(fine, key), getattr(reporting, key), rtol=1e-10, err_msg=key
        )

    by_npairs = b_modes._weight_binning_matrix(
        fine.meanr, fine.npairs, reporting.left_edges, reporting.right_edges
    )
    assert not np.allclose(by_npairs @ fine.xip, reporting.xip, rtol=1e-6, atol=0)


def test_pure_eb_operator_refuses_a_reporting_floor_at_the_grid_edge(pure_eb_xi):
    """Reporting bins that reach the fine grid's floor raise, never return NaN.

    The ξ− integrals over [tmin, t] have fewer than interp_order + 1 nodes for
    the first few fine nodes, so those operator rows are under-determined.
    """
    theta_int = pure_eb_xi["theta_int"]
    with pytest.raises(ValueError, match="under-determined"):
        b_modes.pure_eb_operator(
            theta_int,
            pure_eb_xi["weight_int"],
            *b_modes.log_bin_edges(theta_int[0], 70.0, 6),
        )


# ---------------------------------------------------------------------------
# 6. Grid edges and the COSEBIs covariance seam
# ---------------------------------------------------------------------------


def test_log_bin_edges_matches_the_grid_stub():
    """Edges reconstructed from a binning are the ones TreeCorr would report.

    A part stores bin centres only, so a consumer rebuilds the edges from the
    binning it was measured on; the two must agree bin for bin.
    """
    left, right = b_modes.log_bin_edges(1.0, 100.0, _NBINS_GRID)
    gg = _grid_gg()
    npt.assert_allclose(left, gg.left_edges)
    npt.assert_allclose(right, gg.right_edges)
    # ...and they index scale cuts identically.
    assert b_modes.bins_from_edges(left, right, 2.0, 50.0) == (2, 8)


def test_cosebis_scan_propagates_the_supplied_covariance(monkeypatch):
    """The COSEBIs covariance is the ξ± covariance through the same kernel.

    The kernel is stubbed, so what is pinned is the seam: which ξ± covariance
    sub-block is handed to the transform (the scale cut's, in [ξ+; ξ−] order)
    and that Hartlap uses the supplied npatch.
    """
    nbins, nmodes = _NBINS_GRID, 3
    theta = np.geomspace(1.2, 90.0, nbins)
    cov_xipm = np.diag(np.arange(1.0, 2 * nbins + 1))
    seen = {}

    class _StubCOSEBIS:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def cosebis_from_xipm(self, theta_cut, xip_cut, xim_cut, parallel=True):
            seen["n_theta"] = len(theta_cut)
            return np.ones(nmodes), np.full(nmodes, 2.0)

        def cosebis_covariance_from_xipm_covariance(self, theta_cut, cov_cut):
            seen["cov_cut"] = cov_cut
            return np.eye(2 * nmodes)

    module = types.ModuleType("cosmo_numba.B_modes.cosebis")
    module.COSEBIS = _StubCOSEBIS
    monkeypatch.setitem(
        __import__("sys").modules, "cosmo_numba.B_modes.cosebis", module
    )

    left, right = b_modes.log_bin_edges(1.0, 100.0, nbins)
    results = b_modes.cosebis_scan_from_xi(
        theta,
        np.arange(nbins) * 1e-5,
        np.arange(nbins) * 2e-5,
        cov_xipm,
        left,
        right,
        nmodes=nmodes,
        scale_cuts=[(2.0, 50.0)],
        npatch=100,
    )

    (result,) = results.values()
    # The cut is bins 2..8, so the covariance sub-block is those rows/cols in
    # both the ξ+ and the ξ− half.
    inds = np.concatenate([np.arange(2, 8), np.arange(2, 8) + nbins])
    npt.assert_array_equal(seen["cov_cut"], cov_xipm[np.ix_(inds, inds)])
    assert seen["n_theta"] == 6
    npt.assert_allclose(result["hartlap_factor"], (100 - 2 * nmodes - 2) / 99)
    # χ² carries the Hartlap factor: modes are 1, cov is the identity.
    npt.assert_allclose(result["chi2_E"], nmodes * result["hartlap_factor"])


def test_cosebis_scan_theory_covariance_skips_hartlap(monkeypatch):
    """A theory covariance has no realisations to debias, so Hartlap is 1."""
    nbins, nmodes = _NBINS_GRID, 2

    class _StubCOSEBIS:
        def __init__(self, **kwargs):
            pass

        def cosebis_from_xipm(self, theta_cut, xip_cut, xim_cut, parallel=True):
            return np.ones(nmodes), np.ones(nmodes)

        def cosebis_covariance_from_xipm_covariance(self, theta_cut, cov_cut):
            return np.eye(2 * nmodes)

    module = types.ModuleType("cosmo_numba.B_modes.cosebis")
    module.COSEBIS = _StubCOSEBIS
    monkeypatch.setitem(
        __import__("sys").modules, "cosmo_numba.B_modes.cosebis", module
    )

    left, right = b_modes.log_bin_edges(1.0, 100.0, nbins)
    (result,) = b_modes.cosebis_scan_from_xi(
        np.geomspace(1.2, 90.0, nbins),
        np.zeros(nbins),
        np.zeros(nbins),
        np.eye(2 * nbins),
        left,
        right,
        nmodes=nmodes,
        npatch=None,
    ).values()
    assert result["hartlap_factor"] == 1


def test_pure_eb_npz_carries_what_the_summary_reads(tmp_path):
    """The .npz keys cv_summarize_bmodes reads are the ones the writer emits.

    The two live in different rules, so the contract between them — the PTE
    matrices under ``pte_matrices_{stat}`` and the jackknife patch count under
    ``npatch`` — is pinned here rather than discovered on a cluster run.
    """
    results, nbins = _eb_inputs()
    results.update(
        {key: np.zeros(nbins) for key in b_modes._EB_KEYS if key not in results}
    )
    results = b_modes.calculate_eb_statistics(results)

    out = tmp_path / "pure_eb_data.npz"
    b_modes.save_pure_eb_results(results, str(out))
    saved = np.load(out)

    for stat in ("xip_B", "xim_B", "combined"):
        assert f"pte_matrices_{stat}" in saved
        assert saved[f"pte_matrices_{stat}"].shape == (nbins, nbins)
    assert saved["npatch"] == results["npatch"]
    npt.assert_allclose(saved["theta"], results["theta"])
    for key in b_modes._EB_KEYS:
        assert key in saved

    # The summary reads the fiducial cut out of those matrices through the same
    # helper the plots use, so a valid cut must resolve to a finite PTE.
    edges = b_modes.log_bin_edges(1.0, 100.0, nbins)
    pte = b_modes._get_pte_from_scale_cut(
        saved["pte_matrices_xip_B"], edges, (1.0, 100.0)
    )
    assert np.isfinite(pte)
