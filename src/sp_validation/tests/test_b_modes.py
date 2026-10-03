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
        0.00012671119742922553,
        0.00011647263203301798,
        0.00011277680462627065,
        0.00011334741455438306,
        9.407280835053578e-05,
        8.500501372536384e-05,
    ],
    "xim_E": [
        7.684864690558884e-06,
        -2.9296916553309855e-07,
        1.1150412226648596e-06,
        5.81495542664638e-06,
        -5.8802091546090845e-06,
        5.245220040191294e-06,
    ],
    "xip_B": [
        -5.872804581564311e-05,
        -5.3898600777518044e-05,
        -7.01611878338831e-05,
        -6.447148201049941e-05,
        -6.422794082425538e-05,
        -5.7688865883825626e-05,
    ],
    "xim_B": [
        -1.2625808436389906e-05,
        8.906960918149856e-07,
        3.6497760325039334e-06,
        9.0173210532702e-06,
        5.367045395488404e-06,
        4.3985417696428825e-06,
    ],
    "xip_amb": [
        8.975404658793264e-05,
        8.870470395597095e-05,
        8.695112377370022e-05,
        8.401987012227805e-05,
        7.914207248993659e-05,
        7.098386083077631e-05,
    ],
    "xim_amb": [
        1.5068975296876175e-06,
        9.035986620575128e-07,
        5.414736226368331e-07,
        3.2422141650144327e-07,
        1.9445163845662097e-07,
        1.1646044607564388e-07,
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

    The reported ξ± variances are the same pair-weighted average pushed
    through the ξ+ and ξ− blocks of C, and the reported edges are the snapped
    ones the operator used.
    """
    n_fine = len(pure_eb_xi["theta_int"])
    cov_xi = _spd(2 * n_fine, seed=7)
    r = b_modes.calculate_pure_eb_correlation(**pure_eb_xi, cov_xi=cov_xi, npatch=40)
    K, P, edges = b_modes.pure_eb_operator(
        pure_eb_xi["weight_int"],
        pure_eb_xi["edges_int"],
        pure_eb_xi["reporting_edges"],
    )
    npt.assert_allclose(r["cov"], K @ cov_xi @ K.T, rtol=1e-12)
    npt.assert_allclose(r["cov"], r["cov"].T, rtol=1e-12)
    npt.assert_allclose(r["var_xip"], np.diag(P @ cov_xi[:n_fine, :n_fine] @ P.T))
    npt.assert_allclose(r["var_xim"], np.diag(P @ cov_xi[n_fine:, n_fine:] @ P.T))
    npt.assert_array_equal(r["left_edges"], edges[:-1])
    npt.assert_array_equal(r["right_edges"], edges[1:])
    assert np.all(np.isin(edges, pure_eb_xi["edges_int"]))
    assert r["npatch"] == 40

    with pytest.raises(ValueError, match="npatch > 1"):
        b_modes.calculate_pure_eb_correlation(**pure_eb_xi, cov_xi=cov_xi, npatch=1)


def test_pure_eb_reporting_bins_are_unions_of_fine_bins():
    """Requested edges snap to the nearest fine edge; rows pool whole fine bins.

    Rows sum to one and weight their fine bins by the pair weight; a fine bin
    with no weight carries none; equal weights give the plain mean. Edges that
    snap together, edges outside the fine grid and empty bins raise.
    """
    edges_int = np.geomspace(1.0, 100.0, 41)
    weight = np.arange(1.0, 41.0)
    weight[20] = 0.0
    requested = np.array([2.1, 6.0, 20.0, 49.0])
    P, edges = b_modes._reporting_binning(weight, edges_int, requested)

    snap = [np.argmin(np.abs(np.log(edges_int / e))) for e in requested]
    npt.assert_array_equal(edges, edges_int[snap])
    npt.assert_allclose(P.sum(axis=1), 1.0)
    for row, (lo, hi) in enumerate(zip(snap[:-1], snap[1:])):
        inside = np.zeros(40, dtype=bool)
        inside[lo:hi] = True
        npt.assert_allclose(P[row, inside], weight[inside] / weight[inside].sum())
        assert np.all(P[row, ~inside] == 0)
    assert np.all(P[:, 20] == 0)

    flat, _ = b_modes._reporting_binning(np.ones(40), edges_int, requested)
    npt.assert_allclose(flat[0, snap[0] : snap[1]], 1.0 / (snap[1] - snap[0]))

    with pytest.raises(ValueError, match="same fine edge"):
        b_modes._reporting_binning(weight, edges_int, [2.0, 2.05, 20.0])
    with pytest.raises(ValueError, match="outside the fine grid"):
        b_modes._reporting_binning(weight, edges_int, [2.0, 20.0, 200.0])
    with pytest.raises(ValueError, match="hold no integration-grid pairs"):
        b_modes._reporting_binning(np.zeros(40), edges_int, requested)


def _weighted_catalogue(treecorr):
    """A weighted flat shear catalogue, separations in arcmin."""
    rng = np.random.default_rng(2024)
    n_gal = 3000
    x, y = rng.uniform(0.0, 300.0, (2, n_gal))
    g1, g2 = 0.02 + 0.05 * rng.standard_normal((2, n_gal))
    return treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, w=rng.uniform(0.2, 1.0, n_gal))


@pytest.mark.parametrize(
    "requested",
    [np.geomspace(15.0, 70.0, 7), np.geomspace(14.0, 73.0, 7)],
    ids=["nested", "snapped"],
)
def test_pure_eb_binning_reproduces_the_reporting_measurement(requested):
    """P on the fine grid gives TreeCorr's ξ± and meanr on the snapped bins.

    TreeCorr's ξ± and meanr in a bin are averages over its pairs weighted by
    ``w_i w_j``, and every reporting bin is a union of fine bins, so pooling
    them with the fine ``weight`` is the same sum. A weighted catalogue with
    exact binning makes that hold to round-off, whether or not the requested
    edges fall on fine edges; weighting by ``npairs`` instead does not.
    """
    treecorr = pytest.importorskip("treecorr")
    cat = _weighted_catalogue(treecorr)
    exact = {"bin_slop": 0, "angle_slop": 0}

    # Eight fine bins per [15, 70]′ / 6 reporting bin, plus four either side.
    step = np.log(70.0 / 15.0) / 48
    fine = treecorr.GGCorrelation(
        min_sep=15.0 * np.exp(-4 * step),
        max_sep=70.0 * np.exp(4 * step),
        nbins=56,
        **exact,
    )
    fine.process(cat)
    edges_int = np.append(fine.left_edges, fine.right_edges[-1])

    P, edges = b_modes._reporting_binning(fine.weight, edges_int, requested)
    assert np.all(np.isin(edges, edges_int))
    measured = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        gg = treecorr.GGCorrelation(min_sep=lo, max_sep=hi, nbins=1, **exact)
        gg.process(cat)
        measured.append(gg)
    for key in ("xip", "xim", "meanr"):
        npt.assert_allclose(
            P @ getattr(fine, key),
            [getattr(gg, key)[0] for gg in measured],
            rtol=1e-10,
            err_msg=key,
        )

    by_npairs, _ = b_modes._reporting_binning(fine.npairs, edges_int, requested)
    assert not np.allclose(
        by_npairs @ fine.xip, [gg.xip[0] for gg in measured], rtol=1e-6, atol=0
    )


def test_pure_eb_transform_needs_a_log_uniform_grid():
    """The transform runs on log-uniform nodes and refuses irregular edges.

    cosmo_numba's interpolator places samples on a regular grid in log θ, so
    edges that are not log-uniform would silently mis-place them.
    """
    edges = np.geomspace(1.0, 300.0, 61)
    edges[30] *= 1.001
    with pytest.raises(ValueError, match="log-uniform"):
        b_modes._fixed_quadrature_operator(edges)


def test_pure_eb_operator_refuses_a_reporting_floor_at_the_grid_edge(pure_eb_xi):
    """Reporting bins that reach the fine grid's floor raise, never return NaN.

    The ξ− integrals over [tmin, t] have fewer than interp_order + 1 nodes for
    the first few fine nodes, so those operator rows are under-determined.
    """
    edges_int = pure_eb_xi["edges_int"]
    with pytest.raises(ValueError, match="under-determined"):
        b_modes.pure_eb_operator(
            pure_eb_xi["weight_int"],
            edges_int,
            np.geomspace(edges_int[0], 70.0, 7),
        )


# ---------------------------------------------------------------------------
# 6. Grid edges and the COSEBIs covariance seam
# ---------------------------------------------------------------------------


def test_scale_cuts_select_the_same_bins_on_every_fine_grid():
    """A cut on nominal edges picks the same bins whatever grid they snapped to.

    [1, 250]′ in 20 bins snapped onto the cosmo_val (0.08–300′) and Paper II
    (0.5–300′) fine grids moves edge 9 (11.997′) to either side of 12′, so
    exact containment would disagree; snapping the cut to the nearest edge
    selects bins 9–15 for [12, 83]′ and all bins for [1, 250]′ on both.
    """
    requested = np.geomspace(1.0, 250.0, 21)
    for lo in (0.08, 0.5):
        fine = np.geomspace(lo, 300.0, 1001)
        _, edges = b_modes._reporting_binning(np.ones(1000), fine, requested)
        left, right = edges[:-1], edges[1:]
        assert b_modes.bins_from_scale_cut(left, right, (12.0, 83.0)) == (9, 16)
        assert b_modes.bins_from_scale_cut(left, right, (1.0, 250.0)) == (0, 20)
        pte = np.arange(400.0).reshape(20, 20)
        assert (
            b_modes._get_pte_from_scale_cut(pte, (left, right), (12, 83)) == pte[9, 15]
        )

    with pytest.raises(RuntimeError, match="selects no bins"):
        b_modes.bins_from_scale_cut(left, right, (12.0, 12.5))


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
    matrices under ``pte_matrices_{stat}``, the reporting edges they are
    indexed on and the jackknife patch count under ``npatch`` — is pinned here
    rather than discovered on a cluster run.
    """
    results, nbins = _eb_inputs()
    results.update(
        {key: np.zeros(nbins) for key in b_modes._EB_KEYS if key not in results}
    )
    results["left_edges"], results["right_edges"] = b_modes.log_bin_edges(
        1.0, 100.0, nbins
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
    # helper the plots use, on the saved edges, so a valid cut must resolve to
    # a finite PTE.
    edges = (saved["left_edges"], saved["right_edges"])
    npt.assert_array_equal(edges[0], results["left_edges"])
    pte = b_modes._get_pte_from_scale_cut(
        saved["pte_matrices_xip_B"], edges, (1.0, 100.0)
    )
    assert np.isfinite(pte)
