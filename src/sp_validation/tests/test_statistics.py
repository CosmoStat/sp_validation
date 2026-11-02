"""VALUE-DRIFT CHARACTERIZATION TESTS FOR statistics.py.

This module pins the numeric behaviour of the cosmology-independent
statistical helpers in :mod:`sp_validation.statistics`. The inputs are
fully deterministic (seeded RNG, no cluster data) and the outputs are
committed as literals with a tight ``rtol``. A refactor that changes the
numbers must turn this file red.

:Author: cdaley

"""

import numpy as np
import numpy.testing as npt
from scipy import stats

from sp_validation.statistics import (
    calibrate_min_pte,
    chi2_and_pte,
    corr_from_cov,
    cov_from_one_covariance,
    effective_number_of_tests,
    jackknif_weighted_average2,
    jackknife_weighted_mean,
)


def test_jackknif_weighted_average2_mean_and_error():
    """Pin the jackknife weighted average + error for a seeded RNG draw.

    WHAT IS PINNED: ``jackknif_weighted_average2`` draws ``n_realization``
    bootstrap-style subsamples (size = (1 - remove_size) * N, sampled WITH
    replacement via ``np.random.choice``) and returns
    (mean over realizations, std over realizations) of the per-subsample
    weighted average. With ``np.random.seed`` fixed before the call, the
    draws are deterministic, so both numbers are pinned as literals
    obtained from an actual run.

    WHY TEETH: the function depends on the data values, the weights, and
    the sampling. The companion assertion changes a single weight (under
    the same seed and therefore the same index draws) and asserts the mean
    moves, proving the weighting is load-bearing and not ignored.
    """
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    np.random.seed(1234)
    mean, err = jackknif_weighted_average2(
        data, weights, remove_size=0.1, n_realization=50
    )

    npt.assert_allclose(mean, _PINNED_JK_MEAN, rtol=1e-10)
    npt.assert_allclose(err, _PINNED_JK_ERR, rtol=1e-10)

    # TEETH: same seed (same index draws) but a perturbed weight -> the
    # weighted average changes, so the returned mean must differ.
    weights_perturbed = weights.copy()
    weights_perturbed[0] = 100.0
    np.random.seed(1234)
    mean_p, _ = jackknif_weighted_average2(
        data, weights_perturbed, remove_size=0.1, n_realization=50
    )
    assert not np.isclose(mean, mean_p)


# Literals pinned from an observed run inside the container; see the test
# docstring for the seed and parameters that reproduce them.
_PINNED_JK_MEAN = 6.248279984721161
_PINNED_JK_ERR = 0.971349020459367


def test_corr_from_cov_pins_correlation_matrix():
    """Pin the correlation matrix derived from a fixed 3x3 covariance.

    WHAT IS PINNED: ``corr_from_cov`` normalises each entry by the geometric
    mean of the two diagonal standard deviations,
    ``corr[i, j] = cov[i, j] / (sigma_i sigma_j)``. With the hand-picked
    covariance below the off-diagonals are exact rationals
    (1/sqrt(4*9) = 1/6, -2/sqrt(4*16) = -1/4, 3/sqrt(9*16) = 1/4), pinned as
    literals from an observed container run.

    WHY TEETH: the defining property of a correlation matrix is a UNIT
    DIAGONAL, and every off-diagonal must lie in [-1, 1]; both are asserted
    independently of the literals. A companion check reconstructs the matrix
    from the closed form ``cov / outer(std, std)`` so a refactor that, e.g.,
    divided by the variances instead of the standard deviations would break
    the unit-diagonal property and the closed form simultaneously.
    """
    cov = np.array([[4.0, 1.0, -2.0], [1.0, 9.0, 3.0], [-2.0, 3.0, 16.0]])
    corr = corr_from_cov(cov)

    npt.assert_allclose(
        corr,
        [[1.0, 1.0 / 6.0, -0.25], [1.0 / 6.0, 1.0, 0.25], [-0.25, 0.25, 1.0]],
        rtol=1e-12,
    )

    # TEETH: a correlation matrix has unit diagonal and |off-diagonal| <= 1.
    npt.assert_allclose(np.diag(corr), 1.0, rtol=1e-12)
    assert np.all(np.abs(corr) <= 1.0)

    # And independently: the documented closed form cov / outer(std, std).
    std = np.sqrt(np.diag(cov))
    npt.assert_allclose(corr, cov / np.outer(std, std), rtol=1e-12)


def test_chi2_and_pte_diagonal_reduces_to_sum_of_squares():
    """Pin chi2/reduced-chi2/PTE and prove the diagonal-cov reduction.

    WHAT IS PINNED: for a diagonal covariance ``diag(sigma^2)`` the quadratic
    form collapses to ``chi2 = sum((d / sigma)^2)`` with ``dof = len(d)``,
    ``reduced_chi2 = chi2 / dof`` and ``pte = 1 - chi2.cdf(chi2, dof)``. All
    three are pinned as literals from an observed container run.

    WHY TEETH: the diagonal case has a closed form, so chi2 is asserted equal
    to ``sum((d / sigma)^2)`` and the PTE to the matching ``scipy`` survival
    value -- a refactor that dropped the inverse, used the covariance instead
    of its inverse, or mis-set the dof would break the reduction. The
    companion check feeds the same data through a NON-diagonal covariance and
    asserts the chi2 changes, proving the full quadratic form is exercised.
    """
    d = np.array([1.0, -2.0, 0.5, 3.0])
    sigma = np.array([2.0, 1.0, 0.5, 4.0])
    cov_diag = np.diag(sigma**2)

    chi2, reduced_chi2, pte = chi2_and_pte(d, cov_diag)

    npt.assert_allclose(chi2, 5.8125, rtol=1e-12)
    npt.assert_allclose(reduced_chi2, 1.453125, rtol=1e-12)
    npt.assert_allclose(pte, 0.21359530221841705, rtol=1e-12)

    # TEETH: the diagonal quadratic form is exactly sum((d / sigma)^2), and
    # the PTE is the matching scipy chi2 survival probability on dof = len(d).
    chi2_closed = np.sum((d / sigma) ** 2)
    npt.assert_allclose(chi2, chi2_closed, rtol=1e-12)
    npt.assert_allclose(reduced_chi2, chi2_closed / len(d), rtol=1e-12)
    npt.assert_allclose(pte, 1 - stats.chi2.cdf(chi2_closed, len(d)), rtol=1e-12)

    # TEETH: off-diagonal covariance terms change the quadratic form, so the
    # full d^T C^-1 d path (not just the diagonal shortcut) is load-bearing.
    cov_full = np.array(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.2, 0.0],
            [0.0, 0.2, 0.25, 0.1],
            [0.0, 0.0, 0.1, 16.0],
        ]
    )
    chi2_full, _, pte_full = chi2_and_pte(d, cov_full)
    npt.assert_allclose(chi2_full, 13.526036131774706, rtol=1e-12)
    npt.assert_allclose(pte_full, 0.00897199803545734, rtol=1e-12)
    assert not np.isclose(chi2_full, chi2)


def test_cov_from_one_covariance_selects_gaussian_column():
    """Pin the reshaped matrix and prove the gaussian column selection."""
    # obs, ell1, ell2, s1, s2, tomoi, tomoj, tomok, tomol, cov, covg, covng, covssc
    one_cov = np.array(
        [
            [0.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 99.0, 100.0, 0.0, 0.0],
            [0.0, 10.0, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 101.0, 0.0, 0.0],
            [0.0, 20.0, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 110.0, 111.0, 0.0, 0.0],
        ]
    )

    cov_gauss = cov_from_one_covariance(one_cov, gaussian=True)
    cov_nongauss = cov_from_one_covariance(one_cov, gaussian=False)

    npt.assert_allclose(cov_gauss, [[100.0, 101.0], [101.0, 111.0]], rtol=1e-12)
    npt.assert_allclose(cov_nongauss, [[99.0, 100.0], [100.0, 110.0]], rtol=1e-12)

    # TEETH: the gaussian flag shifts the column by one, so every entry of
    # the gaussian matrix exceeds its non-gaussian counterpart by exactly 1.
    npt.assert_allclose(cov_gauss - cov_nongauss, 1.0, rtol=1e-12)


def test_cov_from_one_covariance_orders_tomo_blocks():
    """Pin the tomo-block ordering against ``combinations_with_replacement``."""
    # obs, ell1, ell2, s1, s2, tomoi, tomoj, tomok, tomol, cov, covg, covng, covssc
    one_cov = np.array(
        [
            [
                0.0,
                10.0,
                10.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                0.0,
                0.0,
                0.0,
            ],  # (a,b)=(0,0)
            [
                0.0,
                10.0,
                10.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                2.0,
                0.0,
                1.0,
                0.0,
                0.0,
            ],  # (0,1)
            [
                0.0,
                10.0,
                10.0,
                1.0,
                1.0,
                1.0,
                1.0,
                2.0,
                2.0,
                1.0,
                2.0,
                0.0,
                0.0,
            ],  # (0,2)
            [
                0.0,
                10.0,
                10.0,
                1.0,
                1.0,
                1.0,
                2.0,
                1.0,
                2.0,
                10.0,
                11.0,
                0.0,
                0.0,
            ],  # (1,1)
            [
                0.0,
                10.0,
                10.0,
                1.0,
                1.0,
                1.0,
                2.0,
                2.0,
                2.0,
                11.0,
                12.0,
                0.0,
                0.0,
            ],  # (1,2)
            [
                0.0,
                10.0,
                10.0,
                1.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                21.0,
                22.0,
                0.0,
                0.0,
            ],  # (2,2)
        ]
    )

    cov_gauss = cov_from_one_covariance(one_cov, gaussian=True)

    npt.assert_allclose(
        cov_gauss,
        [[0.0, 1.0, 2.0], [1.0, 11.0, 12.0], [2.0, 12.0, 22.0]],
        rtol=1e-12,
    )


def test_calibrate_min_pte_independent_uniform_ptes():
    """Independent uniform PTEs recover the Sidak threshold and k_eff = k."""
    rng = np.random.default_rng(1)
    alpha, k = 0.05, 6
    cal = calibrate_min_pte(rng.uniform(size=(200_000, k)), alpha=alpha)
    expected = 1.0 - (1.0 - alpha) ** (1.0 / k)
    npt.assert_allclose(cal.threshold, expected, rtol=0.02)
    npt.assert_allclose(cal.k_eff, k, rtol=0.02)
    assert cal.threshold_interval[0] <= cal.threshold <= cal.threshold_interval[1]
    assert cal.k_eff_interval[0] <= cal.k_eff <= cal.k_eff_interval[1]
    npt.assert_allclose(effective_number_of_tests(expected, alpha), k)


def test_calibrate_min_pte_perfectly_correlated_is_one_test():
    """Identical columns collapse to one test: threshold alpha, k_eff 1."""
    rng = np.random.default_rng(2)
    column = rng.uniform(size=(100_000, 1))
    cal = calibrate_min_pte(np.repeat(column, 8, axis=1), alpha=0.05)
    npt.assert_allclose(cal.threshold, 0.05, rtol=0.03)
    npt.assert_allclose(cal.k_eff, 1.0, rtol=0.03)


def test_calibrate_min_pte_two_sided_independent():
    """Two-sided PTEs 2 min(p, 1 - p) are uniform, so Sidak still holds."""
    rng = np.random.default_rng(3)
    cal = calibrate_min_pte(rng.uniform(size=(200_000, 4)), alpha=0.05, two_sided=True)
    npt.assert_allclose(cal.k_eff, 4.0, rtol=0.03)


def test_min_pte_global_pte():
    """Global p-value is the mock fraction with min PTE <= the data's."""
    mock_ptes = np.array([[0.1, 0.9], [0.5, 0.2], [0.3, 0.7], [0.8, 0.6]])
    cal = calibrate_min_pte(mock_ptes, alpha=0.25)
    # Mock minima: 0.1, 0.2, 0.3, 0.6.
    p, (lo, hi) = cal.global_pte([0.9, 0.2])
    assert p == 0.5
    assert lo < 0.5 < hi
    assert cal.global_pte([0.05, 0.5])[0] == 0.0
    assert cal.global_pte([0.99, 0.95])[0] == 1.0
    # Two-sided: a suspiciously good PTE of 0.99 counts like 0.02.
    two = calibrate_min_pte(mock_ptes, alpha=0.25, two_sided=True)
    assert two.global_pte([0.99, 0.5])[0] == 0.0


def test_global_pte_is_calibrated_under_the_null():
    """For null data the global p-value is uniform: P(p <= alpha) ~ alpha."""
    rng = np.random.default_rng(4)
    mean = np.zeros(5)
    cov = 0.6 * np.ones((5, 5)) + 0.4 * np.eye(5)
    to_pte = lambda z: stats.norm.sf(z)  # noqa: E731
    cal = calibrate_min_pte(to_pte(rng.multivariate_normal(mean, cov, 4000)))
    data = to_pte(rng.multivariate_normal(mean, cov, 4000))
    p = np.array([cal.global_pte(row)[0] for row in data])
    npt.assert_allclose(np.mean(p <= 0.05), 0.05, atol=0.012)
    assert 1.0 < cal.k_eff < 5.0


def test_jackknife_weighted_mean_white_noise_matches_shot_noise():
    """Uncorrelated values: the patch jackknife gives the shot-noise error."""
    rng = np.random.default_rng(1)
    n, npatch, sigma = 200_000, 50, 0.3
    x = rng.normal(0.0, sigma, n)
    w = rng.uniform(0.5, 1.5, n)
    patch = rng.integers(0, npatch, n)

    mean, err = jackknife_weighted_mean(x, w, patch)

    shot = np.sqrt(np.sum(w**2 * (x - np.average(x, weights=w)) ** 2)) / w.sum()
    npt.assert_allclose(mean, np.average(x, weights=w), atol=1e-5)
    npt.assert_allclose(err, shot, rtol=0.25)


def test_jackknife_weighted_mean_sees_patch_offsets():
    """A constant offset per patch raises the error by its scatter / sqrt(K)."""
    rng = np.random.default_rng(2)
    n, npatch = 200_000, 50
    patch = rng.integers(0, npatch, n)
    w = np.ones(n)
    offset = rng.normal(0.0, 0.01, npatch)

    _, err = jackknife_weighted_mean(offset[patch], w, patch)

    npt.assert_allclose(err, offset.std(ddof=1) / np.sqrt(npatch), rtol=0.05)


def test_jackknife_weighted_mean_skips_empty_patches():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.ones(4)
    with_gap = np.array([0, 0, 2, 2])
    without_gap = np.array([0, 0, 1, 1])
    npt.assert_allclose(
        jackknife_weighted_mean(x, w, with_gap),
        jackknife_weighted_mean(x, w, without_gap),
    )
