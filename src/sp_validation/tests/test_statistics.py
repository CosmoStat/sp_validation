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
    chi2_and_pte,
    corr_from_cov,
    cov_from_one_covariance,
    jackknif_weighted_average2,
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
    """Pin the reshaped matrix and prove the gaussian column selection.

    WHAT IS PINNED: ``cov_from_one_covariance`` reads a flat OneCovariance
    table with one row per ``(i, j)`` pair (row-major, ``k = i*n_bins + j``)
    and lays the chosen column into a square matrix. Column 10 carries the
    Gaussian-only term (``gaussian=True``); column 9 the Gaussian+non-Gaussian
    term (``gaussian=False``). With a deterministic table whose entries encode
    their row and column, both reshaped matrices are pinned as literals.

    WHY TEETH: the only difference between the two calls is the column index
    (10 vs 9), so the two pinned matrices differ by exactly 1 in every entry,
    proving the gaussian flag selects the right column. A companion check
    places a unique tag ``10*i + j`` in column 10 and asserts the reshape is
    row-major (``cov[i, j]`` lands at row ``i*n_bins + j``), so a transposed
    or column-major refactor would change the recovered matrix.
    """
    # Two-bin table -> 4 rows; entry (row k, col c) = 10*k + c, so the
    # chosen-column values are distinct and self-documenting. Row k carries
    # the (i, j) pair with k = i*n_bins + j, so rows 0..3 -> (0,0),(0,1),
    # (1,0),(1,1). Column 10 holds values 10, 20, 30, 40; column 9 holds
    # 9, 19, 29, 39.
    one_cov = np.array([np.arange(11.0) + 10.0 * k for k in range(4)])

    cov_gauss = cov_from_one_covariance(one_cov, gaussian=True)
    cov_nongauss = cov_from_one_covariance(one_cov, gaussian=False)

    npt.assert_allclose(cov_gauss, [[10.0, 20.0], [30.0, 40.0]], rtol=1e-12)
    npt.assert_allclose(cov_nongauss, [[9.0, 19.0], [29.0, 39.0]], rtol=1e-12)

    # TEETH: the gaussian flag shifts the column by one, so every entry of the
    # gaussian matrix exceeds its non-gaussian counterpart by exactly 1.
    npt.assert_allclose(cov_gauss - cov_nongauss, 1.0, rtol=1e-12)

    # TEETH: the (i, j) -> row k = i*n_bins + j layout is row-major. Tag
    # column 10 with 10*i + j and check it lands at cov[i, j], not cov[j, i].
    tagged = np.zeros((4, 11))
    for i in range(2):
        for j in range(2):
            tagged[i * 2 + j, 10] = 10.0 * i + j
    npt.assert_allclose(
        cov_from_one_covariance(tagged, gaussian=True),
        [[0.0, 1.0], [10.0, 11.0]],
        rtol=1e-12,
    )
