"""VALUE-DRIFT CHARACTERIZATION TESTS FOR basic.py.

This module pins the numeric behaviour of the metacalibration estimator
and the standalone selection / jackknife helpers in
:mod:`sp_validation.basic`. Every assertion is a *characterization* test:
the inputs are fully deterministic (hand-built arrays / seeded RNG, no
cluster data) and the outputs are committed as literals with a tight
``rtol``. A refactor that changes the numbers must turn this file red.

Each test documents the perturbation that gives it teeth: a deliberate
change to the input that is asserted to produce a different result, so a
reviewer can independently confirm the test bites.

:Author: cdaley

"""

import numpy as np
import numpy.testing as npt
from astropy.table import Table

from sp_validation.basic import (
    metacal,
    mask_gal_size,
    mask_gal_SNR,
    jackknif_weighted_average2,
)


# Metacal sheared-image variants and their NGMIX column suffixes.
_VARIANTS = ["1M", "1P", "2M", "2P", "NOSHEAR"]


def _build_ngmix_catalog(slope_11=2.0, slope_22=3.0, step=0.01):
    """Build a deterministic NGMIX catalogue with a known metacal response.

    The metacal shear response is a finite difference of the measured
    ellipticity between the +shear and -shear sheared images, divided by
    twice the step ``h``:

        R11 = (g1[1P] - g1[1M]) / (2 h)
        R22 = (g2[2P] - g2[2M]) / (2 h)
        R12 = (g1[2P] - g1[2M]) / (2 h)
        R21 = (g2[1P] - g2[1M]) / (2 h)

    We inject a pure diagonal response by making the measured g1 vary
    linearly with the applied component-1 step (slope ``slope_11``) and g2
    vary linearly with the applied component-2 step (slope ``slope_22``),
    with NO cross-response. Concretely, relative to a NOSHEAR baseline of
    (g1, g2) = (0.05, -0.02):

        1P: g1 += slope_11 * step,   1M: g1 -= slope_11 * step
        2P: g2 += slope_22 * step,   2M: g2 -= slope_22 * step

    so the finite difference recovers R11 = slope_11, R22 = slope_22, and
    R12 = R21 = 0 exactly. The selection response is zero because every
    object passes every cut identically (the selection masks are the same
    set for 1P/1M/2P/2M), so the total R equals the injected slope matrix.

    All objects are built to pass the galaxy cuts: flag == 0,
    rel_size = T / Tpsf = 1.0 (inside [0.5, 3.0]), and
    SNR = flux / flux_err = 50 (inside [10, 500]).
    """
    n = 8
    g1_base = 0.05
    g2_base = -0.02

    # Per-variant (g1, g2) shifts: only the matching component moves.
    shifts = {
        "NOSHEAR": (0.0, 0.0),
        "1P": (slope_11 * step, 0.0),
        "1M": (-slope_11 * step, 0.0),
        "2P": (0.0, slope_22 * step),
        "2M": (0.0, -slope_22 * step),
    }

    cols = {}
    for var in _VARIANTS:
        dg1, dg2 = shifts[var]
        ell = np.zeros((n, 2))
        ell[:, 0] = g1_base + dg1
        ell[:, 1] = g2_base + dg2
        cols[f"NGMIX_ELL_{var}"] = ell
        cols[f"NGMIX_FLAGS_{var}"] = np.zeros(n, dtype=int)
        cols[f"NGMIX_FLUX_{var}"] = np.full(n, 50.0)
        cols[f"NGMIX_FLUX_ERR_{var}"] = np.full(n, 1.0)
        # T / Tpsf = 1.0 -> rel_size inside [0.5, 3.0].
        cols[f"NGMIX_T_{var}"] = np.full(n, 1.0)
        cols[f"NGMIX_T_ERR_{var}"] = np.full(n, 0.1)
        cols[f"NGMIX_Tpsf_{var}"] = np.full(n, 1.0)

    # Per-component ellipticity errors -> inverse-variance weights.
    cols["NGMIX_ELL_ERR_NOSHEAR"] = np.full((n, 2), 0.25)

    return Table(cols), n


def test_metacal_R_matrix_recovers_injected_response():
    """Pin the metacal total response matrix R for an injected slope.

    WHAT IS PINNED: with a hand-built NGMIX catalogue whose measured
    ellipticity responds linearly to the applied metacal shear step
    (slope 2.0 on component 1, slope 3.0 on component 2, no cross term),
    the estimator's total response ``mcal.R`` is a 2x2 matrix equal to the
    injected slopes:  R = [[2.0, 0.0], [0.0, 3.0]]. The off-diagonal terms
    and the selection response are exactly zero because every object passes
    every cut identically.

    WHY TEETH: the response is the finite difference R11 = slope_11 by
    construction. If a refactor changed the step normalisation (the 2 h
    divisor), the +/-/component bookkeeping, the selection-response
    subtraction, or the column names read by ``_read_data_ngmix``, R would
    no longer equal the injected matrix. The companion assertion below
    re-runs with slope 5.0 and confirms R11 tracks it (5.0 != 2.0), so a
    change that decouples R from the input numbers fails.

    NOTE: the estimator prints an 'FHP/MK hack' line and an unweighted /
    weighted response line; these are expected stdout, not errors.
    """
    data, n = _build_ngmix_catalog(slope_11=2.0, slope_22=3.0, step=0.01)
    mask = np.ones(n, dtype=bool)

    mcal = metacal(
        data,
        mask,
        masking_type="gal",
        step=0.01,
        prefix="NGMIX",
        size_corr_ell=False,   # avoid the in-place T mutation; clean linear cut
        global_R_weight=None,  # unweighted mean over objects
    )

    assert mcal.R.shape == (2, 2)
    npt.assert_allclose(
        mcal.R,
        np.array([[2.0, 0.0], [0.0, 3.0]]),
        rtol=1e-10,
        atol=1e-12,
    )

    # TEETH: a different injected slope yields a different R11.
    data2, n2 = _build_ngmix_catalog(slope_11=5.0, slope_22=3.0, step=0.01)
    mcal2 = metacal(
        data2,
        np.ones(n2, dtype=bool),
        masking_type="gal",
        step=0.01,
        prefix="NGMIX",
        size_corr_ell=False,
        global_R_weight=None,
    )
    npt.assert_allclose(mcal2.R[0, 0], 5.0, rtol=1e-10)
    assert not np.isclose(mcal2.R[0, 0], mcal.R[0, 0])


def test_metacal_R_matrix_step_normalization():
    """Pin that R is independent of the step h for a fixed injected slope.

    WHAT IS PINNED: the response is divided by 2 h, and the injected
    ellipticity shift is slope * h, so R = slope regardless of h. Building
    the catalogue with step = 0.02 (and reading it back with step = 0.02)
    still gives R11 = 2.0, R22 = 3.0.

    WHY TEETH: this isolates the ``h2 = 2 * self._step`` normalisation in
    ``_shear_response``. If a refactor dropped the factor of 2 or used the
    wrong step, R would scale and this exact-match assertion would fail.
    """
    data, n = _build_ngmix_catalog(slope_11=2.0, slope_22=3.0, step=0.02)
    mcal = metacal(
        data,
        np.ones(n, dtype=bool),
        masking_type="gal",
        step=0.02,
        prefix="NGMIX",
        size_corr_ell=False,
        global_R_weight=None,
    )
    npt.assert_allclose(
        mcal.R,
        np.array([[2.0, 0.0], [0.0, 3.0]]),
        rtol=1e-10,
        atol=1e-12,
    )


def test_mask_gal_size_boolean_mask():
    """Pin the exact boolean size mask spanning below/within/above the cut.

    WHAT IS PINNED: ``mask_gal_size`` keeps objects with
    rel_size_min < T / Tpsf < rel_size_max (strict on both ends). With
    Tpsf = 1.0 everywhere and T spanning the bounds [0.5, 3.0], the
    expected mask is computed element by element below.

    WHY TEETH: T / Tpsf values 0.5 and 3.0 sit exactly on the (strict)
    bounds and are EXCLUDED; 0.49 and 3.01 are also excluded; the interior
    values pass. The companion assertion tightens rel_size_min to 1.0 and
    asserts the 0.75 element flips from True to False.
    """
    T = np.array([0.3, 0.49, 0.5, 0.75, 1.0, 2.999, 3.0, 3.01, 5.0])
    Tpsf = np.ones_like(T)

    mask = mask_gal_size(T, Tpsf, rel_size_min=0.5, rel_size_max=3.0)

    expected = np.array(
        [False, False, False, True, True, True, False, False, False]
    )
    npt.assert_array_equal(mask, expected)

    # TEETH: tightening the lower bound to 1.0 drops the 0.75 element.
    mask_tight = mask_gal_size(T, Tpsf, rel_size_min=1.0, rel_size_max=3.0)
    assert mask[3] and not mask_tight[3]
    assert not np.array_equal(mask, mask_tight)


def test_mask_gal_SNR_boolean_mask():
    """Pin the exact boolean SNR mask spanning below/within/above the cut.

    WHAT IS PINNED: ``mask_gal_SNR`` keeps objects with
    snr_min < SNR < snr_max (strict on both ends). With bounds [10, 500],
    the boundary values 10 and 500 are EXCLUDED and the interior passes.

    WHY TEETH: shifting the lower bound from 10 to 12 flips the SNR = 11
    element from True to False; the companion assertion confirms it.
    """
    SNR = np.array([5.0, 10.0, 11.0, 100.0, 499.0, 500.0, 600.0])

    mask = mask_gal_SNR(SNR, snr_min=10.0, snr_max=500.0)

    expected = np.array([False, False, True, True, True, False, False])
    npt.assert_array_equal(mask, expected)

    # TEETH: raising snr_min to 12 drops the SNR = 11 element.
    mask_shifted = mask_gal_SNR(SNR, snr_min=12.0, snr_max=500.0)
    assert mask[2] and not mask_shifted[2]
    assert not np.array_equal(mask, mask_shifted)


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
