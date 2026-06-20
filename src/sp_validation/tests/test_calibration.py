"""UNIT TESTS FOR CALIBRATION SUBPACKAGE.

This module contains value-drift CHARACTERIZATION tests for
``sp_validation.calibration``: the m/c bias linear algebra
(``get_calibrated_quantities``, ``get_calibrated_m_c``), the ``metacal``
response class, and the galaxy selection masks (``mask_gal_size``,
``mask_gal_SNR``).

The m/c functions consume a ``gal_metacal`` object through exactly three
attributes -- ``.R`` (2x2 response matrix), ``.ns`` (dict of 'g1'/'g2'/'w'
arrays), and ``.mask_dict`` (dict with key 'ns' -> boolean mask). For those
we build a LIGHTWEIGHT FAKE ``gal_metacal`` (a ``SimpleNamespace``) so we pin
the bias-correction maths in isolation; the ``metacal`` tests below exercise
the real class directly.

The literals below were produced by running the real estimator once in the
container and copying the observed output; they are pinned with a tight rtol
so any future refactor that changes the numbers fails loudly.

:Author: cdaley

"""

import types

import numpy as np
import numpy.testing as npt
import pytest
from astropy.table import Table

from sp_validation import calibration
from sp_validation.calibration import metacal, mask_gal_size, mask_gal_SNR


# Fixed, compact inputs shared across the tests.
#
# The response matrix is intentionally non-diagonal so that inv(R) mixes the
# two shear components (a diagonal R would let a transposed inverse pass
# silently). The mask drops indices 2 and 5, so the calibration operates on
# the 4 "kept" objects only.
R = np.array([[0.7, 0.05], [0.03, 0.72]])
G1 = np.array([0.10, -0.20, 0.30, 0.05, -0.15, 0.25])
G2 = np.array([-0.05, 0.15, -0.25, 0.20, 0.10, -0.30])
W = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 0.25])
MASK = np.array([True, True, False, True, True, False])


def make_fake_gal_metacal(R=R, g1=G1, g2=G2, w=W, mask=MASK):
    """Build a lightweight fake ``gal_metacal``.

    Only the three attributes the calibration functions actually touch are
    populated, so the test exercises the pure linear algebra and nothing else.
    """
    return types.SimpleNamespace(
        R=np.asarray(R, dtype=float),
        ns={
            "g1": np.asarray(g1, dtype=float),
            "g2": np.asarray(g2, dtype=float),
            "w": np.asarray(w, dtype=float),
        },
        mask_dict={"ns": np.asarray(mask, dtype=bool)},
    )


@pytest.fixture
def gal_metacal():
    return make_fake_gal_metacal()


def test_get_calibrated_quantities_pins_inv_R_application(gal_metacal):
    """Pin g_corr = inv(R) @ g_uncorr on the masked uncalibrated shears.

    WHAT is pinned: the masked uncalibrated shears (drop indices 2 and 5),
    the calibrated shears obtained by left-multiplying with inv(R), and the
    masked weights and the returned mask.

    TEETH: g_corr is asserted equal both to committed literals (rtol 1e-12)
    AND to a fresh inv(R) @ g_uncorr -- a refactor that, e.g., swapped inv(R)
    for R, transposed inv(R), or dropped the mask would change these numbers.
    A complementary teeth test below perturbs R and asserts g_corr moves.
    """
    g_corr, g_uncorr, w, mask = calibration.get_calibrated_quantities(gal_metacal)

    # Masked uncalibrated shears: indices 0, 1, 3, 4 of G1 / G2.
    npt.assert_allclose(
        g_uncorr,
        [[0.10, -0.20, 0.05, -0.15], [-0.05, 0.15, 0.20, 0.10]],
        rtol=1e-12,
    )

    # Calibrated shears: committed literals from an observed estimator run.
    npt.assert_allclose(
        g_corr,
        [
            [
                0.1482587064676617,
                -0.30149253731343284,
                0.05174129353233831,
                -0.22487562189054724,
            ],
            [
                -0.07562189054726369,
                0.2208955223880597,
                0.2756218905472637,
                0.1482587064676617,
            ],
        ],
        rtol=1e-12,
    )

    # And independently: g_corr is exactly inv(R) applied to g_uncorr.
    npt.assert_allclose(g_corr, np.linalg.inv(R).dot(g_uncorr), rtol=1e-12)

    # Masked weights and the propagated mask.
    npt.assert_allclose(w, [1.0, 2.0, 1.5, 3.0], rtol=1e-12)
    npt.assert_array_equal(mask, MASK)


def test_get_calibrated_quantities_teeth_R_changes_g_corr(gal_metacal):
    """TEETH: perturbing R perturbs g_corr.

    Confirms the headline calibration genuinely depends on R, so the pinned
    literals above are not accidentally robust to the response matrix. We
    nudge R[0, 0] from 0.70 to 0.65 and assert g_corr moves by a finite,
    asserted-different amount (here ~0.023 max abs change).
    """
    g_corr_ref, *_ = calibration.get_calibrated_quantities(gal_metacal)

    perturbed = make_fake_gal_metacal(R=[[0.65, 0.05], [0.03, 0.72]])
    g_corr_pert, *_ = calibration.get_calibrated_quantities(perturbed)

    assert np.abs(g_corr_ref - g_corr_pert).max() > 1e-3
    # The reference run must still match its pinned first element.
    npt.assert_allclose(g_corr_ref[0, 0], 0.1482587064676617, rtol=1e-12)


def test_get_calibrated_m_c_pins_additive_bias_and_corrected_shear(gal_metacal):
    """Pin c, c_err, and the m+c-corrected shear g_corr_mc.

    WHAT is pinned:
      - c[comp]     == mean(g_uncorr[comp]) over the masked sample,
      - c_err[comp] == std(g_uncorr[comp]) (population std, ddof=0),
      - g_corr_mc   == inv(R) @ g_uncorr - inv(R) @ c, componentwise.

    TEETH: every quantity is asserted against committed literals (rtol 1e-12)
    AND g_corr_mc is reconstructed from the documented closed form, so a
    refactor that changed the additive-bias subtraction (e.g. subtracting the
    uncorrected c instead of inv(R) @ c) would break the closed-form check.
    """
    g_corr_mc, g_uncorr, w, mask, c, c_err = calibration.get_calibrated_m_c(gal_metacal)

    # Additive bias = component-wise mean of the masked uncalibrated shears.
    npt.assert_allclose(c, [-0.05, 0.10], rtol=1e-12)
    npt.assert_allclose(c, np.mean(g_uncorr, axis=1), rtol=1e-12)

    # Error = population std (ddof=0).
    npt.assert_allclose(c_err, [0.12747548783981963, 0.09354143466934854], rtol=1e-12)
    npt.assert_allclose(c_err, np.std(g_uncorr, axis=1), rtol=1e-12)

    # m+c-corrected shear: committed literals.
    npt.assert_allclose(
        g_corr_mc,
        [
            [
                0.22985074626865673,
                -0.21990049751243781,
                0.13333333333333333,
                -0.14328358208955222,
            ],
            [
                -0.21791044776119406,
                0.07860696517412935,
                0.13333333333333336,
                0.005970149253731349,
            ],
        ],
        rtol=1e-12,
    )

    # And independently: the documented closed form inv(R) @ g_uncorr
    # minus the calibrated additive bias inv(R) @ c.
    Rinv = np.linalg.inv(R)
    closed_form = Rinv.dot(g_uncorr) - Rinv.dot(c)[:, None]
    npt.assert_allclose(g_corr_mc, closed_form, rtol=1e-12)


def test_get_calibrated_m_c_teeth_offset_tracks_in_c(gal_metacal):
    """TEETH: a constant ellipticity offset shifts c by exactly that offset.

    The additive bias c is the mean of the uncalibrated shears, so injecting a
    constant offset (delta1, delta2) into every g1/g2 must shift c by exactly
    (delta1, delta2). This is the defining property of an additive-bias
    estimator; a refactor that weighted, recentred, or rescaled c would break
    this exact tracking. We assert c moves by the injected offset to rtol
    1e-12, and that the m+c-corrected shear stays consistent with its own
    closed form on the offset input.
    """
    _, _, _, _, c_ref, _ = calibration.get_calibrated_m_c(gal_metacal)

    delta1, delta2 = 0.03, -0.07
    offset = make_fake_gal_metacal(g1=G1 + delta1, g2=G2 + delta2)
    g_corr_mc, g_uncorr, _, _, c_off, _ = calibration.get_calibrated_m_c(offset)

    # The additive bias tracks the injected offset exactly.
    npt.assert_allclose(c_off - c_ref, [delta1, delta2], rtol=1e-12)

    # The m+c estimate remains consistent with its closed form after the shift.
    Rinv = np.linalg.inv(R)
    closed_form = Rinv.dot(g_uncorr) - Rinv.dot(c_off)[:, None]
    npt.assert_allclose(g_corr_mc, closed_form, rtol=1e-12)


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
        size_corr_ell=False,  # avoid the in-place T mutation; clean linear cut
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

    expected = np.array([False, False, False, True, True, True, False, False, False])
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
