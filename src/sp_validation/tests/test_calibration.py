"""UNIT TESTS FOR CALIBRATION SUBPACKAGE.

This module contains value-drift CHARACTERIZATION tests for the pure
calibration linear algebra in ``sp_validation.calibration``:
``get_calibrated_quantities`` and ``get_calibrated_m_c``.

Both functions consume a ``gal_metacal`` object through exactly three
attributes -- ``.R`` (2x2 response matrix), ``.ns`` (dict of 'g1'/'g2'/'w'
arrays), and ``.mask_dict`` (dict with key 'ns' -> boolean mask). We build a
LIGHTWEIGHT FAKE ``gal_metacal`` (a ``SimpleNamespace``) so we pin the
multiplicative/additive bias correction maths without coupling to the
metacal-class internals (``test_basic.py`` owns the class itself).

The literals below were produced by running the real estimator once in the
container and copying the observed output; they are pinned with a tight rtol
so any future refactor that changes the numbers fails loudly.

:Author: cdaley

"""

import types

import numpy as np
import numpy.testing as npt
import pytest

from sp_validation import calibration


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
        ns={"g1": np.asarray(g1, dtype=float),
            "g2": np.asarray(g2, dtype=float),
            "w": np.asarray(w, dtype=float)},
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
        [[0.10, -0.20, 0.05, -0.15],
         [-0.05, 0.15, 0.20, 0.10]],
        rtol=1e-12,
    )

    # Calibrated shears: committed literals from an observed estimator run.
    npt.assert_allclose(
        g_corr,
        [[0.1482587064676617, -0.30149253731343284,
          0.05174129353233831, -0.22487562189054724],
         [-0.07562189054726369, 0.2208955223880597,
          0.2756218905472637, 0.1482587064676617]],
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
    g_corr_mc, g_uncorr, w, mask, c, c_err = calibration.get_calibrated_m_c(
        gal_metacal
    )

    # Additive bias = component-wise mean of the masked uncalibrated shears.
    npt.assert_allclose(c, [-0.05, 0.10], rtol=1e-12)
    npt.assert_allclose(c, np.mean(g_uncorr, axis=1), rtol=1e-12)

    # Error = population std (ddof=0).
    npt.assert_allclose(
        c_err, [0.12747548783981963, 0.09354143466934854], rtol=1e-12
    )
    npt.assert_allclose(c_err, np.std(g_uncorr, axis=1), rtol=1e-12)

    # m+c-corrected shear: committed literals.
    npt.assert_allclose(
        g_corr_mc,
        [[0.22985074626865673, -0.21990049751243781,
          0.13333333333333333, -0.14328358208955222],
         [-0.21791044776119406, 0.07860696517412935,
          0.13333333333333336, 0.005970149253731349]],
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
