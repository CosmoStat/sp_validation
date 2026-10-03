"""Metacal selection response is the centered finite difference of NOSHEAR means
on the +/- sheared masks. Selected means use configured response weights, and
sheared ellipticities must not enter this term.
"""

import numpy as np
import pytest

from sp_validation.calibration import metacal

pytestmark = [
    pytest.mark.fast,
    pytest.mark.decision("calibration.response_estimator"),
]


def _make_metacal(global_weight, sheared_offset=0.0):
    """Build the four-row synthetic estimator without running catalogue setup."""
    rng = np.random.default_rng(314159)
    estimator = metacal.__new__(metacal)
    estimator._step = 0.01
    estimator._global_R_weight = global_weight
    estimator.ns = {
        "g1": np.array([0.001, 0.003, -0.001, -0.003]),
        "g2": np.array([0.004, -0.002, 0.002, -0.004]),
        "w": np.array([1.0, 3.0, 1.0, 3.0]),
    }
    estimator.mask_dict = {
        "ns": np.array([True, True, True, True]),
        "p1": np.array([True, True, True, False]),
        "m1": np.array([True, False, True, True]),
        "p2": np.array([True, True, False, False]),
        "m2": np.array([False, False, True, True]),
    }
    estimator.p1 = {key: rng.normal(size=4) + sheared_offset for key in ("g1", "g2")}
    estimator.m1 = {key: rng.normal(size=4) + sheared_offset for key in ("g1", "g2")}
    estimator.p2 = {key: rng.normal(size=4) + sheared_offset for key in ("g1", "g2")}
    estimator.m2 = {key: rng.normal(size=4) + sheared_offset for key in ("g1", "g2")}
    return estimator


def _weighted_mean(values, weights, mask):
    """Calculate a selected weighted mean using explicit finite sums."""
    selected = [index for index, keep in enumerate(mask) if keep]
    return sum(values[index] * weights[index] for index in selected) / sum(
        weights[index] for index in selected
    )


def _reference_selection(global_weight):
    """Independently calculate all four centered finite differences."""
    estimator = _make_metacal(global_weight)
    weights = estimator.ns[global_weight] if global_weight is not None else np.ones(4)
    masks = estimator.mask_dict
    h2 = 2 * estimator._step
    return np.array(
        [
            [
                (
                    _weighted_mean(estimator.ns["g1"], weights, masks["p1"])
                    - _weighted_mean(estimator.ns["g1"], weights, masks["m1"])
                )
                / h2,
                (
                    _weighted_mean(estimator.ns["g1"], weights, masks["p2"])
                    - _weighted_mean(estimator.ns["g1"], weights, masks["m2"])
                )
                / h2,
            ],
            [
                (
                    _weighted_mean(estimator.ns["g2"], weights, masks["p1"])
                    - _weighted_mean(estimator.ns["g2"], weights, masks["m1"])
                )
                / h2,
                (
                    _weighted_mean(estimator.ns["g2"], weights, masks["p2"])
                    - _weighted_mean(estimator.ns["g2"], weights, masks["m2"])
                )
                / h2,
            ],
        ]
    )


def test_selection_response_is_centered_difference_of_noshear_means():
    """Unweighted R_selection is the centered finite difference of NOSHEAR means
    on the +/- masks, and sheared ellipticities never enter it. The identity is
    exact; rtol=0 and atol=1e-12 allow only floating-point roundoff.
    """
    expected = _reference_selection(None)
    np.testing.assert_allclose(
        expected, [[0.10, 0.20], [1 / 30, 0.10]], rtol=0, atol=1e-12
    )
    unweighted = _make_metacal(None)
    unweighted._selection_response()
    np.testing.assert_allclose(unweighted.R_selection, expected, rtol=0, atol=1e-12)

    shifted = _make_metacal(None, sheared_offset=10.0)
    shifted._selection_response()
    np.testing.assert_allclose(
        shifted.R_selection, unweighted.R_selection, rtol=0, atol=1e-12
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "assay:response-weight-mismatch: _selection_response ignores the "
        "configured response weight that _total_response applies to R_shear"
    ),
)
def test_selection_response_uses_the_shear_response_weight():
    """With global_R_weight set, R_selection uses the same weights as R_shear.
    The identity is exact; rtol=0 and atol=1e-12 allow only roundoff.
    """
    expected = _reference_selection("w")
    np.testing.assert_allclose(
        expected, [[0.18, 0.25], [0.06, 0.10]], rtol=0, atol=1e-12
    )
    weighted = _make_metacal("w")
    weighted._selection_response()
    np.testing.assert_allclose(weighted.R_selection, expected, rtol=0, atol=1e-12)
