"""Kish effective density is the weighted Kish count divided by square arcminutes.

A global positive weight scaling and row reordering leave it unchanged.
"""

import numpy as np
import pytest

from sp_validation import survey

pytestmark = [
    pytest.mark.fast,
    pytest.mark.decision("shear_field.covariance_survey_inputs"),
]


def test_neff_is_kish_count_per_square_arcminute():
    """False-alarm probability is 0; tolerance admits only fixture rounding error."""
    rng = np.random.default_rng(2026)
    weights = np.array([1.0, 2.0])
    weights = weights[rng.permutation(weights.size)]
    area_deg2 = 1.0
    expected_density = 1.0 / 2000.0
    tolerances = {"rtol": 1e-12, "atol": 1e-15}

    np.testing.assert_allclose(
        survey.n_eff_density(weights, area_deg2),
        expected_density,
        err_msg="n_eff must be (sum w)^2 / (sum w^2 * area_arcmin2)",
        **tolerances,
    )
    np.testing.assert_allclose(
        survey.n_eff_density(weights * 7.0, area_deg2),
        expected_density,
        err_msg="a global positive weight scaling must not change n_eff",
        **tolerances,
    )
    np.testing.assert_allclose(
        survey.n_eff_density(weights[::-1], area_deg2),
        expected_density,
        err_msg="reordering catalogue rows must not change n_eff",
        **tolerances,
    )
    np.testing.assert_allclose(
        survey.n_eff_density(np.ones(4), area_deg2),
        4.0 / 3600.0,
        err_msg="unit weights must give N / area_arcmin2",
        **tolerances,
    )
    np.testing.assert_allclose(
        survey.n_eff_density(np.array([]), area_deg2),
        0.0,
        err_msg="empty weights must give zero density",
        **tolerances,
    )
    np.testing.assert_allclose(
        survey.n_eff_density(np.concatenate([weights, np.zeros(5)]), area_deg2),
        expected_density,
        err_msg="appending zero-weight rows must not change n_eff",
        **tolerances,
    )
    np.testing.assert_allclose(
        survey.n_eff_density(np.zeros(3), area_deg2),
        0.0,
        err_msg="zero-weight rows must not count toward effective density",
        **tolerances,
    )
