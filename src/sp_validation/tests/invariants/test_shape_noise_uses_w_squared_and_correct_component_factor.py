"""Uncentred shape RMS uses w² and separates summed and per-component sigma.

The summed-component sigma squared is twice the per-component sigma squared;
both are invariant to common weight scaling and linear in ellipticity scaling.
"""

import math

import numpy as np
import pytest

from sp_validation import survey

pytestmark = [
    pytest.mark.fast,
    pytest.mark.decision("shear_field.covariance_survey_inputs"),
]


def test_shape_noise_uses_w_squared_and_correct_component_factor():
    """The deterministic identity has zero statistical false-alarm probability;
    rtol=1e-12 and atol=1e-14 allow only floating-point error on this fixture.
    """
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 2.0])
    w = np.array([1.0, 2.0])
    area_deg2 = 1.0

    expected_sigma_pc = math.sqrt(17 / 10)
    expected_sigma_summed = math.sqrt(17 / 5)
    expected_n_eff = 9 / 18000

    sigma_pc = survey.ellipticity_dispersion(e1, e2, w)
    stats = survey.effective_survey_stats(e1, e2, w, area_deg2)
    sigma_summed = stats["sigma_e"]

    assert sigma_pc == pytest.approx(expected_sigma_pc, rel=1e-12, abs=1e-14)
    assert sigma_summed == pytest.approx(expected_sigma_summed, rel=1e-12, abs=1e-14)
    assert sigma_summed**2 == pytest.approx(2 * sigma_pc**2, rel=1e-12, abs=1e-14)
    assert stats["n_eff"] == pytest.approx(expected_n_eff, rel=1e-12, abs=1e-14)

    weight_scaled_sigma_pc = survey.ellipticity_dispersion(e1, e2, 7 * w)
    weight_scaled_stats = survey.effective_survey_stats(e1, e2, 7 * w, area_deg2)
    assert weight_scaled_sigma_pc == pytest.approx(
        expected_sigma_pc, rel=1e-12, abs=1e-14
    )
    assert weight_scaled_stats["sigma_e"] == pytest.approx(
        expected_sigma_summed, rel=1e-12, abs=1e-14
    )
    assert weight_scaled_stats["n_eff"] == pytest.approx(
        expected_n_eff, rel=1e-12, abs=1e-14
    )

    swapped_sigma_pc = survey.ellipticity_dispersion(e2, e1, w)
    swapped_stats = survey.effective_survey_stats(e2, e1, w, area_deg2)
    assert swapped_sigma_pc == pytest.approx(expected_sigma_pc, rel=1e-12, abs=1e-14)
    assert swapped_stats["sigma_e"] == pytest.approx(
        expected_sigma_summed, rel=1e-12, abs=1e-14
    )

    scaled_sigma_pc = survey.ellipticity_dispersion(0.1 * e1, 0.1 * e2, w)
    scaled_stats = survey.effective_survey_stats(0.1 * e1, 0.1 * e2, w, area_deg2)
    assert scaled_sigma_pc == pytest.approx(
        0.1 * expected_sigma_pc, rel=1e-12, abs=1e-14
    )
    assert scaled_stats["sigma_e"] == pytest.approx(
        0.1 * expected_sigma_summed, rel=1e-12, abs=1e-14
    )
