"""UNIT TESTS FOR SURVEY SUBPACKAGE.

This module contains unit tests for the module package
sp_validation.survey.

:Author: Martin Kilbinger <martin.kilbinger@cea.fr>

"""

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from sp_validation import survey


class SurveyTestCase(TestCase):
    def setUp(self):

        self._dd = np.array(
            [(270.283, 1), (270.283, 0), (188.308, 0)],
            dtype=[("TILE_ID", "f8"), ("FLAGS", "i2")],
        )
        self._area_tile = 0.5
        self._verbose = False
        self._area_deg2 = 1
        self._area_amin2 = 3600
        self._tile_IDs = (270.283, 188.308)

    def tearDown(self):

        self.number_tile = None
        self.number_exp = None
        self.number_int = None

    def test_get_area(self):
        """Test ``sp_validation.survey_get_area`` method."""

        # Test return values
        area_deg2, area_amin2, tile_IDs = survey.get_area(
            self._dd, self._area_tile, self._verbose
        )
        npt.assert_almost_equal(
            area_deg2,
            self._area_deg2,
        )
        npt.assert_almost_equal(
            area_amin2,
            self._area_amin2,
        )
        self.assertTrue(
            sorted(tile_IDs) == sorted(self._tile_IDs),
            msg=f"{tile_IDs}!={self._tile_IDs}",
        )


def test_additive_bias_errors_shape_noise_and_jackknife():
    rng = np.random.default_rng(3)
    n, npatch, R = 100_000, 20, 0.7
    e1 = rng.normal(0.0, 0.25, n)
    e2 = rng.normal(0.0, 0.25, n)
    w = np.ones(n)
    patch = rng.integers(0, npatch, n)

    errors = survey.additive_bias_errors(e1, e2, w, R, patch=patch)

    # Unit weights: sigma_e / sqrt(n), not var / sqrt(n)
    npt.assert_allclose(errors["sn"], (np.std(e1 / R), np.std(e2 / R)) / np.sqrt(n))
    npt.assert_allclose(errors["jk"], errors["sn"], rtol=0.4)
    assert survey.additive_bias_errors(e1, e2, w, R)["jk"] is None
