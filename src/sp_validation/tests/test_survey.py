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

        self._dd = np.array([
            (270.283, 1),
            (270.283, 0),
            (188.308, 0)],
            dtype=[('TILE_ID', 'f8'), ('FLAGS', 'i2')]
        )
        self._area_tile = 0.5
        self._verbose = False
        self._area_deg2 = 1
        self._area_amin2 = 3600
        self._tile_IDs = (270.283, 188.308)

        self._ra = np.array([240.0])
        self._dec = np.array([32.0])
        self._patch = ['P5']

    def tearDown(self):

        self.number_tile = None
        self.number_exp = None
        self.number_int = None

    def test_get_area(self):
        """Test ``sp_validation.survey_get_area`` method.

        """

        # Test return values
        area_deg2, area_amin2, tile_IDs = survey.get_area(
            self._dd,
            self._area_tile,
            self._verbose
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
            msg=f'{tile_IDs}!={self._tile_IDs}'
        )

    def test_get_footprint(self):
        """Test ``sp_validation.survey_get_footprint`` method.

        """
        for patch in self._patch:
            coords = survey.get_footprint(patch, self._ra, self._dec)
            self.assertTrue(coords[0])
