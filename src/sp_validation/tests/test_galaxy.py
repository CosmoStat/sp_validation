"""UNIT TESTS FOR GALAXY SUBPACKAGE.

This module contains unit tests for the module package
sp_validation.galaxy.

:Author: Cail Daley <cailmdaley@gmail.com>

"""

from unittest import TestCase

import numpy.testing as npt


class GalaxyTestCase(TestCase):

    def test_galaxy_imports(self):
        """Test that the galaxy module imports.

        The package-level imports in ``sp_validation/__init__.py`` are
        commented out, so without this smoke test a broken dependency
        (e.g. a cs_util without ``cs_util.size``) passes CI silently.
        """
        from sp_validation import galaxy  # noqa: F401

    def test_T_to_fwhm_is_dimensionally_correct(self):
        """Test FWHM(T) = 2 sqrt(2 ln 2) sqrt(T / 2) = 2.35482 sigma.

        T = 2 sigma^2 is an area; the conversion to the length FWHM
        carries a square root. The previous local implementation
        (T / 1.17741 * 2.355) was linear in T and only coincided with
        the correct value near sigma = 0.5.
        """
        from sp_validation.galaxy import T_to_fwhm, sigma_to_fwhm

        # unit-sigma Gaussian: T = 2, FWHM = 2.35482
        npt.assert_allclose(T_to_fwhm(2.0), 2.3548200450, rtol=1e-6)
        npt.assert_allclose(sigma_to_fwhm(1.0), 2.3548200450, rtol=1e-6)
        # the old linear form would give 4.0 here
        npt.assert_allclose(T_to_fwhm(2.0), sigma_to_fwhm(1.0))
