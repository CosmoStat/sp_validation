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

    def test_never_fit_objects_are_rejected(self):
        """Test that NGMIX_N_EPOCH == 0 objects are cut.

        ShapePipe's make_cat pre-fills the NGMIX_* columns with
        sentinels (G1/G2 = -10, T/FLUX = 0) and overwrites them only
        for objects present in the ngmix output. An object ngmix never
        fit therefore keeps NGMIX_MCAL_FLAGS == 0 and passes a
        flag-only selection, carrying e1 = -10 into the shear
        statistics. In final_cat_smk-g7.hdf5 that is 18,983 of
        1,851,100 objects (1.03%).
        """
        import numpy as np

        from sp_validation.galaxy import classification_galaxy_ngmix

        # row 0: never fit (sentinels, flags clean); row 1: a good fit
        dd = {
            "NGMIX_N_EPOCH": np.array([0.0, 3.0]),
            "NGMIX_MCAL_FLAGS": np.array([0.0, 0.0]),
            "NGMIX_G1_PSF_ORIG_NOSHEAR": np.array([-10.0, 0.02]),
            "NGMIX_MCAL_TYPES_FAIL": np.array([0.0, 0.0]),
            "NGMIX_G1_NOSHEAR": np.array([-10.0, 0.1]),
        }
        cut_common = np.array([True, True])

        keep = classification_galaxy_ngmix(dd, cut_common)
        npt.assert_array_equal(keep, [False, True])

        # the N_EPOCH cut must stand on its own: even if the -10
        # sentinel were to change, the never-fit row stays rejected
        dd["NGMIX_G1_PSF_ORIG_NOSHEAR"] = np.array([-99.0, 0.02])
        keep = classification_galaxy_ngmix(dd, cut_common)
        npt.assert_array_equal(keep, [False, True])
