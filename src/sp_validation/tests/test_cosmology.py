"""UNIT TESTS FOR COSMOLOGY SUBPACKAGE.

This module contains unit tests for the module package
sp_validation.cosmology.

:Author: Claude Code Assistant

"""

import numpy as np
import numpy.testing as npt
import pyccl as ccl
import pytest

from sp_validation import cosmology

# Test performance markers
pytestmark = pytest.mark.fast  # Mark all tests as fast by default

# Standard test cosmology used across all tests
TEST_COSMOLOGY = {
    "Omega_m": 0.3,
    "Omega_b": 0.05,
    "h": 0.7,
    "sig8": 0.8,
    "ns": 0.96,
    "w0": -1.0,
    "wa": 0.0,
}

# Derived CAMB format from TEST_COSMOLOGY
TEST_CAMB = {
    "H0": TEST_COSMOLOGY["h"] * 100,  # 70.0
    "ombh2": TEST_COSMOLOGY["Omega_b"] * TEST_COSMOLOGY["h"] ** 2,
    "omch2": (
        (TEST_COSMOLOGY["Omega_m"] - TEST_COSMOLOGY["Omega_b"])
        * TEST_COSMOLOGY["h"] ** 2
    ),
    "ns": TEST_COSMOLOGY["ns"],  # 0.96
    "sigma8": TEST_COSMOLOGY["sig8"],  # 0.8
    "w": TEST_COSMOLOGY["w0"],  # -1.0
    "wa": TEST_COSMOLOGY["wa"],  # 0.0
}

# Derived CosmoCov format from TEST_COSMOLOGY
TEST_COSMOCOV = {
    "Omega_m": TEST_COSMOLOGY["Omega_m"],  # 0.3
    "omb": TEST_COSMOLOGY["Omega_b"],  # 0.05
    "h0": TEST_COSMOLOGY["h"],  # 0.7
    "sigma_8": TEST_COSMOLOGY["sig8"],  # 0.8
    "n_spec": TEST_COSMOLOGY["ns"],  # 0.96
    "w0": TEST_COSMOLOGY["w0"],  # -1.0
    "wa": TEST_COSMOLOGY["wa"],  # 0.0
}

# Fast test parameters (reduced precision for speed)
FAST_ELL_MAX = 2000
FAST_N_ELL = 100
FAST_Z_POINTS = 20

# Realistic test parameters (full precision for one integration test)
REALISTIC_ELL_MAX = 10_000
REALISTIC_N_ELL = 500


class TestCosmologyHelpers:
    """Test helper functions for parameter conversion."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        # Use standard test cosmology definitions
        self.camb_params = {
            k: v
            for k, v in TEST_CAMB.items()
            if k in ["H0", "ombh2", "omch2", "ns", "sigma8"]
        }
        self.camb_params_full = TEST_CAMB.copy()
        self.cosmocov_params = {
            k: v
            for k, v in TEST_COSMOCOV.items()
            if k in ["Omega_m", "omb", "h0", "sigma_8", "n_spec"]
        }
        self.cosmocov_params_full = TEST_COSMOCOV.copy()

    def test_ccl_to_camb_basic(self):
        """Test CCL to CAMB parameter conversion."""
        # Create a test cosmology using TEST_COSMOLOGY values
        omega_c = TEST_COSMOLOGY["Omega_m"] - TEST_COSMOLOGY["Omega_b"]
        cosmo = ccl.Cosmology(
            Omega_c=omega_c,
            Omega_b=TEST_COSMOLOGY["Omega_b"],
            h=TEST_COSMOLOGY["h"],
            sigma8=TEST_COSMOLOGY["sig8"],
            n_s=TEST_COSMOLOGY["ns"],
        )

        # Convert to CAMB parameters
        camb_params = cosmology._ccl_to_camb(cosmo)

        # Check basic parameters against expected TEST_CAMB values
        npt.assert_almost_equal(camb_params["H0"], TEST_CAMB["H0"], decimal=2)
        npt.assert_almost_equal(camb_params["ombh2"], TEST_CAMB["ombh2"], decimal=3)
        npt.assert_almost_equal(camb_params["omch2"], TEST_CAMB["omch2"], decimal=3)
        npt.assert_almost_equal(camb_params["ns"], TEST_CAMB["ns"], decimal=4)
        # Check that As is present and positive (sigma8 converted to As)
        assert "As" in camb_params
        assert camb_params["As"] > 0

    def test_camb_to_ccl_basic(self):
        """Test CAMB to CCL parameter conversion."""
        ccl_params = cosmology._camb_to_ccl(self.camb_params)

        # Check conversion against expected TEST_COSMOLOGY values
        assert isinstance(ccl_params, dict)
        npt.assert_almost_equal(
            ccl_params["Omega_b"], TEST_COSMOLOGY["Omega_b"], decimal=3
        )
        npt.assert_almost_equal(
            ccl_params["Omega_c"],
            TEST_COSMOLOGY["Omega_m"] - TEST_COSMOLOGY["Omega_b"],
            decimal=3,
        )
        npt.assert_almost_equal(ccl_params["h"], TEST_COSMOLOGY["h"], decimal=4)
        npt.assert_almost_equal(ccl_params["n_s"], TEST_COSMOLOGY["ns"], decimal=4)
        npt.assert_almost_equal(ccl_params["sigma8"], TEST_COSMOLOGY["sig8"], decimal=4)

    def test_camb_to_ccl_with_optional_params(self):
        """Test CAMB to CCL conversion with optional parameters."""
        ccl_params = cosmology._camb_to_ccl(self.camb_params_full)

        # Check optional parameters against TEST_CAMB values
        npt.assert_almost_equal(ccl_params["w0"], TEST_CAMB["w"], decimal=4)
        npt.assert_almost_equal(ccl_params["wa"], TEST_CAMB["wa"], decimal=4)

    def test_camb_to_ccl_missing_normalization(self):
        """Test error when neither As nor sigma8 provided."""
        bad_params = {
            "H0": TEST_CAMB["H0"],
            "ombh2": TEST_CAMB["ombh2"],
            "omch2": TEST_CAMB["omch2"],
            "ns": TEST_CAMB["ns"],
        }

        with pytest.raises(ValueError):
            cosmology._camb_to_ccl(bad_params)

    def test_cosmocov_to_ccl_basic(self):
        """Test CosmoCov to CCL parameter conversion."""
        ccl_params = cosmology._cosmocov_to_ccl(self.cosmocov_params)

        # Check conversion against TEST_COSMOCOV values
        assert isinstance(ccl_params, dict)
        expected_omega_c = TEST_COSMOCOV["Omega_m"] - TEST_COSMOCOV["omb"]
        npt.assert_almost_equal(ccl_params["Omega_c"], expected_omega_c, decimal=4)
        npt.assert_almost_equal(ccl_params["Omega_b"], TEST_COSMOCOV["omb"], decimal=4)
        npt.assert_almost_equal(ccl_params["h"], TEST_COSMOCOV["h0"], decimal=4)
        npt.assert_almost_equal(
            ccl_params["sigma8"], TEST_COSMOCOV["sigma_8"], decimal=4
        )
        npt.assert_almost_equal(ccl_params["n_s"], TEST_COSMOCOV["n_spec"], decimal=4)

    def test_cosmocov_to_ccl_with_optional_params(self):
        """Test CosmoCov to CCL conversion with optional parameters."""
        ccl_params = cosmology._cosmocov_to_ccl(self.cosmocov_params_full)

        # Check optional parameters against TEST_COSMOCOV values
        npt.assert_almost_equal(ccl_params["w0"], TEST_COSMOCOV["w0"], decimal=4)
        npt.assert_almost_equal(ccl_params["wa"], TEST_COSMOCOV["wa"], decimal=4)

    def test_cosmocov_to_ccl_missing_params(self):
        """Test error when required CosmoCov parameters are missing."""
        incomplete_params = {
            "Omega_m": TEST_COSMOCOV["Omega_m"],
            "omb": TEST_COSMOCOV["omb"],
            # Missing h0, sigma_8, n_spec
        }

        with pytest.raises(KeyError):
            cosmology._cosmocov_to_ccl(incomplete_params)


class TestGetCosmo:
    """Test get_cosmo function."""

    def test_default_cosmology(self):
        """Test that default parameters are Planck18 (not our test values)."""
        cosmo = cosmology.get_cosmo()

        # Check that we get a CCL cosmology object
        assert isinstance(cosmo, ccl.Cosmology)

        # Check that defaults are NOT our test values (should be Planck18)
        assert cosmo["h"] != TEST_COSMOLOGY["h"]
        assert cosmo["Omega_b"] != TEST_COSMOLOGY["Omega_b"]

    def test_individual_parameters(self):
        """Test overriding individual parameters."""
        cosmo = cosmology.get_cosmo(h=TEST_COSMOLOGY["h"], sig8=TEST_COSMOLOGY["sig8"])

        # Check custom parameters
        assert cosmo["h"] == pytest.approx(TEST_COSMOLOGY["h"], abs=1e-4)
        assert cosmo["sigma8"] == pytest.approx(TEST_COSMOLOGY["sig8"], abs=1e-4)

        # Check other parameters use defaults (not our test values)
        assert cosmo["Omega_b"] != TEST_COSMOLOGY["Omega_b"]

    def test_cosmocov_params(self):
        """Test using CosmoCov parameter format."""
        cosmo = cosmology.get_cosmo(cosmocov_params=TEST_COSMOCOV)

        # Check parameters against TEST_COSMOCOV constants
        assert cosmo["Omega_b"] == pytest.approx(TEST_COSMOCOV["omb"], abs=1e-4)
        assert cosmo["h"] == pytest.approx(TEST_COSMOCOV["h0"], abs=1e-4)
        assert cosmo["sigma8"] == pytest.approx(TEST_COSMOCOV["sigma_8"], abs=1e-4)
        assert cosmo["n_s"] == pytest.approx(TEST_COSMOCOV["n_spec"], abs=1e-4)

    def test_camb_params(self):
        """Test using CAMB parameter format."""
        cosmo = cosmology.get_cosmo(camb_params=TEST_CAMB)

        # Check parameters against TEST_CAMB constants
        expected_h = TEST_CAMB["H0"] / 100.0
        expected_omega_b = TEST_CAMB["ombh2"] / expected_h**2

        assert cosmo["h"] == pytest.approx(expected_h, abs=1e-4)
        assert cosmo["Omega_b"] == pytest.approx(expected_omega_b, abs=1e-4)
        assert cosmo["sigma8"] == pytest.approx(TEST_CAMB["sigma8"], abs=1e-4)

    def test_mutually_exclusive_params(self):
        """Test that cosmocov_params and camb_params are mutually exclusive."""
        with pytest.raises(ValueError):
            cosmology.get_cosmo(cosmocov_params=TEST_COSMOCOV, camb_params=TEST_CAMB)

    def test_transfer_function_options(self):
        """Test different transfer function options."""
        cosmo1 = cosmology.get_cosmo(transfer_function="boltzmann_camb")
        cosmo2 = cosmology.get_cosmo(transfer_function="boltzmann_class")

        # Both should create valid cosmologies
        assert isinstance(cosmo1, ccl.Cosmology)
        assert isinstance(cosmo2, ccl.Cosmology)


# Shared test fixtures to avoid recomputation
@pytest.fixture
def fast_redshift_data():
    """Fast redshift distribution for most tests."""
    z = np.linspace(0.1, 2.0, FAST_Z_POINTS)
    nz = np.exp(-(((z - 0.8) / 0.2) ** 2))  # Gaussian around z=0.8
    nz /= np.trapezoid(nz, z)  # Normalize
    return z, nz


@pytest.fixture
def fast_ell_array():
    """Fast ell array for most tests."""
    return np.array([10, 100, 1000])  # Reduced from [10, 50, 100, 500, 1000]


@pytest.fixture
def realistic_redshift_data():
    """Realistic redshift distribution for integration tests."""
    z = np.linspace(0.1, 2.0, 50)
    nz = np.exp(-(((z - 0.8) / 0.2) ** 2))
    nz /= np.trapezoid(nz, z)
    return z, nz


class TestGetTheoCell:
    """Test get_theo_c_ell function."""

    def test_ccl_backend_basic(self, fast_ell_array, fast_redshift_data):
        """Test C_ell calculation with CCL backend - basic functionality."""
        z, nz = fast_redshift_data
        cosmo = cosmology.get_cosmo()

        cl = cosmology.get_theo_c_ell(fast_ell_array, z, nz, backend="ccl", cosmo=cosmo)

        # Check output shape and positivity
        assert len(cl) == len(fast_ell_array)
        assert np.all(cl > 0)

    def test_ccl_backend_with_params(self, fast_ell_array, fast_redshift_data):
        """Test C_ell calculation with CCL backend using individual parameters."""
        z, nz = fast_redshift_data
        cl = cosmology.get_theo_c_ell(
            fast_ell_array, z, nz, backend="ccl", **TEST_COSMOLOGY
        )

        # Check output shape and positivity
        assert len(cl) == len(fast_ell_array)
        assert np.all(cl > 0)

    def test_ccl_backend_default_params(self, fast_ell_array, fast_redshift_data):
        """Test C_ell calculation with CCL backend using default parameters."""
        z, nz = fast_redshift_data
        cl = cosmology.get_theo_c_ell(fast_ell_array, z, nz, backend="ccl")

        # Check output shape and positivity
        assert len(cl) == len(fast_ell_array)
        assert np.all(cl > 0)

    def test_camb_backend(self, fast_ell_array, fast_redshift_data):
        """Test C_ell calculation with CAMB backend."""
        z, nz = fast_redshift_data
        try:
            cl = cosmology.get_theo_c_ell(fast_ell_array, z, nz, backend="camb")

            # Check output shape and positivity
            assert len(cl) == len(fast_ell_array)
            assert np.all(cl > 0)
        except ImportError:
            pytest.skip("CAMB not available")

    def test_invalid_backend(self, fast_ell_array, fast_redshift_data):
        """Test error with invalid backend."""
        z, nz = fast_redshift_data
        with pytest.raises(ValueError):
            cosmology.get_theo_c_ell(fast_ell_array, z, nz, backend="invalid")

    def test_small_ell_array(self, fast_redshift_data):
        """Test behavior with small ell array."""
        z, nz = fast_redshift_data
        small_ell = np.array([10, 100])
        cl = cosmology.get_theo_c_ell(small_ell, z, nz, backend="ccl")

        assert len(cl) == 2
        assert np.all(cl > 0)

    @pytest.mark.slow
    def test_backend_consistency_fast(self, fast_ell_array, fast_redshift_data):
        """Test that CCL and CAMB backends give consistent results (fast version)."""
        z, nz = fast_redshift_data
        try:
            # Calculate with both backends using fast parameters
            cl_ccl = cosmology.get_theo_c_ell(fast_ell_array, z, nz, backend="ccl")
            cl_camb = cosmology.get_theo_c_ell(fast_ell_array, z, nz, backend="camb")

            # Check that both have same shape
            assert len(cl_ccl) == len(cl_camb) == len(fast_ell_array)

            # Check relative differences between cosmology codes
            # CCL and CAMB should agree well since they compute the same physics
            # Observed differences: ~4.8% max, ~2.8% average
            rel_diff = np.abs(cl_camb - cl_ccl) / cl_ccl
            max_rel_diff = rel_diff.max()

            assert (
                max_rel_diff < 0.08
            ), f"CCL and CAMB backends differ by >8%: {max_rel_diff:.1%}"
        except ImportError:
            pytest.skip("CAMB not available")


@pytest.fixture
def fast_theta_array():
    """Fast theta array for most tests."""
    return np.array([5.0, 20.0, 100.0])  # Reduced from logspace(0, 2, 10)


class TestGetTheoXi:
    """Test get_theo_xi function."""

    def test_backwards_compatibility(self, fast_theta_array, fast_redshift_data):
        """Test backwards compatible parameter interface with fast parameters."""
        z, nz = fast_redshift_data
        xip, xim = cosmology.get_theo_xi(
            fast_theta_array,
            z,
            nz,
            **TEST_COSMOLOGY,
            ell_min=10,
            ell_max=FAST_ELL_MAX,
            n_ell=FAST_N_ELL,
            backend="ccl",
        )

        # Check output shapes
        assert len(xip) == len(fast_theta_array)
        assert len(xim) == len(fast_theta_array)

        # Check that xi+ is positive for typical scales
        assert np.any(xip > 0)

    def test_default_parameters(self, fast_theta_array, fast_redshift_data):
        """Test with default cosmological parameters using fast settings."""
        z, nz = fast_redshift_data
        xip, xim = cosmology.get_theo_xi(
            fast_theta_array, z, nz, ell_min=10, ell_max=FAST_ELL_MAX, n_ell=FAST_N_ELL
        )

        # Check output shapes
        assert len(xip) == len(fast_theta_array)
        assert len(xim) == len(fast_theta_array)

    @pytest.mark.slow
    def test_different_ell_ranges(self, fast_theta_array, fast_redshift_data):
        """Test that different ell ranges give consistent results."""
        z, nz = fast_redshift_data

        # Coarse integration (lower ell_max)
        xip1, xim1 = cosmology.get_theo_xi(
            fast_theta_array, z, nz, ell_min=10, ell_max=1000, n_ell=50
        )

        # Fine integration (higher ell_max)
        xip2, xim2 = cosmology.get_theo_xi(
            fast_theta_array, z, nz, ell_min=10, ell_max=FAST_ELL_MAX, n_ell=FAST_N_ELL
        )

        # Scale-dependent tolerance:
        # Xi+ converges well at all scales (~4% max differences)
        # Xi- has poor convergence at small scales due to oscillatory Hankel
        # transforms and sensitivity to high-ell power (up to ~15% at θ<10 arcmin)
        # but excellent convergence at large scales (~0.2% at θ>20 arcmin)

        small_scale_mask = fast_theta_array <= 10.0  # arcmin
        large_scale_mask = fast_theta_array > 10.0  # arcmin

        # Xi+ has good convergence at all scales
        npt.assert_allclose(xip1, xip2, rtol=0.06)  # 6% tolerance for xi+

        # Xi- has scale-dependent convergence
        if np.any(small_scale_mask):
            # Small scales (θ ≤ 10 arcmin): physics-limited precision
            npt.assert_allclose(
                xim1[small_scale_mask], xim2[small_scale_mask], rtol=0.18
            )  # 18% tolerance for xi- at small scales

        if np.any(large_scale_mask):
            # Large scales (θ > 10 arcmin): excellent convergence
            npt.assert_allclose(
                xim1[large_scale_mask], xim2[large_scale_mask], rtol=0.05
            )  # 5% tolerance for xi- at large scales


class TestCosmologyIntegration:
    """Integration tests for cosmology functions."""

    def test_end_to_end_pipeline_fast(self, fast_redshift_data):
        """Test complete pipeline from parameters to xi (fast version)."""
        z, nz = fast_redshift_data
        # Create cosmology using TEST_CAMB parameters
        camb_subset = {
            k: v
            for k, v in TEST_CAMB.items()
            if k in ["H0", "ombh2", "omch2", "ns", "sigma8"]
        }
        cosmo = cosmology.get_cosmo(camb_params=camb_subset)

        # Calculate C_ell with fast parameters
        ell = np.logspace(1, 3, 15)  # Reduced from 30 points
        cl = cosmology.get_theo_c_ell(ell, z, nz, cosmo=cosmo)

        # Calculate xi
        theta = np.array([5.0, 10.0, 20.0])
        xip, xim = cosmology.c_ell_to_xi(cosmo, theta, ell, cl)

        # Sanity checks
        assert len(cl) == len(ell)
        assert len(xip) == len(theta)
        assert len(xim) == len(theta)
        assert np.all(cl > 0)

    @pytest.mark.slow
    def test_realistic_precision_pipeline(self, realistic_redshift_data):
        """Test complete pipeline with realistic precision for production use."""
        z, nz = realistic_redshift_data

        # Create cosmology with realistic parameters
        cosmo = cosmology.get_cosmo(**TEST_COSMOLOGY)

        # Calculate C_ell with realistic precision
        ell = np.logspace(1, np.log10(REALISTIC_ELL_MAX), REALISTIC_N_ELL)
        cl = cosmology.get_theo_c_ell(ell, z, nz, cosmo=cosmo)

        # Test xi calculation with realistic theta range
        theta = np.geomspace(1, 300, 20)  # 1 to 300 arcmin, 20 points
        xip, xim = cosmology.get_theo_xi(
            theta,
            z,
            nz,
            ell_min=10,
            ell_max=REALISTIC_ELL_MAX,
            n_ell=REALISTIC_N_ELL,
            **TEST_COSMOLOGY,
        )

        # Comprehensive checks for realistic test
        assert len(cl) == len(ell)
        assert len(xip) == len(theta)
        assert len(xim) == len(theta)
        assert np.all(cl > 0)

        # Check that xi+ is positive at small scales (expected for cosmic shear)
        small_scale_mask = theta < 10  # arcmin
        assert np.any(xip[small_scale_mask] > 0)

        # Check that xi+ amplitude is reasonable (order of magnitude check)
        max_xip = np.max(np.abs(xip))
        assert 1e-6 < max_xip < 1e-2, f"xi+ amplitude unrealistic: {max_xip}"

    def test_parameter_format_consistency(self):
        """Test that different parameter formats give consistent results."""
        # Individual parameters
        cosmo1 = cosmology.get_cosmo(**TEST_COSMOLOGY)

        # CosmoCov format
        cosmo2 = cosmology.get_cosmo(cosmocov_params=TEST_COSMOCOV)

        # CAMB format
        cosmo3 = cosmology.get_cosmo(camb_params=TEST_CAMB)

        # All should give consistent results
        npt.assert_allclose(cosmo1["Omega_b"], cosmo2["Omega_b"], rtol=1e-4)
        npt.assert_allclose(cosmo1["Omega_b"], cosmo3["Omega_b"], rtol=1e-4)
        npt.assert_allclose(cosmo1["h"], cosmo2["h"], rtol=1e-4)
        npt.assert_allclose(cosmo1["h"], cosmo3["h"], rtol=1e-4)
