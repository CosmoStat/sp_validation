"""UNIT TESTS FOR COSMOLOGY VALIDATION CLASS.

This module contains integration tests for the CosmologyValidation class,
specifically testing the ellipticity_suffix parameter functionality for
handling leak-corrected ellipticity columns.

:Author: cdaley

"""

import os

import pytest

from sp_validation.cosmo_val import CosmologyValidation


class TestCosmologyValidation:
    """Test CosmologyValidation initialization and additive bias calculation."""

    @pytest.fixture
    def base_config(self, tmp_path):
        """Common configuration parameters for tests."""
        # Get the path to the repo root (3 levels up from this test file)
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        catalog_config = os.path.join(
            repo_root, "notebooks", "cosmo_val", "cat_config.yaml"
        )

        # Use temporary directory for outputs
        output_dir = tmp_path / "test_output"
        output_dir.mkdir()

        return {
            "catalog_config": catalog_config,
            "output_dir": str(output_dir),
            "npatch": 1,
            "theta_min": 1.0,
            "theta_max": 250.0,
            "nbins": 20,
        }

    @pytest.mark.parametrize(
        "version,e1_col,e2_col",
        [
            ("SP_v1.4.5", "e1", "e2"),
            ("SP_v1.4.6", "e1", "e2"),
            ("SP_v1.4.5_glass_mock", "e1", "e2"),
            ("SP_v1.4.5_bright", "e1", "e2"),
            ("SP_v1.4.5_faint", "e1", "e2"),
            ("SP_v1.4.5_intermediate", "e1", "e2"),
            ("SP_v1.4.5.A", "g1", "g2"),
            ("SP_v1.4.7", "e1", "e2"),
            ("SP_v1.4.8", "e1", "e2"),
        ],
    )
    def test_additive_bias_base_columns(self, base_config, version, e1_col, e2_col):
        """Test additive bias calculation using base ellipticity columns.

        This test initializes CosmologyValidation without an ellipticity_suffix,
        which means it will use the default columns defined in the catalog
        configuration.
        """
        cv = CosmologyValidation(
            versions=[version],
            **base_config,
        )

        # Verify version names remain unchanged
        assert cv.versions == [version]

        # Verify the ellipticity columns are the base columns
        assert cv.cc[version]["shear"]["e1_col"] == e1_col
        assert cv.cc[version]["shear"]["e2_col"] == e2_col

        # Calculate additive bias
        cv.calculate_additive_bias()

        # Verify c1 and c2 were calculated and stored
        assert hasattr(cv, "_c1") and hasattr(cv, "_c2")
        assert version in cv.c1
        assert version in cv.c2

        # Verify the values are numeric (not NaN or None)
        assert isinstance(cv.c1[version], float)
        assert isinstance(cv.c2[version], float)

    @pytest.mark.parametrize("version", ["SP_v1.4.5"])
    def test_additive_bias_leak_corrected_columns(self, base_config, version):
        """Test additive bias calculation using leak-corrected columns.

        This test requests a leak-corrected version by passing "{version}_leak_corr"
        as the version name. The CosmologyValidation class automatically detects
        the _leak_corr suffix and creates the config entry using e1_col_corrected
        and e2_col_corrected from the base version.
        """
        version_leak_corr = f"{version}_leak_corr"

        cv = CosmologyValidation(
            versions=[version_leak_corr],
            **base_config,
        )

        # Verify version names include the _leak_corr suffix
        assert cv.versions == [version_leak_corr]

        # Verify the leak-corrected config was auto-created with corrected columns
        assert cv.cc[version_leak_corr]["shear"]["e1_col"] == "e1_leak_corrected"
        assert cv.cc[version_leak_corr]["shear"]["e2_col"] == "e2_leak_corrected"

        # Verify original config entry remains unchanged
        assert cv.cc[version]["shear"]["e1_col"] == "e1"
        assert cv.cc[version]["shear"]["e2_col"] == "e2"

        # Calculate additive bias
        cv.calculate_additive_bias()

        # Verify c1 and c2 were calculated and stored with leak-corrected name
        assert hasattr(cv, "_c1") and hasattr(cv, "_c2")
        assert version_leak_corr in cv.c1
        assert version_leak_corr in cv.c2

        # Verify the values are numeric
        assert isinstance(cv.c1[version_leak_corr], float)
        assert isinstance(cv.c2[version_leak_corr], float)
