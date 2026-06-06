"""UNIT TESTS FOR COSMOLOGY VALIDATION CLASS.

This module contains integration tests for the CosmologyValidation class,
specifically testing the ellipticity_suffix parameter functionality for
handling leak-corrected ellipticity columns.

:Author: cdaley

"""

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, Tuple

import pytest
import yaml

from sp_validation.cosmo_val import CosmologyValidation

# These tests load real UNIONS catalogues from the cluster filesystem. Skip them
# when that data isn't mounted (e.g. in CI / off-cluster) so the suite still runs
# its environment-independent unit tests; on a cluster node they run as normal.
requires_catalog_data = pytest.mark.skipif(
    not Path("/n17data").exists(),
    reason="UNIONS catalog data (/n17data) not mounted — running off-cluster",
)


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

    @staticmethod
    def _make_seed_config(tmp_path, shear_filename):
        """Create a minimal catalog config for seed variant testing."""
        base_version = "TestCatalog"
        base_dir = tmp_path / "catalog"
        base_dir.mkdir()
        (base_dir / shear_filename).touch()

        star_filename = "star_seed_1234.fits"
        (base_dir / star_filename).touch()

        nz_dir = tmp_path / "nz"
        nz_dir.mkdir()
        (nz_dir / "dndz.txt").write_text("0.1 1.0\n")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config_path = tmp_path / "seed_config.yaml"
        config_data = {
            "nz": {
                "subdir": str(nz_dir),
                "dndz": {"blind": "A", "path": "dndz.txt"},
            },
            "paths": {"output": str(output_dir)},
            base_version: {
                "subdir": str(base_dir),
                "pipeline": "SP",
                "shear": {
                    "path": shear_filename,
                    "w_col": "w",
                    "e1_col": "e1",
                    "e2_col": "e2",
                    "e1_col_corrected": "e1_corr",
                    "e2_col_corrected": "e2_corr",
                },
                "star": {"path": star_filename},
            },
        }
        config_path.write_text(yaml.dump(config_data, sort_keys=False))

        params = {
            "catalog_config": str(config_path),
            "output_dir": str(output_dir),
            "npatch": 1,
            "theta_min": 1.0,
            "theta_max": 250.0,
            "nbins": 20,
        }
        return params, base_version

    @pytest.mark.slow
    @requires_catalog_data
    def test_additive_bias_base_columns(self, base_config):
        """Test additive bias calculation using base ellipticity columns.

        This test initializes CosmologyValidation without an ellipticity_suffix,
        which means it will use the default columns defined in the catalog
        configuration. Tests SP_v1.4.5 with full additive bias computation.
        """
        version = "SP_v1.4.5"
        e1_col = "e1"
        e2_col = "e2"

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

    @pytest.mark.slow
    @requires_catalog_data
    def test_additive_bias_leak_corrected_columns(self, base_config):
        """Test additive bias calculation using leak-corrected columns.

        This test requests a leak-corrected version by passing "SP_v1.4.6_leak_corr"
        as the version name. The CosmologyValidation class automatically detects
        the _leak_corr suffix and creates the config entry using e1_col_corrected
        and e2_col_corrected from the base version.
        """
        base_version = "SP_v1.4.6"
        version_leak_corr = f"{base_version}_leak_corr"

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
        assert cv.cc[base_version]["shear"]["e1_col"] == "e1"
        assert cv.cc[base_version]["shear"]["e2_col"] == "e2"

        # Calculate additive bias
        cv.calculate_additive_bias()

        # Verify c1 and c2 were calculated and stored with leak-corrected name
        assert hasattr(cv, "_c1") and hasattr(cv, "_c2")
        assert version_leak_corr in cv.c1
        assert version_leak_corr in cv.c2

        # Verify the values are numeric
        assert isinstance(cv.c1[version_leak_corr], float)
        assert isinstance(cv.c2[version_leak_corr], float)

    @staticmethod
    def _iter_catalog_entries(config: Dict[str, Dict]) -> Iterator[Tuple[str, Dict]]:
        """Yield (name, entry) pairs for catalog-like entries in the config."""
        for name, entry in config.items():
            if not isinstance(entry, dict):
                continue
            if "subdir" not in entry:
                continue
            yield name, entry

    @staticmethod
    def _resolve(base: Path, candidate: str) -> Path:
        """Return an absolute path given a base directory and a candidate string."""
        candidate_path = Path(candidate)
        return candidate_path if candidate_path.is_absolute() else base / candidate_path

    @pytest.mark.slow
    @requires_catalog_data
    def test_catalog_paths_exist(self, base_config):
        """Verify that catalog paths for active versions exist on disk.

        This is a lightweight test that checks that all files referenced in the
        catalog configuration for UNIONS analysis versions actually exist. It
        discovers versions programmatically from cat_config.yaml rather than
        using hardcoded lists.
        """
        # Get the path to catalog config
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        catalog_config_path = os.path.join(
            repo_root, "notebooks", "cosmo_val", "cat_config.yaml"
        )

        config = yaml.safe_load(Path(catalog_config_path).read_text())

        # This integrity check needs the real catalogs on disk (cluster only).
        # Skip where the data directories aren't mounted — e.g. CI running
        # inside the docker image, which has cat_config.yaml but no catalogs.
        if not any(
            Path(entry["subdir"]).is_dir()
            for _, entry in self._iter_catalog_entries(config)
        ):
            pytest.skip("catalog data directories not present (not on cluster)")

        working = []
        nonfunctional = defaultdict(set)

        for version, entry in self._iter_catalog_entries(config):
            # Skip nz entries and versions already tested in heavy tests
            if version == "nz":
                continue

            base = Path(entry["subdir"])
            version_missing = set()

            # Check shear, star, and psf files
            for block_name in ("shear", "star", "psf"):
                block = entry.get(block_name)
                if not block:
                    continue
                resolved_path = self._resolve(base, block["path"])
                if not resolved_path.is_file():
                    version_missing.add(block_name)

            if version_missing:
                nonfunctional[version] = version_missing
            else:
                working.append(version)

        # Print summary
        print(f"\n✓ Working versions ({len(working)}):")
        for v in sorted(working):
            print(f"  - {v}")

        if nonfunctional:
            print(f"\n✗ Non-functional versions ({len(nonfunctional)}):")
            for v in sorted(nonfunctional.keys()):
                print(f"  - {v}: missing {nonfunctional[v]}")

        assert not nonfunctional, (
            "Catalog configuration references missing files: "
            f"{dict(nonfunctional)}"
        )

    def test_seed_variant_updates_shear_path(self, tmp_path):
        """Seeded versions should materialize a seed-specific shear path."""
        params, base_version = self._make_seed_config(
            tmp_path, shear_filename="shear_seed_1234.fits"
        )
        seed_version = f"{base_version}_seed007"

        cv = CosmologyValidation(versions=[seed_version], **params)

        assert cv.versions == [seed_version]
        assert seed_version in cv.cc
        assert cv.cc[seed_version]["shear"]["path"].endswith("shear_seed_007.fits")

    def test_seed_leak_corr_materializes_seed_first(self, tmp_path):
        """_seed<N>_leak_corr should clone the seed variant before leak fixes."""
        params, base_version = self._make_seed_config(
            tmp_path, shear_filename="shear_seed_1234.fits"
        )
        leak_version = f"{base_version}_seed007_leak_corr"
        seed_version = f"{base_version}_seed007"

        cv = CosmologyValidation(versions=[leak_version], **params)

        assert cv.versions == [leak_version]
        assert seed_version in cv.cc
        assert cv.cc[seed_version]["shear"]["path"].endswith("shear_seed_007.fits")
        assert cv.cc[leak_version]["shear"]["e1_col"] == "e1_corr"
        assert cv.cc[leak_version]["shear"]["e2_col"] == "e2_corr"

    def test_seed_variant_without_token_errors(self, tmp_path):
        """Missing seed token in shear path should raise a descriptive error."""
        params, base_version = self._make_seed_config(
            tmp_path, shear_filename="shear_base.fits"
        )
        seed_version = f"{base_version}_seed123"

        with pytest.raises(ValueError, match="seed"):
            CosmologyValidation(versions=[seed_version], **params)

    def test_v1_4_6_glass_mock_seed_variant(self, base_config):
        """Test that v1.4.6 glass mock seed variant loads with correct path."""
        seed = 9
        seed_version = f"SP_v1.4.6_glass_mock_seed{seed}"

        cv = CosmologyValidation(
            versions=[seed_version],
            **base_config,
        )

        # Verify version was created
        assert cv.versions == [seed_version]
        assert seed_version in cv.cc

        # Verify seed was substituted in shear path
        expected_filename = f"unions_glass_sim_{seed:05d}_4096.fits"
        assert expected_filename in cv.cc[seed_version]["shear"]["path"]

        # Verify path points to v1.4.6 glass mock directory
        assert "glass_mock_v1.4.6" in cv.cc[seed_version]["shear"]["path"]

    def test_v1_4_6_glass_mock_default_seed(self, base_config):
        """Test that glass mock without seed suffix uses the default seed_00001."""
        cv = CosmologyValidation(
            versions=["SP_v1.4.6_glass_mock"],
            **base_config,
        )

        # Verify version loads without seed suffix
        assert cv.versions == ["SP_v1.4.6_glass_mock"]
        assert "SP_v1.4.6_glass_mock" in cv.cc

        # Verify it uses the default path (seed_00001 for v1.4.6)
        path = cv.cc["SP_v1.4.6_glass_mock"]["shear"]["path"]
        assert "unions_glass_sim_00001_4096.fits" in path
        assert "glass_mock_v1.4.6" in path
