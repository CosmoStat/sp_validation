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

import numpy as np
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
        catalog_config = os.path.join(repo_root, "cosmo_val", "cat_config.yaml")

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
        catalog_config_path = os.path.join(repo_root, "cosmo_val", "cat_config.yaml")

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
            f"Catalog configuration references missing files: {dict(nonfunctional)}"
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

    # ------------------------------------------------------------------
    # Synthetic-catalog integration ("glue") tests
    #
    # These run the real compute seams end-to-end on a small, deterministic
    # toy catalog written to disk, asserting that sp_validation wires the
    # catalog/config/estimator together correctly and that the chain produces
    # output of the right shape with finite values. They do NOT re-test the
    # underlying numerical libraries (treecorr, cosmo_numba): no specific
    # numerical values are asserted. These are the back-pressure that catches
    # config-path / wiring breakage during restructuring. They can be tightened
    # to allclose-against-a-committed-reference later for value-drift coverage.
    #
    # Environment-independent: the catalog is synthesized in a tmp dir, so no
    # cluster data is needed. They do require the scientific stack (treecorr,
    # shear_psf_leakage, cosmo_numba), i.e. they run in the container.
    # ------------------------------------------------------------------

    @staticmethod
    def _write_synthetic_catalogs(
        tmp_path,
        n_gal=2000,
        n_star=800,
        ra_range=(10.0, 14.0),
        dec_range=(10.0, 14.0),
        seed=1234,
        coherent_shear=False,
        with_psf=False,
    ):
        """Write small deterministic FITS catalogs + dndz, return a config dict.

        Builds a synthetic shear catalog (RA/Dec/e1/e2/w), a PSF star catalog
        with the columns the leakage/rho-tau seams read, and a cs_util-readable
        dndz file. Returns ``(params, version)`` ready to hand to
        ``CosmologyValidation``.

        Parameters
        ----------
        coherent_shear : bool
            If True, inject a smooth position-dependent shear pattern on top of
            shape noise so that xi+/- is smooth (needed for the pure-E/B
            integral to be numerically well-posed).
        with_psf : bool
            If True, add a ``psf`` config block (rho/tau / pseudo-Cl read it via
            ``get_params_rho_tau``).
        """
        from astropy.table import Table

        rng = np.random.default_rng(seed)
        version = "TestCatalog"

        cat_dir = tmp_path / "catalog"
        cat_dir.mkdir()
        nz_dir = tmp_path / "nz"
        nz_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        ra = rng.uniform(*ra_range, n_gal)
        dec = rng.uniform(*dec_range, n_gal)
        if coherent_shear:
            # Smooth E-mode-like pattern so xi+/- is smooth, plus shape noise.
            e1 = 0.02 * np.cos(np.radians(ra) * 40) + rng.normal(0, 0.05, n_gal)
            e2 = 0.02 * np.sin(np.radians(dec) * 40) + rng.normal(0, 0.05, n_gal)
        else:
            e1 = rng.normal(0, 0.25, n_gal)
            e2 = rng.normal(0, 0.25, n_gal)
        w = rng.uniform(0.5, 1.0, n_gal)

        shear_path = cat_dir / "shear.fits"
        Table({"RA": ra, "Dec": dec, "e1": e1, "e2": e2, "w": w}).write(
            shear_path, overwrite=True
        )

        star_path = cat_dir / "star.fits"
        Table(
            {
                "RA": rng.uniform(*ra_range, n_star),
                "Dec": rng.uniform(*dec_range, n_star),
                "HSM_G1_PSF": rng.normal(0, 0.03, n_star),
                "HSM_G2_PSF": rng.normal(0, 0.03, n_star),
                "HSM_G1_STAR": rng.normal(0, 0.03, n_star),
                "HSM_G2_STAR": rng.normal(0, 0.03, n_star),
                "HSM_T_PSF": rng.uniform(0.4, 0.6, n_star),
                "HSM_T_STAR": rng.uniform(0.4, 0.6, n_star),
                "HSM_FLAG_PSF": np.zeros(n_star, dtype=int),
                "HSM_FLAG_STAR": np.zeros(n_star, dtype=int),
            }
        ).write(star_path, overwrite=True)

        # dndz in the commented-header format cs_util.read_dndz expects:
        # column "z" holds bin edges (n+1), "dn_dz" the densities.
        z_edges = np.linspace(0.05, 3.0, 31)
        dndz = np.exp(-(((z_edges - 0.7) / 0.3) ** 2))
        dndz_lines = ["# z dn_dz"] + [f"{zz} {nn}" for zz, nn in zip(z_edges, dndz)]
        (nz_dir / "dndz_SP_A.txt").write_text("\n".join(dndz_lines) + "\n")

        shear_cfg = {
            "path": "shear.fits",
            "w_col": "w",
            "e1_col": "e1",
            "e2_col": "e2",
            "R": 1.0,
            "e1_col_corrected": "e1",
            "e2_col_corrected": "e2",
        }
        star_cfg = {
            "path": "star.fits",
            "ra_col": "RA",
            "dec_col": "Dec",
            "e1_col": "HSM_G1_PSF",
            "e2_col": "HSM_G2_PSF",
        }
        version_cfg = {
            "subdir": str(cat_dir),
            "pipeline": "SP",
            "shear": shear_cfg,
            "star": star_cfg,
        }
        if with_psf:
            version_cfg["psf"] = {
                "path": "star.fits",
                "hdu": 1,
                "ra_col": "RA",
                "dec_col": "Dec",
                "e1_PSF_col": "HSM_G1_PSF",
                "e2_PSF_col": "HSM_G2_PSF",
                "e1_star_col": "HSM_G1_STAR",
                "e2_star_col": "HSM_G2_STAR",
                "PSF_size": "HSM_T_PSF",
                "star_size": "HSM_T_STAR",
                "PSF_flag": "HSM_FLAG_PSF",
                "star_flag": "HSM_FLAG_STAR",
            }

        config_data = {
            "nz": {
                "subdir": str(nz_dir),
                "dndz": {"blind": "A", "path": "dndz"},
            },
            "paths": {"output": str(output_dir)},
            version: version_cfg,
        }
        config_path = tmp_path / "synthetic_config.yaml"
        config_path.write_text(yaml.dump(config_data, sort_keys=False))

        params = {
            "catalog_config": str(config_path),
            "output_dir": str(output_dir),
        }
        return params, version

    def test_calculate_2pcf_runs_on_synthetic_catalog(self, tmp_path):
        """calculate_2pcf wires catalog+config into treecorr GGCorrelation.

        Smoke-integration: assert the xi+/- data vector is computed with the
        configured number of angular bins and is finite. Numerical values are
        deliberately not asserted (could be tightened to allclose vs. a
        committed reference later for value-drift coverage).
        """
        pytest.importorskip("treecorr")
        params, version = self._write_synthetic_catalogs(tmp_path)

        nbins = 8
        cv = CosmologyValidation(
            versions=[version],
            npatch=1,
            theta_min=5.0,
            theta_max=100.0,
            nbins=nbins,
            **params,
        )

        gg = cv.calculate_2pcf(version)

        # treecorr GGCorrelation with xi+/- on the configured angular grid
        assert gg.xip.shape == (nbins,)
        assert gg.xim.shape == (nbins,)
        assert np.all(np.isfinite(gg.xip))
        assert np.all(np.isfinite(gg.xim))
        # The additive-bias subtraction in the pipeline must have run.
        assert version in cv.c1 and version in cv.c2

    def test_calculate_scale_dependent_leakage_runs_on_synthetic_catalog(
        self, tmp_path
    ):
        """calculate_scale_dependent_leakage wires shear x PSF into treecorr.

        Smoke-integration: assert the galaxy-PSF and PSF-PSF correlations and
        the assembled alpha leakage / xi_sys are computed on the configured
        angular grid and are finite. No numerical values asserted (could be
        tightened to allclose vs. a committed reference later).
        """
        pytest.importorskip("treecorr")
        params, version = self._write_synthetic_catalogs(tmp_path)

        nbins = 8
        cv = CosmologyValidation(
            versions=[version],
            npatch=1,
            theta_min=5.0,
            theta_max=100.0,
            nbins=nbins,
            **params,
        )

        cv.calculate_scale_dependent_leakage()

        res = cv.results[version]
        # shear x PSF and PSF x PSF correlations on the configured grid
        assert res.r_corr_gp.xip.shape == (nbins,)
        assert res.r_corr_pp.xip.shape == (nbins,)
        assert np.all(np.isfinite(res.r_corr_gp.xip))
        # alpha leakage and the systematic xi_sys assembled from them
        assert np.shape(res.alpha_leak) == (nbins,)
        assert np.all(np.isfinite(res.alpha_leak))
        assert hasattr(res, "C_sys_p") and hasattr(res, "C_sys_m")

    def test_calculate_pure_eb_runs_on_synthetic_catalog(self, tmp_path):
        """calculate_pure_eb wires xi+/- into cosmo_numba's Schneider E/B split.

        Integration test of the headline B-mode seam: the pure E/B/amb
        decomposition runs end-to-end via cosmo_numba and returns vectors of the
        configured length, all bins finite, with a jackknife covariance of the
        right shape -- AND the deterministic mode vectors match pinned reference
        values, so a refactor that silently changes the numerical B-modes fails
        rather than staying green on finiteness alone.

        Two layers of teeth:

        1. Finiteness on EVERY reporting bin (not just the interior). The
           Schneider (2022) pure E/B estimator evaluates singular kernel
           integrals (Eq. 42-43, 55-56) at each reporting theta, integrating the
           fine ``gg_int`` xi+/- over [tmin, tmax]. At the extreme reporting bins
           the evaluation point sits at the integration boundary, where the
           integrand is near-singular; a *coarse* integration grid fails to
           resolve it and the mode goes NaN. The real bmodes workflow
           (papers/bmodes/config.yaml) avoids this with a broad-and-fine grid --
           reporting [1, 250] arcmin, integration [0.5, 300] with nbins_int=1000
           -- so the integration range brackets the reporting range AND the grid
           is fine enough that the boundary integrals converge. This test mirrors
           that: reporting [15, 70] arcmin, integration [1, 300] arcmin (brackets
           on both ends) with nbins_int=600. Confirmed directly that nbins_int~80
           over this range NaNs the last xip_E bin and the first xim_E bin, so
           coarsening the integration grid back toward ~80 reintroduces edge NaNs
           and fails -- this is the finiteness teeth.

        2. Value-drift pins on the four deterministic mode vectors (xip/xim,
           E/B). These come from a seeded synthetic catalog -> full-sample
           treecorr xi+/- (no RNG) -> Schneider linear transform, so they are
           reproducible. Verified bitwise-stable across two separate container
           processes to a worst-case relative drift of ~1.4e-11 (pure float64
           reduction-order noise; absolute drift ~1.5e-17). The pins use
           rtol=1e-6 / atol=1e-12 -- ~5 orders of magnitude above that float-noise
           floor (no flakiness margin consumed) yet tight enough that a sub-
           percent change in any mode bites. The jackknife COVARIANCE depends on
           treecorr's kmeans patch assignment and is NOT pinned by value -- only
           its shape is asserted.
        """
        pytest.importorskip("treecorr")
        pytest.importorskip("cosmo_numba")
        # Coherent shear -> smooth xi+/-, so the pure-E/B integral is well-posed.
        params, version = self._write_synthetic_catalogs(
            tmp_path, n_gal=4000, coherent_shear=True
        )

        npatch = 8
        nbins = 6
        cv = CosmologyValidation(
            versions=[version],
            npatch=npatch,
            theta_min=15.0,
            theta_max=70.0,
            nbins=nbins,
            **params,
        )

        # Integration range strictly brackets the reporting range [15, 70] on
        # both ends (1 << 15, 300 >> 70) AND uses a fine grid (nbins_int=600), so
        # the near-singular boundary-bin Schneider integrals converge. This
        # mirrors the bmodes workflow's broad-and-fine integration grid; every
        # reporting bin is well-defined (no edge NaNs). nbins_int~80 here would
        # NaN the edge bins -- confirmed -- which is the finiteness teeth.
        results = cv.calculate_pure_eb(
            version,
            npatch=npatch,
            min_sep_int=1.0,
            max_sep_int=300.0,
            nbins_int=600,
        )

        # Reference mode vectors from the seeded synthetic catalog + Schneider
        # transform. Deterministic (full-sample treecorr, no RNG); regenerate by
        # running calculate_pure_eb with the setup above and printing repr() of
        # results[key]. Tolerances justified in the docstring.
        expected = {
            "xip_E": np.array(
                [
                    1.6688018692521218e-06,
                    -1.8392317186434428e-05,
                    1.4170916007248522e-06,
                    8.1454486560987474e-06,
                    6.2050467269160570e-06,
                    2.6649478149110497e-06,
                ]
            ),
            "xim_E": np.array(
                [
                    -4.4552381788304276e-05,
                    -1.1082898248960663e-04,
                    -9.2495668600755951e-05,
                    -5.8456322151105526e-05,
                    -4.4270469941501174e-05,
                    -2.4236697154723798e-05,
                ]
            ),
            "xip_B": np.array(
                [
                    1.8508958599700601e-05,
                    3.8264056862769537e-05,
                    -1.0482698132038303e-05,
                    -7.3081832089716533e-06,
                    -9.1621105374021936e-06,
                    -6.4075815485457576e-06,
                ]
            ),
            "xim_B": np.array(
                [
                    -1.1129938750754923e-04,
                    -4.7967477760986883e-05,
                    -3.4334760596175194e-05,
                    -1.4776328993077835e-05,
                    -4.0078671892721522e-06,
                    -8.3202301900417799e-07,
                ]
            ),
        }

        for key in ("xip_E", "xim_E", "xip_B", "xim_B"):
            vec = np.asarray(results[key])
            assert vec.shape == (nbins,)
            # All reporting bins are well-defined under the widened integration
            # range (no edge NaNs) -- the finiteness teeth.
            assert np.all(np.isfinite(vec)), f"{key} not finite"
            # Value-drift pins -- the deterministic-mode teeth.
            np.testing.assert_allclose(
                vec,
                expected[key],
                rtol=1e-6,
                atol=1e-12,
                err_msg=f"{key} drifted from pinned reference",
            )

        # Jackknife covariance over the 6 stats (xip/xim x E/B/amb) x nbins.
        # Patch (kmeans) assignment isn't guaranteed deterministic, so only the
        # shape is pinned, not the values.
        cov = np.asarray(results["cov"])
        assert cov.shape == (6 * nbins, 6 * nbins)
        assert results["n_eff"] == npatch
