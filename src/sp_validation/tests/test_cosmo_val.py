"""UNIT TESTS FOR COSMOLOGY VALIDATION CLASS.

This module contains integration tests for the CosmologyValidation class,
specifically testing the ellipticity_suffix parameter functionality for
handling leak-corrected ellipticity columns.

:Author: cdaley

"""

import os
from pathlib import Path

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
    # underlying numerical libraries (treecorr, cosmo_numba); only the pure-E/B
    # test pins values. These are the back-pressure that catches config-path /
    # wiring breakage during restructuring.
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
        """calculate_pure_eb carries ξ± through cosmo_numba's pure-E/B split.

        Every reporting bin is finite, and the four mode vectors match pins.

        Finiteness: the Schneider (2022) integrals are near-singular where a
        reporting bin meets the integration boundary, so the integration grid
        [1, 300]′ brackets the reporting grid [15, 70]′ on both ends and is fine
        (600 bins); about 80 integration bins NaN the edge bins.

        Pins: TreeCorr's default bin_slop/angle_slop approximate separations from
        its tree, whose top-level split follows the jackknife patches and, through
        min_top, the thread count TreeCorr takes from cpu_count(). On this
        catalogue that moves the reporting ξ− by up to 16% between 4 and 48
        threads. Exact binning makes ξ± a plain pair sum, so the pins move only
        when sp_validation does; rtol=1e-6 is far above its 1e-12 reduction-order
        noise and far below a sub-percent change in any mode. The E/B transform
        alone is pinned on fixed ξ± in ``test_b_modes``.
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
        cv.treecorr_config.update(bin_slop=0, angle_slop=0)

        results = cv.calculate_pure_eb(
            version,
            npatch=npatch,
            min_sep_int=1.0,
            max_sep_int=300.0,
            nbins_int=600,
        )

        # Regenerate by printing repr(results[key]) from the setup above.
        expected = {
            "xip_E": np.array(
                [
                    -2.9831529669572382e-06,
                    -1.5008524620255579e-05,
                    3.221623968699465e-07,
                    1.1797672310854472e-05,
                    5.715510692580715e-06,
                    8.825804523810145e-07,
                ]
            ),
            "xim_E": np.array(
                [
                    -4.7375580917647536e-05,
                    -0.00010853189443992296,
                    -9.094825175031718e-05,
                    -5.826599101284305e-05,
                    -4.6464054157485714e-05,
                    -1.997802892533382e-05,
                ]
            ),
            "xip_B": np.array(
                [
                    1.706912124226553e-05,
                    3.059889782372767e-05,
                    -4.880539925382135e-06,
                    -6.999262696331131e-06,
                    -1.267200698975249e-05,
                    -1.2141491389775202e-06,
                ]
            ),
            "xim_B": np.array(
                [
                    -0.00011478091634543392,
                    -5.4451120021397075e-05,
                    -3.100806652947136e-05,
                    -1.0940424256752644e-05,
                    -5.755185146639151e-06,
                    -1.628217762504009e-06,
                ]
            ),
        }

        for key in ("xip_E", "xim_E", "xip_B", "xim_B"):
            vec = np.asarray(results[key])
            assert vec.shape == (nbins,)
            assert np.all(np.isfinite(vec)), f"{key} not finite"
            np.testing.assert_allclose(
                vec,
                expected[key],
                rtol=1e-6,
                atol=1e-12,
                err_msg=f"{key} drifted from pinned reference",
            )

        # Jackknife covariance over the 6 stats (xip/xim x E/B/amb) x nbins.
        cov = np.asarray(results["cov"])
        assert cov.shape == (6 * nbins, 6 * nbins)
        assert results["n_eff"] == npatch
