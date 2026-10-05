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
                "dndz": {"path": "dndz.txt"},
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
    # output of the right shape with finite values; a test that also compares
    # values says against what in its docstring. They do NOT re-test the
    # underlying numerical libraries (treecorr, cosmo_numba). These are the
    # back-pressure that catches config-path / wiring breakage during
    # restructuring.
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
        with_tomography=False,
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
        with_tomography : bool
            If True, add two tomographic bins to the shear catalogue.
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
        shear_data = {"RA": ra, "Dec": dec, "e1": e1, "e2": e2, "w": w}
        if with_tomography:
            shear_data["tomo_bin_id"] = rng.integers(1, 3, n_gal)
        Table(shear_data).write(shear_path, overwrite=True)

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
            "redshift_path": str(nz_dir / "dndz_SP_A.txt"),
            "w_col": "w",
            "e1_col": "e1",
            "e2_col": "e2",
            "R": 1.0,
            "e1_col_corrected": "e1",
            "e2_col_corrected": "e2",
        }
        if with_tomography:
            shear_cfg["tomo_bin_col"] = "tomo_bin_id"
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
            "colour": "tab:blue",
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
                "dndz": {"path": "dndz_{pipeline}_A.txt"},
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

        ggs = cv.calculate_2pcf_version(version)

        # treecorr GGCorrelation with xi+/- on the configured angular grid
        # LG:Hardcoded to be non-tomographic for now
        assert ggs["tomo_bin_all_tomo_bin_all"].xip.shape == (nbins,)
        assert ggs["tomo_bin_all_tomo_bin_all"].xim.shape == (nbins,)
        assert np.all(np.isfinite(ggs["tomo_bin_all_tomo_bin_all"].xip))
        assert np.all(np.isfinite(ggs["tomo_bin_all_tomo_bin_all"].xim))
        # The additive-bias subtraction in the pipeline must have run.
        assert version in cv.c1 and version in cv.c2

    @pytest.mark.parametrize("npatch", [1, 4])
    def test_xi_part_carries_the_covariance_the_measurement_estimated(
        self, tmp_path, npatch
    ):
        """run_2pcf's ξ± part carries TreeCorr's covariance, whoever calls it.

        The jackknife covariance with patches, the shot-noise diagonal without,
        so every part has variances whether the rule or the CLI measured it.
        """
        import importlib.util

        import sacc

        from sp_validation import sacc_io

        script = Path(__file__).resolve().parents[3] / "workflow/scripts/run_2pcf.py"
        spec = importlib.util.spec_from_file_location("run_2pcf_part", script)
        run_2pcf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_2pcf)

        params, version = self._write_synthetic_catalogs(tmp_path)
        part = tmp_path / "part.sacc"
        kwargs = dict(
            ver=version,
            min_sep=5.0,
            max_sep=100.0,
            nbins=6,
            npatch=npatch,
            cat_config=params["catalog_config"],
            output_dir=params["output_dir"],
            sacc_out=str(part),
        )
        # The second producer call encounters its own columns-only text cache.
        for _ in range(2):
            gg = run_2pcf.run_2pcf(**kwargs)
            cov = sacc_io.load(str(part), allow_unblinded=True).covariance
            if npatch > 1:
                assert gg.results
                assert isinstance(cov, sacc.covariance.FullCovariance)
                np.testing.assert_array_equal(cov.dense, gg.cov)
            else:
                assert isinstance(cov, sacc.covariance.DiagonalCovariance)
                np.testing.assert_array_equal(
                    cov.diag, np.concatenate([gg.varxip, gg.varxim])
                )

    def test_a_patched_xi_dump_remeasures_covariance(self, tmp_path):
        """Columns-only dumps cannot supply the dense jackknife covariance."""
        params, version = self._write_synthetic_catalogs(tmp_path)
        binning = dict(npatch=4, min_sep=5.0, max_sep=100.0, nbins=6, num_threads=1)
        pair = "tomo_bin_all_tomo_bin_all"
        measured = CosmologyValidation(
            versions=[version], **params
        ).calculate_2pcf_version(version, **binning)[pair]
        dumps = list(Path(params["output_dir"]).glob(f"xi_{version}_*.txt"))
        assert len(dumps) == 1
        read = CosmologyValidation(versions=[version], **params).calculate_2pcf_version(
            version, **binning
        )[pair]
        for column in ("meanr", "npairs", "xip", "xim", "varxip", "varxim", "cov"):
            np.testing.assert_array_equal(
                getattr(read, column), getattr(measured, column)
            )
        assert read.results

    def test_xi_plotting_reads_the_published_columns_without_remeasurement(
        self, tmp_path, monkeypatch
    ):
        params, version = self._write_synthetic_catalogs(tmp_path)
        cv = CosmologyValidation(
            [version], npatch=4, theta_min=5, theta_max=100, nbins=6, **params
        )
        pair = "tomo_bin_all_tomo_bin_all"
        measured = cv.calculate_2pcf_version(version, num_threads=1)[pair]

        def no_catalogue_read(*args):
            raise AssertionError("plotting must reuse its declared measurement input")

        monkeypatch.setattr(cv, "_shear_columns", no_catalogue_read)
        cached = cv.calculate_2pcf_version(version, read_cached=True)[pair]
        for column in ("xip", "xim", "meanr", "meanlogr", "weight", "npairs"):
            np.testing.assert_array_equal(
                getattr(cached, column), getattr(measured, column)
            )
        np.testing.assert_allclose(cached.varxip, measured.varxip, rtol=1e-15)
        assert not cached.results

    def test_calculate_2pcf_is_reproducible_across_machines(
        self, tmp_path, monkeypatch
    ):
        """Fresh calculate_2pcf_version runs share patches and ξ± across CPU counts."""
        import treecorr
        import treecorr.field

        patches = []
        process = treecorr.GGCorrelation.process

        def recording_process(gg, cat, *args, **kwargs):
            if cat.npatch > 1:
                patches.append(np.array(cat.patch))
            return process(gg, cat, *args, **kwargs)

        monkeypatch.setattr(treecorr.GGCorrelation, "process", recording_process)

        xi, var, counts = {}, {}, {}
        for tree, n_cpu in (("a", 4), ("b", 16)):
            monkeypatch.setattr(treecorr.field, "get_omp_threads", lambda n=n_cpu: n)
            run_dir = tmp_path / tree
            run_dir.mkdir()
            params, version = self._write_synthetic_catalogs(
                run_dir,
                n_gal=4000,
                ra_range=(0.0, 60.0),
                dec_range=(-10.0, 30.0),
                coherent_shear=True,
            )
            gg = CosmologyValidation(
                versions=[version],
                npatch=100,
                theta_min=15.0,
                theta_max=70.0,
                nbins=6,
                **params,
            ).calculate_2pcf_version(version, num_threads=n_cpu)[
                "tomo_bin_all_tomo_bin_all"
            ]
            xi[tree] = np.concatenate([gg.xip, gg.xim])
            var[tree] = np.concatenate([gg.varxip, gg.varxim])
            counts[tree] = {key: result.npairs for key, result in gg.results.items()}

        np.testing.assert_array_equal(patches[0], patches[1])
        assert counts["a"].keys() == counts["b"].keys()
        for key in counts["a"]:
            np.testing.assert_array_equal(counts["a"][key], counts["b"][key])
        np.testing.assert_allclose(xi["a"], xi["b"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(var["a"], var["b"], rtol=1e-10)

    def test_cross_tomographic_pair_shares_patch_centres(self, tmp_path, monkeypatch):
        """Both catalogues in a cross-bin pair use the full-sample centres."""
        import treecorr

        from sp_validation.statistics import jackknife_patch_centers

        params, version = self._write_synthetic_catalogs(
            tmp_path, n_gal=400, with_tomography=True
        )
        cv = CosmologyValidation(versions=[version], npatch=4, **params)
        cols = cv._shear_columns(version, compute_tomography=True)
        full_catalog = treecorr.Catalog(
            ra=cols["ra"],
            dec=cols["dec"],
            w=cols["w"],
            ra_units=cv.treecorr_config["ra_units"],
            dec_units=cv.treecorr_config["dec_units"],
        )
        expected_centers = jackknife_patch_centers(full_catalog, 4)
        catalogs = []
        make_catalog = cv._bin_catalog

        def recording_bin_catalog(cols, bin_id, npatch, patch_centers=None):
            catalog = make_catalog(cols, bin_id, npatch, patch_centers)
            catalogs.append((bin_id, patch_centers, np.array(catalog._centers)))
            return catalog

        monkeypatch.setattr(cv, "_bin_catalog", recording_bin_catalog)
        cv.calculate_2pcf_version(version, npatch=4, compute_tomography=True)

        assert len(catalogs) == 4
        cross_pair_catalogs = catalogs[1:3]
        assert [entry[0] for entry in cross_pair_catalogs] == [1, 2]
        np.testing.assert_array_equal(
            cross_pair_catalogs[0][1], cross_pair_catalogs[1][1]
        )
        np.testing.assert_array_equal(
            cross_pair_catalogs[0][2], cross_pair_catalogs[1][2]
        )
        np.testing.assert_allclose(
            cross_pair_catalogs[0][2], expected_centers, rtol=0, atol=1e-14
        )
        np.testing.assert_allclose(
            cross_pair_catalogs[1][2], expected_centers, rtol=0, atol=1e-14
        )

    @pytest.mark.parametrize("b_target", [0.01, 0.02])
    def test_bin_slop_follows_the_grid(self, tmp_path, b_target):
        params, version = self._write_synthetic_catalogs(tmp_path)
        cv = CosmologyValidation(
            [version],
            theta_min=1,
            theta_max=250,
            nbins=20,
            b_target=b_target,
            **params,
        )
        expected = b_target / (np.log(250) / 20)
        assert cv.treecorr_config["bin_slop"] == pytest.approx(expected)
        assert cv._binning()["bin_slop"] == pytest.approx(expected)
        assert cv._binning(0.08, 300, 1000)["bin_slop"] == 1
        assert cv._binning(bin_slop=0)["bin_slop"] == 0
        assert "angle_slop" not in cv.treecorr_config

    @pytest.mark.parametrize("tomography", [False, True])
    def test_published_means_are_layout_independent(
        self, tmp_path, monkeypatch, tomography
    ):
        """Both auto- and cross-bin means ignore the covariance patch layout."""
        import treecorr

        from sp_validation.statistics import jackknife_patch_centers

        params, version = self._write_synthetic_catalogs(
            tmp_path,
            n_gal=4000,
            with_tomography=tomography,
            coherent_shear=True,
        )
        cv = CosmologyValidation(
            [version], npatch=8, theta_min=5, theta_max=200, nbins=8, **params
        )
        # Small trees exercise cell aggregation without a large catalogue.
        cv.treecorr_config.update(min_top=2, num_threads=1)
        cols = cv._shear_columns(version, tomography)
        cat = cv._bin_catalog(cols, "all", 1)
        centers = jackknife_patch_centers(cat, 8)
        angle = np.deg2rad(0.35)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        layouts = [centers, centers @ rotation.T]
        assert not np.array_equal(
            cv._bin_catalog(cols, "all", 8, layouts[0]).patch,
            cv._bin_catalog(cols, "all", 8, layouts[1]).patch,
        )
        measured = []
        for centers in layouts:
            monkeypatch.setattr(cv, "_patch_centers", lambda *args, c=centers: c)
            measured.append(
                cv.calculate_2pcf_version(version, compute_tomography=tomography)
            )
        for pair in measured[0]:
            a, b = measured[0][pair], measured[1][pair]
            for column in (
                "xip",
                "xim",
                "xip_im",
                "xim_im",
                "meanr",
                "meanlogr",
                "weight",
                "npairs",
            ):
                np.testing.assert_array_equal(getattr(a, column), getattr(b, column))
            assert not np.array_equal(a.cov, b.cov)
            # Covariance is still exactly the direct patched measurement.
            bin1, bin2 = (
                (1, 2)
                if pair == "tomo_bin_1_tomo_bin_2"
                else (
                    (1, 1)
                    if pair == "tomo_bin_1_tomo_bin_1"
                    else (2, 2)
                    if pair == "tomo_bin_2_tomo_bin_2"
                    else ("all", "all")
                )
            )
            covariance_config = cv._binning()
            covariance_config.pop("bin_slop")
            direct = treecorr.GGCorrelation(covariance_config)
            direct.process(
                cv._bin_catalog(cols, bin1, 8, layouts[0]),
                cat2=(
                    cv._bin_catalog(cols, bin2, 8, layouts[0]) if bin1 != bin2 else None
                ),
            )
            np.testing.assert_array_equal(a.cov, direct.cov)
            np.testing.assert_array_equal(a.estimate_cov("jackknife"), direct.cov)

    def test_rho_tau_means_are_layout_independent(self, tmp_path, monkeypatch):
        from sp_validation import rho_tau

        params, version = self._write_synthetic_catalogs(
            tmp_path,
            n_gal=1200,
            n_star=1000,
            with_psf=True,
        )
        cv = CosmologyValidation(
            [version], npatch=8, theta_min=5, theta_max=200, nbins=8, **params
        )
        cv.treecorr_config["num_threads"] = 1
        cv.cc[version]["patch_number"] = 8
        from shear_psf_leakage.rho_tau_stat import Catalogs

        from sp_validation.statistics import jackknife_patch_centers

        cols = cv._shear_columns(version, False)
        centers = jackknife_patch_centers(cv._bin_catalog(cols, "all", 1), 8)
        angle = np.deg2rad(0.35)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        build = Catalogs.build_catalog
        outputs = []
        for i, layout in enumerate((centers, centers @ rotation.T)):

            def with_layout(self, *args, layout=layout, **kwargs):
                kwargs["patch_centers"] = layout
                return build(self, *args, **kwargs)

            monkeypatch.setattr(Catalogs, "build_catalog", with_layout)
            out = tmp_path / f"rho_tau_{i}"
            out.mkdir()
            rho, tau = rho_tau.get_jackknife_cov(
                cv.cc,
                version,
                cv.treecorr_config,
                str(out),
                "test",
                "test",
                npatch=8,
                ncov=1,
            )
            outputs.append(
                (
                    rho.rho_stats.copy(),
                    tau.tau_stats.copy(),
                    np.load(out / "cov_tau_test_jk.npy"),
                )
            )
        for i in (0, 1):
            for column in outputs[0][i].dtype.names:
                if not column.startswith("var"):
                    np.testing.assert_array_equal(
                        outputs[0][i][column], outputs[1][i][column]
                    )
        assert not np.array_equal(outputs[0][2], outputs[1][2])
        assert not np.array_equal(
            outputs[0][1]["vartau_0_p"], outputs[1][1]["vartau_0_p"]
        )

    def test_aperture_mass_means_are_layout_independent(self, tmp_path, monkeypatch):
        from sp_validation.statistics import jackknife_patch_centers

        params, version = self._write_synthetic_catalogs(tmp_path, n_gal=3000)
        cv = CosmologyValidation([version], npatch=8, **params)
        cv.treecorr_config.update(min_top=2, num_threads=1)
        cols = cv._shear_columns(version, False)
        cat = cv._bin_catalog(cols, "all", 1)
        centers = jackknife_patch_centers(cat, 8)
        angle = np.deg2rad(0.35)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        measured = []
        for layout in (centers, centers @ rotation.T):
            monkeypatch.setattr(cv, "_patch_centers", lambda *args, c=layout: c)
            cv.calculate_aperture_mass_dispersion(
                theta_min=5, theta_max=200, nbins=30, npatch=8
            )
            measured.append(cv.map2[version]["tomo_bin_all_tomo_bin_all"])
        for key in ("mapsq", "mapsq_im", "mxsq", "mxsq_im"):
            np.testing.assert_array_equal(measured[0][key], measured[1][key])
        assert not np.array_equal(measured[0]["varmapsq"], measured[1]["varmapsq"])

    def test_treecorr_runs_on_the_cpus_the_process_holds(self, tmp_path):
        """By default TreeCorr takes the process's CPU affinity, not the node's count."""
        import treecorr

        params, version = self._write_synthetic_catalogs(tmp_path)
        CosmologyValidation(
            versions=[version], npatch=1, **params
        ).calculate_2pcf_version(version)
        assert treecorr.get_omp_threads() == len(os.sched_getaffinity(0))

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

    def test_calculate_pure_eb_runs_on_synthetic_catalog(self, tmp_path, pure_eb_xi):
        """calculate_pure_eb measures the fine ξ± and pushes it through the operator.

        The ξ± it measures equal the committed ``pure_eb_xi`` (``test_b_modes``
        pins the operator on the same ξ±, so a failure names the step that
        moved), and the jackknife covariance of the modes is the jackknife ξ±
        covariance through the operator: for a linear estimator that equals
        TreeCorr's per-patch jackknife of the modes themselves.

        ξ±: exact binning (bin_slop = angle_slop = 0) makes ξ± a plain pair sum,
        independent of the tree and so of the jackknife patches, whose k-means
        centres this test does not fix.
        """
        pytest.importorskip("treecorr")
        pytest.importorskip("cosmo_numba")
        import treecorr

        from sp_validation import b_modes

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

        integration = dict(min_sep_int=1.0, max_sep_int=300.0, nbins_int=600)
        results = cv.calculate_pure_eb(version, npatch=npatch, **integration)[
            "tomo_bin_all_tomo_bin_all"
        ]

        measured = {
            key: results[key]
            for key in ("theta_int", "xip_int", "xim_int", "weight_int", "edges_int")
        }
        measured["reporting_edges"] = np.geomspace(15.0, 70.0, nbins + 1)
        # Regenerate the fixture with np.savez(conftest.PURE_EB_XI, **measured).
        for key, value in measured.items():
            np.testing.assert_allclose(
                value, pure_eb_xi[key], rtol=1e-10, atol=0, err_msg=key
            )

        operator, _, edges = b_modes.pure_eb_operator(
            *(measured[k] for k in ("weight_int", "edges_int", "reporting_edges"))
        )
        np.testing.assert_array_equal(results["left_edges"], edges[:-1])
        modes = operator @ np.concatenate([measured["xip_int"], measured["xim_int"]])
        for i, key in enumerate(b_modes._EB_KEYS):
            vec = np.asarray(results[key])
            assert vec.shape == (nbins,)
            assert np.all(np.isfinite(vec)), f"{key} not finite"
            np.testing.assert_allclose(
                vec, modes[i * nbins : (i + 1) * nbins], rtol=1e-12, err_msg=key
            )

        cov = np.asarray(results["cov"])
        assert cov.shape == (6 * nbins, 6 * nbins)
        assert results["npatch"] == npatch
        # The integration-grid ξ± again, measured on the jackknife patches the
        # first measurement wrote (its columns-only dump carries no patches).
        for dump in Path(params["output_dir"]).glob(f"xi_{version}_*.txt"):
            dump.unlink()
        gg_int = cv.calculate_2pcf_version(
            version,
            npatch=npatch,
            min_sep=integration["min_sep_int"],
            max_sep=integration["max_sep_int"],
            nbins=integration["nbins_int"],
        )["tomo_bin_all_tomo_bin_all"]
        jackknife_of_modes = treecorr.estimate_multi_cov(
            [gg_int],
            "jackknife",
            func=lambda corrs: operator @ np.concatenate([corrs[0].xip, corrs[0].xim]),
            cross_patch_weight="match",
        )
        np.testing.assert_allclose(
            cov, jackknife_of_modes, rtol=0, atol=1e-10 * np.abs(cov).max()
        )

    def test_plot_2pcf_writes_the_non_tomographic_figures(self, tmp_path):
        """plot_2pcf draws the ("all", "all") ξ± through plot_2pcf_tomography."""
        params, version = self._write_synthetic_catalogs(tmp_path)
        cv = CosmologyValidation(
            versions=[version],
            npatch=1,
            theta_min=5.0,
            theta_max=100.0,
            nbins=6,
            **params,
        )
        cv.plot_2pcf(show=False)

        out = Path(params["output_dir"])
        for name in ("xi_pm_tomography_False", "xi_pm_theta_tomography_False"):
            assert (out / f"{name}.png").is_file(), name

    def test_plot_ratio_xi_sys_xi_writes_the_declared_figure(self, tmp_path):
        """plot_ratio_xi_sys_xi divides the pair's ξ^{PSF, sys}± by its ξ±.

        ξ^{PSF, sys} is set on the instance, so the test needs no ρ/τ fit; the
        figure lands where the cv_ratio_xi_sys_xi rule declares it.
        """
        params, version = self._write_synthetic_catalogs(tmp_path)
        nbins = 6
        cv = CosmologyValidation(
            versions=[version],
            npatch=1,
            theta_min=5.0,
            theta_max=100.0,
            nbins=nbins,
            **params,
        )
        sys = np.full(nbins, 1e-6)
        cv._xi_psf_sys = {
            version: {
                "tomo_bin_all_tomo_bin_all": {
                    "mean_plus": sys,
                    "var_plus": sys**2,
                    "mean_minus": sys,
                    "var_minus": sys**2,
                }
            }
        }
        cv.plot_ratio_xi_sys_xi(show=False)

        assert (Path(params["output_dir"]) / "ratio_xi_sys_xi.png").is_file()
        with pytest.raises(ValueError, match="compute_tomography"):
            cv.plot_ratio_xi_sys_xi(tomography=True, show=False)

    def test_summarize_bmodes_reads_the_all_pair(self, tmp_path):
        """Each statistic's ("all", "all") result reaches the summary.

        A result whose shape the summary cannot read raises rather than
        silently leaving its column empty.
        """
        from types import SimpleNamespace

        from sp_validation.statistics import chi2_and_pte

        params, version = self._write_synthetic_catalogs(tmp_path)
        cv = CosmologyValidation(versions=[version], **params)
        pair = "tomo_bin_all_tomo_bin_all"

        edges = np.geomspace(1.0, 100.0, 6)
        cv._pure_eb_results[version] = {
            pair: {
                "left_edges": edges[:-1],
                "right_edges": edges[1:],
                "pte_matrices": {
                    stat: np.full((5, 5), pte)
                    for stat, pte in (("xip_B", 0.1), ("xim_B", 0.2), ("combined", 0.3))
                },
                "npatch": 8,
            }
        }
        cv._cosebis_results[version] = {pair: {"pte_B": 0.4}}
        cl_bb = np.array([1.0, -0.5, 0.25])
        cov_bb = np.diag([2.0, 1.0, 0.5])
        cv._pseudo_cls = {
            version: {
                pair: {
                    "pseudo_cl": {"BB": cl_bb},
                    "cov": {"COVAR_BB_BB": SimpleNamespace(data=cov_bb)},
                }
            }
        }

        row = cv.summarize_bmodes(fiducial_scale_cut=(1.0, 100.0))[version]

        assert row == pytest.approx(
            {
                "xip_B": 0.1,
                "xim_B": 0.2,
                "combined": 0.3,
                "COSEBIS": 0.4,
                "C_l_BB": chi2_and_pte(cl_bb, cov_bb)[2],
            }
        )

        del cv._pure_eb_results[version][pair]["pte_matrices"]
        with pytest.raises(KeyError):
            cv.summarize_bmodes(fiducial_scale_cut=(1.0, 100.0))

    def test_sacc_nz_is_the_summed_tomographic_nz(self, tmp_path):
        """A multi-column n(z) file gives the SACC parts its whole-survey sum."""
        params, version = self._write_synthetic_catalogs(tmp_path)
        cv = CosmologyValidation(versions=[version], **params)

        z = np.linspace(0.0, 2.0, 11)
        nz_bins = np.stack([np.exp(-((z - mu) ** 2)) for mu in (0.5, 1.0, 1.5)])
        nz_path = tmp_path / "nz_tomo.txt"
        np.savetxt(nz_path, np.column_stack([z, *nz_bins]))
        cv.cc[version]["shear"]["redshift_path"] = str(nz_path)

        [(bin_id, (z_read, nz_read))] = cv.sacc_nz(version).items()
        assert bin_id == 0
        np.testing.assert_allclose(z_read, z)
        np.testing.assert_allclose(nz_read, nz_bins.sum(axis=0))

    @pytest.mark.parametrize("pol_factor", [True, False, 0, 2])
    def test_pol_factor_must_be_plus_or_minus_one(self, tmp_path, pol_factor):
        """pol_factor is a ±1 e2 multiplier; a bool would pass `in (-1, 1)`."""
        params, version = self._write_synthetic_catalogs(tmp_path)
        with pytest.raises(ValueError, match="pol_factor"):
            CosmologyValidation(versions=[version], pol_factor=pol_factor, **params)
