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
from _synthetic import write_synthetic_catalogs

from sp_validation import sacc_io
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

    def test_calculate_2pcf_runs_on_synthetic_catalog(self, tmp_path):
        """calculate_2pcf wires catalog+config into a ξ± part.

        Smoke-integration: the part's ξ± has the configured number of angular
        bins and is finite, and the part reads back with the binning's edges,
        by which scale cuts select bins. Numerical values are deliberately not
        asserted.
        """
        from sp_validation.b_modes import log_bin_edges

        pytest.importorskip("treecorr")
        params, version = write_synthetic_catalogs(tmp_path)

        nbins = 8
        cv = CosmologyValidation(
            versions=[version],
            npatch=1,
            theta_min=5.0,
            theta_max=100.0,
            nbins=nbins,
            **params,
        )

        gg = sacc_io.xi_correlation(cv.calculate_2pcf(version))

        edges = log_bin_edges(5.0, 100.0, nbins)
        np.testing.assert_allclose(gg.left_edges, edges[0], rtol=1e-12)
        np.testing.assert_allclose(gg.right_edges, edges[1], rtol=1e-12)
        assert gg.xip.shape == (nbins,)
        assert gg.xim.shape == (nbins,)
        assert np.all(np.isfinite(gg.xip))
        assert np.all(np.isfinite(gg.xim))
        # The additive-bias subtraction in the pipeline must have run.
        assert version in cv.c1 and version in cv.c2

    def test_calculate_2pcf_does_not_depend_on_thread_count(self, tmp_path):
        """calculate_2pcf's ξ± is the same on 4 and on 48 TreeCorr threads.

        Production binning (default bin_slop/angle_slop), both runs on the
        jackknife patches the first one writes, each from a fresh Catalog; they
        must agree to far below the jackknife σ.
        """
        import treecorr

        params, version = write_synthetic_catalogs(
            tmp_path, n_gal=4000, coherent_shear=True
        )
        cv = CosmologyValidation(
            versions=[version],
            npatch=8,
            theta_min=15.0,
            theta_max=70.0,
            nbins=6,
            **params,
        )

        xi = {}
        for n_threads in (4, 48):
            gg = sacc_io.xi_correlation(
                cv.calculate_2pcf(version, num_threads=n_threads)
            )
            assert treecorr.get_omp_threads() == n_threads  # the count took effect
            xi[n_threads] = np.concatenate([gg.xip, gg.xim])
            sigma = np.sqrt(np.concatenate([gg.varxip, gg.varxim]))

        shift = np.max(np.abs(xi[48] - xi[4]) / sigma)
        assert shift < 1e-6, f"ξ± moves by {shift:.3g}σ between 4 and 48 threads"

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
        params, version = write_synthetic_catalogs(tmp_path)

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

    def test_calculate_pure_eb_runs_on_synthetic_catalog(
        self, tmp_path, pure_eb_xi, monkeypatch
    ):
        """calculate_pure_eb carries ξ± through cosmo_numba's pure-E/B split.

        The ξ± it measures equal the committed ``pure_eb_xi``, its modes are
        ``pure_eb_from_xi`` of those ξ± and edges, and every reporting bin is
        finite. ``test_b_modes`` pins the transform itself on the same ξ±, so a
        failure names the step that moved: measurement, wiring or transform.
        Its jackknife covariance, the integration part's pushed through the
        kernel, matches TreeCorr's jackknife of the modes in each statistic's
        total variance, loosely: on this sparse toy ξ± the kernel's ξ−
        quadrature is additive to only ~30%.

        Finiteness: the Schneider (2022) integrals are near-singular where a
        reporting bin meets the integration boundary, so the integration grid
        [1, 300]′ brackets the reporting grid [15, 70]′ on both ends and is fine
        (600 bins); about 80 integration bins NaN the edge bins.

        ξ±: exact binning (bin_slop = angle_slop = 0) makes ξ± a plain pair sum,
        independent of the tree and so of the jackknife patches, whose k-means
        centres depend on the machine.
        """
        pytest.importorskip("treecorr")
        pytest.importorskip("cosmo_numba")
        from sp_validation import b_modes

        # Coherent shear -> smooth xi+/-, so the pure-E/B integral is well-posed.
        params, version = write_synthetic_catalogs(
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

        import treecorr

        kernel = b_modes.pure_eb_from_xi
        process, correlations = treecorr.GGCorrelation.process, []
        monkeypatch.setattr(
            treecorr.GGCorrelation,
            "process",
            lambda gg, *a, **kw: correlations.append(gg) or process(gg, *a, **kw),
        )
        results = cv.calculate_pure_eb(
            version,
            npatch=npatch,
            min_sep_int=1.0,
            max_sep_int=300.0,
            nbins_int=600,
        )

        measured = {
            "theta_report": results["theta"],
            "xip_report": results["xip"],
            "xim_report": results["xim"],
            "theta_int": results["theta_int"],
            "xip_int": results["xip_int"],
            "xim_int": results["xim_int"],
            "tmin": results["left_edges"][0],
            "tmax": results["right_edges"][-1],
        }
        # Regenerate the fixture with np.savez(conftest.PURE_EB_XI, **measured).
        for key, value in measured.items():
            np.testing.assert_allclose(
                value, pure_eb_xi[key], rtol=1e-10, atol=0, err_msg=key
            )

        modes = b_modes.pure_eb_from_xi(**measured)
        for key in b_modes._EB_KEYS:
            vec = np.asarray(results[key])
            assert vec.shape == (nbins,)
            assert np.all(np.isfinite(vec)), f"{key} not finite"
            np.testing.assert_allclose(vec, modes[key], rtol=1e-10, err_msg=key)

        # TreeCorr's jackknife of the modes, from the two measurements.
        def modes_of(pair):
            gg, gg_int = pair
            return b_modes._eb_vector(
                kernel(
                    gg.meanr,
                    gg.xip,
                    gg.xim,
                    gg_int.meanr,
                    gg_int.xip,
                    gg_int.xim,
                    gg.left_edges[0],
                    gg.right_edges[-1],
                )
            )

        reference = treecorr.estimate_multi_cov(
            correlations, "jackknife", func=modes_of, cross_patch_weight="match"
        )
        cov = np.asarray(results["cov"])
        assert cov.shape == reference.shape == (6 * nbins, 6 * nbins)

        def block_variances(c):
            return np.diag(c).reshape(6, nbins).sum(axis=1)

        np.testing.assert_allclose(
            block_variances(cov), block_variances(reference), rtol=0.4
        )
        assert results["n_eff"] == npatch


# --------------------------------------------------------------------------- #
# I14: a blinded catalogue's ξ± leaves CosmologyValidation only concealed
# --------------------------------------------------------------------------- #
@pytest.fixture
def blinded_and_twin(tmp_path):
    """TOY, blinded under `toy`, and TOY_OPEN: the same galaxies, unblinded."""
    import json

    from sp_validation import blinding

    params, _ = write_synthetic_catalogs(
        tmp_path,
        n_gal=4000,
        coherent_shear=True,
        catalogues={"TOY": None, "TOY_OPEN": "unblinded"},
    )
    fast = tmp_path / "fast.json"
    fast.write_text(json.dumps({"theory": {"transfer_function": "eisenstein_hu"}}))
    blinding.main(
        [
            "init",
            "toy",
            "TOY",
            "--cat-config",
            params["catalog_config"],
            "--config",
            str(fast),
        ]
    )
    return CosmologyValidation(
        versions=["TOY", "TOY_OPEN"],
        npatch=1,
        theta_min=5.0,
        theta_max=60.0,
        nbins=6,
        **params,
    )


def test_a_blinded_catalogues_xi_leaves_concealed(blinded_and_twin):
    """[signal-leaves-sealed] calculate_2pcf returns, and caches, a blinded
    catalogue's ξ± shifted from its unblinded twin's by exactly the blind's shift."""
    from sp_validation import blinding

    cv = blinded_and_twin
    blinded, twin = (cv.calculate_2pcf(v) for v in ("TOY", "TOY_OPEN"))
    shift = blinding.conceal(twin, blinding.open_blind(cv.custody("TOY"))).mean
    shift = shift - twin.mean

    assert np.all(shift != 0)
    np.testing.assert_allclose(
        blinded.mean - twin.mean, shift, rtol=1e-8, atol=1e-12 * np.abs(shift).max()
    )
    assert blinded.metadata["blinding"] == "blinded"
    assert cv.xi_parts["TOY", "reporting"] is blinded
