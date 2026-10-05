"""Exactly nested, pair-weighted reporting means and their measurement routes."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import treecorr

from sp_validation import sacc_io
from sp_validation.angular_binning import validate_nested_grids
from sp_validation.correlation import rebin_gg_means
from sp_validation.cosmo_val import CosmologyValidation
from sp_validation.statistics import jackknife_patch_centers
from sp_validation.tests.test_cosmo_val import (
    TestCosmologyValidation as _CatalogFactory,
)

REPORT = dict(min_sep=1.0, max_sep=250.0, nbins=20)
FINE = dict(min_sep=0.07939053012074941, max_sep=301.3330747561176, nbins=1015)
FIELDS = ("xip", "xim", "xip_im", "xim_im", "meanr", "meanlogr", "weight", "npairs")


def _small_grids():
    report = dict(min_sep=5.0, max_sep=200.0, nbins=8)
    step = np.log(40) / (8 * 4)
    fine = dict(min_sep=5 * np.exp(-5 * step), max_sep=200 * np.exp(5 * step), nbins=42)
    return report, fine


def test_shipped_nested_grid_edges():
    assert validate_nested_grids(REPORT, FINE) == 34
    edges = np.geomspace(FINE["min_sep"], FINE["max_sep"], 1016)
    np.testing.assert_allclose(edges[312:993:34], np.geomspace(1, 250, 21), rtol=1e-14)
    assert FINE["min_sep"] < 0.08 and FINE["max_sep"] > 300


@pytest.mark.parametrize(
    "fine",
    [
        dict(min_sep=0.08, max_sep=300, nbins=1000),
        {
            **FINE,
            "min_sep": FINE["min_sep"] * np.exp(0.000811979546744448),
            "max_sep": FINE["max_sep"] * np.exp(0.000811979546744448),
        },
        dict(min_sep=np.exp(np.log(250) / (20 * 34)), max_sep=250, nbins=679),
    ],
)
def test_non_nested_grids_fail_clearly(fine):
    with pytest.raises(ValueError, match="must nest"):
        validate_nested_grids(REPORT, fine)


def test_exact_weighted_rebin_matches_direct_treecorr():
    rng = np.random.default_rng(4)
    cat = treecorr.Catalog(
        x=rng.random(500),
        y=rng.random(500),
        g1=rng.normal(size=500),
        g2=rng.normal(size=500),
        w=np.exp(rng.normal(size=500)),
    )
    # Both slops zero force leaf-pair counting, independent of cell aggregation.
    report = dict(
        min_sep=0.01,
        max_sep=1.0,
        nbins=8,
        bin_slop=0,
        angle_slop=0,
        min_top=3,
        num_threads=1,
    )
    step = np.log(100) / (8 * 4)
    fine_config = {
        **report,
        "min_sep": 0.01 * np.exp(-5 * step),
        "max_sep": np.exp(5 * step),
        "nbins": 42,
    }
    direct, fine = treecorr.GGCorrelation(report), treecorr.GGCorrelation(fine_config)
    direct.process(cat)
    fine.process(cat)
    rebinned = rebin_gg_means(fine, report)
    for name in FIELDS:
        np.testing.assert_allclose(
            getattr(rebinned, name), getattr(direct, name), rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize("tomography", [False, True])
def test_reporting_path_uses_fine_means_and_only_a_patched_reporting_pass(
    tmp_path, monkeypatch, tomography
):
    report, fine_grid = _small_grids()
    params, ver = _CatalogFactory._write_synthetic_catalogs(
        tmp_path, n_gal=2000, with_tomography=tomography
    )
    cv = CosmologyValidation(
        [ver],
        theta_min=report["min_sep"],
        theta_max=report["max_sep"],
        nbins=report["nbins"],
        npatch=8,
        integration=fine_grid,
        **params,
    )
    cv.treecorr_config.update(min_top=2, num_threads=1)
    fine = cv.calculate_2pcf_version(
        ver, npatch=1, compute_tomography=tomography, **fine_grid
    )
    cols = cv._shear_columns(ver, tomography)
    centers = jackknife_patch_centers(cv._bin_catalog(cols, "all", 1), 8)
    angle = np.deg2rad(0.35)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    calls = []
    process = treecorr.GGCorrelation.process

    def recording(gg, cat, *args, **kwargs):
        calls.append((gg.nbins, cat.npatch))
        return process(gg, cat, *args, **kwargs)

    monkeypatch.setattr(treecorr.GGCorrelation, "process", recording)
    outputs = []
    for layout in (centers, centers @ rotation.T):
        monkeypatch.setattr(cv, "_patch_centers", lambda *args, c=layout: c)
        outputs.append(
            cv.calculate_2pcf_version(
                ver, compute_tomography=tomography, fine_correlations=fine
            )
        )
    assert calls == [(8, 8)] * (6 if tomography else 2)
    for pair in fine:
        a, b = outputs[0][pair], outputs[1][pair]
        expected = rebin_gg_means(fine[pair], cv._binning())
        for name in FIELDS:
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
            np.testing.assert_array_equal(getattr(a, name), getattr(expected, name))
        assert not np.array_equal(a.cov, b.cov)


def test_sacc_fine_product_drives_the_reporting_producer(tmp_path):
    report, fine_grid = _small_grids()
    params, ver = _CatalogFactory._write_synthetic_catalogs(tmp_path, n_gal=1200)
    script = Path(__file__).resolve().parents[3] / "workflow/scripts/run_2pcf.py"
    spec = importlib.util.spec_from_file_location("nested_run_2pcf", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shared = dict(
        ver=ver, cat_config=params["catalog_config"], output_dir=params["output_dir"]
    )
    fine_path, report_path = tmp_path / "fine.sacc", tmp_path / "report.sacc"
    fine = module.run_2pcf(
        **shared, **fine_grid, npatch=1, grid="integration", sacc_out=fine_path
    )
    measured = module.run_2pcf(
        **shared,
        **report,
        npatch=8,
        integration=fine_grid,
        fine_xi=fine_path,
        sacc_out=report_path,
    )
    expected = rebin_gg_means(fine, {**report, "sep_units": "arcmin"})
    for name in FIELDS:
        np.testing.assert_array_equal(getattr(measured, name), getattr(expected, name))
    part = sacc_io.load(str(report_path), allow_unblinded=True)
    np.testing.assert_array_equal(part.covariance.dense, measured.cov)
    np.testing.assert_array_equal(
        sacc_io.get_xi_aux(part, (0, 0), grid="reporting")["meanlogr"],
        measured.meanlogr,
    )
    # A raw-catalogue covariance cannot be paired with a concealed mean input.
    part = sacc_io.load(str(fine_path), allow_unblinded=True)
    part.metadata["concealed"] = True
    sacc_io.save(part, fine_path, type="data")
    with pytest.raises(ValueError, match="pre-blinding"):
        module.run_2pcf(
            **shared, **report, npatch=8, integration=fine_grid, fine_xi=fine_path
        )
