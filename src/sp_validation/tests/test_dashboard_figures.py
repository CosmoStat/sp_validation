"""Geometry and binning helpers behind the dashboard figures."""

import healpy as hp
import numpy as np
import pandas as pd
import pytest

from sp_validation.dashboard.field_maps import (
    leakage_cells,
    weight_cell_index,
    weight_grid_edges,
)
from sp_validation.dashboard.object_bias import bias_fit, leakage_fit, slopes
from sp_validation.dashboard.psf import xi_sys_comparison
from sp_validation.dashboard.sky import Zenithal, coarsen, footprint_parts, quantity


def test_xi_sys_comparison_uses_fit_grid_without_tau(tmp_path):
    rt = {
        "fit": {
            "theta": np.array([1.0, 2.0]),
            "samples_plus": np.ones((3, 2)),
            "samples_minus": np.ones((3, 2)),
        }
    }
    out = tmp_path / "plot.png"
    xi_sys_comparison(rt, out)
    assert out.exists()


def test_leakage_quantile_bins():
    rng = np.random.default_rng(17)
    data = rng.normal(size=(6400, 7))
    data[:, 0] = np.exp(data[:, 0])
    data[:, 2] = 1
    data[:, 4] = 0.2 * data[:, 3] + 0.4
    data[:, 6] = -0.1 * data[:, 5] - 0.2
    _, _, sums = leakage_cells(data)
    np.testing.assert_array_equal(sums[0, 6], np.full((8, 8), 100))
    np.testing.assert_allclose(slopes(sums[0]), 0.2, atol=1e-12)
    np.testing.assert_allclose(slopes(sums[1]), -0.1, atol=1e-12)


def test_weighted_leakage_regression_has_intercept():
    x = np.array([-2.0, -1.0, 1.0, 2.0])
    y = 0.3 * x + 0.7
    w = np.array([1.0, 2.0, 3.0, 4.0])
    sums = np.array(
        [w.sum(), (w * x).sum(), (w * y).sum(), (w * x * x).sum(), (w * x * y).sum()]
    )
    np.testing.assert_allclose(slopes(sums), 0.3)


def test_leakage_fit_matches_statsmodels():
    import statsmodels.api as sm

    rng = np.random.default_rng(42)
    x = rng.normal(size=200)
    y = 0.03 * x + 0.4 + rng.normal(size=200)
    w = rng.uniform(0.1, 3, size=200)
    s = np.array(
        [
            w.sum(),
            (w * x).sum(),
            (w * y).sum(),
            (w * x * x).sum(),
            (w * x * y).sum(),
            (w * y * y).sum(),
            len(x),
        ]
    )
    value, error = leakage_fit(np.tile(s, (2, 1)))
    fit = sm.WLS(y, sm.add_constant(x), weights=w).fit()
    np.testing.assert_allclose(value, fit.params[1])
    np.testing.assert_allclose(error, fit.bse[1])


def test_binned_bias_fit():
    x = np.array([-1.0, 0.0, 1.0])
    y = 0.4 * x + 0.2
    s = np.array(
        [
            np.full(3, 10.0),
            10 * x,
            10 * y,
            10 * x * x,
            10 * x * y,
            10 * (y * y + 0.1),
            np.full(3, 10.0),
            np.full(3, 10.0),
            10 * y,
            10 * (y * y + 0.1),
        ]
    )
    slope, error = bias_fit(s)
    np.testing.assert_allclose(slope, 0.4 * np.sqrt(2 / 3))
    assert error > 0


def test_size_residual_uses_model_normalised_sum():
    maps = {
        "nside": 128,
        "pixels": np.array([0]),
        "n_star": np.array([10.0]),
        "sum_dT_over_T_psf": np.array([-2.0]),
    }
    pixels, values = quantity(maps, "dT_over_T", 128)
    np.testing.assert_array_equal(pixels, [0])
    np.testing.assert_allclose(values, [-0.2])


def test_weight_cells_match_calibration_binning():
    """Cell indices reproduce cut_to_bins: right-closed, outside -> last cell."""
    snr_edges, _ = weight_grid_edges(20, (10, 500), (0.707, 3))
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.uniform(5, 700, 2000), snr_edges, [np.nan]])
    expected = pd.cut(values, snr_edges, labels=False)
    expected = np.where(np.isnan(expected), 19, expected).astype(int)
    np.testing.assert_array_equal(weight_cell_index(values.copy(), snr_edges), expected)


@pytest.mark.parametrize("rotation", [0.0, 37.0, -80.0])
def test_zenithal_round_trip(rotation):
    proj = Zenithal(229.0, 57.0, rotation)
    rng = np.random.default_rng(2)
    ra = rng.uniform(150, 290, 500)
    dec = rng.uniform(25, 85, 500)
    x, y = proj.forward(ra, dec)
    ra2, dec2, valid = proj.inverse(x, y)
    assert valid.all()
    np.testing.assert_allclose(dec2, dec, atol=1e-8)
    np.testing.assert_allclose(np.mod(ra2 - ra + 180, 360) - 180, 0, atol=1e-8)


def test_zenithal_ra_increases_to_the_left():
    x, _ = Zenithal(200.0, 40.0).forward(
        np.array([199.0, 201.0]), np.array([40.0, 40.0])
    )
    assert x[1] < x[0]


def test_footprint_parts_separates_patch_and_strip():
    nside = 256
    patch = hp.query_disc(
        nside, hp.ang2vec(220, 55, lonlat=True), np.radians(10), nest=True
    )
    ra = np.mod(np.linspace(-30, 60, 4000), 360)
    strip = np.unique(
        np.concatenate(
            [
                hp.ang2pix(nside, ra, np.full_like(ra, d), lonlat=True, nest=True)
                for d in np.arange(30, 34, 0.2)
            ]
        )
    )
    parts = footprint_parts(np.concatenate([patch, strip]), nside)
    assert len(parts) == 2
    assert parts[0].sum() == len(patch)


def test_coarsen_sums_children():
    pixels = np.arange(64)
    out, sums = coarsen(pixels, {"n": np.ones(64)}, nside=4, nside_out=2)
    np.testing.assert_array_equal(out, np.arange(16))
    np.testing.assert_array_equal(sums["n"], np.full(16, 4))


def test_cosebis_grid_panel_and_blind_check(tmp_path):
    from sp_validation.dashboard.products import load_cosebis_grid
    from sp_validation.dashboard.scale_cuts import config_space

    edges = np.geomspace(1, 250, 21)
    n = 20
    tri = np.where(np.triu(np.ones((n, n))) > 0, 0.5, np.nan)
    path = tmp_path / "cosebis_grid_V.npz"
    np.savez(
        path,
        left_edges=edges[:-1],
        right_edges=edges[1:],
        pte_B_n6=tri,
        pte_B_n20=tri,
        blind=np.array("unions"),
    )
    with pytest.raises(ValueError, match="another blind"):
        load_cosebis_grid(path, "other")
    cg = load_cosebis_grid(path, "unions")
    pe = {
        "left_edges": edges[:-1],
        "right_edges": edges[1:],
        "pte_matrices": {"xip_B": tri, "xim_B": tri},
    }
    for layout in ("wide", "narrow"):
        out = tmp_path / f"ptes_{layout}.png"
        config_space(pe, cg, out, (12, 83), layout=layout)
        assert out.exists()
