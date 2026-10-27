"""Geometry and binning helpers behind the dashboard figures."""

import healpy as hp
import numpy as np
import pandas as pd
import pytest

from sp_validation.dashboard.field_maps import weight_cell_index, weight_grid_edges
from sp_validation.dashboard.sky import Zenithal, coarsen, footprint_parts, quantity


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
