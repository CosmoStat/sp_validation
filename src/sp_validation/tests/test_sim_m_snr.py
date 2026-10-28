"""SNR-bin selections, covariance propagation and product persistence."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from sp_validation import sim_m_snr


def test_branch_specific_selection():
    gm = SimpleNamespace(mask_dict={b: np.arange(3) for b in ("ns", "p1", "m1")})
    gm.ns = {"flux": np.array([14.0, 15.0, 16.0]), "flux_err": np.ones(3)}
    gm.p1 = {"flux": np.array([16.0, 15.0, 14.0]), "flux_err": np.ones(3)}
    gm.m1 = {"flux": np.array([14.0, 16.0, 16.0]), "flux_err": np.ones(3)}
    masks = sim_m_snr.snr_masks(gm, 10, 15)
    np.testing.assert_array_equal(masks["ns"], [0, 1])
    np.testing.assert_array_equal(masks["p1"], [1, 2])
    np.testing.assert_array_equal(masks["m1"], [0])


def test_effective_m_and_covariance():
    product = {
        "edges": [10, 15, 20],
        "value": [[1, 0, 2, 0], [3, 0, 6, 0]],
        "covariance": np.diag(np.arange(1.0, 9.0)),
    }
    m, cov, share = sim_m_snr.effective_m(
        product,
        np.array([12, 15, 18]),
        np.array([1, 2, 1]),
        np.array([[1, 2], [1, 2], [3, 1]]),
    )
    np.testing.assert_allclose(share, [[0.5, 6 / 7], [0.5, 1 / 7]])
    np.testing.assert_allclose(m, [2, 18 / 7])
    hm, hc, hs = sim_m_snr.effective_m(product, histogram=[[3, 6], [3, 1]])
    np.testing.assert_allclose(hm, m)
    np.testing.assert_allclose(hc, cov)
    np.testing.assert_allclose(hs, share)
    np.testing.assert_allclose(np.diag(cov), [1.5, (36 * 3 + 7) / 49])
    with pytest.raises(ValueError, match="covered"):
        sim_m_snr.effective_m(product, [9], [1], [1])


def test_flat_m_bins_sum_to_full_numerator():
    # In a flat-m toy, summing each bin's response-weighted numerator
    # exactly recovers the full-population numerator in either component.
    weights = np.array([1.0, 4.0, 2.0, 3.0])
    r = np.array([[0.3, 0.7], [0.6, 0.5], [0.8, 0.9], [1.0, 0.4]])
    m = np.array([0.02, -0.01])
    product = {
        "edges": [10, 15, 20],
        "value": [[m[0], 0, m[1], 0]] * 2,
        "covariance": np.eye(8),
    }
    result, _, shares = sim_m_snr.effective_m(product, [11, 14, 17, 19], weights, r)
    np.testing.assert_allclose(result, m)
    total_response = np.sum(weights[:, None] * r, axis=0)
    np.testing.assert_allclose(
        (shares * m).sum(axis=0) * total_response,
        np.sum(weights[:, None] * r * m, axis=0),
    )


def test_grid_combination_and_product_roundtrip(tmp_path):
    products = [
        dict(
            edges=np.array([10, 15, 20]),
            value=np.full((2, 4), v),
            covariance=np.eye(8) * var,
        )
        for v, var in [(1.0, 1.0), (3.0, 4.0)]
    ]
    result = sim_m_snr.combine_grids(products)
    np.testing.assert_allclose(result["value"], 1.4)
    np.testing.assert_allclose(result["covariance"], np.eye(8) * 0.8)
    path = tmp_path / "snr.fits"
    sim_m_snr.write_product(path, result, {"name": "data", "path": "/table"})
    recovered = sim_m_snr.read_product(path)
    np.testing.assert_allclose(recovered["value"], result["value"])
    np.testing.assert_allclose(recovered["covariance"], result["covariance"])
    with fits.open(path) as hdus:
        assert hdus[0].header["WPRODUCT"] == "data"
        assert hdus[0].header["WGHTSCH"] == "des"
