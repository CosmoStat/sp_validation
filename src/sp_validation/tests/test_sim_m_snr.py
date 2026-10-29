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


def test_effective_m_total_response_and_covariance():
    product = {
        "edges": [10, 15, 20],
        "value": [[0.1, 0, 0.1, 0], [0, 0, 0, 0]],
        "covariance": np.diag(np.arange(1.0, 9.0)),
    }
    # Equal W and R_shear=0.5; selection response is 0.5 in the first bin.
    responses = np.array([np.eye(2), np.eye(2) * 0.5])
    m, cov, share = sim_m_snr.effective_m(product, [1, 1], responses)
    np.testing.assert_allclose(share, [[2 / 3, 2 / 3], [1 / 3, 1 / 3]])
    np.testing.assert_allclose(m * 1000, [66.6666666667, 66.6666666667])
    np.testing.assert_allclose(np.diag(cov), [1.0, 19 / 9])
    with pytest.raises(ValueError, match="Expected bin weights"):
        sim_m_snr.effective_m(product, [1], responses)
    with pytest.raises(ValueError, match="nonnegative"):
        sim_m_snr.effective_m(product, [-1, 1], responses)


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
