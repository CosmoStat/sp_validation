"""Minimum-count DES moments and identical full/deleted training rules."""

import numpy as np
import pytest
from astropy.io import fits

from sp_validation import calibration, response
from sp_validation.tests.test_consistent_response import catalogue


def sparse_fixture():
    # Two nearly round objects would otherwise get W=640000.
    snr = np.r_[np.repeat(20.0, 40), [100.0, 100.0]]
    size = np.ones(len(snr))
    e = np.full((2, len(snr)), 0.2)
    e[:, -2:] = 0.001
    r = np.full(len(snr), 0.8)
    edges = np.array([10.0, 50.0, 500.0]), np.array([0.5, 2.0, 3.0])
    return snr, size, e, r, *edges


def test_sparse_cell_pools_moments_not_weights():
    args = sparse_fixture()
    raw = response.fit_w(*args)
    pooled = response.fit_w(*args, min_count=3)
    assert raw[2][1, 0] > 600000
    expected = 0.8**2 / np.mean(args[2] ** 2)
    np.testing.assert_allclose(pooled[2][1, 0], expected)
    assert pooled[2][1, 0] < 17
    # Dense cell is bitwise unchanged; training counts stay unmerged.
    assert pooled[2][0, 0] == raw[2][0, 0]
    np.testing.assert_array_equal(pooled[3], raw[3])
    # A newly occupied artificial-branch cell uses the same pooled table.
    np.testing.assert_allclose(response.evaluate_w(pooled, [100.0], [2.5]), expected)


def test_sparse_deletion_is_deterministic_and_order_independent():
    snr, size, e, r, es, er = sparse_fixture()
    for deleted in (0, 40, 41):
        keep = np.arange(len(snr)) != deleted
        args = (snr[keep], size[keep], e[:, keep], r[keep], es, er)
        first = response.fit_w(*args, min_count=30)
        second = response.fit_w(*args, min_count=30)
        reverse = response.fit_w(
            snr[keep][::-1],
            size[keep][::-1],
            e[:, keep][:, ::-1],
            r[keep][::-1],
            es,
            er,
            min_count=30,
        )
        np.testing.assert_array_equal(first[2], second[2])
        np.testing.assert_allclose(first[2], reverse[2], rtol=1e-14)
        assert first[2][1, 0] < 17


def test_fully_populated_table_is_bitwise_unchanged():
    snr = np.repeat([20.0, 20.0, 100.0, 100.0], 30)
    size = np.repeat([1.0, 2.5, 1.0, 2.5], 30)
    e = np.array([np.linspace(0.1, 0.4, len(snr))] * 2)
    args = (
        snr,
        size,
        e,
        np.full(len(snr), 0.8),
        np.array([10.0, 50.0, 500.0]),
        np.array([0.5, 2.0, 3.0]),
    )
    for actual, expected in zip(
        response.fit_w(*args, min_count=30), response.fit_w(*args), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_neighbourhood_grows_until_threshold_and_handles_small_total():
    snr = np.r_[15.0, np.repeat(150.0, 10), np.repeat(1500.0, 40)]
    size = np.ones(len(snr))
    e = np.array([np.r_[0.001, np.repeat(0.1, 10), np.repeat(0.3, 40)]] * 2)
    args = (
        snr,
        size,
        e,
        np.ones(len(snr)),
        np.array([10.0, 100.0, 1000.0, 10000.0]),
        np.array([0.5, 2.0]),
    )
    near = response.fit_w(*args, min_count=10)
    far = response.fit_w(*args, min_count=30)
    np.testing.assert_allclose(near[2][0, 0], 1 / np.mean(e[:, :11] ** 2))
    np.testing.assert_allclose(far[2][0, 0], 1 / np.mean(e**2))
    small = response.fit_w(*args, min_count=100)
    np.testing.assert_allclose(small[2], 1 / np.mean(e**2))
    with pytest.raises(ValueError, match="nonnegative"):
        response.fit_w(*args, min_count=-1)


def test_production_refit_uses_same_minimum_count_rule(tmp_path):
    d, _, _ = catalogue()
    gm = calibration.metacal(
        d,
        np.ones(len(d), bool),
        weight_scheme="des",
        size_corr_ell=False,
        des_num_bins=2,
        des_min_count=30,
    )
    keep = np.arange(len(d)) % 4 != 0
    refitted = gm.refit_weights(keep)
    rebuilt = calibration.metacal(
        d[keep],
        np.ones(keep.sum(), bool),
        weight_scheme="des",
        size_corr_ell=False,
        des_num_bins=2,
        des_min_count=30,
    )
    for b in response.BRANCHES:
        np.testing.assert_allclose(refitted[b][keep], rebuilt.branch_weights[b])
    path = tmp_path / "response.fits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(path)
    calibration.save_response_product(path, gm)
    with fits.open(path) as hdus:
        assert hdus[0].header["DESNMIN"] == 30
        assert "pooled moments" in hdus[0].header["DESMERGE"]
        np.testing.assert_array_equal(hdus["W_COUNT"].data, gm.W[3])
