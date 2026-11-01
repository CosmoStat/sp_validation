"""Branch IV and refitted-table regression tests."""

import numpy as np
import pytest

from sp_validation import calibration, response
from sp_validation.tests.test_consistent_response import catalogue


def test_extraction_carries_available_branch_errors_but_allows_historical_des():
    data, _, _ = catalogue()
    carried = response.branch_error_columns(data)
    assert len(carried) == 8
    for key, value in carried.items():
        np.testing.assert_array_equal(value, data[key])
    historical = data[[k for k in data.dtype.names if k not in carried]]
    assert response.branch_error_columns(historical) == {}
    # Requiring these columns for DES would break existing merged inputs.
    calibration.metacal(
        historical,
        np.ones(len(historical), bool),
        weight_scheme="des",
        size_corr_ell=False,
        des_num_bins=2,
    )


def test_iv_branch_variances_and_normalization():
    data, _, _ = catalogue()
    data["NGMIX_G1_ERR_1P"] = np.linspace(0.1, 0.8, len(data))
    data["NGMIX_G2_ERR_1M"] = np.linspace(0.8, 0.1, len(data))
    gm = calibration.metacal(
        data, np.ones(len(data), bool), weight_scheme="iv", size_corr_ell=False
    )
    for b, suffix in (("p1", "1P"), ("m1", "1M")):
        expected = 1 / (
            2 * 0.34**2
            + data[f"NGMIX_G1_ERR_{suffix}"] ** 2
            + data[f"NGMIX_G2_ERR_{suffix}"] ** 2
        )
        np.testing.assert_allclose(gm.branch_weights[b], expected)
    expected = (
        np.average(gm.branches["ns"], weights=gm.branch_weights["p1"], axis=1)
        - np.average(gm.branches["ns"], weights=gm.branch_weights["m1"], axis=1)
    ) / 0.02
    np.testing.assert_allclose(gm.R_selection[:, 0], expected)
    assert abs(expected[0]) > 0.01


def test_refit_w_jackknife_matches_retained_training_sample():
    data, _, _ = catalogue()
    # Unequal group noise at the same SNR changes fitted W on deletion.
    for b in ("NOSHEAR", "1P", "1M", "2P", "2M"):
        data[f"NGMIX_G1_{b}"][:20] += 0.3
    gm = calibration.metacal(
        data,
        np.ones(len(data), bool),
        weight_scheme="des",
        size_corr_ell=False,
        des_num_bins=2,
    )
    groups = np.repeat(np.arange(4), 20)
    labels, jk, cov = gm.response_jackknife(groups, refit_w=True)
    for j, label in enumerate(labels):
        keep = groups != label
        retained = data[keep]
        other = calibration.metacal(
            retained,
            np.ones(len(retained), bool),
            weight_scheme="des",
            size_corr_ell=False,
            des_num_bins=2,
        )
        np.testing.assert_allclose(jk["c_g"][j], other.c_g, atol=1e-13)
        np.testing.assert_allclose(jk["total"][j], other.R, atol=1e-13)
    fixed = gm.response_jackknife(groups)[1]
    assert np.max(abs(jk["c_g"] - fixed["c_g"])) > 1e-3
    vectors = np.concatenate([x.reshape(len(labels), -1) for x in jk.values()], axis=1)
    delta = vectors - vectors.mean(axis=0)
    np.testing.assert_allclose(cov, 0.75 * delta.T @ delta)
    weights = gm.refit_weights(np.ones(len(data), bool))
    for b in response.BRANCHES:
        np.testing.assert_allclose(weights[b], gm.branch_weights[b])


def test_iv_missing_branch_errors_requires_reextraction():
    data, _, _ = catalogue()
    data = data[[k for k in data.dtype.names if k != "NGMIX_G1_ERR_1P"]]
    with pytest.raises(ValueError, match="regenerate.*comprehensive"):
        calibration.metacal(
            data, np.ones(len(data), bool), weight_scheme="iv", size_corr_ell=False
        )
