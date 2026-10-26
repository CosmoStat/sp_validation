"""Branch IV and refitted-table regression tests."""

import numpy as np
import pytest

from sp_validation import calibration
from sp_validation.tests.test_consistent_response import catalogue


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


def test_iv_missing_branch_errors_requires_reextraction():
    data, _, _ = catalogue()
    data = data[[k for k in data.dtype.names if k != "NGMIX_G1_ERR_1P"]]
    with pytest.raises(ValueError, match="regenerate.*comprehensive"):
        calibration.metacal(
            data, np.ones(len(data), bool), weight_scheme="iv", size_corr_ell=False
        )
