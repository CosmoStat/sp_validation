"""Each occupied SNR/size cell's ``w_des`` is the squared mean diagonal response
divided by mean two-component shape noise.
Per-row weights stay in input order and scale as response squared or ellipticity
inverse-squared.
"""

import numpy as np
import numpy.testing as npt
import pytest

from sp_validation.calibration import get_w_des

pytestmark = [pytest.mark.fast, pytest.mark.decision("calibration.shape_weights")]


def test_gatti_weights_use_squared_response_and_both_shape_components():
    """At rtol=atol=1e-12, this identity has zero statistical
    false-alarm probability; the tolerance only allows floating-point error.
    """
    cell_snr = np.repeat([20.0, 20.0, 80.0, 80.0], 4)
    cell_size = np.repeat([1.0, 4.0, 1.0, 4.0], 4)
    response_scale = np.repeat([1.0, 1.5, 2.0, 2.5], 4)
    ellipticity_scale = np.repeat([1.0, 2.0, 3.0, 4.0], 4)
    # Unit-mean jitter: the cell-mean response is unchanged, but the mean of
    # squared per-row responses differs from the squared mean response.
    response_jitter = np.tile([0.9, 1.1, 0.95, 1.05], 4)

    catalogue = {
        "snr": cell_snr,
        "size_ratio": cell_size,
        # Nonzero cell mean of e1, so variance differs from mean square.
        "e1_uncal": np.tile([0.1, 0.1, 0.2, -0.2], 4) * ellipticity_scale,
        "e2_uncal": np.tile([0.3, -0.3, 0.1, -0.1], 4) * ellipticity_scale,
        "R_g11": 0.6 * response_scale * response_jitter,
        "R_g22": 0.8 * response_scale * response_jitter,
    }
    cell_weights = (196.0 / 15.0) * (response_scale / ellipticity_scale) ** 2
    permutation = np.random.default_rng(1447).permutation(16)
    shuffled_catalogue = {
        name: values[permutation] for name, values in catalogue.items()
    }

    weights = get_w_des(
        shuffled_catalogue,
        num_bins=2,
        snr_min=10.0,
        snr_max=160.0,
        size_ratio_min=0.5,
        size_ratio_max=8.0,
    )
    expected_shuffled = cell_weights[permutation]
    npt.assert_allclose(weights, expected_shuffled, rtol=1e-12, atol=1e-12)

    original_weights = get_w_des(
        catalogue,
        num_bins=2,
        snr_min=10.0,
        snr_max=160.0,
        size_ratio_min=0.5,
        size_ratio_max=8.0,
    )
    npt.assert_allclose(
        weights[np.argsort(permutation)], original_weights, rtol=1e-12, atol=1e-12
    )
    # First cell: shape noise = 0.0375, mean response = 0.7, hence w = 196/15.
    npt.assert_allclose(original_weights[:4], 196.0 / 15.0, rtol=1e-12, atol=1e-12)

    response_scaled = {
        name: values.copy() for name, values in shuffled_catalogue.items()
    }
    response_scaled["R_g11"] *= 1.25
    response_scaled["R_g22"] *= 1.25
    response_scaled_weights = get_w_des(
        response_scaled,
        num_bins=2,
        snr_min=10.0,
        snr_max=160.0,
        size_ratio_min=0.5,
        size_ratio_max=8.0,
    )
    npt.assert_allclose(
        response_scaled_weights,
        weights * 1.25**2,
        rtol=1e-12,
        atol=1e-12,
    )

    ellipticity_scaled = {
        name: values.copy() for name, values in shuffled_catalogue.items()
    }
    ellipticity_scaled["e1_uncal"] *= 1.5
    ellipticity_scaled["e2_uncal"] *= 1.5
    ellipticity_scaled_weights = get_w_des(
        ellipticity_scaled,
        num_bins=2,
        snr_min=10.0,
        snr_max=160.0,
        size_ratio_min=0.5,
        size_ratio_max=8.0,
    )
    npt.assert_allclose(
        ellipticity_scaled_weights,
        weights / 1.5**2,
        rtol=1e-12,
        atol=1e-12,
    )
