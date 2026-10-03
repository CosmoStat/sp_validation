"""Object-wise leakage alphas equal an independent two-pass weighted regression.
The invariant includes both shear components and original per-row output alignment.
"""

import numpy as np
import pytest
from scipy.linalg import lstsq

from sp_validation import calibration

pytestmark = [
    pytest.mark.fast,
    pytest.mark.decision("calibration.objectwise_leakage_correction"),
]


def _synthetic_catalogue():
    """Build the fixed 512-row heteroscedastic fixture from the assay plan."""
    rng = np.random.default_rng(47)
    n_rows = 512
    snr = np.exp(rng.uniform(np.log(10), np.log(200), n_rows))
    size_ratio = rng.uniform(0.2, 0.8, n_rows)
    e1_psf, e2_psf = rng.normal(0, 0.03, size=(2, n_rows))
    w_des = rng.uniform(0.5, 2.0, n_rows)
    e1 = (0.04 + 0.3 * size_ratio**3 + 0.4 * np.exp(-snr / 20)) * e1_psf + rng.normal(
        0, 0.005 * (1 + 40 / snr)
    )
    e2 = (-0.02 + 0.2 * size_ratio**2) * e2_psf + rng.normal(0, 0.01 * (1 + 20 / snr))
    return {
        "e1": e1,
        "e2": e2,
        "e1_PSF": e1_psf,
        "e2_PSF": e2_psf,
        "snr": snr,
        "w_des": w_des,
        "NGMIX_T_PSF_RECONV_NOSHEAR": np.ones(n_rows),
        "NGMIX_T_NOSHEAR": 1 / size_ratio - 1,
    }


def _weighted_slope_and_error(values, psf, weights):
    """Fit [1, PSF] by independent sqrt(weight)-scaled SciPy least squares."""
    design = np.column_stack((np.ones(values.size), psf))
    row_scale = np.sqrt(weights)
    scaled_design = design * row_scale[:, None]
    scaled_values = values * row_scale
    coefficients, _, rank, _ = lstsq(
        scaled_design, scaled_values, lapack_driver="gelsd"
    )
    assert rank == 2
    residual = values - design @ coefficients
    variance = np.sum(weights * residual**2) / (values.size - 2)
    covariance = np.linalg.inv(scaled_design.T @ scaled_design) * variance
    return coefficients[1], np.sqrt(covariance[1, 1])


def _trend_basis(snr, size_ratio):
    """Return the five-term SNR/size basis used by the production estimator."""
    inverse_snr_squared = snr**-2
    return np.column_stack(
        (
            np.ones(snr.size),
            inverse_snr_squared,
            snr**-3,
            size_ratio,
            size_ratio * inverse_snr_squared,
        )
    )


def _fit_trend_predictions(
    fit_snr, fit_size, slopes, errors, predict_snr, predict_size
):
    """Fit inverse-variance trend WLS with equilibrated columns and predict."""
    design = _trend_basis(fit_snr, fit_size)
    row_scale = 1 / errors
    scaled_design = design * row_scale[:, None]
    column_scale = np.linalg.norm(scaled_design, axis=0)
    equilibrated_design = scaled_design / column_scale
    scaled_slopes = slopes * row_scale
    scaled_coefficients, _, rank, _ = lstsq(
        equilibrated_design, scaled_slopes, lapack_driver="gelsd"
    )
    assert rank == design.shape[1], "trend design must have full column rank"
    coefficients = scaled_coefficients / column_scale
    return _trend_basis(predict_snr, predict_size) @ coefficients


def _independent_two_pass_reference(catalogue, num_bins=4):
    """Reconstruct both trend and residual corrections without pandas/statsmodels."""
    snr = catalogue["snr"]
    size_ratio = catalogue["NGMIX_T_PSF_RECONV_NOSHEAR"] / (
        catalogue["NGMIX_T_NOSHEAR"] + catalogue["NGMIX_T_PSF_RECONV_NOSHEAR"]
    )
    weights = catalogue["w_des"]
    n_rows = snr.size

    # Assign quantile cells by sorted row IDs, independently of pandas.qcut.
    size_cell = np.empty(n_rows, dtype=int)
    for cell, rows in enumerate(np.array_split(np.argsort(size_ratio), num_bins)):
        size_cell[rows] = cell
    snr_cell = np.empty(n_rows, dtype=int)
    for size_bin in range(num_bins):
        rows = np.flatnonzero(size_cell == size_bin)
        ordered_rows = rows[np.argsort(snr[rows])]
        for cell, cell_rows in enumerate(np.array_split(ordered_rows, num_bins)):
            snr_cell[cell_rows] = cell

    cell_rows = [
        np.flatnonzero((size_cell == size_bin) & (snr_cell == snr_bin))
        for size_bin in range(num_bins)
        for snr_bin in range(num_bins)
    ]
    assert len(cell_rows) == num_bins**2
    assert all(rows.size > 2 for rows in cell_rows)

    trend_features = []
    first_pass_slopes = [[], []]
    first_pass_errors = [[], []]
    for rows in cell_rows:
        trend_features.append(
            (
                np.average(snr[rows], weights=weights[rows]),
                np.average(size_ratio[rows], weights=weights[rows]),
            )
        )
        for component, psf_key in enumerate(("e1_PSF", "e2_PSF")):
            slope, error = _weighted_slope_and_error(
                catalogue[f"e{component + 1}"][rows],
                catalogue[psf_key][rows],
                weights[rows],
            )
            first_pass_slopes[component].append(slope)
            first_pass_errors[component].append(error)

    trend_features = np.asarray(trend_features)
    trend_snr, trend_size = trend_features.T
    trend_alpha = [
        _fit_trend_predictions(
            trend_snr,
            trend_size,
            np.asarray(first_pass_slopes[component]),
            np.asarray(first_pass_errors[component]),
            snr,
            size_ratio,
        )
        for component in range(2)
    ]

    corrected_values = [
        catalogue[f"e{component + 1}"]
        - trend_alpha[component] * catalogue[f"e{component + 1}_PSF"]
        for component in range(2)
    ]
    residual_slopes = [np.empty(num_bins**2), np.empty(num_bins**2)]
    for cell_index, rows in enumerate(cell_rows):
        for component, psf_key in enumerate(("e1_PSF", "e2_PSF")):
            residual_slopes[component][cell_index] = _weighted_slope_and_error(
                corrected_values[component][rows],
                catalogue[psf_key][rows],
                weights[rows],
            )[0]

    final_alpha = [np.empty(n_rows), np.empty(n_rows)]
    for cell_index, rows in enumerate(cell_rows):
        for component in range(2):
            final_alpha[component][rows] = (
                trend_alpha[component][rows] + residual_slopes[component][cell_index]
            )
    return final_alpha


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#383: the trend fit scales rows by "
        "1/SE^2 instead of 1/SE, so it minimizes a 1/SE^4 objective"
    ),
)
def test_object_leakage_two_pass_coefficients_match_independent_wls():
    """With rtol=1e-8 and atol=1e-7, the identity has statistical false-alarm bound 0."""
    catalogue = _synthetic_catalogue()
    expected_alpha1, expected_alpha2 = _independent_two_pass_reference(catalogue)

    reverse_rows = np.arange(catalogue["snr"].size - 1, -1, -1)
    reversed_catalogue = {
        key: values[reverse_rows] for key, values in catalogue.items()
    }
    actual_alpha1, actual_alpha2 = calibration.get_alpha_leakage_per_object(
        reversed_catalogue, num_bins=4
    )
    actual_alpha1 = np.asarray(actual_alpha1)[reverse_rows]
    actual_alpha2 = np.asarray(actual_alpha2)[reverse_rows]

    np.testing.assert_allclose(
        actual_alpha1,
        expected_alpha1,
        rtol=1e-8,
        atol=1e-7,
        err_msg="component-1 trend-plus-residual alpha differs from independent WLS",
    )
    np.testing.assert_allclose(
        actual_alpha2,
        expected_alpha2,
        rtol=1e-8,
        atol=1e-7,
        err_msg="component-2 trend-plus-residual alpha differs from independent WLS",
    )
