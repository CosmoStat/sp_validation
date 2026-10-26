"""A DES table fitted on the measured population suppresses its shear signal."""

from itertools import product

import numpy as np
import pytest

from sp_validation import response


@pytest.mark.parametrize("occupancy", [3, 4, 8])
def test_self_fitted_table_bias_is_inverse_cell_occupancy(occupancy):
    # Enumerate every Rademacher sample instead of drawing Monte Carlo trials.
    # Each sample is one cell; duplicate it for two independent training signs.
    signs = np.array(list(product((-1.0, 1.0), repeat=occupancy)))
    x = np.repeat(signs, 2, axis=0) * 0.25
    e = np.stack((x, np.full_like(x, 0.25)))
    independent = e.copy()
    independent[0] = np.tile([-0.25, 0.25], len(signs))[:, None]
    cells = len(x)
    snr = np.repeat(np.arange(cells) + 0.5, occupancy)
    size = np.ones(len(snr))
    edges_s = np.arange(cells + 1)
    edges_r = np.array([0.0, 2.0])
    step = 1e-5
    biases = []
    for training in (e, independent):
        means = []
        for shear in (-step, step):
            shifted = training.copy()
            shifted[0] += shear
            table = response.fit_w(
                snr,
                size,
                shifted.reshape(2, -1),
                np.ones(len(snr)),
                edges_s,
                edges_r,
                min_count=3,
            )
            weights = response.evaluate_w(table, snr, size)
            means.append(np.average((e[0] + shear).ravel(), weights=weights))
        biases.append((means[1] - means[0]) / (2 * step) - 1)
    assert biases[0] == pytest.approx(-1 / occupancy, abs=3e-9)
    assert biases[1] == pytest.approx(0, abs=3e-9)
