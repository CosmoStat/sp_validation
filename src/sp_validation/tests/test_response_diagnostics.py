"""Brute-force checks of the explicitly shear-only #23 stage-1 diagnostic."""

import numpy as np

from sp_validation.response_diagnostics import pair_products


def test_all_pair_tensor_products_against_brute_force():
    ra = np.array([0.0, 0.03, 0.07, 0.13, 0.21, 0.32])
    dec = np.zeros(6)
    w = np.array([1.0, 4.0, 2.0, 3.0, 1.0, 2.0])
    r = np.array(
        [
            np.linspace(0.6, 1.1, 6),
            np.linspace(-0.1, 0.2, 6),
            np.linspace(0.2, -0.05, 6),
            np.linspace(0.8, 1.2, 6),
        ]
    )
    result = pair_products(
        ra, dec, w, r.reshape(2, 2, 6), min_sep=1, max_sep=30, nbins=3
    )
    sums = np.zeros((3, 4, 4))
    den = np.zeros(3)
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            theta = abs(ra[i] - ra[j]) * 60
            if not 1 <= theta < 30:
                continue
            k = int(np.log(theta) / (np.log(30) / 3))
            sums[k] += w[i] * w[j] * np.outer(r[:, i], r[:, j])
            den[k] += w[i] * w[j]
    np.testing.assert_allclose(
        result["products"], sums / den[:, None, None], rtol=2e-7, atol=1e-9
    )
    # Spatial variation is measurable; the global product need not suffice.
    assert (
        np.max(abs(result["products"][:, 0, 0] / result["global_products"][0, 0] - 1))
        > 0.02
    )


def test_constant_full_matrix_has_exact_pair_product():
    matrix = np.array([[0.8, 0.1], [-0.02, 0.9]])
    rg = np.repeat(matrix[:, :, None], 6, axis=2)
    result = pair_products(
        np.arange(6) * 0.05,
        np.zeros(6),
        np.arange(6) + 1.0,
        rg,
        min_sep=1,
        max_sep=30,
        nbins=3,
    )
    expected = np.outer(matrix.ravel(), matrix.ravel())
    np.testing.assert_allclose(
        result["products"], np.broadcast_to(expected, (3, 4, 4)), rtol=2e-7, atol=1e-9
    )
