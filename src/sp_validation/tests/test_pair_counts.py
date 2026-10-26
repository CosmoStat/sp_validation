"""Weighted-pair normalization, selection, and shape-free I/O."""

import numpy as np
import pytest
from astropy.io import fits

from sp_validation.pair_counts import measure_pair_counts, ordered_effective_pairs


def test_uniform_limit_and_dispersion_convention():
    # Exact expectation for a uniform Poisson field, far from boundaries.
    # All unordered pairs have the same product weight, which must cancel.
    edges = np.geomspace(1, 83, 31)
    area, density, s, pair_weight = 2000 * 3600, 5.0, 0.33, 7.0**2
    unordered = area * density**2 * np.pi * np.diff(edges**2) / 2
    W = unordered * pair_weight
    Q = unordered * pair_weight**2
    ordered = ordered_effective_pairs(W, Q)
    np.testing.assert_allclose(ordered, 2 * unordered)
    np.testing.assert_allclose(4 * s**4 / ordered, 2 * s**4 * Q / W**2)
    sigma_total = np.sqrt(2) * s
    np.testing.assert_allclose(sigma_total**4 / (2 * unordered), 4 * s**4 / ordered)


@pytest.mark.parametrize(
    "W,Q",
    [
        ([0], [1]),
        ([1], [0]),
        ([1], [-1]),
        ([np.nan], [1]),
        ([1], [np.inf]),
        ([1, 2], [1]),
    ],
)
def test_invalid_counts_fail_closed(W, Q):
    with pytest.raises(ValueError):
        ordered_effective_pairs(W, Q)


def test_positions_weights_only_and_unordered_treecorr(tmp_path):
    # There are deliberately no shape or blind columns/records in this FITS.
    ra = np.array([0.0, 0.01, 0.03, 0.07])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    cat = tmp_path / "positions.fits"
    fits.BinTableHDU.from_columns(
        [
            fits.Column(name="RA", format="D", array=ra),
            fits.Column(name="Dec", format="D", array=np.zeros(4)),
            fits.Column(name="w", format="D", array=weights),
        ]
    ).writeto(cat)
    out = tmp_path / "pairs.npz"
    measure_pair_counts(cat, out, min_sep=0.1, max_sep=10.0, nbins=1)
    products = np.array(
        [weights[i] * weights[j] for i in range(4) for j in range(i + 1, 4)]
    )
    with np.load(out) as z:
        np.testing.assert_allclose(z["weight"], products.sum())
        np.testing.assert_allclose(z["weight_squared"], (products**2).sum())
        np.testing.assert_allclose(z["npairs"], 6)
        np.testing.assert_allclose(z["theta"], [1.0])
        np.testing.assert_allclose(
            np.loadtxt(out.with_suffix(".txt"))[1],
            2 * products.sum() ** 2 / (products**2).sum(),
        )
