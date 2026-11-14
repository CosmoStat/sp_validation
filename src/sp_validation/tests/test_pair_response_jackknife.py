"""Stage-1 paired/global covariance, checked against explicit pair sums."""

import numpy as np
import pytest

from sp_validation.response_diagnostics import scalar_pair_response


@pytest.mark.parametrize("varying", [False, True])
def test_scalar_response_shared_patch_deletions(varying):
    ra = np.arange(12, dtype=float) * 0.01
    dec = np.zeros(12)
    w = np.arange(12, dtype=float) + 1
    r = np.linspace(0.5, 1.1, 12) if varying else np.full(12, 0.8)
    patch = np.repeat(np.arange(4), 3)
    measured = scalar_pair_response(
        ra,
        dec,
        w,
        r,
        patch=patch,
        min_sep=0.1,
        max_sep=10,
        nbins=1,
        bin_slop=0,
    )

    def brute(keep):
        ii, jj = np.triu_indices(len(ra), 1)
        selected = keep[ii] & keep[jj]
        ww = w[ii[selected]] * w[jj[selected]]
        pair = np.average(r[ii[selected]] * r[jj[selected]], weights=ww)
        return pair / np.average(r[keep], weights=w[keep]) ** 2 - 1

    expected = np.array([brute(patch != j) for j in range(4)])
    # TreeCorr scalar fields are stored at float32 precision.
    np.testing.assert_allclose(
        measured["fraction"], brute(np.ones(12, bool)), atol=1e-7
    )
    np.testing.assert_allclose(measured["deleted"][:, 0], expected, atol=1e-7)
    np.testing.assert_allclose(
        measured["error"][0],
        np.sqrt(0.75 * np.sum((expected - expected.mean()) ** 2)),
        atol=1e-7,
    )
    if not varying:
        np.testing.assert_allclose(measured["fraction"], 0, atol=1e-7)
        np.testing.assert_allclose(measured["error"], 0, atol=1e-7)
