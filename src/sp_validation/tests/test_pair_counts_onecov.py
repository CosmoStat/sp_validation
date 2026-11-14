"""Opt-in probe of the external OneCovariance table reader and noise kernel.

Run with ONECOVARIANCE_PATH pointing to the checkout used by the workflow.
Only the cosmological Fourier terms are stubbed; the real table reader and
real-space noise contractions execute from that checkout.
"""

import os

import numpy as np
import pytest
from astropy.io import fits

from sp_validation.pair_counts import measure_pair_counts


@pytest.mark.skipif(
    not os.environ.get("ONECOVARIANCE_PATH"),
    reason="Set ONECOVARIANCE_PATH to the workflow's external checkout",
)
def test_onecov_tomographic_noise_kernel(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(os.environ["ONECOVARIANCE_PATH"])
    from onecov.cov_input import FileInput
    from onecov.cov_theta_space import CovTHETASpace

    cat = tmp_path / "positions.fits"
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ids = np.array([1, 1, 1, 2, 2, 2])
    fits.BinTableHDU.from_columns(
        [
            fits.Column(name="RA", format="D", array=np.arange(6) * 0.01),
            fits.Column(name="Dec", format="D", array=np.zeros(6)),
            fits.Column(name="w", format="D", array=weights),
            fits.Column(name="bin", format="J", array=ids),
        ]
    ).writeto(cat)
    pairs = [(1, 1), (1, 2), (2, 2)]
    files, expected = [], {}
    s = np.array([0.23, 0.37])
    for i, j in pairs:
        out = tmp_path / f"pairs_{i}_{j}.npz"
        measure_pair_counts(
            cat,
            out,
            min_sep=0.1,
            max_sep=10,
            nbins=1,
            tomo_bin_col="bin",
            bin_pair=(i, j),
        )
        files.append(out.with_suffix(".txt").name)
        products = np.array(
            [
                weights[a] * weights[b]
                for a in range(6)
                for b in range(6)
                if ids[a] == i and ids[b] == j and (i != j or a < b)
            ]
        )
        expected[i - 1, j - 1] = (
            2
            * s[i - 1] ** 2
            * s[j - 1] ** 2
            * (products**2).sum()
            / products.sum() ** 2
        )

    reader = FileInput({})
    reader.clustering, reader.ggl, reader.cosmicshear = False, False, True
    reader.n_tomo_lens = 2
    reader._FileInput__get_npair_tabs(
        {
            "tabulated inputs files": {
                "npair_directory": str(tmp_path),
                "npair_mm_file": ", ".join(files),
            }
        }
    )
    assert reader.npair_mm.shape == (1, 2, 2, 1)

    cov = CovTHETASpace.__new__(CovTHETASpace)
    cov.cov_dict = {"gauss": True, "split_gauss": True, "sn_only": True}
    cov.csmf, cov.gg, cov.gm, cov.mm = False, False, False, True
    cov.n_tomo_lens = 2
    cov.sample_dim = 1
    cov.xi_pp, cov.xi_pm, cov.xi_mm = True, True, True
    cov.theta_bins_lensing = np.array([1.0])
    cov.ellrange = np.array([10.0])
    cov.npair_mm = reader.npair_mm.transpose(0, 3, 1, 2)
    zero = np.zeros((1, 1, 1, 1, 2, 2, 2, 2))
    cov.covELL_gaussian = lambda *args: [zero] * 18
    parts = cov._CovTHETASpace__covTHETA_split_gaussian(
        {}, {"ellipticity_dispersion": s}, False
    )
    for offset in (23, 29):
        noise = parts[offset][0, 0, 0, 0]
        for (i, j), value in expected.items():
            np.testing.assert_allclose(noise[i, j, i, j], value, rtol=1e-12)
            for k, ell in expected:
                if (i, j) != (k, ell):
                    assert noise[i, j, k, ell] == 0
    assert parts[26] == 0  # Pure-noise xi+ / xi- cross covariance vanishes.
