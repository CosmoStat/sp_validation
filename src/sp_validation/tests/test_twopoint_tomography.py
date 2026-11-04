"""Round-trip through the same twopoint reader used by CSL's 2pt_like."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from sp_validation import sacc_io as sio


def _reader():
    path = Path(os.environ.get("CSL_DIR", "/opt/cosmosis-standard-library"))
    path /= "likelihood/2pt/twopoint.py"
    if not path.exists():
        pytest.skip("CSL twopoint reader is not installed")
    spec = importlib.util.spec_from_file_location("csl_twopoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("n_bins", [1, 4])
def test_twopoint_roundtrip(tmp_path, n_bins):
    z = np.linspace(0.01, 2.01, 101)
    nz = {i: (z, np.exp(-(((z - 0.3 * (i + 1)) / 0.2) ** 2))) for i in range(n_bins)}
    s = sio.new_sacc(nz)
    pairs = [(i, j) for i in range(n_bins) for j in range(i, n_bins)]
    theta, ell = np.geomspace(1, 250, 5), np.array([100.0, 300.0, 900.0])
    # Reverse insertion order so a converter assuming canonical SACC order fails.
    for k, pair in enumerate(reversed(pairs)):
        sio.add_xi(s, pair, theta, k + theta, -k - theta, grid="reporting")
        sio.add_pseudo_cl(
            s, pair, ell, k + ell, window_ells=ell, window_weights=np.eye(len(ell))
        )
        sio.add_xi(s, pair, theta * 0.01, theta * 99, theta * 99, grid="integration")
    n = len(s.mean)
    # Asymmetric row/column encoding detects every wrong permutation.
    cov = np.arange(n * n, dtype=float).reshape(n, n)
    s.add_covariance(cov)
    path = tmp_path / "two.fits"
    sio.sacc_to_twopoint_fits(s, path, n_bins=n_bins)
    reader = _reader()
    # The legacy file has separate ξ and Cℓ covariances. Read spectra without
    # assigning errors, and read each covariance with CSL's own covariance reader.
    two = reader.TwoPointFile.from_fits(path, covmat_name=None)
    harmonic = two
    from astropy.io import fits

    with fits.open(path) as hdus:
        xi_cov = reader.CovarianceMatrixInfo.from_fits(hdus["COVMAT"])
        cl_cov = reader.CovarianceMatrixInfo.from_fits(hdus["COVMAT_CELL"])

    def rows(dtype):
        return np.array(
            [
                k
                for pair in pairs
                for k, p in enumerate(s.data)
                if p.data_type == dtype
                and p.tracers == sio._pair(pair)
                and p.tags["grid"] == "reporting"
            ]
        )

    for name, dtype, product, angle in [
        ("XI_PLUS", sio.XI_PLUS, two, theta),
        ("XI_MINUS", sio.XI_MINUS, two, theta),
        ("CELL_EE", sio.CL_EE, harmonic, ell),
    ]:
        spectrum = product.get_spectrum(name)
        np.testing.assert_array_equal(spectrum.value, s.mean[rows(dtype)])
        np.testing.assert_array_equal(
            spectrum.bin1, np.repeat([i + 1 for i, j in pairs], len(angle))
        )
        np.testing.assert_array_equal(
            spectrum.bin2, np.repeat([j + 1 for i, j in pairs], len(angle))
        )
        np.testing.assert_array_equal(
            spectrum.angular_bin, np.tile(np.arange(1, len(angle) + 1), len(pairs))
        )
        np.testing.assert_array_equal(spectrum.angle, np.tile(angle, len(pairs)))
    xi_rows = np.concatenate([rows(sio.XI_PLUS), rows(sio.XI_MINUS)])
    np.testing.assert_array_equal(xi_cov.covmat, cov[np.ix_(xi_rows, xi_rows)])
    cl_rows = rows(sio.CL_EE)
    np.testing.assert_array_equal(cl_cov.covmat, cov[np.ix_(cl_rows, cl_rows)])
    kernel = two.get_kernel("NZ_SOURCE")
    np.testing.assert_array_equal(kernel.z, z)
    for i in range(n_bins):
        np.testing.assert_array_equal(kernel.nzs[i], nz[i][1])
