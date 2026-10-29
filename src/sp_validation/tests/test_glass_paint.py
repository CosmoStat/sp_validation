"""Low-resolution painting checks, including cosmo_val's own NaMaster path."""

import json

import h5py
import healpy as hp
import numpy as np
import pytest

from sp_validation.glass_mock import (
    convergence_to_shear,
    draw_b_mode_maps,
    paint_catalogue,
    paint_shear,
)


@pytest.fixture
def mock_maps():
    nside = 16
    npix = hp.nside2npix(nside)
    return np.zeros(npix), np.full(npix, 0.02), np.full(npix, -0.01)


def test_rotation_preserves_amplitude_and_reduced_shear_inverse(mock_maps):
    rng = np.random.default_rng(123)
    size = 1000
    e1, e2 = rng.normal(0, 0.2, (2, size))
    mock_maps[0][:] = 0.1
    result = paint_shear(np.zeros(size), np.zeros(size), e1, e2, mock_maps, rng)
    noise = result["e1_noise"] + 1j * result["e2_noise"]
    g = result["g1_true"] + 1j * result["g2_true"]
    e = result["e1"] + 1j * result["e2"]
    np.testing.assert_allclose(np.abs(noise), np.hypot(e1, e2), atol=1e-15)
    np.testing.assert_allclose(g, (0.02 + 0.01j) / 0.9)
    np.testing.assert_allclose((e - g) / (1 - np.conj(g) * e), noise, atol=1e-15)


def test_reproducible_and_injection(mock_maps):
    inputs = [np.zeros(100)] * 2 + [np.full(100, 0.2), np.full(100, 0.1)]
    first = paint_shear(*inputs, mock_maps, np.random.default_rng(6))
    second = paint_shear(*inputs, mock_maps, np.random.default_rng(6))
    third = paint_shear(
        *inputs,
        mock_maps,
        np.random.default_rng(7),
        psf=(np.full(100, 0.3), np.full(100, 0.4)),
        alpha=0.2,
        additive=(0.01, -0.02),
    )
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert not np.array_equal(first["e1"], third["e1"])
    np.testing.assert_allclose(third["e1_injected"], 0.07)
    np.testing.assert_allclose(third["e2_injected"], 0.06)


def test_zero_signal_noise_only(mock_maps):
    for m in mock_maps:
        m[:] = 0
    inputs = [np.zeros(100)] * 2 + [np.full(100, 0.2), np.full(100, 0.1)]
    result = paint_shear(*inputs, mock_maps, np.random.default_rng(9))
    for component in ("e1", "e2"):
        np.testing.assert_array_equal(result[component], result[component + "_noise"])


def test_streamed_catalogue_and_chunk_independence(tmp_path, mock_maps):
    from sp_validation.io import read_catalogue

    for name, values in zip(("kappa", "gamma1", "gamma2"), mock_maps, strict=True):
        np.save(tmp_path / f"{name}.npy", values)
    source = np.zeros(
        100,
        dtype=[
            (name, "f8")
            for name in (
                "RA",
                "Dec",
                "e1",
                "e2",
                "w_des",
                "e1_PSF",
                "e2_PSF",
                "e1_leak_corrected",
                "e2_leak_corrected",
            )
        ],
    )
    source["e1"] = 0.2
    source["w_des"] = np.arange(100)
    source["e1_PSF"] = 0.03
    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as dst:
        dst["data"] = source
    for chunk in (17, 100):
        paint_catalogue(
            path, tmp_path / f"mock_{chunk}.h5", tmp_path, 123, chunk_size=chunk
        )
    first = read_catalogue(tmp_path / "mock_17.h5")
    second = read_catalogue(tmp_path / "mock_100.h5")
    np.testing.assert_array_equal(first, second)
    for key in ("RA", "Dec", "w_des", "e1_PSF", "e2_PSF"):
        np.testing.assert_array_equal(first[key], source[key])
    np.testing.assert_array_equal(first["e1_leak_corrected"], first["e1"])
    assert not np.array_equal(first["e1"], source["e1"])
    with h5py.File(tmp_path / "mock_17.h5") as src:
        assert src.attrs["is_mock"]
        assert json.loads(src.attrs["mock_metadata"])["rotation_seed"] == 123


def pure_e_paint(nside=32):
    """Sample a fixed non-axisymmetric pure-E field at every pixel centre."""
    lmax = 2 * nside - 1
    alm = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
    for ell, emm in ((12, 3), (19, 7), (24, 10), (30, 2)):
        alm[hp.Alm.getidx(lmax, ell, emm)] = (1 + 0.3j) * 0.01
    kappa = hp.alm2map(alm, nside, lmax=lmax)
    gamma1, gamma2 = convergence_to_shear(kappa, lmax)
    ra, dec = hp.pix2ang(nside, np.arange(len(kappa)), lonlat=True)
    # Kappa is zero only for the paint step: test gamma, not reduced shear.
    result = paint_shear(
        ra,
        dec,
        np.zeros(len(ra)),
        np.zeros(len(ra)),
        (np.zeros_like(kappa), gamma1, gamma2),
        np.random.default_rng(1),
    )
    return ra, dec, result, gamma1, gamma2


def test_sign_convention_through_pipeline_namaster():
    from sp_validation.pseudo_cl import get_pseudo_cls_map

    ra, dec, result, gamma1, gamma2 = pure_e_paint()
    nside = hp.npix2nside(len(ra))
    shear = result["e1"] + 1j * result["e2"]
    ell, cls, _ = get_pseudo_cls_map(
        shear, np.ones(len(ra)), nside, "linear", pol_factor=-1, ell_step=8
    )
    _, wrong, _ = get_pseudo_cls_map(
        shear, np.ones(len(ra)), nside, "linear", pol_factor=1, ell_step=8
    )
    assert np.sum(np.abs(cls[3])) / np.sum(cls[0]) < 1e-8
    assert np.sum(wrong[3]) / np.sum(cls[0]) > 0.1
    np.testing.assert_allclose(result["e1"], gamma1)
    np.testing.assert_allclose(result["e2"], -gamma2)
    assert np.all(ell > 0)


def test_b_injection_is_pure_b():
    from sp_validation.pseudo_cl import get_pseudo_cls_map

    nside, lmax = 16, 31
    cl = np.zeros(lmax + 1)
    cl[8:24] = 1e-7
    b1, b2 = draw_b_mode_maps(nside, cl, 58)
    same = draw_b_mode_maps(nside, cl, 58)
    np.testing.assert_array_equal(b1, same[0])
    _, measured, _ = get_pseudo_cls_map(
        b1 - 1j * b2, np.ones(len(b1)), nside, "linear", ell_step=8
    )
    assert np.sum(np.abs(measured[0])) / np.sum(measured[3]) < 1e-6
