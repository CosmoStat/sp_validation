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
    result = paint_shear(
        np.zeros(size), np.zeros(size), e1, e2, mock_maps, rng, shear="reduced"
    )
    noise = result["e1_noise"] + 1j * result["e2_noise"]
    g = result["g1_true"] + 1j * result["g2_true"]
    e = result["e1"] + 1j * result["e2"]
    np.testing.assert_allclose(np.abs(noise), np.hypot(e1, e2), atol=1e-15)
    np.testing.assert_allclose(g, (0.02 + 0.01j) / 0.9)
    np.testing.assert_allclose((e - g) / (1 - np.conj(g) * e), noise, atol=1e-15)


def test_gamma_default_and_reduced_switch(mock_maps):
    mock_maps[0][:] = 0.2
    inputs = [np.zeros(10)] * 4
    gamma = paint_shear(*inputs, mock_maps, np.random.default_rng(1))
    reduced = paint_shear(*inputs, mock_maps, np.random.default_rng(1), shear="reduced")
    np.testing.assert_allclose(gamma["g1_true"], 0.02)
    np.testing.assert_allclose(reduced["g1_true"], 0.02 / 0.8)
    np.testing.assert_allclose(gamma["g2_true"], 0.01)
    with pytest.raises(ValueError, match="shear must"):
        paint_shear(*inputs, mock_maps, np.random.default_rng(1), shear="invalid")


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


@pytest.mark.parametrize("layout", ["hdf5", "chunked_hdf5", "columnar_hdf5", "fits"])
def test_streamed_catalogue_and_chunk_independence(tmp_path, mock_maps, layout):
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
    path = tmp_path / ("input.fits" if layout == "fits" else "input.h5")
    if layout == "fits":
        from astropy.io import fits

        fits.BinTableHDU(source).writeto(path)
    else:
        with h5py.File(path, "w") as dst:
            if layout == "columnar_hdf5":
                for name in source.dtype.names:
                    dst[name] = source[name]
            elif layout == "chunked_hdf5":
                group = dst.create_group("tiles")
                group["00001"] = source[:37]
                group["00002"] = source[37:]
            else:
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
        metadata = json.loads(src.attrs["mock_metadata"])
        assert metadata["rotation_seed"] == 123
        assert metadata["shear"] == "gamma"


def test_paint_injections_share_noise_and_save_realized_b_truth(tmp_path, mock_maps):
    from sp_validation.io import read_catalogue

    for name, values in zip(("kappa", "gamma1", "gamma2"), mock_maps, strict=True):
        np.save(tmp_path / f"{name}.npy", values)
    source = np.zeros(
        100,
        dtype=[
            (name, "f8")
            for name in ("RA", "Dec", "w_des", "e1", "e2", "e1_PSF", "e2_PSF")
        ],
    )
    source["e1"], source["e2"], source["w_des"] = 0.2, 0.1, 1.0
    source["e1_PSF"], source["e2_PSF"] = 0.03, -0.02
    path = tmp_path / "source.h5"
    with h5py.File(path, "w") as dst:
        dst["data"] = source
    cl = np.zeros(32)
    cl[8:24] = 1e-7
    for tag, injection in (
        ("clean", {}),
        ("psf", {"alpha": 0.1}),
        ("b", {"b_cl": cl, "b_seed": 401}),
    ):
        paint_catalogue(
            path, tmp_path / f"{tag}.h5", tmp_path, 1, chunk_size=17, **injection
        )
    clean, psf, b = [
        read_catalogue(tmp_path / f"{tag}.h5") for tag in ("clean", "psf", "b")
    ]
    for component in (1, 2):
        name = f"e{component}"
        for injected in (psf, b):
            np.testing.assert_array_equal(
                clean[f"{name}_noise"], injected[f"{name}_noise"]
            )
            np.testing.assert_array_equal(
                clean[f"g{component}_true"], injected[f"g{component}_true"]
            )
        np.testing.assert_allclose(
            psf[name] - clean[name], 0.1 * source[f"{name}_PSF"], atol=1e-15
        )
        np.testing.assert_allclose(
            b[name] - clean[name], b[f"{name}_injected"], atol=1e-8
        )
    with np.load(tmp_path / "b.h5.bmode_spectra.npz") as spectra:
        np.testing.assert_array_equal(spectra["target_bb"], cl)
        assert int(spectra["seed"]) == 401
        assert (
            np.sum(spectra["realization_ee"]) / np.sum(spectra["realization_bb"]) < 1e-6
        )
    metadata = json.loads((tmp_path / "b.h5.json").read_text())
    assert metadata["b_realization_spectra"] == str(tmp_path / "b.h5.bmode_spectra.npz")
    expected_sigma = np.sqrt(0.5 * (source["e1"] ** 2 + source["e2"] ** 2).mean())
    assert metadata["shape_noise"]["noise"]["weight_power_2"][
        "isotropic_sigma"
    ] == pytest.approx(expected_sigma)


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


def test_parallel_solver_matches_glass_and_restores_threads(monkeypatch):
    import glass

    from sp_validation.glass_mock import solve_shell_spectra

    fields = [glass.grf.Lognormal(), glass.grf.Lognormal()]
    cl = 1e-5 / (1 + np.arange(17)) ** 2
    cls = [cl, cl * 2, cl * 0.4]
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    gaussian, _, info = solve_shell_spectra(fields, cls, workers=2)
    import os

    assert os.environ["OMP_NUM_THREADS"] == "4"
    for first, second in zip(
        gaussian, glass.solve_gaussian_spectra(fields, cls), strict=True
    ):
        np.testing.assert_array_equal(first, second)
    assert np.all(info > 0)


def test_projected_maps_seed_and_cache_reproducibility(tmp_path):
    from dataclasses import replace

    from sp_validation.glass_mock import GlassMockConfig, generate_lensing_maps

    nz = tmp_path / "nz.txt"
    np.savetxt(nz, np.column_stack((np.linspace(0.01, 0.39, 30), np.ones(30))))
    config = GlassMockConfig(
        nside=8,
        harmonic_lmax=16,
        seed=7,
        zmax=0.4,
        dx=500,
        nz_path=str(nz),
        limber=True,
    )
    first = generate_lensing_maps(config, tmp_path / "first", tmp_path / "cache")
    second = generate_lensing_maps(config, tmp_path / "second", tmp_path / "cache")
    third = generate_lensing_maps(
        replace(config, seed=8), tmp_path / "third", tmp_path / "cache"
    )
    assert first["spectra_cache"] == second["spectra_cache"] == third["spectra_cache"]
    for key in ("kappa", "gamma1", "gamma2"):
        a = np.load(tmp_path / "first" / f"{key}.npy")
        b = np.load(tmp_path / "second" / f"{key}.npy")
        c = np.load(tmp_path / "third" / f"{key}.npy")
        assert a.dtype == np.float32
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)
    with np.load(tmp_path / "first" / "spectra.npz") as spectra:
        assert spectra["realization_ee"].shape == (17,)
        assert np.all(spectra["cl_ee_pixel"][2:] > 0)


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


def test_painted_mock_carries_no_input_shape(tmp_path, mock_maps):
    from sp_validation.glass_mock import SHAPE_COLUMN
    from sp_validation.io import read_catalogue

    for name, values in zip(("kappa", "gamma1", "gamma2"), mock_maps, strict=True):
        np.save(tmp_path / f"{name}.npy", values)
    names = ("RA", "Dec", "w_des", "e1", "e2", "e1_PSF", "e2_PSF", "snr")
    shaped = ("e1_uncal", "e2_uncal", "R_g11", "R_g22", "e1_leak_corrected")
    source = np.zeros(100, dtype=[(name, "f8") for name in names + shaped])
    rng = np.random.default_rng(5)
    for name in ("e1", "e2", *shaped):
        source[name] = rng.uniform(-0.3, 0.3, len(source))
    source["w_des"], source["snr"] = 1.0, 20.0
    path = tmp_path / "source.h5"
    with h5py.File(path, "w") as dst:
        dst["data"] = source
    paint_catalogue(path, tmp_path / "mock.h5", tmp_path, 7, keep=("snr",))
    mock = read_catalogue(tmp_path / "mock.h5")
    assert not set(shaped[:-1]) & set(mock.dtype.names)
    assert {"RA", "Dec", "w_des", "e1_PSF", "e2_PSF", "snr"} <= set(mock.dtype.names)
    for column in mock.dtype.names:
        for name in ("e1", "e2", *shaped):
            assert not np.array_equal(mock[column], source[name]), (column, name)
    for name in ("e1_uncal", "NGMIX_G1_NOSHEAR", "NGMIX_ELL_PSFo", "R_g11"):
        assert SHAPE_COLUMN.search(name), name
    assert not SHAPE_COLUMN.search("snr") and not SHAPE_COLUMN.search("FLUX_RADIUS")
    with pytest.raises(ValueError, match="galaxy shapes"):
        paint_catalogue(
            path, tmp_path / "leak.h5", tmp_path, 7, keep=("e1_uncal",), overwrite=True
        )
