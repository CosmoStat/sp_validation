"""Low-resolution painting checks, including cosmo_val's own NaMaster path.

The sky is a convergence alm evaluated exactly at each galaxy; the exactness
and known-answer tests here fail for a nearest-pixel lookup of the maps.
"""

import json

import h5py
import healpy as hp
import numpy as np
import pytest

from sp_validation.glass_mock import (
    convergence_to_shear,
    draw_b_mode_alm,
    evaluate_alm,
    paint_catalogue,
    paint_shear,
    shear_alm,
)


@pytest.fixture
def mock_maps():
    """Constant (kappa, gamma1, gamma2) signal at 1000 galaxy positions."""
    size = 1000
    return np.zeros(size), np.full(size, 0.02), np.full(size, -0.01)


def gaussian_kappa_alm(lmax, seed=3, amplitude=1e-4):
    """Band-limited Gaussian convergence alm with a red, l>=2 spectrum."""
    ell, emm = hp.Alm.getlm(lmax)
    rng = np.random.default_rng(seed)
    alm = rng.normal(size=len(ell)) + 1j * rng.normal(size=len(ell))
    alm *= np.sqrt(amplitude / (1 + ell) ** 1.5 / 2)
    alm[emm == 0] = alm[emm == 0].real * np.sqrt(2)
    alm[ell < 2] = 0
    return alm


def write_sky(directory, nside=16, lmax=47, seed=3):
    """A ``generate_lensing_maps``-shaped sky directory, without GLASS."""
    alm = gaussian_kappa_alm(lmax, seed)
    np.savez(directory / "kappa_alm.npz", alm=alm, nside=nside)
    return alm


def test_rotation_preserves_amplitude_and_reduced_shear_inverse(mock_maps):
    rng = np.random.default_rng(123)
    e1, e2 = rng.normal(0, 0.2, (2, len(mock_maps[0])))
    mock_maps[0][:] = 0.1
    result = paint_shear(e1, e2, mock_maps, rng, shear="reduced")
    noise = result["e1_noise"] + 1j * result["e2_noise"]
    g = result["g1_true"] + 1j * result["g2_true"]
    e = result["e1"] + 1j * result["e2"]
    np.testing.assert_allclose(np.abs(noise), np.hypot(e1, e2), atol=1e-15)
    np.testing.assert_allclose(g, (0.02 + 0.01j) / 0.9)
    np.testing.assert_allclose((e - g) / (1 - np.conj(g) * e), noise, atol=1e-15)


def test_rotation_draws_one_uniform_per_row_in_order(mock_maps):
    e1, e2 = np.full(1000, 0.3), np.full(1000, -0.1)
    result = paint_shear(e1, e2, mock_maps, np.random.default_rng(11))
    angle = np.random.default_rng(11).uniform(0, np.pi, 1000)
    expected = np.hypot(e1, e2) * np.exp(2j * angle)
    np.testing.assert_allclose(result["e1_noise"], expected.real, atol=1e-15)
    np.testing.assert_allclose(result["e2_noise"], expected.imag, atol=1e-15)


def test_gamma_default_and_reduced_switch(mock_maps):
    mock_maps[0][:] = 0.2
    inputs = [np.zeros(1000)] * 2
    gamma = paint_shear(*inputs, mock_maps, np.random.default_rng(1))
    reduced = paint_shear(*inputs, mock_maps, np.random.default_rng(1), shear="reduced")
    np.testing.assert_allclose(gamma["g1_true"], 0.02)
    np.testing.assert_allclose(reduced["g1_true"], 0.02 / 0.8)
    np.testing.assert_allclose(gamma["g2_true"], 0.01)
    with pytest.raises(ValueError, match="shear must"):
        paint_shear(*inputs, mock_maps, np.random.default_rng(1), shear="invalid")


def test_reproducible_and_injection(mock_maps):
    inputs = [np.full(1000, 0.2), np.full(1000, 0.1)]
    first = paint_shear(*inputs, mock_maps, np.random.default_rng(6))
    second = paint_shear(*inputs, mock_maps, np.random.default_rng(6))
    third = paint_shear(
        *inputs,
        mock_maps,
        np.random.default_rng(7),
        psf=(np.full(1000, 0.3), np.full(1000, 0.4)),
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
    inputs = [np.full(1000, 0.2), np.full(1000, 0.1)]
    result = paint_shear(*inputs, mock_maps, np.random.default_rng(9))
    for component in ("e1", "e2"):
        np.testing.assert_array_equal(result[component], result[component + "_noise"])


@pytest.mark.parametrize("layout", ["hdf5", "chunked_hdf5", "columnar_hdf5", "fits"])
def test_streamed_catalogue_and_block_independence(tmp_path, layout):
    from sp_validation.io import read_catalogue

    write_sky(tmp_path)
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
    source["RA"] = np.linspace(0, 359, 100)
    source["Dec"] = np.linspace(-80, 80, 100)
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
    for chunk, block in ((17, 23), (100, 10**7)):
        paint_catalogue(
            path,
            tmp_path / f"mock_{chunk}.h5",
            tmp_path,
            123,
            chunk_size=chunk,
            block_size=block,
            nthreads=2,
        )
    first = read_catalogue(tmp_path / "mock_17.h5")
    second = read_catalogue(tmp_path / "mock_100.h5")
    for name in first.dtype.names:
        np.testing.assert_allclose(first[name], second[name], rtol=0, atol=1e-12)
    for key in ("RA", "Dec", "w_des", "e1_PSF", "e2_PSF"):
        np.testing.assert_array_equal(first[key], source[key])
    np.testing.assert_array_equal(first["e1_leak_corrected"], first["e1"])
    assert not np.array_equal(first["e1"], source["e1"])
    with h5py.File(tmp_path / "mock_17.h5") as src:
        assert src.attrs["is_mock"]
        metadata = json.loads(src.attrs["mock_metadata"])
        assert metadata["rotation_seed"] == 123
        assert metadata["shear"] == "gamma"
        assert metadata["painting"].startswith("exact alm evaluation")
        assert metadata["block_size"] == 23


def test_paint_injections_share_noise_and_save_realized_b_truth(tmp_path):
    from sp_validation.io import read_catalogue

    write_sky(tmp_path)
    source = np.zeros(
        100,
        dtype=[
            (name, "f8")
            for name in ("RA", "Dec", "w_des", "e1", "e2", "e1_PSF", "e2_PSF")
        ],
    )
    source["e1"], source["e2"], source["w_des"] = 0.2, 0.1, 1.0
    source["RA"] = np.random.default_rng(2).uniform(0, 360, 100)
    source["Dec"] = np.random.default_rng(3).uniform(-60, 60, 100)
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
    b_alm = draw_b_mode_alm(cl, 401)
    b1, b2 = evaluate_alm([np.zeros_like(b_alm), b_alm], source["RA"], source["Dec"])
    np.testing.assert_allclose(b["e1_injected"], b1, atol=1e-7)
    np.testing.assert_allclose(b["e2_injected"], -b2, atol=1e-7)
    with np.load(tmp_path / "b.h5.bmode_spectra.npz") as spectra:
        np.testing.assert_array_equal(spectra["target_bb"], cl)
        assert int(spectra["seed"]) == 401
        np.testing.assert_allclose(spectra["realization_bb"], hp.alm2cl(b_alm))
        assert np.all(spectra["realization_bb"][8:24] > 0)
        assert np.all(spectra["realization_bb"][24:] == 0)
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
    elm = shear_alm(alm, nside)
    gamma1, gamma2 = hp.alm2map_spin([elm, 0 * elm], nside, 2, lmax)
    ra, dec = hp.pix2ang(nside, np.arange(len(gamma1)), lonlat=True)
    # Kappa is zero only for the paint step: test gamma, not reduced shear.
    result = paint_shear(
        np.zeros(len(ra)),
        np.zeros(len(ra)),
        (np.zeros(len(ra)), *evaluate_alm([elm, 0 * elm], ra, dec)),
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
    np.testing.assert_allclose(result["e1"], gamma1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["e2"], -gamma2, rtol=0, atol=1e-12)
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
        realization_ee = spectra["realization_ee"]
    # The persisted alm is the sky: its spectrum is the known answer, and its
    # exact evaluation at pixel centres reproduces the float32 maps.
    with np.load(tmp_path / "first" / "kappa_alm.npz") as sky:
        alm, nside = sky["alm"], int(sky["nside"])
    elm = shear_alm(alm, nside)
    np.testing.assert_allclose(realization_ee, hp.alm2cl(elm))
    ra, dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    gamma = evaluate_alm([elm, 0 * elm], ra, dec)
    for index, name in enumerate(("gamma1", "gamma2")):
        stored = np.load(tmp_path / "first" / f"{name}.npy")
        np.testing.assert_allclose(gamma[index], stored, rtol=1e-6, atol=1e-9)
    # The kappa map is GLASS's pixel field; the alm is its transform to lmax.
    np.testing.assert_allclose(
        evaluate_alm(alm, ra, dec)[0], hp.alm2map(alm, nside), rtol=0, atol=1e-12
    )


def test_b_injection_is_pure_b():
    from sp_validation.pseudo_cl import get_pseudo_cls_map

    nside, lmax = 16, 31
    cl = np.zeros(lmax + 1)
    cl[8:24] = 1e-7
    alm = draw_b_mode_alm(cl, 58)
    np.testing.assert_array_equal(alm, draw_b_mode_alm(cl, 58))
    ra, dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    b1, b2 = evaluate_alm([np.zeros_like(alm), alm], ra, dec)
    _, measured, _ = get_pseudo_cls_map(
        b1 - 1j * b2, np.ones(len(b1)), nside, "linear", ell_step=8
    )
    assert np.sum(np.abs(measured[0])) / np.sum(measured[3]) < 1e-6


def test_painted_mock_carries_no_input_shape(tmp_path):
    from sp_validation.glass_mock import SHAPE_COLUMN
    from sp_validation.io import read_catalogue

    write_sky(tmp_path)
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


def test_exact_evaluation_off_pixel_centres():
    """Random positions against a direct spherical-harmonic sum (spin 0) and
    foreign-grid pixel centres against healpy's spin-2 synthesis."""
    from scipy.special import sph_harm_y

    nside, lmax = 32, 95
    alm = gaussian_kappa_alm(lmax)
    rng = np.random.default_rng(4)
    ra, dec = (
        rng.uniform(-180, 540, 300),
        np.degrees(np.arcsin(rng.uniform(-1, 1, 300))),
    )
    theta, phi = np.radians(90 - dec), np.radians(ra)
    ell, emm = hp.Alm.getlm(lmax)
    ylm = sph_harm_y(ell[:, None], emm[:, None], theta[None], phi[None])
    weight = np.where(emm == 0, 1.0, 2.0)[:, None]
    direct = np.sum(weight * (alm[:, None] * ylm).real, axis=0)
    np.testing.assert_allclose(evaluate_alm(alm, ra, dec)[0], direct, atol=1e-12)
    elm = shear_alm(alm, nside)
    for grid in (nside, 3 * nside):
        maps = hp.alm2map_spin([elm, 0 * elm], grid, 2, lmax)
        ra, dec = hp.pix2ang(grid, np.arange(hp.nside2npix(grid)), lonlat=True)
        exact = evaluate_alm([elm, 0 * elm], ra, dec, nthreads=2)
        np.testing.assert_allclose(exact, maps, rtol=0, atol=1e-9 * np.max(maps))


def test_painted_catalogue_shear_is_the_alm_not_the_nearest_pixel(tmp_path):
    """The painted signal equals exact evaluation; the nside map lookup does not."""
    from sp_validation.io import read_catalogue

    nside, lmax = 32, 95
    alm = write_sky(tmp_path, nside, lmax)
    elm = shear_alm(alm, nside)
    rng = np.random.default_rng(8)
    size = 20_000
    source = np.zeros(
        size, dtype=[(n, "f8") for n in ("RA", "Dec", "e1", "e2", "w_des")]
    )
    source["RA"] = rng.uniform(0, 360, size)
    source["Dec"] = np.degrees(np.arcsin(rng.uniform(-1, 1, size)))
    source["w_des"] = 1.0
    with h5py.File(tmp_path / "source.h5", "w") as dst:
        dst["data"] = source
    paint_catalogue(tmp_path / "source.h5", tmp_path / "mock.h5", tmp_path, 1)
    mock = read_catalogue(tmp_path / "mock.h5")
    gamma1, gamma2 = evaluate_alm([elm, 0 * elm], source["RA"], source["Dec"])
    kappa = evaluate_alm(alm, source["RA"], source["Dec"])[0]
    scale = np.std(gamma1)
    np.testing.assert_allclose(mock["g1_true"], gamma1, rtol=0, atol=1e-6 * scale)
    np.testing.assert_allclose(mock["g2_true"], -gamma2, rtol=0, atol=1e-6 * scale)
    np.testing.assert_allclose(mock["kappa_true"], kappa, rtol=0, atol=1e-6 * scale)
    map1, _ = hp.alm2map_spin([elm, 0 * elm], nside, 2, lmax)
    nearest = map1[hp.ang2pix(nside, source["RA"], source["Dec"], lonlat=True)]
    assert np.std(nearest - mock["g1_true"]) > 0.1 * scale


def test_painted_spectrum_is_the_alm_spectrum_without_pixel_window(tmp_path):
    """Paint the centres of a finer grid: its EE is the sky alm's, not x pw0^2."""
    from sp_validation.io import read_catalogue

    nside, lmax, fine = 32, 95, 128
    alm = write_sky(tmp_path, nside, lmax)
    known = hp.alm2cl(shear_alm(alm, nside))
    ra, dec = hp.pix2ang(fine, np.arange(hp.nside2npix(fine)), lonlat=True)
    source = np.zeros(
        len(ra), dtype=[(n, "f8") for n in ("RA", "Dec", "e1", "e2", "w_des")]
    )
    source["RA"], source["Dec"], source["w_des"] = ra, dec, 1.0
    with h5py.File(tmp_path / "source.h5", "w") as dst:
        dst["data"] = source
    paint_catalogue(tmp_path / "source.h5", tmp_path / "mock.h5", tmp_path, 1)
    mock = read_catalogue(tmp_path / "mock.h5")
    zero = np.zeros(len(ra))
    g1, g2 = np.asarray(mock["g1_true"], float), -np.asarray(mock["g2_true"], float)
    measured = hp.anafast([zero, g1, g2], lmax=lmax, pol=True, iter=3)
    ratio = measured[1, 2:] / known[2:]
    np.testing.assert_allclose(ratio, 1, atol=1e-4)
    assert np.max(measured[2, 2:] / known[2:]) < 1e-6
    # A nearest-pixel lookup carries one extra spin-0 window of the sky's grid.
    map1, map2 = convergence_to_shear(hp.alm2map(alm, nside, lmax=lmax), lmax)
    pix = hp.ang2pix(nside, ra, dec, lonlat=True)
    ngp = hp.anafast([zero, map1[pix], map2[pix]], lmax=lmax, pol=True, iter=3)
    assert np.min(ngp[1, 2:] / known[2:]) < 0.8
