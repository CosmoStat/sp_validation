"""Shared-shell tomographic skies and row-wise exact painting."""
from dataclasses import replace

import h5py
import healpy as hp
import numpy as np
import pytest

from sp_validation.glass_mock import (
    GlassMockConfig, generate_lensing_maps, paint_catalogue,
    read_sky, shear_alm, evaluate_alm, projected_tomographic_theory,
)


def test_two_bin_sky_and_single_source_identity(tmp_path):
    z = np.linspace(0.01, 0.39, 30)
    nz = np.column_stack([z, np.ones(30), z**2])
    np.savetxt(tmp_path / "tomo.txt", nz)
    np.savetxt(tmp_path / "single.txt", nz[:, :2])
    config = GlassMockConfig(nside=8, harmonic_lmax=16, seed=7, zmax=0.4,
                             dx=500, nz_path=str(tmp_path / "tomo.txt"), limber=True)
    generate_lensing_maps(config, tmp_path / "tomo", tmp_path / "cache")
    generate_lensing_maps(replace(config, nz_path=str(tmp_path / "single.txt")),
                          tmp_path / "single", tmp_path / "cache")
    alms, nside = read_sky(tmp_path / "tomo")
    single, _ = read_sky(tmp_path / "single")
    np.testing.assert_array_equal(alms[0], single)
    assert alms.shape == (2, hp.Alm.getsize(16))
    assert not list((tmp_path / "tomo").glob("*.npy"))
    elms = [shear_alm(a, nside) for a in alms]
    with np.load(tmp_path / "tomo/spectra.npz") as spectra, np.load(tmp_path / "single/spectra.npz") as old:
        for key in ("cl_ee", "cl_ee_pixel", "cl_ee_achieved_pixel", "realization_ee"):
            np.testing.assert_allclose(spectra[key][0, 0], old[key], rtol=1e-14)
        for i in range(2):
            for j in range(2):
                np.testing.assert_array_equal(spectra["realization_ee"][i, j],
                                              hp.alm2cl(elms[i], elms[j]))
                np.testing.assert_array_equal(spectra["realization_kk"][i, j],
                                              hp.alm2cl(alms[i], alms[j]))
        assert not np.array_equal(spectra["cl_ee"][0, 0], spectra["cl_ee"][1, 1])
        assert np.all(spectra["cl_ee"][0, 1, 2:] > 0)


def test_cross_theory_is_bilinear_shell_projection():
    import glass
    ell = np.arange(17)
    cls = [1e-5 / (1 + ell)**2, 2e-5 / (1 + ell)**2, 0.4e-5 / (1 + ell)**2]
    transfer = np.array([[0.2, 0.7], [0.6, 0.1]])
    theory = projected_tomographic_theory(cls, transfer, 8, 16)
    for a in range(2):
        for b in range(2):
            expected = sum(transfer[a, i] * transfer[b, j] * glass.getcl(cls, i, j)
                           for i in range(2) for j in range(2))
            np.testing.assert_allclose(theory["cl_kk"][a, b], expected)


def test_paint_rows_use_own_bin_and_drop_non_sources(tmp_path):
    from sp_validation.tests.test_glass_paint import gaussian_kappa_alm
    alms = np.array([gaussian_kappa_alm(31, seed=3), gaussian_kappa_alm(31, seed=4)])
    np.savez(tmp_path / "kappa_alm.npz", alm=alms, nside=16)
    source = np.zeros(120, dtype=[(k, "f8") for k in ("RA", "Dec", "e1", "e2", "w_des")] + [("TOM_BIN_ID", "i4")])
    source["RA"] = np.linspace(0, 359, len(source))
    source["Dec"] = np.linspace(-60, 60, len(source))
    source["w_des"] = 1
    source["TOM_BIN_ID"] = np.tile([-1, 0, 1, 2], 30)
    with h5py.File(tmp_path / "input.h5", "w") as f:
        f["data"] = source
    with pytest.raises(ValueError, match="tomo_bin_col"):
        paint_catalogue(tmp_path / "input.h5", tmp_path / "bad.h5", tmp_path, 10)
    for chunk, block in [(17, 23), (120, 120)]:
        paint_catalogue(tmp_path / "input.h5", tmp_path / f"mock{chunk}.h5", tmp_path, 10,
                        tomo_bin_col="TOM_BIN_ID", chunk_size=chunk, block_size=block, nthreads=2)
    with h5py.File(tmp_path / "mock17.h5") as f, h5py.File(tmp_path / "mock120.h5") as g:
        rows = f["data"][:]
        np.testing.assert_array_equal(rows, g["data"][:])
    expected = source[source["TOM_BIN_ID"] > 0]
    np.testing.assert_array_equal(rows["RA"], expected["RA"])
    assert len(rows) == 60
    for b, alm in enumerate(alms, 1):
        mask = rows["TOM_BIN_ID"] == b
        elm = shear_alm(alm, 16)
        g1, g2 = evaluate_alm([elm, 0 * elm], rows["RA"][mask], rows["Dec"][mask])
        np.testing.assert_allclose(rows["e1"][mask], g1, atol=1e-12)
        np.testing.assert_allclose(rows["e2"][mask], -g2, atol=1e-12)
