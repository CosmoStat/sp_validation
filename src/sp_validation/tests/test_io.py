"""The catalogue reader is independent of container and column convention.

A catalogue whose columns follow no ShapePipe convention reads to the same
table from FITS, a flat HDF5 dataset, a comprehensive ``data`` + ``data_ext``
HDF5 and an HDF5 group of row chunks; its names pass through unless a
``column_map`` renames them. The rho/tau loader reads a star catalogue from
HDF5 by the column names its config gives.
"""

import h5py
import numpy as np
import numpy.testing as npt
import pytest
from astropy.io import fits

from sp_validation import grammar, io
from sp_validation.rho_tau import _CatalogueLoader

N = 60


@pytest.fixture
def foreign():
    """A star catalogue in a naming convention of its own."""
    rng = np.random.default_rng(3)
    table = np.empty(
        N,
        dtype=[
            ("alpha", "f8"),
            ("delta", "f8"),
            ("psf_e1", "f8"),
            ("psf_e2", "f8"),
            ("star_e1", "f8"),
            ("star_e2", "f8"),
            ("psf_size", "f4"),
            ("star_size", "f4"),
            ("flag", "i2"),
        ],
    )
    for name in table.dtype.names:
        table[name] = rng.normal(size=N).astype(table.dtype[name])
    return table


def _write(table, path, layout):
    if layout == "fits":
        fits.BinTableHDU(table).writeto(path)
    elif layout == "dataset":
        with h5py.File(path, "w") as handle:
            handle.create_dataset("stars", data=table)
    elif layout == "comprehensive":
        names = table.dtype.names
        with h5py.File(path, "w") as handle:
            handle.create_dataset("data", data=table[list(names[:4])])
            handle.create_dataset("data_ext", data=table[list(names[4:])])
    elif layout == "chunks":
        with h5py.File(path, "w") as handle:
            group = handle.create_group("exposures")
            for i, start in enumerate(range(0, N, 25)):
                group.create_dataset(f"{2000 + i}p", data=table[start : start + 25])
            handle.attrs["n_exposures"] = len(group)
    return path


LAYOUTS = ("fits", "dataset", "comprehensive", "chunks")


def _assert_table_equal(out, table):
    assert set(out.dtype.names) >= set(table.dtype.names)
    for name in table.dtype.names:
        npt.assert_array_equal(np.asarray(out[name]), table[name], err_msg=name)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_every_container_reads_to_the_same_table(foreign, tmp_path, layout):
    path = _write(foreign, tmp_path / f"stars.{layout}", layout)
    out = io.read_catalogue(path)
    assert tuple(out.dtype.names) == foreign.dtype.names
    _assert_table_equal(out, foreign)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_columns_restrict_the_read(foreign, tmp_path, layout):
    path = _write(foreign, tmp_path / f"stars.{layout}", layout)
    out = io.read_catalogue(path, columns=["star_e2", "alpha"])
    assert out.dtype.names == ("star_e2", "alpha")
    _assert_table_equal(out, foreign[["star_e2", "alpha"]])


def test_container_is_detected_from_contents_not_extension(foreign, tmp_path):
    hdf5 = _write(foreign, tmp_path / "stars.fits", "dataset")
    fits_file = _write(foreign, tmp_path / "stars.hdf5", "fits")
    for path in (hdf5, fits_file):
        _assert_table_equal(io.read_catalogue(path), foreign)


def test_key_column_records_chunk_names(foreign, tmp_path):
    path = _write(foreign, tmp_path / "stars.hdf5", "chunks")
    out = io.read_catalogue(path, key_column="EXPID")
    npt.assert_array_equal(out["EXPID"], np.repeat([2000, 2001, 2002], [25, 25, 10]))
    flat = _write(foreign, tmp_path / "flat.hdf5", "dataset")
    assert "EXPID" not in io.read_catalogue(flat, key_column="EXPID").dtype.names


@pytest.mark.parametrize("layout", LAYOUTS)
def test_column_map_presents_canonical_names(foreign, tmp_path, layout):
    path = _write(foreign, tmp_path / f"stars.{layout}", layout)
    column_map = {"RA": "alpha", "DEC": "delta", "HSM_G*_PSF": "psf_e*"}
    out = io.read_catalogue(path, column_map=column_map)
    assert {"RA", "DEC", "HSM_G1_PSF", "HSM_G2_PSF"} <= set(out.dtype.names)
    assert not {"alpha", "delta", "psf_e1", "psf_e2"} & set(out.dtype.names)
    npt.assert_array_equal(out["RA"], foreign["alpha"])
    npt.assert_array_equal(out["HSM_G2_PSF"], foreign["psf_e2"])
    npt.assert_array_equal(out["star_e1"], foreign["star_e1"])


def test_column_map_overrides_and_composes_with_v1_detection():
    v1 = np.zeros(3, dtype=[("SIGMA_PSF_HSM", "f8"), ("E1_PSF_HSM", "f8"), ("x", "f8")])
    v1["SIGMA_PSF_HSM"] = 0.5
    v1["x"] = [1, 2, 3]
    view = grammar.adapt(v1, column_map={"HSM_G1_PSF": "x"})
    # the map wins for HSM_G1_PSF; v1 detection still converts sigma to T
    npt.assert_array_equal(view["HSM_G1_PSF"], [1, 2, 3])
    npt.assert_allclose(view["HSM_T_PSF"], 2 * 0.5**2)
    assert "E1_PSF_HSM" in view.names


def test_bad_pattern_raises():
    table = np.zeros(2, dtype=[("a", "f8")])
    with pytest.raises(ValueError, match=r"exactly one '\*'"):
        grammar.adapt(table, column_map={"NGMIX_*_*": "MY_*"})


@pytest.mark.parametrize("layout", ("dataset", "chunks"))
def test_rho_tau_loader_reads_hdf5_stars_by_config_names(foreign, tmp_path, layout):
    path = _write(foreign, tmp_path / "stars.hdf5", layout)
    params = {
        "ra_PSF_col": "alpha",
        "dec_PSF_col": "delta",
        "e1_PSF_col": "psf_e1",
        "e2_PSF_col": "psf_e2",
        "e1_star_col": "star_e1",
        "e2_star_col": "star_e2",
        "PSF_size": "psf_size",
        "star_size": "star_size",
        "PSF_flag": "flag",
        "star_flag": "flag",
    }
    with _CatalogueLoader({"psf": {"path": str(path)}}, params) as load:
        fits_path = load("psf")
        assert fits_path != str(path)
        out = fits.getdata(fits_path, 1)
        assert set(out.dtype.names) == set(params.values())
        _assert_table_equal(out, foreign[list(dict.fromkeys(params.values()))])


def test_rho_tau_loader_passes_plain_fits_through(foreign, tmp_path):
    path = _write(foreign, tmp_path / "stars.fits", "fits")
    params = {"ra_PSF_col": "alpha", "dec_PSF_col": "delta"}
    with _CatalogueLoader({"psf": {"path": str(path)}}, params) as load:
        assert load("psf") == str(path)
    entry = {"path": str(path), "column_map": {"RA": "alpha"}}
    with _CatalogueLoader({"psf": entry}, {"ra_PSF_col": "RA"}) as load:
        assert load("psf") != str(path)
        npt.assert_array_equal(fits.getdata(load("psf"), 1)["RA"], foreign["alpha"])
