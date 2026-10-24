"""Comprehensive catalogues preserve metacal failure bits exactly."""

import runpy
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.io import fits

from sp_validation.catalog import write_shape_catalog
from sp_validation.catalog_builders import JointCat

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "params_path",
    ["scripts/calibration/params.py", "workflow/image_sims/params_im_sim.py"],
    ids=["data", "image-sims"],
)
@pytest.mark.parametrize("extension", [".fits", ".hdf5"])
@pytest.mark.parametrize("reduce_mem", [False, True])
def test_metacal_flags_roundtrip(tmp_path, params_path, extension, reduce_mem):
    """Contract metacal-flag-width: neither output nor merging loses flag bits.

    Exercise ngmix bits 12 and 15, their combinations, and a high bit 30
    value with bit 3. Float64 inputs match ShapePipe's final catalogue.
    Signed int16 cannot hold bit 15, and float32 loses bit 3 next to bit 30.
    """
    with np.printoptions():
        params = runpy.run_path(str(ROOT / params_path))
    flags = np.array(
        [0, 2**12, 2**15, 2**15 | 2**12 | 8, 2**16 - 1, 2**30, 2**30 | 8],
        dtype=np.float64,
    )
    bit_columns = ["NGMIX_MCAL_FLAGS"] + [
        f"NGMIX_FLAGS_{suffix}" for suffix in ("NOSHEAR", "1P", "1M", "2P", "2M")
    ]
    columns = {name: flags for name in bit_columns}
    columns["NGMIX_MCAL_TYPES_FAIL"] = np.array([0, 1, 5, 2, 5, 0, 1])
    assert set(columns) <= set(params["add_cols_pre_cal"])
    path = tmp_path / f"comprehensive{extension}"
    write_shape_catalog(
        str(path),
        np.zeros(len(flags)),
        np.zeros(len(flags)),
        np.ones(len(flags)),
        add_cols=columns,
        add_cols_format=params["add_cols_pre_cal_format"],
    )
    if extension == ".fits":
        written = fits.getdata(path, 1)
    else:
        with h5py.File(path, "r") as catalog:
            written = catalog["data"][:]

    builder = JointCat()
    builder._params["reduce_mem"] = reduce_mem
    for name, expected in columns.items():
        values = written[name]
        np.testing.assert_array_equal(values, expected, err_msg=name)
        if name in bit_columns:
            assert values.dtype.kind == "i" and values.dtype.itemsize == 4, name
        reduced = values.astype(builder.dtype_out(name, values.dtype))
        np.testing.assert_array_equal(reduced, expected, err_msg=name)


@pytest.mark.parametrize("reduce_mem", [False, True])
def test_reduce_mem_never_narrows_integers(reduce_mem):
    """reduce_mem narrows float64 (except RA/Dec) and leaves integers intact."""
    builder = JointCat()
    builder._params["reduce_mem"] = reduce_mem
    for dtype in (">i4", "<i4", ">i8", "<i2"):
        values = np.array([0, 127, 128, 2**14], dtype=dtype)
        out = builder.dtype_out("N_EPOCH", values.dtype)
        assert out == values.dtype
        np.testing.assert_array_equal(values.astype(out), values)
    for name in ("RA", "Dec"):
        assert builder.dtype_out(name, np.dtype(">f8")) == np.dtype(">f8")
    expected = np.float32 if reduce_mem else np.dtype(">f8")
    assert builder.dtype_out("NGMIX_G1_NOSHEAR", np.dtype(">f8")) == expected


@pytest.mark.parametrize("extension", [".fits", ".hdf5"])
def test_mask_columns_roundtrip_as_booleans(tmp_path, extension):
    """Logical (format "L") mask columns come back as booleans, True = masked.

    astropy holds FITS logicals as the characters 'T'/'F'; the HDF5 writer
    must not store those character codes (84/70), which every mask cut
    (``kind: equal, value: False``) would read as masked.
    """
    with np.printoptions():
        params = runpy.run_path(str(ROOT / "scripts/calibration/params.py"))
    masked = np.array([0, 1, 1, 0], dtype=np.int8)
    columns = {name: masked for name in params["mask_columns"]}
    assert all(params["add_cols_pre_cal_format"][name] == "L" for name in columns)
    # Columns of other formats alongside, as in the real comprehensive catalogue
    others = {
        "TILE_ID": np.array([b"200.300", b"200.300", b"201.300", b"201.300"]),
        "NUMBER": np.arange(len(masked)),
        "FLAGS": np.array([0, 2, 3, 0]),
    }
    path = tmp_path / f"comprehensive{extension}"
    write_shape_catalog(
        str(path),
        np.zeros(len(masked)),
        np.zeros(len(masked)),
        np.ones(len(masked)),
        add_cols=columns | others,
        add_cols_format=params["add_cols_pre_cal_format"],
    )
    if extension == ".fits":
        written = fits.getdata(path, 1)
    else:
        with h5py.File(path, "r") as catalog:
            written = catalog["data"][:]

    np.testing.assert_array_equal(written["NUMBER"], others["NUMBER"])
    np.testing.assert_array_equal(written["FLAGS"], others["FLAGS"])
    for name in columns:
        assert written[name].dtype == np.bool_, name
        np.testing.assert_array_equal(written[name], masked.astype(bool), err_msg=name)
