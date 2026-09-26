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

    Bit 30 marks an absent measurement; bit 3 alongside it must also survive.
    Float64 inputs match ShapePipe's final catalogue. A float32 downcast loses
    bit 3, while int16 and int8 conversions lose bit 30.
    """
    with np.printoptions():
        params = runpy.run_path(str(ROOT / params_path))
    flags = np.array([0, 2**30, 2**30 | 8], dtype=np.float64)
    bit_columns = ["NGMIX_MCAL_FLAGS"] + [
        f"NGMIX_FLAGS_{suffix}" for suffix in ("NOSHEAR", "1P", "1M", "2P", "2M")
    ]
    columns = {name: flags for name in bit_columns}
    columns["NGMIX_MCAL_TYPES_FAIL"] = np.array([0, 5, 1])
    assert set(columns) <= set(params["add_cols_pre_cal"])
    path = tmp_path / f"comprehensive{extension}"
    write_shape_catalog(
        str(path),
        np.zeros(3),
        np.zeros(3),
        np.ones(3),
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
            assert values.dtype.kind == "i" and values.dtype.itemsize == 8, name
        reduced = values.astype(builder.dtype_out(name, values.dtype))
        np.testing.assert_array_equal(reduced, expected, err_msg=name)
