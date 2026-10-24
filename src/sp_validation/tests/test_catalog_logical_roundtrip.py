"""HDF5 masks retain boolean semantics, independent of FITS logical storage."""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from sp_validation import catalog


@pytest.mark.parametrize("values", [
    np.array([False, True, False], dtype=bool),
    np.array([70, 84, 70], dtype=np.uint8),
    np.array([70, 84, 70], dtype=np.int64),
    np.array([b"F", b"T", b"F"], dtype="S1"),
    np.array(["F", "T", "F"], dtype="U1"),
    np.array([0, 1, 0], dtype=np.uint8),
])
def test_logical_hdf5_representations(tmp_path, monkeypatch, values):
    """Exercise the writer with boolean, byte, and integer logical arrays."""
    original = catalog.fits.Column

    def column(*args, **kwargs):
        col = original(*args, **kwargs)
        if str(col.format).endswith("L"):
            return SimpleNamespace(name=col.name, format=col.format, array=values)
        return col

    monkeypatch.setattr(catalog.fits, "Column", column)
    path = tmp_path / "mask.hdf5"
    catalog.write_shape_catalog(
        str(path), np.zeros(3), np.zeros(3), np.ones(3),
        add_cols={"MASK_4_Stars": np.array([False, True, False])},
        add_cols_format={"MASK_4_Stars": "L"},
    )
    with h5py.File(path, "r") as f:
        mask = f["data"]["MASK_4_Stars"]
    assert mask.dtype == np.dtype(bool)
    np.testing.assert_array_equal(mask, [False, True, False])
    assert np.count_nonzero(mask == False) == 2  # noqa: E712


def test_logical_hdf5_real_astropy_roundtrip(tmp_path):
    path = tmp_path / "real-mask.hdf5"
    catalog.write_shape_catalog(
        str(path), np.zeros(2), np.zeros(2), np.ones(2),
        add_cols={"MASK_4_Stars": np.array([False, True])},
        add_cols_format={"MASK_4_Stars": "L"},
    )
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["data"]["MASK_4_Stars"], [False, True])


def test_invalid_logical_rejected():
    with pytest.raises(ValueError, match="Invalid FITS logical"):
        catalog._logical_to_bool(np.array([70, 99], dtype=np.uint8))
