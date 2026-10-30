"""Columnar HDF5 catalogues expose bounded, lazy structured row windows."""

import h5py
import numpy as np
import pytest

from sp_validation import grammar, io


@pytest.mark.parametrize("group_name", ["/", "catalogue"])
def test_columnar_catalogue_lazy_windows_and_column_map(
    tmp_path, monkeypatch, group_name
):
    path = tmp_path / "columns.h5"
    expected = np.zeros(1000, dtype=[("ra", "f4"), ("ell", "f4", 2), ("halo", "?")])
    expected["ra"] = np.arange(1000)
    expected["ell"] = np.arange(2000).reshape(1000, 2)
    expected["halo"] = np.arange(1000) % 2 == 0
    with h5py.File(path, "w") as dst:
        group = dst if group_name == "/" else dst.create_group(group_name)
        for name in expected.dtype.names:
            group[name] = expected[name]
    calls = []
    getitem = h5py.Dataset.__getitem__

    def spy(dataset, key, *args, **kwargs):
        calls.append(key)
        return getitem(dataset, key, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", spy)
    with io.Catalogue(path, column_map={"RA": "ra"}) as cat:
        assert len(cat) == 1000
        assert not cat.chunked
        assert cat.dtype()["ell"].shape == (2,)
        # Header/dtype inspection reads no data beyond zero-length windows.
        assert all(isinstance(key, slice) and key.stop == 0 for key in calls)
        calls.clear()
        rows = grammar.materialise(cat.table()[7:15])
        assert all(
            isinstance(key, slice) and key.start == 7 and key.stop == 15
            for key in calls
        )
        np.testing.assert_array_equal(rows["RA"], expected["ra"][7:15])
        np.testing.assert_array_equal(rows["ell"], expected["ell"][7:15])
        np.testing.assert_array_equal(rows["halo"], expected["halo"][7:15])
    all_rows = io.read_catalogue(path, columns=["halo", "ra"])
    assert all_rows.dtype.names == ("halo", "ra")
    np.testing.assert_array_equal(all_rows["ra"], expected["ra"])


def test_columnar_catalogue_rejects_misaligned_columns(tmp_path):
    path = tmp_path / "columns.h5"
    with h5py.File(path, "w") as dst:
        dst["a"] = np.zeros(3)
        dst["b"] = np.zeros(4)
    with pytest.raises(ValueError, match="Column lengths differ"):
        io.Catalogue(path)
