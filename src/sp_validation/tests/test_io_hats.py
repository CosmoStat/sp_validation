"""A HATS catalogue reads like any other container.

A catalogue written as HATS (by ``lsdb``, as a collection) and as one HDF5
dataset reads to the same rows, matched on a row ID since HATS orders rows by
sky position; each partition is a chunk, the whole catalogue opens as one lazy
``ParquetTable``, and a row window reads only the row groups it touches.
"""

import h5py
import lsdb
import numpy as np
import numpy.testing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sp_validation import grammar, io

N = 3000


@pytest.fixture(scope="module")
def catalogue():
    rng = np.random.default_rng(5)
    table = np.empty(
        N,
        dtype=[
            ("RA", "f8"),
            ("Dec", "f8"),
            ("ROW", "i8"),
            ("NGMIX_G1_NOSHEAR", "f8"),
            ("w", "f4"),
            ("FLAGS", "i2"),
            ("MASK_4_Stars", "?"),
        ],
    )
    table["RA"] = rng.uniform(0, 360, N)
    table["Dec"] = np.degrees(np.arcsin(rng.uniform(-1, 1, N)))
    table["ROW"] = np.arange(N)
    table["NGMIX_G1_NOSHEAR"] = rng.normal(0, 0.3, N)
    table["w"] = rng.uniform(0, 1, N)
    table["FLAGS"] = rng.integers(0, 4, N)
    table["MASK_4_Stars"] = rng.uniform(size=N) < 0.1
    return table


@pytest.fixture(scope="module")
def paths(catalogue, tmp_path_factory):
    root = tmp_path_factory.mktemp("hats")
    hdf5 = root / "cat.hdf5"
    with h5py.File(hdf5, "w") as handle:
        handle.create_dataset("data", data=catalogue)
    frame = pd.DataFrame({name: catalogue[name] for name in catalogue.dtype.names})
    lsdb.from_dataframe(
        frame, ra_column="RA", dec_column="Dec", partition_rows=400
    ).write_catalog(root / "cat_hats", progress_bar=False)
    return hdf5, root / "cat_hats"


def _by_row(table):
    return table[np.argsort(table["ROW"])]


def test_hats_matches_hdf5_row_for_row(paths, catalogue):
    hdf5, hats = paths
    assert io.is_hats(hats)
    expected = io.read_catalogue(hdf5)
    out = _by_row(io.read_catalogue(hats))
    for name in catalogue.dtype.names:
        assert out.dtype[name] == expected.dtype[name], name
        npt.assert_array_equal(out[name], expected[name], err_msg=name)


def test_partitions_are_chunks(paths, catalogue):
    _, hats = paths
    columns = ["ROW", "w"]
    with io.Catalogue(hats) as cat:
        assert len(cat.chunks) > 1
        assert all(key.startswith("Norder=") for key in cat.chunks)
        assert len(cat) == N
        assert cat.dtype(columns).names == tuple(columns)
        chunks = list(cat.iter_chunks(columns))
        npt.assert_array_equal(np.concatenate(chunks), cat.read(columns))
        with pytest.raises(ValueError, match="begin with an integer"):
            cat.read(columns, key_column="PIX")
    npt.assert_array_equal(np.sort(np.concatenate(chunks)["ROW"]), np.arange(N))


def test_open_catalogue_is_lazy_and_maps_columns(paths, catalogue):
    _, hats = paths
    table = io.open_catalogue(hats)
    assert isinstance(table, io.ParquetTable)
    order = table["ROW"]
    npt.assert_array_equal(table["w"], catalogue["w"][order])

    mapped = io.open_catalogue(hats, column_map={"e1": "NGMIX_G1_NOSHEAR"})
    assert isinstance(mapped, grammar.V2View)
    assert "e1" in mapped.names and "NGMIX_G1_NOSHEAR" not in mapped.names
    npt.assert_array_equal(mapped["e1"], catalogue["NGMIX_G1_NOSHEAR"][order])

    stars = table["MASK_4_Stars"]
    selected = table[stars]
    assert isinstance(selected, grammar.V2View)
    npt.assert_array_equal(selected["ROW"], order[stars])
    npt.assert_array_equal(table[["ROW", "FLAGS"]]["FLAGS"], catalogue["FLAGS"][order])


def test_painter_style_windows_pool_partitions(paths, catalogue):
    """Slices of each partition, materialised and pooled, give every row once."""
    _, hats = paths
    needed = ["ROW", "RA", "w"]
    pending = []
    with io.Catalogue(hats) as cat:
        for part in cat.chunks.values():
            for lo in range(0, len(part), 150):
                pending.append(grammar.materialise(part[lo : lo + 150], needed))
    pooled = _by_row(np.concatenate(pending))
    npt.assert_array_equal(pooled["ROW"], np.arange(N))
    npt.assert_array_equal(pooled["w"], catalogue["w"])


def test_windows_read_only_their_row_groups(tmp_path):
    """Windows across row-group and file boundaries read the right rows."""
    values = np.arange(100, dtype="f8")
    files = []
    for i, (lo, hi) in enumerate([(0, 45), (45, 100)]):
        path = tmp_path / f"part{i}.parquet"
        pq.write_table(pa.table({"x": values[lo:hi]}), path, row_group_size=7)
        files.append(io._ParquetFile(str(path), pq.read_metadata(path)))
    table = io.ParquetTable(files, pq.read_schema(files[0].path))
    assert len(table) == 100
    rng = np.random.default_rng(1)
    for start, stop in [(0, 0), (0, 100), (44, 46), (6, 8), (93, 100)] + [
        tuple(sorted(rng.integers(0, 101, 2))) for _ in range(20)
    ]:
        npt.assert_array_equal(table.read_window("x", start, stop), values[start:stop])
    npt.assert_array_equal(table[10:20]["x"], values[10:20])
