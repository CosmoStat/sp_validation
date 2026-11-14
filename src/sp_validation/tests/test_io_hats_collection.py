"""A HATS collection with extensions reads as one table.

The reader takes each column from the catalogue that holds it and joins each
extension to the primary on the join column, within the pixel: a dense
extension holds every primary row (here in the same order), a sparse one only
some, on some pixels.
"""

import shutil
from pathlib import Path

import lsdb
import numpy as np
import numpy.testing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hats.catalog import (
    CollectionProperties,
    ExtensionProperties,
    PartitionInfo,
    TableProperties,
)
from hats.io import paths as hats_paths

from sp_validation import io

N = 2000


def _sparse_table(primary):
    """Spec-z-like columns for every fifth row of one primary pixel file, shuffled."""
    ids = primary["OBJECT_ID"].to_numpy()
    rows = np.flatnonzero(ids % 5 == 0)[::-1]
    return pa.table(
        dict(
            OBJECT_ID=pa.array(ids[rows]),
            Z_SPEC=pa.array(ids[rows] / N),
            N_SPEC=pa.array((ids[rows] % 4).astype(np.int16)),
            SURVEY=pa.array([f"s{i}" for i in ids[rows]]),
        )
    )


def _write_extension(path, name, table, files, columns):
    """Write ``table(file)`` for each primary pixel file, and its properties."""
    primary = path / CollectionProperties.read_from_dir(path).hats_primary_table_url
    extension = path / name
    written = []
    for file in files:
        target = extension / file.relative_to(primary)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table(pq.read_table(file)), target)
        written.append(hats_paths.get_healpix_from_path(str(target)))
    PartitionInfo.from_healpix(written).write_to_file(
        hats_paths.get_partition_info_pointer(extension)
    )
    schema = table(pq.read_table(file)).schema
    pq.write_metadata(schema, extension / "dataset" / "_common_metadata")
    TableProperties.read_from_dir(primary).copy_and_update(
        catalog_name=name, default_columns=None
    ).to_properties_file(extension)
    ExtensionProperties(
        name=name,
        primary_catalog=path.name,
        primary_column="OBJECT_ID",
        join_catalog=name,
        join_column="OBJECT_ID",
        extension_columns=columns,
    ).to_properties_file(path)


def _extension_table(primary):
    """PhotoPipe-like columns for the rows of one primary pixel file."""
    ids = primary["OBJECT_ID"].to_numpy()
    matched = ids % 3 != 0
    return pa.table(
        dict(
            _healpix_29=primary["_healpix_29"],
            OBJECT_ID=primary["OBJECT_ID"],
            Z_B=pa.array(ids / N, mask=~matched),
            FLAG=pa.array((ids % 7).astype(np.int16), mask=~matched),
            STATUS=pa.array(np.where(matched, 0, 1).astype(np.int8)),
            NAME=pa.array([f"t{i}" for i in ids], mask=~matched),
        )
    )


@pytest.fixture(scope="module")
def collection(tmp_path_factory):
    root = tmp_path_factory.mktemp("collection")
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        dict(
            RA=rng.uniform(0, 360, N),
            Dec=np.degrees(np.arcsin(rng.uniform(-1, 1, N))),
            OBJECT_ID=np.arange(N),
            G1=rng.normal(0, 0.3, N),
        )
    )
    lsdb.from_dataframe(
        frame, ra_column="RA", dec_column="Dec", partition_rows=300
    ).write_catalog(root / "coll", catalog_name="shapes", progress_bar=False)
    path = root / "coll"
    primary = path / CollectionProperties.read_from_dir(path).hats_primary_table_url
    files = sorted((primary / "dataset").rglob("Npix=*.parquet"))
    _write_extension(
        path, "photo", _extension_table, files, ["Z_B", "FLAG", "STATUS", "NAME"]
    )
    # The sparse extension has no file for the first pixel.
    _write_extension(
        path, "specz", _sparse_table, files[1:], ["Z_SPEC", "N_SPEC", "SURVEY"]
    )
    properties = CollectionProperties.read_from_dir(path)
    properties.model_copy(
        update=dict(
            all_extensions={
                "photo": "photo.properties",
                "specz": "specz.properties",
            }
        )
    ).to_properties_file(path)
    return path, frame


def test_columns_come_from_either_catalogue(collection):
    path, frame = collection
    out = io.read_catalogue(path, columns=["OBJECT_ID", "G1", "Z_B", "FLAG", "NAME"])
    out = out[np.argsort(out["OBJECT_ID"])]
    ids = frame["OBJECT_ID"].to_numpy()
    matched = ids % 3 != 0

    npt.assert_array_equal(out["G1"], frame["G1"].to_numpy())
    npt.assert_array_equal(out["Z_B"], np.where(matched, ids / N, np.nan))
    # An integer column holding nulls reads as float64, NaN where null.
    assert out.dtype["FLAG"] == np.float64
    npt.assert_array_equal(out["FLAG"], np.where(matched, ids % 7, np.nan))
    assert list(out["NAME"]) == [f"t{i}" if m else None for i, m in zip(ids, matched)]


def test_integer_column_without_nulls_keeps_its_dtype(collection):
    path, _ = collection
    out = io.read_catalogue(path, columns=["STATUS"])
    assert out.dtype["STATUS"] == np.int8


def test_sparse_extension_leaves_rows_without_a_match_null(collection):
    path, frame = collection
    primary = path / "shapes"
    first = sorted((primary / "dataset").rglob("Npix=*.parquet"))[0]
    absent = pq.read_table(first, columns=["OBJECT_ID"])["OBJECT_ID"].to_numpy()
    out = io.read_catalogue(path, columns=["OBJECT_ID", "Z_SPEC", "N_SPEC", "SURVEY"])
    ids = out["OBJECT_ID"]
    has = (ids % 5 == 0) & ~np.isin(ids, absent)

    npt.assert_array_equal(out["Z_SPEC"], np.where(has, ids / N, np.nan))
    # An integer column of an extension that lacks rows reads as float64.
    assert out.dtype["N_SPEC"] == np.float64
    npt.assert_array_equal(out["N_SPEC"], np.where(has, ids % 4, np.nan))
    assert list(out["SURVEY"]) == [f"s{i}" if h else None for i, h in zip(ids, has)]


def test_extension_rows_join_on_the_key_in_any_order(collection, tmp_path):
    path, _ = collection
    copy = Path(shutil.copytree(path, tmp_path / "coll"))
    file = next(iter(sorted((copy / "photo" / "dataset").rglob("Npix=*.parquet"))))
    table = pq.read_table(file)
    pq.write_table(table.take(np.arange(table.num_rows)[::-1]), file)
    columns = ["OBJECT_ID", "Z_B", "STATUS"]
    reordered = io.read_catalogue(copy, columns=columns)
    original = io.read_catalogue(path, columns=columns)
    for name in columns:
        npt.assert_array_equal(reordered[name], original[name])


def test_dense_extension_without_a_key_cannot_null_an_integer(collection, tmp_path):
    path, _ = collection
    copy = Path(shutil.copytree(path, tmp_path / "coll"))
    file = next(iter(sorted((copy / "photo" / "dataset").rglob("Npix=*.parquet"))))
    table = pq.read_table(file)
    ids = table["OBJECT_ID"].to_numpy().copy()
    ids[0] = -1
    table = table.set_column(1, "OBJECT_ID", pa.array(ids))
    pq.write_table(table, file)
    with pytest.raises(ValueError, match="cannot hold null"):
        io.read_catalogue(copy, columns=["G1", "STATUS"])
