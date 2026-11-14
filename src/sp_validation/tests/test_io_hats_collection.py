"""A HATS collection with an extension reads as one table.

The extension's pixel files hold the primary's rows in the same order; the
reader takes each column from the catalogue that holds it and checks the join
column on both sides.
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

    extension = path / "photo"
    for file in sorted((primary / "dataset").rglob("Npix=*.parquet")):
        target = extension / file.relative_to(primary)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(_extension_table(pq.read_table(file)), target)
    pixels = PartitionInfo.read_from_dir(primary).get_healpix_pixels()
    PartitionInfo.from_healpix(pixels).write_to_file(
        hats_paths.get_partition_info_pointer(extension)
    )
    schema = _extension_table(pq.read_table(file)).schema
    pq.write_metadata(schema, extension / "dataset" / "_common_metadata")
    TableProperties.read_from_dir(primary).copy_and_update(
        catalog_name="photo", default_columns=None
    ).to_properties_file(extension)
    ExtensionProperties(
        name="photo",
        primary_catalog="coll",
        primary_column="OBJECT_ID",
        join_catalog="photo",
        join_column="OBJECT_ID",
        extension_columns=["Z_B", "FLAG", "STATUS", "NAME"],
    ).to_properties_file(path)
    properties = CollectionProperties.read_from_dir(path)
    properties.model_copy(
        update=dict(all_extensions={"photo": "photo.properties"})
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


def test_misaligned_extension_raises(collection, tmp_path):
    path, _ = collection
    copy = Path(shutil.copytree(path, tmp_path / "coll"))
    file = next(iter(sorted((copy / "photo" / "dataset").rglob("Npix=*.parquet"))))
    table = pq.read_table(file)
    pq.write_table(table.take(np.arange(table.num_rows)[::-1]), file)
    with pytest.raises(ValueError, match="not row-aligned"):
        io.read_catalogue(copy, columns=["G1", "Z_B"])
