"""The collection step makes shapes, PhotoPipe and spec-z one valid HATS collection."""

import lsdb
import numpy as np
import numpy.testing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from hats.catalog import CollectionProperties, PartitionInfo, TableProperties
from hats.io.parquet_metadata import write_parquet_metadata
from hats.io.paths import get_healpix_from_path
from hats.io.validation import is_valid_catalog, is_valid_collection

from sp_validation import io
from sp_validation.hats_collection import build
from sp_validation.object_id import add_to_catalogue
from sp_validation.photopipe_merge import KEYS

N = 1000


def write_extension(collection, name, rows, columns):
    """Write extension data ``COLLECTION/name/name`` from each shape pixel file.

    ``rows(table)`` picks the rows of a pixel to keep, and ``columns(table)``
    adds the extension's columns; a pixel left with no rows has no file.
    """
    shapes = collection / "shapes"
    data = collection / name / name
    pixels = []
    for file in sorted((shapes / "dataset").rglob("Npix=*.parquet")):
        table = pq.read_table(file, columns=KEYS)
        table = table.take(rows(table))
        if not table.num_rows:
            continue
        for column, values in columns(table).items():
            table = table.append_column(column, pa.array(values))
        target = data / file.relative_to(shapes)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target)
        pixels.append(get_healpix_from_path(str(target)))
    PartitionInfo.from_healpix(pixels).write_to_file(data / "partition_info.csv")
    rows = write_parquet_metadata(data)
    TableProperties.read_from_dir(shapes).copy_and_update(
        catalog_name=name, default_columns=None, total_rows=rows
    ).to_properties_file(data)


def test_build_indexes_shapes_and_lists_the_extensions(tmp_path):
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        dict(
            XWIN_WORLD=rng.uniform(0, 360, N),
            YWIN_WORLD=np.degrees(np.arcsin(rng.uniform(-1, 1, N))),
            TILE_ID=np.full(N, 24.237),
            NUMBER=np.arange(1, N + 1, dtype=np.int32),
            G1=rng.normal(0, 0.3, N),
        )
    )
    lsdb.from_dataframe(
        frame, ra_column="XWIN_WORLD", dec_column="YWIN_WORLD", partition_rows=300
    ).write_catalog(tmp_path / "source", catalog_name="source", progress_bar=False)
    collection = tmp_path / "coll"
    add_to_catalogue(
        tmp_path / "source" / "source", collection / "shapes", name="shapes"
    )

    write_extension(
        collection,
        "photopipe",
        lambda t: np.arange(t.num_rows),
        lambda t: dict(Z_B=t["XWIN_WORLD"].to_numpy() / 360),
    )
    # Spec-z for every third object north of Dec 0.
    write_extension(
        collection,
        "specz",
        lambda t: np.flatnonzero(
            (t["OBJECT_ID"].to_numpy() % 3 == 0) & (t["YWIN_WORLD"].to_numpy() > 0)
        ),
        lambda t: dict(Z_SPEC=t["YWIN_WORLD"].to_numpy() / 90),
    )

    build(collection)

    assert is_valid_collection(collection, strict=True)
    assert is_valid_catalog(collection / "specz" / "specz", strict=True)
    properties = CollectionProperties.read_from_dir(collection)
    assert properties.default_index == "OBJECT_ID"
    # The index names its primary relative to the collection, so a copy still reads.
    index = TableProperties.read_from_dir(collection / "shapes_OBJECT_ID")
    assert index.primary_catalog == "shapes"
    assert properties.all_extensions == {
        "photopipe": "photopipe.properties",
        "specz": "specz.properties",
    }
    found = (
        lsdb.open_catalog(collection, columns=["OBJECT_ID", "G1"])
        .id_search(values={"OBJECT_ID": [24237000007]})
        .compute()
    )
    npt.assert_array_equal(found["G1"], frame["G1"].to_numpy()[[6]])
    out = io.read_catalogue(
        collection, columns=["OBJECT_ID", "XWIN_WORLD", "YWIN_WORLD", "Z_B", "Z_SPEC"]
    )
    npt.assert_array_equal(out["Z_B"], out["XWIN_WORLD"] / 360)
    has = (out["OBJECT_ID"] % 3 == 0) & (out["YWIN_WORLD"] > 0)
    npt.assert_array_equal(out["Z_SPEC"], np.where(has, out["YWIN_WORLD"] / 90, np.nan))
    joined = (
        lsdb.open_catalog(collection, columns=["OBJECT_ID"])
        .load_extension("specz", columns=["Z_SPEC"])
        .compute()
    )
    assert len(joined) == N and joined["Z_SPEC"].notna().sum() == has.sum()
