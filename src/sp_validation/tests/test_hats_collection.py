"""The collection step makes shapes and PhotoPipe one valid HATS collection."""

import lsdb
import numpy as np
import numpy.testing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from hats.catalog import CollectionProperties, PartitionInfo, TableProperties
from hats.io.validation import is_valid_collection

from sp_validation import io
from sp_validation.hats_collection import build
from sp_validation.object_id import add_to_catalogue
from sp_validation.photopipe_merge import KEYS

N = 1000


def test_build_indexes_shapes_and_lists_photopipe(tmp_path):
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

    photopipe = collection / "photopipe" / "photopipe"
    for file in sorted((collection / "shapes" / "dataset").rglob("Npix=*.parquet")):
        shapes = pq.read_table(file, columns=KEYS)
        z = shapes["XWIN_WORLD"].to_numpy() / 360
        target = photopipe / file.relative_to(collection / "shapes")
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(shapes.append_column("Z_B", pa.array(z)), target)
    PartitionInfo.read_from_dir(collection / "shapes").write_to_file(
        photopipe / "partition_info.csv"
    )
    pq.write_metadata(pq.read_schema(target), photopipe / "dataset/_common_metadata")
    TableProperties.read_from_dir(collection / "shapes").copy_and_update(
        catalog_name="photopipe", default_columns=None
    ).to_properties_file(photopipe)

    build(collection)

    assert is_valid_collection(collection, strict=True)
    properties = CollectionProperties.read_from_dir(collection)
    assert properties.default_index == "OBJECT_ID"
    assert properties.all_extensions == {"photopipe": "photopipe.properties"}
    found = (
        lsdb.open_catalog(collection, columns=["OBJECT_ID", "G1"])
        .id_search(values={"OBJECT_ID": [24237000007]})
        .compute()
    )
    npt.assert_array_equal(found["G1"], frame["G1"].to_numpy()[[6]])
    out = io.read_catalogue(collection, columns=["NUMBER", "XWIN_WORLD", "Z_B"])
    npt.assert_array_equal(out["Z_B"], out["XWIN_WORLD"] / 360)
