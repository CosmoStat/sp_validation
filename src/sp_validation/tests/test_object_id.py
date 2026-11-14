import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sp_validation.assembly.object_id import NUMBER_LIMIT, add_to_pixel, object_id


def test_every_tile_id_converts_to_its_integer():
    xxx, yyy = np.meshgrid(np.arange(361), np.arange(1000), indexing="ij")
    tile_id = np.array([float(f"{x}.{y:03d}") for x, y in zip(xxx.flat, yyy.flat)])
    ids = object_id(tile_id, np.zeros(tile_id.size, np.int32)).to_numpy()
    np.testing.assert_array_equal(
        ids, (xxx.ravel() * 1000 + yyy.ravel()) * NUMBER_LIMIT
    )


def test_object_id_packs_tile_and_number():
    ids = object_id([194.248, 24.237, 24.237], np.array([32402, 1, -1], np.int32))
    assert ids.to_pylist() == [194248032402, 24237000001, None]


def test_number_beyond_six_digits_raises():
    with pytest.raises(ValueError, match="NUMBER"):
        object_id([194.248], [NUMBER_LIMIT])


def test_add_to_pixel_keeps_rows_groups_encodings_and_metadata(tmp_path):
    table = pa.table(
        dict(
            _healpix_29=np.arange(5),
            TILE_ID=np.full(5, 24.237),
            NUMBER=np.array([5, 3, 4, 1, 2], np.int32),
            G1=np.linspace(0, 1, 5),
        )
    ).replace_schema_metadata({"origin": "test"})
    source = tmp_path / "in.parquet"
    pq.write_table(
        table,
        source,
        row_group_size=2,
        use_dictionary=["TILE_ID", "NUMBER"],
        column_encoding=dict(_healpix_29="DELTA_BINARY_PACKED", G1="BYTE_STREAM_SPLIT"),
    )
    output = tmp_path / "out" / "Npix=1.parquet"
    add_to_pixel(source, output)

    written = pq.ParquetFile(output)
    assert written.metadata.num_row_groups == 3
    assert written.schema_arrow.names == [
        "_healpix_29",
        "TILE_ID",
        "NUMBER",
        "OBJECT_ID",
        "G1",
    ]
    assert written.schema_arrow.metadata == {b"origin": b"test"}
    group = written.metadata.row_group(0)
    encodings = {
        group.column(i).path_in_schema: group.column(i).encodings
        for i in range(group.num_columns)
    }
    original = pq.ParquetFile(source).metadata.row_group(0)
    for i in range(original.num_columns):
        column = original.column(i)
        assert encodings[column.path_in_schema] == column.encodings
    assert encodings["OBJECT_ID"] == encodings["_healpix_29"]
    assert "DELTA_BINARY_PACKED" in encodings["OBJECT_ID"]
    assert "BYTE_STREAM_SPLIT" in encodings["G1"]
    assert "RLE_DICTIONARY" in encodings["NUMBER"]
    result = written.read()
    assert result.drop_columns("OBJECT_ID").equals(table)
    assert result["OBJECT_ID"].to_pylist() == [24237000000 + n for n in [5, 3, 4, 1, 2]]


def test_add_to_catalogue_writes_a_valid_hats_catalogue(tmp_path):
    import hats
    from hats.catalog import PartitionInfo, TableProperties
    from hats.io.validation import is_valid_catalog
    from hats.pixel_math import HealpixPixel

    from sp_validation.assembly.object_id import add_to_catalogue

    source = tmp_path / "source"
    pixel = source / "dataset/Norder=0/Dir=0/Npix=4.parquet"
    pixel.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            dict(
                _healpix_29=np.arange(3) + 4 * 4**29,
                ra=np.full(3, 10.0),
                dec=np.full(3, 0.0),
                TILE_ID=np.full(3, 24.237),
                NUMBER=np.array([3, 1, 2], np.int32),
            )
        ),
        pixel,
    )
    pq.write_metadata(pq.read_schema(pixel), source / "dataset/_common_metadata")
    PartitionInfo.from_healpix([HealpixPixel(0, 4)]).write_to_file(
        source / "partition_info.csv"
    )
    TableProperties(
        catalog_name="source",
        catalog_type="object",
        total_rows=3,
        ra_column="ra",
        dec_column="dec",
        default_columns=["ra", "dec"],
    ).to_properties_file(source)

    add_to_catalogue(source, tmp_path / "out", name="shapes")

    assert is_valid_catalog(tmp_path / "out", strict=True)
    catalogue = hats.read_hats(tmp_path / "out")
    assert catalogue.catalog_name == "shapes"
    assert catalogue.catalog_info.default_columns == ["ra", "dec", "OBJECT_ID"]
    assert "OBJECT_ID" in catalogue.schema.names
