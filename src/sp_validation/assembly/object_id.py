"""The DR6 object ID as one column, and the collection's shapes that carry it.

Stephen Gwyn's DR6 numbers objects within a tile: ``TILE_ID`` (a double such as
194.248, the tile ``194.248``) and ``NUMBER``. ``OBJECT_ID`` packs the pair into
one int64, ``10**6 * 194248 + NUMBER``, so a HATS index can be built on it. Rows
with no DR6 number (``NUMBER`` < 0) have a null ``OBJECT_ID``.

``write_shapes`` writes the shape catalogue into the collection as
``COLLECTION/shapes``, each pixel file with ``OBJECT_ID`` after ``NUMBER``.
"""

import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from hats.catalog import TableProperties
from hats.io.parquet_metadata import write_parquet_metadata
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN

NAME = "shapes"
OBJECT_ID = "OBJECT_ID"
NUMBER_LIMIT = 10**6


def object_id(tile_id, number):
    """Return the ``OBJECT_ID`` array of ``TILE_ID`` and ``NUMBER`` arrays.

    The tile is ``TILE_ID`` times 1000 rounded to the nearest integer, so a
    tile such as 24.237, stored as 24.236999..., gives 24237.
    """
    tile = np.rint(np.asarray(tile_id, np.float64) * 1000).astype(np.int64)
    number = np.asarray(number, np.int64)
    if number.size and number.max() >= NUMBER_LIMIT:
        raise ValueError(
            f"NUMBER reaches {number.max()}, OBJECT_ID holds < {NUMBER_LIMIT}"
        )
    return pa.array(tile * NUMBER_LIMIT + number, mask=number < 0)


def _encodings(file):
    """Return the ``ParquetWriter`` arguments that reproduce the column encodings
    of ``file``, with ``OBJECT_ID`` encoded as ``_healpix_29`` is."""
    group = file.metadata.row_group(0)
    dictionary, encoding = [], {}
    for column in (group.column(i) for i in range(group.num_columns)):
        if "RLE_DICTIONARY" in column.encodings:
            dictionary.append(column.path_in_schema)
        else:
            # RLE alone is the levels' encoding, unless the data use it too
            data = [e for e in column.encodings if e != "RLE"] or ["RLE"]
            encoding[column.path_in_schema] = data[0]
    if SPATIAL_INDEX_COLUMN in dictionary:
        dictionary.append(OBJECT_ID)
    else:
        encoding[OBJECT_ID] = encoding[SPATIAL_INDEX_COLUMN]
    return dict(use_dictionary=dictionary, column_encoding=encoding)


def add_to_pixel(source, output):
    """Rewrite one pixel file with ``OBJECT_ID``, keeping its rows, row groups,
    column encodings and schema metadata."""
    file = pq.ParquetFile(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    writer = None
    for group in range(file.num_row_groups):
        table = file.read_row_group(group)
        ids = object_id(table["TILE_ID"].to_numpy(), table["NUMBER"].to_numpy())
        table = table.add_column(
            table.schema.get_field_index("NUMBER") + 1, OBJECT_ID, ids
        )
        if writer is None:
            writer = pq.ParquetWriter(
                temporary,
                table.schema,
                compression="zstd",
                compression_level=3,
                **_encodings(file),
            )
        writer.write_table(table, row_group_size=table.num_rows)
    writer.close()
    temporary.replace(output)


def _add_to_pixel(arguments):
    add_to_pixel(*arguments)


def write_shapes(source, collection, workers=1):
    """Write the shape HATS catalogue ``source`` as ``COLLECTION/shapes``.

    Each pixel file gains ``OBJECT_ID`` and keeps its rows in order. The
    partitioning, sky maps and properties are those of ``source``, named
    ``shapes``, with ``OBJECT_ID`` among the default columns, which ``lsdb``'s
    ``id_search`` needs.
    """
    source, output = Path(source), Path(collection) / NAME
    paths = sorted((source / "dataset").glob("Norder=*/Dir=*/Npix=*.parquet"))
    jobs = [(path, output / path.relative_to(source)) for path in paths]
    with ProcessPoolExecutor(workers) as pool:
        for done, _ in enumerate(pool.map(_add_to_pixel, jobs), 1):
            if done % 100 == 0:
                print(f"shapes: {done}/{len(jobs)} pixels", flush=True)

    for path in [source / "partition_info.csv", *source.glob("*.fits")]:
        shutil.copy(path, output / path.name)
    properties = TableProperties.read_from_dir(source)
    rows = write_parquet_metadata(output, create_metadata=False)
    if rows != properties.total_rows:
        raise ValueError(f"wrote {rows} rows, {source} has {properties.total_rows}")
    properties.copy_and_update(
        catalog_name=NAME,
        default_columns=[*(properties.default_columns or []), OBJECT_ID],
        **TableProperties.new_provenance_dict(
            output, builder="sp_validation.assembly.object_id"
        ),
    ).to_properties_file(output)
