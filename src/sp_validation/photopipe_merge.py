"""Give every shape object its PhotoPipe row, on the shape catalogue's HATS partitions.

The inputs are the shape HATS catalogue and the column extract written by
``scripts/assembly/extract_photopipe.py``. The output is a HATS catalogue with
one row for each shape row, in the same pixel, keyed by ``_source_row`` and by
the DR6 object ID ``OBJECT_ID`` (``sp_validation.object_id``); a reader joins
the two on either key within a pixel. By convention the rows of each
pixel file are also in the shape file's order. PhotoPipe columns are null where
``PHOTOPIPE_STATUS`` is not ``MATCHED``.

Both catalogues number objects with Stephen Gwyn's DR6 IDs, so the join is exact
on (tile, ``NUMBER`` = ``SeqNr``). Position plays no part in it:
``MATCH_SEP_ARCSEC`` records the separation of each joined pair, and per-tile
counts of pairs beyond ``SANITY_RADIUS_ARCSEC`` expose a tile whose IDs disagree.
"""

import json
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sp_validation.object_id import with_object_id

SANITY_RADIUS_ARCSEC = 0.8
MATCHED, NO_COUNTERPART, NO_TILE = 0, 1, 2
SHAPE_COLUMNS = [
    "_healpix_29",
    "XWIN_WORLD",
    "YWIN_WORLD",
    "TILE_ID",
    "NUMBER",
    "_source_row",
]
POSITIONS = ["ALPHA_J2000", "DELTA_J2000"]


def tile_name(tile_id):
    """Return the tile string of a shape catalogue's numeric TILE_ID."""
    return f"{tile_id:07.3f}"


def load_extract(extract):
    """Index the extract by tile: shard path, row group and provenance."""
    index = {}
    for manifest in sorted(Path(extract).glob("ra=*.json")):
        content = json.loads(manifest.read_text())
        shard = manifest.with_suffix(".parquet")
        for tile, record in content["tiles"].items():
            index[tile] = dict(record, shard=str(shard))
    return index


def _photopipe_schema(index):
    shard = next(r["shard"] for r in index.values() if r["row_group"] is not None)
    return pq.read_schema(shard)


def _separation_arcsec(ra, dec, other_ra, other_dec):
    delta_ra = (ra - other_ra + 180) % 360 - 180
    return np.hypot(delta_ra * np.cos(np.radians(dec)), dec - other_dec) * 3600


def merge_pixel(shape_path, output_path, index, schema):
    """Write the matched table of one pixel and return per-tile match counts."""
    shapes = pq.read_table(shape_path, columns=SHAPE_COLUMNS)
    ra = shapes["XWIN_WORLD"].to_numpy()
    dec = shapes["YWIN_WORLD"].to_numpy()
    tile_ids, tile_of_row = np.unique(shapes["TILE_ID"].to_numpy(), return_inverse=True)

    status = np.full(len(ra), NO_TILE, np.int8)
    separation = np.full(len(ra), np.nan, np.float32)
    photo_row = np.zeros(len(ra), np.int64)
    pieces, offset, counts = [], 0, []
    for code, tile_id in enumerate(tile_ids):
        tile = tile_name(tile_id)
        rows = np.flatnonzero(tile_of_row == code)
        record = index.get(tile)
        if record is None or record["row_group"] is None:
            counts.append(
                dict(tile=tile, n_shape=len(rows), n_matched=0, n_far=0, has_tile=False)
            )
            continue
        photo = pq.ParquetFile(record["shard"]).read_row_group(record["row_group"])
        ids = photo["SeqNr"].to_numpy()
        numbers = shapes["NUMBER"].to_numpy()[rows]
        position = np.minimum(np.searchsorted(ids, numbers), len(ids) - 1)
        matched = ids[position] == numbers
        pairs = position[matched]
        distance = _separation_arcsec(
            ra[rows[matched]],
            dec[rows[matched]],
            photo["ALPHA_J2000"].to_numpy()[pairs],
            photo["DELTA_J2000"].to_numpy()[pairs],
        )
        status[rows] = np.where(matched, MATCHED, NO_COUNTERPART)
        separation[rows[matched]] = distance
        photo_row[rows[matched]] = pairs + offset
        pieces.append(photo)
        offset += photo.num_rows
        counts.append(
            dict(
                tile=tile,
                n_shape=len(rows),
                n_matched=int(matched.sum()),
                n_far=int((distance > SANITY_RADIUS_ARCSEC).sum()),
                has_tile=True,
            )
        )

    photo = pa.concat_tables(pieces) if pieces else schema.empty_table()
    taken = photo.take(pa.array(photo_row, mask=status != MATCHED))
    table = with_object_id(shapes)
    table = table.append_column("PHOTOPIPE_STATUS", pa.array(status))
    table = table.append_column(
        "MATCH_SEP_ARCSEC", pa.array(separation, mask=status != MATCHED)
    )
    for name in taken.column_names:
        if name not in POSITIONS:
            target = "PHOTOPIPE_SEQNR" if name == "SeqNr" else name
            table = table.append_column(target, taken[name])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    pq.write_table(
        table,
        temporary,
        compression="zstd",
        compression_level=3,
        row_group_size=100_000,
    )
    temporary.replace(output_path)
    return counts, table.schema


def _write_hats_metadata(shapes, catalogue, schema, nrows):
    for name in ("partition_info.csv", "point_map.fits", "skymap.fits"):
        shutil.copy(shapes / name, catalogue / name)
    lines = (shapes / "hats.properties").read_text().splitlines()
    replacements = dict(
        obs_collection="unions_full_photopipe",
        hats_cols_default="XWIN_WORLD YWIN_WORLD PHOTOPIPE_STATUS Z_B",
        hats_builder="sp_validation.photopipe_merge",
    )
    updated = []
    for line in lines:
        key = line.split("=")[0]
        updated.append(f"{key}={replacements[key]}" if key in replacements else line)
    (catalogue / "hats.properties").write_text("\n".join(updated) + "\n")
    shutil.copy(catalogue / "hats.properties", catalogue / "properties")
    pq.write_metadata(schema, catalogue / "dataset" / "_common_metadata")
    if nrows != int(_property(shapes, "hats_nrows")):
        raise ValueError(f"merged {nrows} rows, shape catalogue has a different count")


def _property(catalogue, key):
    for line in (catalogue / "hats.properties").read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1]
    raise KeyError(key)


def _commit():
    root = Path(__file__).parent
    return subprocess.run(
        ["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def _write_provenance(output, shapes, extract, index, counts):
    tiles = sorted(index)
    pq.write_table(
        pa.table(
            dict(
                tile=tiles,
                status=[index[t]["status"] for t in tiles],
                rows=[index[t]["rows"] for t in tiles],
                read_at=[index[t]["read_at"] for t in tiles],
                ext_mtime=[
                    (index[t]["sources"]["ext"] or {}).get("mtime") for t in tiles
                ],
                ext_size=[
                    (index[t]["sources"]["ext"] or {}).get("size") for t in tiles
                ],
                finalmask_mtime=[
                    (index[t]["sources"]["finalmask"] or {}).get("mtime") for t in tiles
                ],
            )
        ),
        output / "photopipe_sources.parquet",
    )
    totals = {}
    for pixel_counts in counts:
        for entry in pixel_counts:
            total = totals.setdefault(
                entry["tile"],
                dict(n_shape=0, n_matched=0, n_far=0, has_tile=entry["has_tile"]),
            )
            total["n_shape"] += entry["n_shape"]
            total["n_matched"] += entry["n_matched"]
            total["n_far"] += entry["n_far"]
    tiles = sorted(totals)
    pq.write_table(
        pa.table(
            dict(
                tile=tiles,
                n_shape=[totals[t]["n_shape"] for t in tiles],
                n_matched=[totals[t]["n_matched"] for t in tiles],
                n_far=[totals[t]["n_far"] for t in tiles],
                has_tile=[totals[t]["has_tile"] for t in tiles],
            )
        ),
        output / "match_by_tile.parquet",
    )
    n_shape = sum(t["n_shape"] for t in totals.values())
    n_matched = sum(t["n_matched"] for t in totals.values())
    build = dict(
        built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        code_commit=_commit(),
        shapes=str(shapes),
        shapes_source_manifest_sha256=_property(shapes, "source_manifest_sha256"),
        extract=str(extract),
        join="(TILE_ID, NUMBER) = (tile, SeqNr)",
        sanity_radius_arcsec=SANITY_RADIUS_ARCSEC,
        n_shape=n_shape,
        n_matched=n_matched,
        n_far=sum(t["n_far"] for t in totals.values()),
        n_tiles_shape=len(totals),
        n_tiles_without_photopipe=sum(not t["has_tile"] for t in totals.values()),
    )
    (output / "BUILD.json").write_text(json.dumps(build, indent=2) + "\n")
    return build


def _merge_worker(arguments):
    return merge_pixel(*arguments)


def merge(shapes, extract, output, workers=1, pixels=None):
    """Build the matched HATS catalogue under ``output/catalogue``."""
    shapes, output = Path(shapes), Path(output)
    catalogue = output / "catalogue"
    index = load_extract(extract)
    schema = _photopipe_schema(index)
    paths = sorted((shapes / "dataset").glob("Norder=*/Dir=*/Npix=*.parquet"))
    if pixels is not None:
        paths = [p for p in paths if int(p.stem.split("=")[1]) in pixels]
    jobs = [
        (path, catalogue / path.relative_to(shapes), index, schema) for path in paths
    ]
    with ProcessPoolExecutor(workers) as pool:
        results = []
        for done, result in enumerate(pool.map(_merge_worker, jobs), 1):
            results.append(result)
            if done % 100 == 0:
                print(f"{done}/{len(jobs)} pixels", flush=True)
    counts = [r[0] for r in results]
    build = _write_provenance(output, shapes, extract, index, counts)
    if pixels is None:
        _write_hats_metadata(shapes, catalogue, results[0][1], build["n_shape"])
    print(json.dumps(build, indent=2))
    return build
