#!/usr/bin/env python
"""Extract PhotoPipe science columns from the DR6 tile tree, next to the data.

Runs in a CANFAR headless session with /arc mounted. Writes one Parquet shard
per RA prefix (``ra=314.parquet``, one row group per tile) and a JSON manifest
beside it naming every file read, with its size, mtime and read time. A shard
whose sources are unchanged is skipped, so rerunning picks up only new vintages.

Everything comes from ``_ugriz_photoz_ext.cat`` (dereddened GAaP magnitudes);
``MASK`` comes from ``_ugriz_photoz_finalmask.cat`` where that file exists and
is null otherwise. These are PhotoPipe's runs on Stephen Gwyn's DR6 detections,
so ``SeqNr`` is his ``NUMBER``, the ID the shape catalogue carries. The
``_SP_ugriz`` files are a separate run on other detections, with their own
numbering.
"""

import argparse
import collections
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fitsio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/arc/projects/unions/catalogues/unions/GAaP_photometry/UNIONS_DR6"
EXT = "_ugriz_photoz_ext.cat"
FINALMASK = "_ugriz_photoz_finalmask.cat"
BANDS = ("u", "g", "r", "i", "z", "z2")
COLUMNS = ["SeqNr", "ALPHA_J2000", "DELTA_J2000"]
for prefix in ("MAG_GAAP", "MAGERR_GAAP", "FLAG_GAAP", "MAG_LIM", "EXTINCTION"):
    COLUMNS += [f"{prefix}_{band}" for band in BANDS]
COLUMNS += [
    "Z_B",
    "Z_B_MIN",
    "Z_B_MAX",
    "T_B",
    "ODDS",
    "Z_ML",
    "T_ML",
    "CHI_SQUARED_BPZ",
    "M_0",
    "BPZ_FILT",
    "NBPZ_FILT",
    "BPZ_NONDETFILT",
    "NBPZ_NONDETFILT",
    "BPZ_FLAGFILT",
    "NBPZ_FLAGFILT",
]


def stat(path):
    try:
        info = os.stat(path)
    except FileNotFoundError:
        return None
    return dict(name=path.name, size=info.st_size, mtime=info.st_mtime)


def sources(root, tile):
    directory = root / f"UNIONS.{tile}"
    return dict(
        ext=stat(directory / f"UNIONS.{tile}{EXT}"),
        finalmask=stat(directory / f"UNIONS.{tile}{FINALMASK}"),
    )


def native(array):
    return array.astype(array.dtype.newbyteorder("="))


def read_tile(root, tile):
    directory = root / f"UNIONS.{tile}"
    data = fitsio.read(directory / f"UNIONS.{tile}{EXT}", ext=1, columns=COLUMNS)
    arrays = {name: pa.array(native(data[name])) for name in COLUMNS}
    path = directory / f"UNIONS.{tile}{FINALMASK}"
    if path.exists():
        masked = fitsio.read(path, ext=1, columns=["SeqNr", "MASK"])
        if not np.array_equal(masked["SeqNr"], data["SeqNr"]):
            raise ValueError("finalmask rows differ from ext")
        arrays["MASK"] = pa.array(native(masked["MASK"]))
    else:
        arrays["MASK"] = pa.nulls(len(data), pa.int32())
    return pa.table(arrays)


def extract_shard(root, output, prefix, tiles):
    shard = output / f"ra={prefix}.parquet"
    manifest_path = output / f"ra={prefix}.json"
    current = {tile: sources(root, tile) for tile in tiles}
    if manifest_path.exists() and shard.exists():
        previous = json.loads(manifest_path.read_text())["tiles"]
        unreadable = any(r["status"] == "unreadable" for r in previous.values())
        if not unreadable and {t: previous[t]["sources"] for t in previous} == current:
            return prefix, "unchanged"
    records = {}
    writer = None
    temporary = shard.with_suffix(".tmp")
    row_group = 0
    for tile in tiles:
        record = dict(sources=current[tile], read_at=time.time(), rows=0, row_group=None)
        records[tile] = record
        if current[tile]["ext"] is None:
            record["status"] = "missing"
            continue
        try:
            table = read_tile(root, tile)
        except (OSError, ValueError) as error:
            record.update(status="unreadable", error=repr(error))
            continue
        record.update(status="ok", rows=table.num_rows)
        if not table.num_rows:
            continue
        if writer is None:
            writer = pq.ParquetWriter(
                temporary, table.schema, compression="zstd", compression_level=3
            )
        writer.write_table(table, row_group_size=table.num_rows)
        record["row_group"] = row_group
        row_group += 1
    if writer is not None:
        writer.close()
        temporary.replace(shard)
    manifest_path.write_text(json.dumps(dict(prefix=prefix, tiles=records)))
    return prefix, "extracted"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    shards = collections.defaultdict(list)
    for entry in sorted(os.listdir(root)):
        if entry.startswith("UNIONS."):
            tile = entry.removeprefix("UNIONS.")
            shards[tile.split(".")[0]].append(tile)
    print(f"{sum(map(len, shards.values()))} tiles in {len(shards)} shards", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [
            pool.submit(extract_shard, root, output, prefix, tiles)
            for prefix, tiles in shards.items()
        ]
        for done, future in enumerate(as_completed(futures), 1):
            prefix, state = future.result()
            print(f"{done}/{len(futures)} {prefix} {state}", flush=True)
    (output / "COMPLETE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


if __name__ == "__main__":
    main()
