# /// script
# requires-python = ">=3.10"
# dependencies = ["pyvo", "astropy", "healpy", "numpy"]
# ///
"""Stage Gaia DR3 stars (G < 20, RUWE < 1.4) over the UNIONS tiles.

Runs on a login node (the query executes on the TAP server); compute nodes
read the staged FITS. The region is every HEALPix level-7 (nside 128,
nested) pixel touched by a tile, from one or more tile lists: a text file of
``XXX.YYY`` names or a JSON manifest whose ``tiles`` mapping is keyed by name.
Gaia ``source_id`` encodes the level-12 nested pixel in its high bits, so each
level-7 pixel is a contiguous ``source_id`` range and the query is a union of
indexed range scans, run synchronously in chunks. Writes ``<out>.fits`` and ``<out>.json`` (ADQL template,
pixel list digest, row count).

Usage::

    uv run scripts/dashboard/stage_gaia.py --tiles tiles.txt MANIFEST.json \\
        --out catpaper/aux/gaia_dr3_G20_ruwe1p4_unions20k
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import healpy as hp
import numpy as np
import pyvo
from astropy.table import vstack

TAP = "https://dc.zah.uni-heidelberg.de/tap"
TABLE = "gaia.dr3lite"
LEVEL = 7
NSIDE = 2**LEVEL
SOURCE_ID_PER_PIXEL = 2**35 * 4 ** (12 - LEVEL)
TILE_HALF_DIAGONAL_DEG = 0.37  # 10000 px x 0.187"/px tiles, half-diagonal
ADQL = (
    "SELECT source_id, ra, dec, phot_g_mean_mag, ruwe FROM {table} "
    "WHERE phot_g_mean_mag < 20 AND ruwe < 1.4 AND ({ranges})"
)


def read_tiles(path):
    path = Path(path)
    if path.suffix == ".json":
        return list(json.loads(path.read_text())["tiles"])
    return [line.strip() for line in path.read_text().split() if line.strip()]


def tile_centre(name):
    """CFIS tile centre (cs_util ``cfis.get_tile_coord_from_nixy``)."""
    xi, yi = (int(part) for part in name.replace("-", ".").split("."))
    dec = yi / 2 - 90
    return xi / 2 / np.cos(np.deg2rad(dec)), dec


def tile_pixels(names):
    pixels = set()
    radius = np.deg2rad(TILE_HALF_DIAGONAL_DEG)
    for name in names:
        ra, dec = tile_centre(name)
        vec = hp.ang2vec(ra, dec, lonlat=True)
        pixels.update(hp.query_disc(NSIDE, vec, radius, inclusive=True, nest=True))
    return np.array(sorted(pixels))


def ranges(pixels):
    """Contiguous pixel runs as (first, last) pairs."""
    breaks = np.flatnonzero(np.diff(pixels) != 1)
    starts = np.r_[pixels[0], pixels[breaks + 1]]
    ends = np.r_[pixels[breaks], pixels[-1]]
    return list(zip(starts, ends))


def where(runs):
    return " OR ".join(
        f"source_id BETWEEN {a * SOURCE_ID_PER_PIXEL} "
        f"AND {(b + 1) * SOURCE_ID_PER_PIXEL - 1}"
        for a, b in runs
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tiles", nargs="+", required=True)
    parser.add_argument("--out", required=True, help="output path without suffix")
    parser.add_argument("--pixels-per-query", type=int, default=300)
    args = parser.parse_args(argv)

    names = sorted({n for path in args.tiles for n in read_tiles(path)})
    pixels = tile_pixels(names)
    print(f"{len(names)} tiles -> {len(pixels)} level-{LEVEL} pixels", flush=True)
    service = pyvo.dal.TAPService(TAP)
    parts = []
    for start in range(0, len(pixels), args.pixels_per_query):
        chunk = pixels[start : start + args.pixels_per_query]
        query = ADQL.format(table=TABLE, ranges=where(ranges(chunk)))
        for attempt in range(5):
            try:
                table = service.run_sync(query, maxrec=16_000_000).to_table()
                break
            except pyvo.dal.DALServiceError as error:
                print(f"retry {attempt + 1}: {error}", flush=True)
                time.sleep(30)
        else:
            raise RuntimeError(f"pixels {start}+ failed after retries")
        print(f"pixels {start}-{start + len(chunk)}: {len(table)} rows", flush=True)
        parts.append(table)
    table = vstack(parts)
    table.meta.clear()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write(out.with_suffix(".fits"), overwrite=True)
    record = {
        "service": TAP,
        "table": TABLE,
        "adql_template": ADQL.format(table=TABLE, ranges="<source_id ranges>"),
        "healpix_level": LEVEL,
        "nest": True,
        "n_tiles": len(names),
        "tile_lists": [str(Path(p).resolve()) for p in args.tiles],
        "n_pixels": int(len(pixels)),
        "pixels_sha256": hashlib.sha256(pixels.astype("<i8").tobytes()).hexdigest(),
        "n_rows": int(len(table)),
        "n_unique_source_id": int(len(np.unique(table["source_id"]))),
    }
    out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
