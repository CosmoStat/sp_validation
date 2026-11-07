#!/usr/bin/env python
"""Compare frozen SOM assignment on the merged magnitudes with reference TOM_BIN_IDs.

The reference comes from the _SP_ugriz PhotoPipe run and the merge from the run
on Stephen Gwyn's detections, so objects near a bin edge legitimately differ.
Row-level agreement measures how far the two photometries are apart.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from sp_validation import nz_calibration as nz

BANDS = ("u", "g", "r", "i", "z", "z2")
Z2_OFFSET = -0.1104888916015625
N_BINS = 5


def row_key(tile_ids, numbers):
    """Combine a tile's integer form (314253) with NUMBER into one sortable key."""
    return tile_ids.astype(np.int64) * (1 << 24) + numbers.astype(np.int64)


def load_reference(path):
    table = pq.read_table(path, columns=["TILE_ID", "NUMBER", "TOM_BIN_ID"])
    tiles = np.char.replace(table["TILE_ID"].to_numpy().astype("U7"), ".", "").astype(np.int64)
    keys = row_key(tiles, table["NUMBER"].to_numpy())
    order = np.argsort(keys)
    return keys[order], table["TOM_BIN_ID"].to_numpy()[order]


def check_pixel(path):
    columns = ["TILE_ID", "NUMBER", "PHOTOPIPE_STATUS", "Z_B"] + [f"MAG_GAAP_{b}" for b in BANDS]
    table = pq.read_table(path, columns=columns)
    tile = np.rint(table["TILE_ID"].to_numpy() * 1000).astype(np.int64)
    keys = row_key(tile, table["NUMBER"].to_numpy())
    position = np.searchsorted(REFERENCE_KEYS, keys)
    position[position == len(REFERENCE_KEYS)] = 0
    shared = np.flatnonzero(REFERENCE_KEYS[position] == keys)
    expected = REFERENCE_BINS[position[shared]]
    status = table["PHOTOPIPE_STATUS"].to_numpy()[shared]
    mag = np.column_stack(
        [table[f"MAG_GAAP_{b}"].to_numpy(zero_copy_only=False)[shared] for b in BANDS]
    ).astype("f4")
    zb = table["Z_B"].to_numpy(zero_copy_only=False)[shared].astype("f8")
    prelim, group, _ = nz.source_groups(
        mag, zb, np.ones(len(shared)), MODEL, offset=Z2_OFFSET, use_z2=True
    )
    found = nz.tom_bin_ids(prelim, group, MODEL["gold"])
    confusion = np.zeros((N_BINS, N_BINS), np.int64)
    np.add.at(confusion, (expected, found), 1)
    unmatched = np.bincount(expected[status != 0], minlength=N_BINS)
    differ = found != expected
    tiles, tile_mismatch = np.unique(tile[shared][differ], return_counts=True)
    return confusion, unmatched, len(shared), dict(zip(tiles.tolist(), tile_mismatch.tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("product", help="merge output directory")
    parser.add_argument("--model", required=True, help="som_assignment_model.npz")
    parser.add_argument("--reference", required=True, help="tom_bin_full.parquet")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    global MODEL, REFERENCE_KEYS, REFERENCE_BINS
    MODEL = dict(np.load(args.model))
    REFERENCE_KEYS, REFERENCE_BINS = load_reference(args.reference)
    product = Path(args.product)
    paths = sorted((product / "catalogue/dataset").glob("Norder=*/Dir=*/Npix=*.parquet"))
    confusion = np.zeros((N_BINS, N_BINS), np.int64)
    unmatched = np.zeros(N_BINS, np.int64)
    shared, by_tile = 0, {}
    with ProcessPoolExecutor(args.workers) as pool:
        for c, u, n, t in pool.map(check_pixel, paths):
            confusion += c
            unmatched += u
            shared += n
            for tile, count in t.items():
                by_tile[tile] = by_tile.get(tile, 0) + count
    mismatches = shared - np.trace(confusion)
    result = dict(
        reference_rows=len(REFERENCE_KEYS),
        shared_rows=shared,
        mismatches=int(mismatches),
        mismatch_fraction=mismatches / shared,
        confusion_reference_by_merged=confusion.tolist(),
        reference_bin_rows_unmatched=unmatched.tolist(),
        mismatch_by_tile={f"{k / 1000:07.3f}": v for k, v in by_tile.items()},
    )
    (product / "tom_bin_check.json").write_text(json.dumps(result, indent=1) + "\n")
    print({k: v for k, v in result.items() if k != "mismatch_by_tile"})


if __name__ == "__main__":
    main()
