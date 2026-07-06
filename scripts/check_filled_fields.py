#!/usr/bin/env python3
"""Check how many entries in selected HDF5 fields are filled (value != -199).

Reads in chunks with a progress bar and periodic fill-fraction updates.
"""

import h5py
import numpy as np
from tqdm import tqdm

HDF5_FILE = "unions_shapepipe_comprehensive_struc_ugriz_2024_v1.6.c.DR6.hdf5"
EMPTY_VALUE = -199
CHUNK_SIZE = 5_000_000   # rows per chunk
REPORT_EVERY = 10        # print running fractions every N chunks

FIELDS = [
    "Z_B",
    "Z_B_MIN",
    "Z_B_MAX",
    "T_B",
    "MAG_GAAP_0p7_u",
    "MAG_GAAP_1p0_u",
    "MAG_GAAP_0p7_g",
    "MAG_GAAP_1p0_g",
    "MAG_GAAP_0p7_r",
    "MAG_GAAP_1p0_r",
    "MAG_GAAP_0p7_i",
    "MAG_GAAP_1p0_i",
    "MAG_GAAP_0p7_z",
    "MAG_GAAP_1p0_z",
    "MAG_GAAP_0p7_z2",
    "MAG_GAAP_1p0_z2",
]

with h5py.File(HDF5_FILE, "r") as f:
    data = f["data"]
    n_total = data.shape[0]
    n_chunks = (n_total + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Total entries : {n_total:,}")
    print(f"Chunk size    : {CHUNK_SIZE:,}  ({n_chunks} chunks)\n")

    counts = {field: 0 for field in FIELDS}

    with tqdm(total=n_total, unit="rows", unit_scale=True, desc="Reading") as pbar:
        for chunk_idx in range(n_chunks):
            start = chunk_idx * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, n_total)

            for field in FIELDS:
                counts[field] += int(np.sum(data[field, start:end] != EMPTY_VALUE))

            pbar.update(end - start)

            # Periodic running-fraction report
            if (chunk_idx + 1) % REPORT_EVERY == 0 or (chunk_idx + 1) == n_chunks:
                rows_done = end
                tqdm.write(f"\n  --- after {rows_done:,} rows ({100*rows_done/n_total:.1f}%) ---")
                tqdm.write(f"  {'Field':<22} {'Filled %':>9}")
                for field in FIELDS:
                    pct = 100.0 * counts[field] / rows_done
                    tqdm.write(f"  {field:<22} {pct:>8.2f}%")

# Final summary
print(f"\n{'='*56}")
print(f"FINAL SUMMARY  (total rows: {n_total:,})")
print(f"{'Field':<22} {'Filled':>12} {'Empty':>12} {'Filled %':>10}")
print("-" * 60)
for field in FIELDS:
    n_filled = counts[field]
    n_empty = n_total - n_filled
    pct = 100.0 * n_filled / n_total
    print(f"{field:<22} {n_filled:>12,} {n_empty:>12,} {pct:>9.2f}%")
