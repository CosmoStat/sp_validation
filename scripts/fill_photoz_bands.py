#!/usr/bin/env python

"""fill_photoz_bands.py

Add PhotoPipe photo-z and multi-band magnitude fields to a ShapePipe
comprehensive HDF5 catalogue and fill them from PhotoPipe FITS tiles.

Two phases, both resumable via a checkpoint file:

  Phase 1 — Create output: copy the input HDF5 and append the new
             columns initialised to EMPTY_VALUE (-199).
  Phase 2 — Fill tiles: for each PhotoPipe FITS tile found in fits_dir,
             write the field values into the output file.

Skips missing FITS files.  Warns (does not error) on missing PhotoPipe
keys.  Supports interrupt + restart at any point.

:Authors: Martin Kilbinger

"""

import json
import os
import sys
import warnings
from collections import defaultdict
from timeit import default_timer as timer

import h5py
import numpy as np
import tqdm
from astropy.io import fits

from cs_util import args as cs_args
from cs_util import logging


FITS_HDU = 1
EMPTY_VALUE = -199
COPY_CHUNK = 2_000_000   # rows per chunk when copying input → output
SCAN_CHUNK = 5_000_000   # rows per chunk when scanning TILE_ID

REQUESTED_KEYS = [
    "Z_B",
    "Z_B_MIN",
    "Z_B_MAX",
    "T_B",
    "Z_ML",
    "MAG_GAAP_u",
    "MAGERR_GAAP_u",
    "MAG_GAAP_g",
    "MAGERR_GAAP_g",
    "MAG_GAAP_r",
    "MAGERR_GAAP_r",
    "MAG_GAAP_i",
    "MAGERR_GAAP_i",
    "MAG_GAAP_z",
    "MAGERR_GAAP_z",
    "MAG_GAAP_z2",
    "MAGERR_GAAP_z2",
]


def params_default():
    """Params Default.

    Return default parameter values and additional information
    about type and command line options.

    Returns
    -------
    tuple
        parameter dict, short_options dict, types dict, help_strings dict

    """
    params = {
        "input": "unions_shapepipe_comprehensive_struc_2024_v1.5.c.hdf5",
        "output": "unions_shapepipe_comprehensive_struc_ugriz_2024_v1.5.c.hdf5",
        "fits_dir": "UNIONS5000",
        "checkpoint": "fill_photoz_bands_checkpoint.json",
        "verbose": False,
    }

    short_options = {
        "input": "-i",
        "output": "-o",
        "fits_dir": "-d",
        "checkpoint": "-c",
    }

    types = {}

    help_strings = {
        "input": "input HDF5 catalogue (no PhotoPipe fields), default={}",
        "output": "output HDF5 catalogue (PhotoPipe fields added and filled), default={}",
        "fits_dir": "directory with PhotoPipe FITS tiles, default={}",
        "checkpoint": "checkpoint JSON file for resume support, default={}",
    }

    return params, short_options, types, help_strings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_dataset_name(hf):
    """Return the first dataset name in an HDF5 file.

    Tries common names first: dat, dat_comb, data.

    Parameters
    ----------
    hf : h5py.File

    Returns
    -------
    str
        Dataset name.

    Raises
    ------
    KeyError
        If no dataset is found.

    """
    for name in ("dat", "dat_comb", "data"):
        if name in hf:
            return name
    # Fall back to first key
    keys = list(hf.keys())
    if not keys:
        raise KeyError("HDF5 file contains no datasets.")
    return keys[0]


def strip_dtype(dtype):
    """Strip h5py metadata from a structured numpy dtype descriptor.

    h5py sometimes adds encoding metadata to dtype descriptions, e.g.
    ``('<f4', {'h5py_encoding': 'ascii'})``.  This function returns a
    clean list of ``(name, base_dtype)`` tuples suitable for constructing
    a new ``np.dtype``.

    Parameters
    ----------
    dtype : numpy.dtype

    Returns
    -------
    list of (str, str) tuples

    """
    cleaned = []
    for name, dt in dtype.descr:
        if isinstance(dt, tuple):
            cleaned.append((name, dt[0]))
        else:
            cleaned.append((name, dt))
    return cleaned


def build_output_dtype(input_dtype, new_keys):
    """Build combined dtype: input fields + new float32 fields.

    Parameters
    ----------
    input_dtype : numpy.dtype
    new_keys : list of str
        Names for new float32 columns.

    Returns
    -------
    numpy.dtype

    """
    fields = strip_dtype(input_dtype)
    existing = {name for name, _ in fields}
    for key in new_keys:
        if key not in existing:
            fields.append((key, np.float32))
    return np.dtype(fields)


def tile_id_to_fits_path(tile_id_bytes, fits_dir):
    """Convert HDF5 TILE_ID (e.g. b'255.348') to FITS file path."""
    tile_str = f"{float(tile_id_bytes):07.3f}"
    return os.path.join(fits_dir, f"UNIONS.{tile_str}_SP_ugriz_photoz_ext.cat")


def check_fits_keys(fits_columns, requested_keys):
    """Return (valid_keys, missing_keys) after checking FITS column names."""
    fits_cols = set(fits_columns)
    valid = [k for k in requested_keys if k in fits_cols]
    missing = [k for k in requested_keys if k not in fits_cols]
    return valid, missing


def write_tile_to_hdf5(dset, hdf5_indices, fits_data, valid_keys):
    """Write valid_keys from fits_data into dset at hdf5_indices.

    Reads the HDF5 range in one chunk, fills fields in memory, writes
    back.  Handles both contiguous and non-contiguous index ranges.

    Parameters
    ----------
    dset : h5py.Dataset
        Compound HDF5 dataset opened in r+ mode.
    hdf5_indices : numpy.ndarray
        Row indices in dset corresponding to this tile (will be sorted).
    fits_data : numpy.recarray
        Data from the PhotoPipe FITS HDU.
    valid_keys : list of str
        Field names to copy from fits_data into dset.

    """
    sorted_idx = np.sort(hdf5_indices)
    idx_min = int(sorted_idx[0])
    idx_max = int(sorted_idx[-1])
    n_range = idx_max - idx_min + 1

    if n_range == len(sorted_idx):
        # Contiguous block: single read-modify-write
        chunk = dset[idx_min : idx_max + 1]
        for key in valid_keys:
            chunk[key] = fits_data[key]
        dset[idx_min : idx_max + 1] = chunk
    else:
        # Non-contiguous: split into contiguous sub-blocks
        gaps = np.where(np.diff(sorted_idx) > 1)[0] + 1
        blocks = np.split(sorted_idx, gaps)
        fits_offset = 0
        for block in blocks:
            b_min, b_max = int(block[0]), int(block[-1])
            n_block = b_max - b_min + 1
            chunk = dset[b_min : b_max + 1]
            for key in valid_keys:
                chunk[key] = fits_data[key][fits_offset : fits_offset + n_block]
            dset[b_min : b_max + 1] = chunk
            fits_offset += n_block


# ---------------------------------------------------------------------------
# Phase 1: create output file
# ---------------------------------------------------------------------------

def create_output_file(input_path, output_path, dataset_name, verbose=False):
    """Create output HDF5 by copying input and appending empty PhotoPipe fields.

    Parameters
    ----------
    input_path : str
    output_path : str
    dataset_name : str
        Dataset name in the input file (used for output too).
    verbose : bool

    """
    print(f"Phase 1: creating output file '{output_path}'")
    t0 = timer()

    with h5py.File(input_path, "r") as hf_in:
        dset_in = hf_in[dataset_name]
        n_total = dset_in.shape[0]
        dtype_out = build_output_dtype(dset_in.dtype, REQUESTED_KEYS)
        new_keys = [k for k in REQUESTED_KEYS if k not in set(dset_in.dtype.names)]

        print(f"  Input rows   : {n_total:,}")
        print(f"  Input fields : {len(dset_in.dtype.names)}")
        print(f"  New fields   : {new_keys}")
        size_mb = n_total * dtype_out.itemsize / 1_048_576
        print(f"  Output size  : ~{size_mb:,.0f} MB")

        with h5py.File(output_path, "w") as hf_out:
            dset_out = hf_out.create_dataset(
                dataset_name,
                shape=(n_total,),
                dtype=dtype_out,
            )

            # Copy input fields chunk by chunk
            input_fields = dset_in.dtype.names
            with tqdm.tqdm(
                total=n_total, unit="rows", unit_scale=True, desc="  Copying"
            ) as pbar:
                for start in range(0, n_total, COPY_CHUNK):
                    end = min(start + COPY_CHUNK, n_total)
                    chunk_in = dset_in[start:end]
                    chunk_out = np.empty(end - start, dtype=dtype_out)
                    for field in input_fields:
                        chunk_out[field] = chunk_in[field]
                    for key in new_keys:
                        chunk_out[key] = EMPTY_VALUE
                    dset_out[start:end] = chunk_out
                    pbar.update(end - start)

            # Copy all other top-level datasets/groups unchanged
            for key in hf_in.keys():
                if key != dataset_name:
                    hf_in.copy(key, hf_out)
                    if verbose:
                        print(f"  Copied group/dataset '{key}' unchanged.")

    elapsed = timer() - t0
    print(f"  Done in {elapsed:.1f}s\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    """Main.

    Main program.

    """
    params, short_options, types, help_strings = params_default()

    options = cs_args.parse_options(params, short_options, types, help_strings)
    params.update(options)

    logging.log_command(argv)

    verbose = params["verbose"]

    if not os.path.exists(params["input"]):
        print(f"ERROR: input file not found: {params['input']}", file=sys.stderr)
        return 1
    if not os.path.isdir(params["fits_dir"]):
        print(f"ERROR: FITS directory not found: {params['fits_dir']}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    if os.path.exists(params["checkpoint"]):
        with open(params["checkpoint"]) as f:
            checkpoint = json.load(f)
        done_tiles = set(checkpoint.get("done_tiles", []))
        print(f"Resuming: {len(done_tiles)} tiles already completed.")
    else:
        done_tiles = set()
        checkpoint = {}

    # ------------------------------------------------------------------
    # Detect input dataset name (cache in checkpoint)
    # ------------------------------------------------------------------
    if "dataset_name" in checkpoint:
        dataset_name = checkpoint["dataset_name"]
    else:
        with h5py.File(params["input"], "r") as hf:
            dataset_name = detect_dataset_name(hf)
        checkpoint["dataset_name"] = dataset_name
        with open(params["checkpoint"], "w") as cf:
            json.dump(checkpoint, cf)

    if verbose:
        print(f"Input dataset: '{dataset_name}'")

    # ------------------------------------------------------------------
    # Phase 1: create output file if needed
    # ------------------------------------------------------------------
    if not checkpoint.get("output_created", False):
        if os.path.exists(params["output"]):
            print(
                f"WARNING: output file '{params['output']}' exists but checkpoint "
                "does not mark it as complete. Overwriting."
            )
        create_output_file(
            params["input"], params["output"], dataset_name, verbose=verbose
        )
        checkpoint["output_created"] = True
        with open(params["checkpoint"], "w") as cf:
            json.dump(checkpoint, cf)
    else:
        if verbose:
            print("Phase 1 already done (output file exists in checkpoint).")

    # ------------------------------------------------------------------
    # Phase 2: fill tiles
    # ------------------------------------------------------------------
    t0 = timer()
    with h5py.File(params["output"], "r+") as hf:
        dset = hf[dataset_name]
        n_total = dset.shape[0]
        print(f"Phase 2: filling tiles in '{params['output']}'")
        print(f"  {n_total:,} rows, dataset '{dataset_name}'")

        hdf5_fields = set(dset.dtype.names)

        # Check all requested keys are present
        for key in REQUESTED_KEYS:
            if key not in hdf5_fields:
                warnings.warn(
                    f"Key '{key}' missing from output dataset — was Phase 1 complete?"
                )

        # Build TILE_ID → output row indices (always scanned; too large for checkpoint)
        print("  Building tile→index map (scans all rows)...")
        tile_index_map_lists = defaultdict(list)
        with tqdm.tqdm(
            total=n_total, unit="rows", unit_scale=True, desc="  Scanning TILE_ID"
        ) as pbar:
            for start in range(0, n_total, SCAN_CHUNK):
                end = min(start + SCAN_CHUNK, n_total)
                tile_chunk = dset[start:end]["TILE_ID"]
                for local_i, tid in enumerate(tile_chunk):
                    tile_index_map_lists[tid].append(start + local_i)
                pbar.update(end - start)

        tile_index_map = {
            tid: np.array(idxs, dtype=np.int64)
            for tid, idxs in tile_index_map_lists.items()
        }
        print(f"  Map built: {len(tile_index_map)} unique tiles.")

        unique_tiles = sorted(tile_index_map.keys())
        n_tiles = len(unique_tiles)

        valid_keys = None   # determined from first available FITS tile
        n_skipped_missing = 0
        n_skipped_done = 0
        n_skipped_size = 0
        n_processed = 0

        print(f"\n  Processing {n_tiles} tiles ({len(done_tiles)} already done)...\n")
        pbar = tqdm.tqdm(unique_tiles, total=n_tiles, unit="tile")

        for tile_id in pbar:
            tile_str = tile_id.decode() if isinstance(tile_id, bytes) else tile_id

            if tile_str in done_tiles:
                n_skipped_done += 1
                continue

            fits_path = tile_id_to_fits_path(tile_id, params["fits_dir"])
            if not os.path.exists(fits_path):
                n_skipped_missing += 1
                done_tiles.add(tile_str)
                pbar.set_postfix({"done": n_processed, "missing": n_skipped_missing})
                continue

            try:
                with fits.open(fits_path, memmap=True) as hdu_list:
                    fits_data = hdu_list[FITS_HDU].data

                    # Validate keys on first successfully opened tile
                    if valid_keys is None:
                        valid_keys, missing_keys = check_fits_keys(
                            fits_data.dtype.names, REQUESTED_KEYS
                        )
                        valid_keys = [k for k in valid_keys if k in hdf5_fields]
                        if missing_keys:
                            warnings.warn(
                                f"Keys absent from PhotoPipe FITS (skipped): {missing_keys}"
                            )
                        tqdm.tqdm.write(f"\n  Keys to fill: {valid_keys}\n")

                    hdf5_indices = tile_index_map[tile_id]
                    n_hdf5 = len(hdf5_indices)
                    n_fits = len(fits_data)

                    if n_hdf5 != n_fits:
                        warnings.warn(
                            f"Tile {tile_str}: HDF5 has {n_hdf5} rows, FITS has "
                            f"{n_fits} — size mismatch, skipping."
                        )
                        n_skipped_size += 1
                        pbar.set_postfix(
                            {"done": n_processed, "size_err": n_skipped_size}
                        )
                        continue

                    write_tile_to_hdf5(dset, hdf5_indices, fits_data, valid_keys)

            except Exception as e:
                warnings.warn(f"Tile {tile_str}: error ({e}), skipping.")
                continue

            n_processed += 1
            done_tiles.add(tile_str)

            if n_processed % 50 == 0:
                checkpoint["done_tiles"] = list(done_tiles)
                with open(params["checkpoint"], "w") as cf:
                    json.dump(checkpoint, cf)

            pbar.set_postfix({"done": n_processed, "missing": n_skipped_missing})

        # Final checkpoint flush
        checkpoint["done_tiles"] = list(done_tiles)
        with open(params["checkpoint"], "w") as cf:
            json.dump(checkpoint, cf)

    elapsed = timer() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Tiles processed       : {n_processed}")
    print(f"  Tiles skipped (done)  : {n_skipped_done}")
    print(f"  FITS files missing    : {n_skipped_missing}")
    print(f"  Size mismatches       : {n_skipped_size}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
