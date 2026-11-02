"""Small non-cosmological diagnostic products shared by compute and plot steps."""

import healpy as hp
import healsparse
import numpy as np


def footprint_counts(ra, dec, nside_coverage=32, nside_sparse=2048):
    """Bin coordinates once, using a finite HealSparse sentinel for valid pixels."""
    pixels, counts = np.unique(
        hp.ang2pix(nside_sparse, ra, dec, lonlat=True, nest=True), return_counts=True
    )
    count_map = healsparse.HealSparseMap.make_empty(
        nside_coverage, nside_sparse, dtype=np.float32
    )
    count_map[pixels] = counts.astype(np.float32)
    return count_map


def save_footprint(count_map, path):
    """Store occupied nested pixels, counts and both HealSparse resolutions.

    A sparse NPZ avoids writing the plotter's NaN sentinel to a FITS header
    (CFITSIO cannot encode it). No per-object coordinates are retained.
    """
    pixels = count_map.valid_pixels
    np.savez_compressed(
        path,
        pixels=pixels,
        counts=count_map[pixels],
        nside_coverage=count_map.nside_coverage,
        nside_sparse=count_map.nside_sparse,
    )


def load_footprint(path):
    """Reconstruct the sparse map from occupied pixels, without a catalogue."""
    with np.load(path, allow_pickle=False) as data:
        count_map = healsparse.HealSparseMap.make_empty(
            int(data["nside_coverage"]),
            int(data["nside_sparse"]),
            dtype=np.float32,
        )
        count_map[data["pixels"]] = data["counts"]
    return count_map
