"""Attach DR6 photometry to selected shear-catalogue rows by sky position."""

from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from astropy.io import fits
from scipy.spatial import cKDTree

BANDS = ("u", "g", "r", "i", "z", "z2")
MATCH_RADIUS_DEG = 0.8 / 3600
DR6_SUFFIX = "_SP_ugriz_photoz_ext.cat"


def _read_dr6_tile(path):
    columns = ["ALPHA_J2000", "DELTA_J2000"]
    columns += [f"MAG_GAAP_{band}" for band in BANDS]
    columns += ["Z_B"]
    with fits.open(path, memmap=True) as hdus:
        data = hdus[1].data
        values = {name: np.asarray(data[name]) for name in columns}
    return (
        values["ALPHA_J2000"].astype("f8"),
        values["DELTA_J2000"].astype("f8"),
        np.column_stack([values[f"MAG_GAAP_{band}"] for band in BANDS]).astype("f4"),
        values["Z_B"].astype("f8"),
    )


def _match_tile(ra, dec, photo_ra, photo_dec):
    """Return nearest DR6 row and separation for each position in one tile."""
    mid_dec = np.median(photo_dec)
    cos_dec = np.cos(np.radians(mid_dec))
    ra_origin = np.median(photo_ra) if np.ptp(photo_ra) < 180 else 0
    photo_x = ((photo_ra - ra_origin + 180) % 360 - 180) * cos_dec
    query_x = ((ra - ra_origin + 180) % 360 - 180) * cos_dec
    return cKDTree(np.c_[photo_x, photo_dec]).query(np.c_[query_x, dec])


def prepare_nz_photometry(selection_path, dr6_root, output_path):
    """Write matched magnitudes, BPZ, shear weights and source row keys.

    The Parquet row order is retained. Missing and unreadable DR6 tiles are
    reported and leave their selection rows unmatched.
    """
    selection = pq.read_table(
        selection_path,
        columns=["TILE_ID", "NUMBER", "RA", "Dec", "w_des"],
    )
    encoded = pc.dictionary_encode(selection["TILE_ID"]).combine_chunks()
    tile_names = np.asarray(encoded.dictionary.to_pylist(), dtype=str)
    tile_codes = encoded.indices.to_numpy()
    name_order = np.argsort(tile_names)
    tile_names = tile_names[name_order]
    code_remap = np.empty_like(name_order)
    code_remap[name_order] = np.arange(len(name_order))
    tile_codes = code_remap[tile_codes]
    row_order = np.argsort(tile_codes, kind="stable")
    tile_offsets = np.searchsorted(
        tile_codes[row_order], np.arange(len(tile_names) + 1)
    )
    nrows = len(tile_codes)
    ra = selection["RA"].to_numpy().astype("f8")
    dec = selection["Dec"].to_numpy().astype("f8")

    mag = np.full((nrows, len(BANDS)), np.nan, dtype="f4")
    zb = np.full(nrows, np.nan, dtype="f8")
    matched_rows = np.zeros(nrows, dtype=bool)
    tile_has_dr6 = np.zeros(len(tile_names), dtype=bool)
    root = Path(dr6_root)

    for tile_index, tile in enumerate(tile_names):
        path = root / f"UNIONS.{tile}{DR6_SUFFIX}"
        if not path.is_file():
            print(f"Missing DR6 tile {tile}: {path}")
            continue
        try:
            photo_ra, photo_dec, photo_mag, photo_zb = _read_dr6_tile(path)
        except (OSError, ValueError, fits.verify.VerifyError) as error:
            print(f"Unreadable DR6 tile {tile}: {error}")
            continue
        tile_has_dr6[tile_index] = True
        rows = row_order[tile_offsets[tile_index] : tile_offsets[tile_index + 1]]
        if not len(rows) or not len(photo_ra):
            continue
        distance, nearest = _match_tile(ra[rows], dec[rows], photo_ra, photo_dec)
        matched = distance < MATCH_RADIUS_DEG
        rows = rows[matched]
        nearest = nearest[matched]
        mag[rows] = photo_mag[nearest]
        zb[rows] = photo_zb[nearest]
        matched_rows[rows] = True

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        tile_names=tile_names,
        tile=tile_codes.astype(np.int32),
        mag=mag,
        zb=zb,
        number=selection["NUMBER"].to_numpy(),
        weight=selection["w_des"].to_numpy().astype("f8"),
    )
    missing = len(tile_names) - tile_has_dr6.sum()
    print(
        f"Matched {matched_rows.sum()} of {nrows} rows; {missing} tiles missing or unreadable"
    )
