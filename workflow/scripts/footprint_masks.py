"""Occupied-pixel footprints of the calibrated sample (HEALPix nside 4096, RING, uint8).

usage: footprint_masks.py <calibrated HATS catalogue> <outdir>

Writes full.fits, the pixels holding any galaxy of the sample, and tomo.fits, the
pixels holding any galaxy on a tile that has at least one galaxy in bins 1-4
(TOM_BIN_ID > 0), so a tile without usable photometry leaves the tomographic
footprint whole.
"""

import glob
import sys
from concurrent.futures import ThreadPoolExecutor

import healpy as hp
import numpy as np
import pyarrow.parquet as pq

NSIDE = 4096


def read(path, ra, dec):
    t = pq.read_table(path, columns=[ra, dec, "TOM_BIN_ID", "TILE_ID"])
    pix = hp.ang2pix(NSIDE, t[ra].to_numpy(), t[dec].to_numpy(), lonlat=True)
    return pix, t["TOM_BIN_ID"].to_numpy(), t["TILE_ID"].to_numpy()


def write(path, pixels, sample):
    m = np.zeros(hp.nside2npix(NSIDE), np.uint8)
    m[pixels] = 1
    header = [("MASKTYPE", "OCCUPIED")] + ([("SAMPLE", sample)] if sample else [])
    hp.write_map(path, m, dtype=np.uint8, overwrite=True, nest=False, extra_header=header)
    area = m.sum() * hp.nside2pixarea(NSIDE, degrees=True)
    print(f"{path}: {int(m.sum())} pixels, {area:.1f} deg2")


def main(catalogue, out):
    files = sorted(glob.glob(f"{catalogue}/dataset/**/*.parquet", recursive=True))
    schema = pq.read_schema(files[0]).names
    dec = "Dec" if "Dec" in schema else "DEC"
    with ThreadPoolExecutor(32) as pool:
        parts = list(pool.map(lambda f: read(f, "RA", dec), files))
    pix, tom_bin, tile = (np.concatenate([p[i] for p in parts]) for i in range(3))
    write(f"{out}/full.fits", pix, None)
    binned_tiles = np.isin(tile, np.unique(tile[tom_bin > 0]))
    write(f"{out}/tomo.fits", pix[binned_tiles], "TOM_BIN_ID>0 tiles")


if __name__ == "__main__":
    main(*sys.argv[1:3])
