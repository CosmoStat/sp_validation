"""Inputs of the SOM n(z): source photometry and the spectroscopic calibrators.

Both come from PhotoPipe's run on Stephen Gwyn's DR6 detections, so sources
and calibrators share one photometry. Sources are the rows of a cosmo_val
shear catalogue; each is found in the ShapePipe HATS catalogue by exact
position, and the PhotoPipe HATS catalogue is row-aligned with those shapes
(joined on tile and DR6 ID). Calibrators are the spectra PhotoPipe matched to
its own objects (``UNIONS.<tile>_ugriz_photoz_ext_specz.cat``), named by
survey through the redshift compilation beside them.
"""

import glob
from multiprocessing import Pool
from pathlib import Path

import fitsio
import healpy as hp
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import nz_calibration as nz

BANDS = ("u", "g", "r", "i", "z", "z2")
MAGS = [f"MAG_GAAP_{band}" for band in BANDS]
HEALPIX_ORDER = 29


def _compilation(path):
    """Read a redshift compilation (ra, dec, zspec, zspec_err, wspec, ref)."""
    table = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=1,
        header=None,
        names=["index", "ra", "dec", "z", "z_err", "w", "survey"],
    )
    return dict(zip(zip(table["ra"].round(7), table["dec"].round(7)), table["survey"]))


def calibrators(specz_dir, compilation, surveys, z_min, offset):
    """Return the calibration sample: spectra from ``surveys`` with complete ugriz.

    ``specz_dir`` holds, per tile, ``UNIONS.<tile>_ugriz_photoz_ext_specz.cat`` and
    ``UNIONS.<tile>_redshifts-<compilation>.asc``. A spectrum matched in several
    overlapping tiles is kept once. Missing z is filled from z2 + ``offset``, as
    for sources.
    """
    columns = ["SeqNr", "ALPHA_J2000_spec", "DELTA_J2000_spec", "z_spec_spec", "Z_B"]
    parts = []
    for path in sorted(glob.glob(f"{specz_dir}/UNIONS.*_ugriz_photoz_ext_specz.cat")):
        tile = Path(path).name.split("_")[0].removeprefix("UNIONS.")
        data = fitsio.read(path, ext=1, columns=columns + MAGS)
        if len(data) == 0:
            continue
        names = _compilation(f"{specz_dir}/UNIONS.{tile}_redshifts-{compilation}.asc")
        survey = np.array(
            [
                names.get((round(ra, 7), round(dec, 7)), "")
                for ra, dec in zip(data["ALPHA_J2000_spec"], data["DELTA_J2000_spec"])
            ]
        )
        parts.append((tile, data, survey))
    data = np.concatenate([d for _, d, _ in parts])
    survey = np.concatenate([s for _, _, s in parts])
    tile = np.concatenate([[t] * len(d) for t, d, _ in parts])
    mag, _ = nz.z_magnitudes(
        np.column_stack([data[m] for m in MAGS]).astype("f8"), offset
    )
    z = data["z_spec_spec"].astype("f8")
    keep = (
        np.isin(survey, surveys)
        & nz.detected(mag).all(1)
        & np.isfinite(data["Z_B"])
        & (z > z_min)
    )
    position = np.round(data["ALPHA_J2000_spec"], 7) * 1e4 + np.round(
        data["DELTA_J2000_spec"], 7
    )
    rows = np.flatnonzero(keep)
    rows = np.sort(rows[np.unique(position[rows], return_index=True)[1]])
    return dict(
        z=z[rows],
        photoz=data["Z_B"][rows].astype("f8"),
        mag=mag[rows],
        survey=survey[rows],
        ra=data["ALPHA_J2000_spec"][rows].astype("f8"),
        dec=data["DELTA_J2000_spec"][rows].astype("f8"),
        tile=tile[rows],
        seqnr=data["SeqNr"][rows],
    )


_SHARED = {}


def _match_pixel(path):
    """Match catalogue rows to one shape HATS pixel file by exact position."""
    shapes, photometry = _SHARED["shapes"], _SHARED["photometry"]
    order, pixel, ra, dec = (_SHARED[k] for k in ("order", "pixel", "ra", "dec"))
    shape = pq.read_table(
        path, columns=["_healpix_29", "XWIN_WORLD", "YWIN_WORLD", "TILE_ID", "NUMBER"]
    )
    h = shape["_healpix_29"].to_numpy()
    lo, hi = np.searchsorted(pixel[order], [h.min(), h.max() + 1])
    if hi == lo:
        return None
    rows = order[lo:hi]
    sort = np.argsort(h, kind="stable")
    start = np.searchsorted(h[sort], pixel[rows], side="left")
    stop = np.searchsorted(h[sort], pixel[rows], side="right")
    x, y = shape["XWIN_WORLD"].to_numpy(), shape["YWIN_WORLD"].to_numpy()
    match = np.full(len(rows), -1, np.int64)
    for k in range(int((stop - start).max())):
        candidate = sort[np.minimum(start + k, len(sort) - 1)]
        same = (
            (start + k < stop)
            & (match < 0)
            & (x[candidate] == ra[rows])
            & (y[candidate] == dec[rows])
        )
        match[same] = candidate[same]
    found = match >= 0
    rows, match = rows[found], match[found]
    photo = pq.read_table(
        Path(photometry) / Path(path).relative_to(shapes),
        columns=["PHOTOPIPE_STATUS", "Z_B"] + MAGS,
    )

    def column(table, name):
        return table[name].to_numpy(zero_copy_only=False)[match]

    return dict(
        row=rows,
        tile=column(shape, "TILE_ID"),
        number=column(shape, "NUMBER"),
        status=np.nan_to_num(column(photo, "PHOTOPIPE_STATUS"), nan=-1),
        photoz=column(photo, "Z_B"),
        mag=np.column_stack([column(photo, m) for m in MAGS]),
    )


def sources(catalogue, shapes, photometry, workers=1):
    """Return PhotoPipe magnitudes, BPZ and ``w_des`` for every catalogue row.

    ``catalogue`` is a cosmo_val shear catalogue (FITS with RA, Dec, w_des);
    ``shapes`` and ``photometry`` are the ShapePipe and PhotoPipe HATS
    catalogue directories. Rows absent from the shapes keep ``number`` -1 and
    NaN photometry; ``status`` is PhotoPipe's join state (0 joined).
    """
    data = fitsio.read(catalogue, ext=1, columns=["RA", "Dec", "w_des"])
    ra, dec = data["RA"].astype("f8"), data["Dec"].astype("f8")
    weight = data["w_des"].astype("f8")
    del data
    pixel = hp.ang2pix(2**HEALPIX_ORDER, ra, dec, nest=True, lonlat=True)
    _SHARED.update(
        shapes=Path(shapes),
        photometry=Path(photometry),
        order=np.argsort(pixel, kind="stable"),
        pixel=pixel,
        ra=ra,
        dec=dec,
    )
    n = len(ra)
    out = dict(
        mag=np.full((n, len(BANDS)), np.nan, np.float32),
        photoz=np.full(n, np.nan),
        tile=np.full(n, np.nan),
        number=np.full(n, -1, np.int32),
        status=np.full(n, -1, np.int8),
    )
    hits = np.zeros(n, np.int8)
    paths = sorted(Path(shapes).glob("dataset/Norder=*/Dir=*/Npix=*.parquet"))
    with Pool(workers) as pool:
        for part in pool.imap_unordered(_match_pixel, paths, chunksize=4):
            if part is None:
                continue
            rows = part.pop("row")
            hits[rows] += 1
            for key, value in part.items():
                out[key][rows] = value
    _SHARED.clear()
    if (hits > 1).any():
        raise ValueError(f"{(hits > 1).sum()} catalogue rows match several shapes")
    out["weight"] = weight
    return out
