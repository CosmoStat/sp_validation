"""Inputs of the SOM n(z): source photometry and the spectroscopic calibrators.

Both come from PhotoPipe's run on Stephen Gwyn's DR6 detections, so sources
and calibrators share one photometry. Sources are the rows of a calibrated
shear catalogue; each takes its photometry from the v2 collection (shapes
with their PhotoPipe extension, :mod:`sp_validation.hats_collection`) by its
DR6 object ID. Calibrators are the spectra PhotoPipe matched to its own
objects (``UNIONS.<tile>_ugriz_photoz_ext_specz.cat``), named by survey
through the redshift compilation beside them.

@sc [decision:nz.photometry_as_delivered]
"""

import glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitsio
import numpy as np
import pandas as pd

from ..io import Catalogue, read_catalogue
from ..object_id import OBJECT_ID, object_id
from . import calibration as nz

BANDS = ("u", "g", "r", "i", "z", "z2")
MAGS = [f"MAG_GAAP_{band}" for band in BANDS]
#: ``status`` of a source the photometry was not looked up for or lacks.
NO_ID, UNMATCHED = -2, -1


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

    @sc [decision:nz.calibrator_set]
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


def sources(catalogue, photometry, workers=1):
    """Return PhotoPipe magnitudes, BPZ and ``w_des`` for every catalogue row.

    ``catalogue`` is a calibrated shear catalogue with ``TILE_ID``, ``NUMBER``
    and ``w_des``; ``photometry`` is the collection holding ``OBJECT_ID`` and
    the PhotoPipe columns. ``status`` is PhotoPipe's join state for the object
    (0 photometry, 1 its tile has PhotoPipe but not the object, 2 its tile has
    no PhotoPipe), ``NO_ID`` for a row with no DR6 number and ``UNMATCHED``
    for an ID the collection lacks; both keep NaN photometry.

    @sc [decision:nz.source_photometry_join]
    @sc [label:convention] sources-row-aligned
    Every returned array is in the catalogue's row order, rows without
    photometry included: ``tom_bin.parquet`` is written in this order, and row
    order is the only key it has for rows with no ``OBJECT_ID``.
    """
    shear = read_catalogue(catalogue, columns=["TILE_ID", "NUMBER", "w_des"])
    ids = np.asarray(object_id(shear["TILE_ID"], shear["NUMBER"]).fill_null(-1))
    n = len(ids)
    order = np.argsort(ids, kind="stable")
    ordered = ids[order]
    identified = ordered >= 0
    if (np.diff(ordered[identified]) == 0).any():
        raise ValueError(f"{catalogue} repeats an OBJECT_ID")
    out = dict(
        object_id=ids,
        weight=shear["w_des"].astype("f8"),
        mag=np.full((n, len(BANDS)), np.nan, np.float32),
        photoz=np.full(n, np.nan),
        status=np.where(ids < 0, NO_ID, UNMATCHED).astype(np.int8),
    )
    del shear
    hits = np.zeros(n, np.int8)
    columns = [OBJECT_ID, "PHOTOPIPE_STATUS", "Z_B", *MAGS]

    def match(view):
        part = view.read(columns)
        at = np.minimum(np.searchsorted(ordered, part[OBJECT_ID]), n - 1)
        found = (ordered[at] == part[OBJECT_ID]) & (part[OBJECT_ID] >= 0)
        return order[at[found]], {name: part[name][found] for name in columns[1:]}

    with Catalogue(photometry) as collection, ThreadPoolExecutor(workers) as pool:
        for rows, part in pool.map(match, collection.chunks.values()):
            np.add.at(hits, rows, 1)
            out["status"][rows] = part["PHOTOPIPE_STATUS"]
            out["photoz"][rows] = part["Z_B"]
            out["mag"][rows] = np.column_stack([part[m] for m in MAGS])
    if (hits > 1).any():
        raise ValueError(f"{photometry} repeats {(hits > 1).sum()} OBJECT_IDs")
    return out


def match_report(status, mag, weight):
    """Return where the shear weight without complete photometry is lost.

    Fractions of the total shear weight, exclusive and in this order: rows
    with no object ID, IDs absent from the photometry, objects in a tile
    without PhotoPipe, objects PhotoPipe lacks in a tile it has, and objects
    with photometry but a band undetected (``incomplete``); the rest is
    ``complete``. ``undetected`` gives, of the total, the weight of objects
    with photometry undetected in each band (an object can miss several);
    ``z`` there is z and z2 both.
    """
    total = weight.sum()

    def share(rows):
        return float(weight[rows].sum() / total)

    seen = nz.detected(mag)
    seen = np.column_stack([seen[:, :4], seen[:, 4] | seen[:, 5]])
    joined = status == 0
    return dict(
        no_object_id=share(status == NO_ID),
        unmatched_by_id=share(status == UNMATCHED),
        tile_without_photopipe=share(status == 2),
        object_without_photopipe=share(status == 1),
        incomplete=share(joined & ~seen.all(1)),
        complete=share(joined & seen.all(1)),
        undetected={
            band: share(joined & ~seen[:, k]) for k, band in enumerate(BANDS[:5])
        },
    )
