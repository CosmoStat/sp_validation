"""Inputs of the SOM n(z): source photometry and the spectroscopic calibrators.

Both come from PhotoPipe's run on Stephen Gwyn's DR6 detections, so sources
and calibrators share one photometry. Sources are the rows of a calibrated
shear catalogue; each takes its photometry from the v2 collection (shapes
with their PhotoPipe extension, :mod:`sp_validation.hats_collection`) by its
DR6 object ID. Calibrators are the objects of the same collection that its
spec-z extension gives a spectrum from the recipe's surveys.

@sc [decision:nz.photometry_as_delivered]
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..io import Catalogue, read_catalogue
from ..object_id import OBJECT_ID, object_id
from . import calibration as nz

BANDS = ("u", "g", "r", "i", "z", "z2")
MAGS = [f"MAG_GAAP_{band}" for band in BANDS]
#: ``status`` of a source the photometry was not looked up for or lacks.
NO_ID, UNMATCHED = -2, -1


def calibrators(photometry, surveys, field, z_min, offset, workers=1):
    """Return the calibration sample: spectra from ``surveys`` in ``field`` with
    complete ugriz.

    ``photometry`` is the collection, whose spec-z extension gives each object
    its matched spectrum (``Z_SPEC``, ``Z_SPEC_SURVEY``, and the spectrum's own
    position ``Z_SPEC_RA``, ``Z_SPEC_DEC``). ``field`` bounds that position,
    ``{"ra": [min, max], "dec": [min, max]}`` in degrees: a survey can observe
    several fields, selected differently. ``Z_B`` and the magnitudes come
    from the PhotoPipe extension of the same collection (the reader joins the
    extensions on ``OBJECT_ID``). A spectrum matched to several objects, as in
    overlapping tiles, is kept once, at the lowest ``OBJECT_ID``; the copies
    are told apart by their position, which the extension carries bit for bit
    from the compilation. Missing z is filled from z2 + ``offset``, as for
    sources. Rows come in ``OBJECT_ID`` order.

    @sc [decision:nz.calibrator_set]
    """
    spectrum = ["Z_SPEC", "Z_SPEC_SURVEY", "Z_SPEC_RA", "Z_SPEC_DEC"]
    columns = [OBJECT_ID, *spectrum, "Z_B", *MAGS]

    def select(view):
        survey = view.read(["Z_SPEC_SURVEY"])["Z_SPEC_SURVEY"]
        rows = np.flatnonzero(np.isin(survey.astype(str), surveys))
        return view.read(columns, rows) if len(rows) else None

    with Catalogue(photometry) as collection, ThreadPoolExecutor(workers) as pool:
        parts = [p for p in pool.map(select, collection.chunks.values()) if p]
    data = {name: np.concatenate([p[name] for p in parts]) for name in columns}
    mag, _ = nz.z_magnitudes(
        np.column_stack([data[m] for m in MAGS]).astype("f8"), offset
    )
    ra, dec = data["Z_SPEC_RA"], data["Z_SPEC_DEC"]
    keep = (
        nz.detected(mag).all(1)
        & np.isfinite(data["Z_B"])
        & (data["Z_SPEC"] > z_min)
        & (ra >= field["ra"][0])
        & (ra <= field["ra"][1])
        & (dec >= field["dec"][0])
        & (dec <= field["dec"][1])
    )
    rows = np.flatnonzero(keep)
    rows = rows[np.argsort(data[OBJECT_ID][rows])]
    position = np.column_stack([data["Z_SPEC_RA"][rows], data["Z_SPEC_DEC"][rows]])
    rows = rows[np.sort(np.unique(position, axis=0, return_index=True)[1])]
    return dict(
        z=data["Z_SPEC"][rows].astype("f8"),
        photoz=data["Z_B"][rows].astype("f8"),
        mag=mag[rows],
        survey=data["Z_SPEC_SURVEY"][rows].astype(str),
        object_id=data[OBJECT_ID][rows],
        ra=data["Z_SPEC_RA"][rows],
        dec=data["Z_SPEC_DEC"][rows],
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
