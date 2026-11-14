"""Compare two calibrated samples object by object.

Both are HATS catalogues with the same pixels. ``compare`` pairs their rows
pixel by pixel on the DR6 object ID (``OBJECT_ID``, or ``TILE_ID`` and
``NUMBER``), and a row without an ID on its exact position, then reports the
rows only one sample holds and, per column, the largest difference between
paired rows and how many exceed a tolerance.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .. import io
from ..object_id import NUMBER_LIMIT, OBJECT_ID
from .pipeline import tile_key

#: Columns copied from the shape catalogue or derived from it row by row.
COPIED = ("RA", "Dec", "e1_PSF", "e2_PSF", "SIZE_RATIO", "SNR", "N_EPOCH", "TILE_KEY")
#: Columns that depend on the whole sample, through its weight table, R and c.
CALIBRATED = ("e1", "e2", "w_des")
#: Edges of the histograms of log10 |difference| of the calibrated columns;
#: equal values fall in the first bin.
LOG_EDGES = np.arange(-20.0, 0.5, 0.5)


def identities(data):
    """Return the object ID of each row of ``data``, -1 for a row without one."""
    if OBJECT_ID in data.dtype.names:
        ids = data[OBJECT_ID]
        return np.where(np.isfinite(ids), ids, -1).astype(np.int64)
    number = data["NUMBER"].astype(np.int64)
    return np.where(number >= 0, tile_key(data["TILE_ID"]) * NUMBER_LIMIT + number, -1)


def pair(a, b):
    """Return the indices of the paired rows of ``a`` and ``b``, and how many
    were paired on their ID."""
    ida, idb = identities(a), identities(b)
    _, ia, ib = np.intersect1d(ida, idb, return_indices=True)
    keep = ida[ia] >= 0
    ia, ib = ia[keep], ib[keep]
    rest_a = np.setdiff1d(np.arange(len(a)), ia)
    rest_b = np.setdiff1d(np.arange(len(b)), ib)
    position = [("RA", "f8"), ("Dec", "f8")]
    pa, pb = (
        np.array(list(zip(x["RA"][rest], x["Dec"][rest])), dtype=position)
        for x, rest in ((a, rest_a), (b, rest_b))
    )
    _, ja, jb = np.intersect1d(pa, pb, return_indices=True)
    return np.r_[ia, rest_a[ja]], np.r_[ib, rest_b[jb]], len(ia)


def _columns(catalogue):
    names = catalogue.dtype().names
    ids = [OBJECT_ID] if OBJECT_ID in names else ["TILE_ID", "NUMBER"]
    return [*COPIED, *CALIBRATED, *ids]


_state = {}


def _start(new, reference):
    _state.update(new=io.Catalogue(new), reference=io.Catalogue(reference))


def compare_pixel(pixel):
    """Return the comparison of one pixel: row counts, and per column the
    largest difference and the histogram of log10 |difference|."""
    a, b = (
        cat.chunks[pixel].to_structured(_columns(cat))
        if pixel in cat.chunks
        else np.empty(0, dtype=cat.dtype(_columns(cat)))
        for cat in (_state["new"], _state["reference"])
    )
    ia, ib, by_id = pair(a, b)
    result = dict(
        new=len(a),
        reference=len(b),
        by_id=by_id,
        by_position=len(ia) - by_id,
        largest={},
        histogram={},
    )
    for name in (*COPIED, *CALIBRATED):
        difference = np.abs(a[name][ia].astype(float) - b[name][ib].astype(float))
        if name == "w_des":
            difference /= np.abs(b[name][ib])
        result["largest"][name] = float(difference.max(initial=0))
        if name in CALIBRATED:
            logs = np.log10(np.maximum(difference, 10 ** LOG_EDGES[0]))
            result["histogram"][name] = np.histogram(logs, LOG_EDGES)[0]
    return result


def compare(new, reference, *, tolerance=1e-10, workers=1):
    """Compare the samples at ``new`` and ``reference``.

    Returns
    -------
    dict
        ``rows`` of each sample; ``paired`` on ID and on position; rows
        ``only_new`` and ``only_reference``; ``largest_difference`` per column
        (absolute, relative for ``w_des``); ``histogram``, of log10 of the
        calibrated columns' differences on ``log_edges``; and ``agree``:
        every row paired, the copied columns equal and the calibrated ones
        within ``tolerance``
    """
    with io.Catalogue(new) as a, io.Catalogue(reference) as b:
        pixels = sorted(set(a.chunks) | set(b.chunks))
    with ProcessPoolExecutor(
        workers, initializer=_start, initargs=(new, reference)
    ) as pool:
        results = list(pool.map(compare_pixel, pixels))
    counts = ("new", "reference", "by_id", "by_position")
    total = {key: sum(r[key] for r in results) for key in counts}
    histogram = {
        name: sum(r["histogram"][name] for r in results).tolist() for name in CALIBRATED
    }
    largest = {
        name: max(r["largest"][name] for r in results) for name in results[0]["largest"]
    }
    paired = total["by_id"] + total["by_position"]
    summary = dict(
        rows=dict(new=total["new"], reference=total["reference"]),
        paired=dict(by_id=total["by_id"], by_position=total["by_position"]),
        only_new=total["new"] - paired,
        only_reference=total["reference"] - paired,
        largest_difference=largest,
        tolerance=tolerance,
        histogram=histogram,
        log_edges=LOG_EDGES.tolist(),
    )
    summary["agree"] = bool(
        summary["only_new"] == 0
        and summary["only_reference"] == 0
        and all(largest[name] == 0 for name in COPIED)
        and all(largest[name] <= tolerance for name in CALIBRATED)
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("new", help="HATS catalogue of the sample just built")
    parser.add_argument("reference", help="HATS catalogue it should reproduce")
    parser.add_argument("--output", required=True, help="JSON file to write")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    summary = compare(
        args.new, args.reference, tolerance=args.tolerance, workers=args.workers
    )
    Path(args.output).write_text(json.dumps(summary, indent=1) + "\n")
    del summary["histogram"], summary["log_edges"]
    print(json.dumps(summary, indent=1))
    raise SystemExit(0 if summary["agree"] else 1)
