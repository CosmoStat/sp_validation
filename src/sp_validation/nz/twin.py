"""A twin of the SOM n(z) calibration, built on the catalogue itself.

Sources and calibrators keep everything the recipe reads: PhotoPipe
magnitudes, BPZ, shear weights and sky positions. Only their redshifts are
painted, by a stated model of p(z | photometry). The recipe
(:func:`.pipeline.estimate`) calibrates the twin, and each bin k is scored by

    Δ⟨z⟩_k = ⟨z⟩ of the calibrated n(z) − the shear-weighted mean painted
             redshift of the sources given TOM_BIN_ID = k,

the calibration's error under that model (:func:`score`). BPZ is the real
BPZ of the real photometry, so it knows exactly what the photometry knows.

A model draws each galaxy's redshift from its ``k`` nearest members of a
library of spectra, in a feature space standardised by the library
(:func:`neighbours`, :func:`paint`):

- ``colour``: the recipe's SOM features (ten colours and r). The galaxy takes
  a neighbour's spectroscopic redshift: p(z | photometry) is what the
  library's spectra say at those colours and magnitude, and no galaxy lies
  outside the library's redshift range.
- ``bpz``: BPZ and r. The galaxy takes its own BPZ plus a neighbour's
  spec-z − BPZ, kept inside [0, ``z_max``]: at fixed BPZ and r the redshift does not
  depend on colour, and BPZ's error is what the library's is.

A galaxy that is itself in the library never draws its own spectrum.

Effects injected on the twin move the photometry and BPZ the recipe reads,
never the painted redshifts: zero-point offsets as maps on the sky
(:func:`zero_point_offsets`), carried into BPZ along its local gradient in
each SOM cell (:func:`photoz_gradient`).
"""

import numpy as np
from scipy.spatial import cKDTree

from . import calibration as nz
from .pipeline import output_grid

SPACES = ("colour", "bpz")


def space(kind, mag, photoz, offset):
    """Unstandardised features of model ``kind`` for (n, 5 or 6) magnitudes."""
    m, _ = nz.z_magnitudes(mag, offset)
    if kind == "colour":
        return nz.features(m)
    return np.column_stack([photoz, m[:, 2]])


def neighbours(kind, library, mag, photoz, object_id, *, k, offset, workers=1):
    """Library rows (n, k) nearest each galaxy in model ``kind``'s space.

    ``library`` holds ``mag``, ``photoz`` and ``object_id`` of the spectra; a
    galaxy whose ``object_id`` is in the library skips its own row.
    """
    lib = space(kind, library["mag"], library["photoz"], offset)
    center, scale = lib.mean(0), lib.std(0)
    tree = cKDTree((lib - center) / scale)
    out = np.empty((len(photoz), k), np.int32)
    for start in range(0, len(photoz), 4_000_000):
        rows = slice(start, start + 4_000_000)
        x = (space(kind, mag[rows], photoz[rows], offset) - center) / scale
        near = tree.query(x, k + 1, workers=workers)[1]
        own = library["object_id"][near] == object_id[rows, None]
        drop = np.where(own.any(1), own.argmax(1), k)
        out[rows] = near[np.arange(k + 1) != drop[:, None]].reshape(len(x), k)
    return out


def paint(kind, library, near, photoz, rng, z_max=5.0):
    """One redshift per galaxy, from a random one of its library neighbours."""
    pick = near[np.arange(len(near)), rng.integers(near.shape[1], size=len(near))]
    if kind == "colour":
        return library["z"][pick].astype("f8")
    return np.clip(photoz + library["z"][pick] - library["photoz"][pick], 0, z_max)


def score(result, weight, z):
    """Calibrated, true and Δ⟨z⟩ per bin of a :func:`.pipeline.estimate` result.

    ``z`` is the painted redshift of every source row.
    """
    tom = result["tom"]
    out = []
    for b in result["est"]["bins"]:
        sel = tom == b["bin"]
        truth = float(np.average(z[sel], weights=weight[sel]))
        out.append(dict(calibrated=b["mean"], truth=truth, delta=b["mean"] - truth))
    return out


def recalibrate(config, result, src, cal):
    """:func:`.pipeline.estimate`'s calibration step again, for new calibrator
    redshifts or a subset of the calibrators, with the SOM, its groups, the
    edges and the sources' cells kept."""
    model = result["model"]
    cal_group = model["group_map"][nz.assign(cal["mag"], model)[1]]
    threshold = nz.gold_threshold(cal["z"], cal["photoz"], **config["gold_qc"])
    est = nz.calibrate(
        result["prelim"],
        result["group"],
        src["weight"],
        src["photoz"],
        cal["z"],
        cal["photoz"],
        cal_group,
        result["edges"],
        config["groups"],
        output_grid(config),
        threshold=threshold,
    )
    tom = nz.tom_bin_ids(result["prelim"], result["group"], est["gold"])
    return dict(result, est=est, tom=tom, threshold=threshold)


def zero_point_offsets(
    ra, dec, tile, sigma, rng, *, coherent=0.5, fwhm=10.0, nside=64, patch=None
):
    """Magnitude offsets (n, 5) in u g r i z for objects at (``ra``, ``dec``).

    Each band's offset has total standard deviation ``sigma[b]`` over the
    objects' sky, a fraction ``coherent`` of its variance in a Gaussian field
    smoothed to ``fwhm`` degrees and the rest independent per ``tile``. A
    ``patch`` ``{"band": b, "ra": [lo, hi], "dec": [lo, hi], "offset": δ}``
    adds δ in band b inside that box.
    """
    import healpy as hp

    pix = hp.ang2pix(nside, ra, dec, lonlat=True)
    seen = np.unique(pix)
    tiles, at = np.unique(tile, return_inverse=True)
    out = np.zeros((len(ra), 5))
    for b, s in enumerate(sigma):
        if s == 0:
            continue
        field = hp.smoothing(
            rng.normal(size=hp.nside2npix(nside)), fwhm=np.radians(fwhm)
        )
        field = (field - field[seen].mean()) / field[seen].std()
        per_tile = rng.normal(size=len(tiles))
        out[:, b] = s * (
            np.sqrt(coherent) * field[pix] + np.sqrt(1 - coherent) * per_tile[at]
        )
    if patch:
        inside = (
            (ra >= patch["ra"][0])
            & (ra <= patch["ra"][1])
            & (dec >= patch["dec"][0])
            & (dec <= patch["dec"][1])
        )
        out[inside, "ugriz".index(patch["band"])] += patch["offset"]
    return out


def offset_magnitudes(mag, dm):
    """Magnitudes (n, 6) in u g r i z z2 moved by (n, 5) offsets; z2 moves as z."""
    return mag + np.column_stack([dm, dm[:, 4]]).astype(mag.dtype)


def photoz_gradient(mag, photoz, cell, n_cells, *, offset, per_cell=2000, seed=0):
    """d BPZ / d m (n_cells, 5): a least-squares plane of BPZ in ugriz through
    up to ``per_cell`` of the objects in each SOM cell (zero below 20)."""
    m, _ = nz.z_magnitudes(mag, offset)
    ok = cell >= 0
    rows = np.flatnonzero(ok)
    rows = rows[np.random.default_rng(seed).permutation(len(rows))]
    rows = rows[np.argsort(cell[rows], kind="stable")]
    starts = np.searchsorted(cell[rows], np.arange(n_cells + 1))
    grad = np.zeros((n_cells, 5))
    for c in range(n_cells):
        r = rows[starts[c] : min(starts[c + 1], starts[c] + per_cell)]
        if len(r) < 20:
            continue
        a = np.column_stack([m[r] - m[r].mean(0), np.ones(len(r))])
        grad[c] = np.linalg.lstsq(a, photoz[r], rcond=None)[0][:5]
    return grad
