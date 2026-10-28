"""SNR-resolved image-simulation calibration and effective multiplicative bias.

The caller supplies metacal objects constructed from the full parent catalogues.
The fixed external weight table is held fixed in all spatial deletions.
"""

import numpy as np
from astropy.io import fits

from sp_validation import response

DEFAULT_SNR_EDGES = (10, 15, 20, 30, 50, 100, 500)
ORDER = ("m1", "c1", "m2", "c2")
PAIRS = (("1p2z", "1m2z"), ("1z2p", "1z2m"))


def snr_masks(gm, lo, hi):
    """Intersect each metacal branch's selection with its own SNR bin.

    Bins are left-open and right-closed, matching the production lower cut
    and assigning internal boundaries to exactly one bin.
    """
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("SNR bin edges must be finite and increasing")
    masks = {}
    for branch, rows in gm.mask_dict.items():
        v = getattr(gm, branch)
        snr = v["flux"][rows] / v["flux_err"][rows]
        masks[branch] = rows[(snr > lo) & (snr <= hi)]
    return masks


def measure_bin(gm, groups, lo, hi):
    """Return calibrated mean and deletions for one catalogue and SNR bin."""
    if gm._des_weight_product is None or gm.weight_scheme != "des":
        raise ValueError("SNR calibration requires a fixed external DES table")
    masks = snr_masks(gm, lo, hi)
    pos = np.searchsorted(gm.mask_dict["ns"], masks["ns"])
    if not np.array_equal(gm.mask_dict["ns"][pos], masks["ns"]):
        raise ValueError("No-shear mask is not sorted")
    rg = gm.R_shear[:, :, pos]
    point = response.responses(gm.branches, masks, gm.branch_weights, rg, gm._step)
    labels, jk, _ = response.jackknife(
        gm.branches, masks, gm.branch_weights, rg, gm._step, groups
    )
    return point["c_g"], jk["c_g"], labels, {b: len(v) for b, v in masks.items()}


def grid_product(catalogues, shear_amplitude, edges=DEFAULT_SNR_EDGES):
    """Measure all bins on full populations and retain cross-bin covariance.

    ``catalogues`` maps each external-shear name to ``(gm, parent_tiles)``.
    Covariance uses the same tile deletion on all external and metacal branches.
    Values have shape (bins, 4), ordered m1,c1,m2,c2.
    """
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("SNR edges must be strictly increasing")
    if not np.isfinite(shear_amplitude) or shear_amplitude <= 0:
        raise ValueError("shear_amplitude must be positive")
    points, draws, counts = [], [], []
    labels = None
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        values, deleted, n = {}, {}, {}
        for name in sum(PAIRS, ()):
            gm, tiles = catalogues[name]
            values[name], deleted[name], current, n[name] = measure_bin(
                gm, tiles, lo, hi
            )
            if labels is None:
                labels = current
            if not np.array_equal(labels, current):
                raise ValueError("External branches must share jackknife tiles")
        point, jk = [], []
        for i, (plus, minus) in enumerate(PAIRS):
            point.extend(
                (
                    (values[plus][i] - values[minus][i]) / (2 * shear_amplitude) - 1,
                    (values[plus][i] + values[minus][i]) / 2,
                )
            )
            jk.extend(
                (
                    (deleted[plus][:, i] - deleted[minus][:, i]) / (2 * shear_amplitude)
                    - 1,
                    (deleted[plus][:, i] + deleted[minus][:, i]) / 2,
                )
            )
        points.append(point)
        draws.append(np.stack(jk, axis=1))
        counts.append(n)
    draws = np.stack(draws, axis=1)
    flat = draws.reshape(len(labels), -1)
    delta = flat - flat.mean(axis=0)
    cov = (len(labels) - 1) / len(labels) * delta.T @ delta
    return dict(
        edges=edges,
        value=np.asarray(points),
        covariance=cov,
        deleted=draws,
        labels=labels,
        counts=counts,
    )


def combine_grids(products):
    """Combine independent grids coordinate-wise, retaining cross-bin covariance."""
    if not products:
        raise ValueError("At least one grid is required")
    edges = products[0]["edges"]
    values = np.array([p["value"].ravel() for p in products])
    covs = np.array([p["covariance"] for p in products])
    if any(not np.array_equal(p["edges"], edges) for p in products):
        raise ValueError("Grid bin edges differ")
    variance = np.diagonal(covs, axis1=1, axis2=2)
    if not np.all(np.isfinite(variance) & (variance > 0)):
        raise ValueError("Grid variances must be positive and finite")
    weights = 1 / variance
    weights /= weights.sum(axis=0)
    cov = np.sum(covs * weights[:, :, None] * weights[:, None, :], axis=0)
    return dict(
        edges=edges,
        value=np.sum(weights * values, axis=0).reshape(-1, 4),
        covariance=cov,
        grid_weights=weights,
    )


def effective_m(product, bin_weights, total_response):
    """Compute diagonal-response-weighted effective m for a data selection.

    ``bin_weights`` is the no-shear sum of weights in each bin and
    ``total_response`` is the selection's (bins, 2, 2) ensemble response,
    including both shear and selection response measured for that bin.
    This diagonal approximation assumes linear response and neglects matrix
    mixing and bin-fraction derivative terms. Recalibrate the response for
    each data sub-selection; separately corrected bins use different weights.
    The covariance propagates simulation-bin uncertainty conditional on the
    supplied data weights and responses.
    """
    weights = np.asarray(bin_weights, dtype=float)
    responses = np.asarray(total_response, dtype=float)
    bins = len(product["edges"]) - 1
    if weights.shape != (bins,) or responses.shape != (bins, 2, 2):
        raise ValueError("Expected bin weights (bins,) and total response (bins,2,2)")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Bin weights must be finite and nonnegative")
    if not np.all(np.isfinite(responses)):
        raise ValueError("Total responses must be finite")
    return _effective_from_histogram(
        product, weights[:, None] * np.diagonal(responses, axis1=1, axis2=2)
    )


def _effective_from_histogram(product, histogram):
    shares = np.array(histogram, dtype=float, copy=True)
    if np.any(shares.sum(axis=0) <= 0):
        raise ValueError("Response-weighted total must be positive")
    shares /= shares.sum(axis=0)
    edges = np.asarray(product["edges"])
    transform = np.zeros((2, (len(edges) - 1) * 4))
    transform[0, 0::4] = shares[:, 0]
    transform[1, 2::4] = shares[:, 1]
    value = transform @ np.asarray(product["value"]).ravel()
    cov = transform @ np.asarray(product["covariance"]) @ transform.T
    return value, cov, shares


def write_product(path, product, weight_product, overwrite=False):
    """Write SNR bins and full bin-major (m1,c1,m2,c2) covariance to FITS."""
    edges = np.asarray(product["edges"])
    values = np.asarray(product["value"])
    header = fits.Header()
    header["WGHTSCH"] = "des"
    header["WPRODUCT"] = weight_product["name"]
    header["WPRODPTH"] = weight_product["path"]
    header["WPOLICY"] = "fixed-external"
    header["COVORDER"] = "bin-major: m1,c1,m2,c2"
    cols = [
        fits.Column(name="SNR_LO", format="D", array=edges[:-1]),
        fits.Column(name="SNR_HI", format="D", array=edges[1:]),
    ]
    cols.extend(
        fits.Column(name=k, format="D", array=values[:, i]) for i, k in enumerate(ORDER)
    )
    hdus = [
        fits.PrimaryHDU(header=header),
        fits.BinTableHDU.from_columns(cols, name="SNR_BINS"),
        fits.ImageHDU(np.asarray(product["covariance"]), name="MC_COV"),
    ]
    if "grid_weights" in product:
        hdus.append(fits.ImageHDU(np.asarray(product["grid_weights"]), name="GRID_W"))
    fits.HDUList(hdus).writeto(path, overwrite=overwrite)


def read_product(path):
    """Read the SNR-bin FITS product and its covariance."""
    with fits.open(path) as hdus:
        tab = hdus["SNR_BINS"].data
        edges = np.r_[tab["SNR_LO"], tab["SNR_HI"][-1]]
        if not np.array_equal(tab["SNR_LO"][1:], tab["SNR_HI"][:-1]):
            raise ValueError("SNR bins are not contiguous")
        return dict(
            edges=edges,
            value=np.stack([tab[k] for k in ORDER], axis=1),
            covariance=hdus["MC_COV"].data.copy(),
        )
