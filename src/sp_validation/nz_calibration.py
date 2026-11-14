"""Provisional SOM direct calibration of source redshift distributions.

Photometry is detected when finite and between 0 and 50 magnitudes.
The map uses all ten ugriz colours and r, standardized per feature.
Each BPZ bin has its own gold mask and source-weight/calibrator-count ratios.
The spectroscopic calibration sample's fiducial Z_B is supplied as ``cal_z``;
``photoz`` always means DR6 BPZ Z_B.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


def detected(mag):
    """Select detected DR6 magnitudes, excluding -99 and 99 sentinels."""
    return np.isfinite(mag) & (mag > 0) & (mag < 50)


def features(mag):
    """Return the ten ugriz colours and r magnitude, in the training order."""
    return np.column_stack(
        [mag[:, i] - mag[:, j] for i in range(5) for j in range(i + 1, 5)] + [mag[:, 2]]
    )


def train(mag, side=101, epochs=60, seed=42, workers=1):
    """Train a planar batch SOM with PCA initialization and Gaussian smoothing."""
    x = features(mag)
    center, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    x = (x - center) / scale
    _, s, v = np.linalg.svd(x, full_matrices=False)
    grid = np.stack(
        np.meshgrid(np.linspace(-2, 2, side), np.linspace(-2, 2, side)), axis=-1
    ).reshape(-1, 2)
    code = grid @ (v[:2] * s[:2, None] / np.sqrt(len(x)))
    code += np.random.default_rng(seed).normal(0, 0.001, code.shape)
    for sigma in np.geomspace(side / 5, 0.5, epochs):
        _, cell = cKDTree(code).query(x, workers=workers)
        counts = np.bincount(cell, minlength=side * side).reshape(side, side)
        den = gaussian_filter(counts.astype(float), sigma, mode="constant")
        sums = np.column_stack(
            [
                np.bincount(cell, weights=x[:, k], minlength=side * side)
                for k in range(x.shape[1])
            ]
        ).reshape(side, side, -1)
        num = gaussian_filter(sums, (sigma, sigma, 0), mode="constant")
        good = den.ravel() > 1e-15
        code[good] = num.reshape(-1, x.shape[1])[good] / den.ravel()[good, None]
    return dict(code=code, center=center, scale=scale, side=np.array(side))


def assign(mag, model, workers=1):
    """Return map distance and cell for detected five-band magnitudes."""
    x = (features(mag) - model["center"]) / model["scale"]
    return cKDTree(model["code"]).query(x, workers=workers)


def group_cells(model, n_groups=2000):
    """Ward-cluster map code vectors into zero-based calibration groups."""
    return (
        fcluster(linkage(model["code"], method="ward"), n_groups, criterion="maxclust")
        - 1
    )


def z_magnitudes(mag, offset, use_z2=True):
    """Return ugriz and z-channel IDs (0 missing, 1 canonical z, 2 mapped z2).

    Canonical z takes precedence; z2 + offset enters only when z is undetected.
    Five-column inputs contain canonical z only.
    """
    out = mag[:, :5].astype("f8")
    channel = detected(out[:, 4]).astype(np.int8)
    out[channel == 0, 4] = np.nan
    if use_z2 and mag.shape[1] == 6:
        extra = (channel == 0) & detected(mag[:, 5])
        out[extra, 4] = mag[extra, 5] + offset
        channel[extra] = 2
    return out, channel


def source_groups(
    mag, photoz, weight, model, *, offset, use_z2=True, chunk=1_000_000, workers=1
):
    """Assign eligible sources to BPZ bins and SOM cells in bounded chunks.

    Returns the preliminary bin, the map cell and the z channel of each source;
    ineligible sources have bin and cell -1.
    BPZ intervals include the lower edge and exclude the upper edge.
    """
    edges = model["edges"]
    prelim = np.full(len(weight), -1, np.int8)
    cells = np.full(len(weight), -1, np.int32)
    channel = np.zeros(len(weight), np.int8)
    tree = cKDTree(model["code"])
    for start in range(0, len(weight), chunk):
        stop = min(start + chunk, len(weight))
        m, channel[start:stop] = z_magnitudes(mag[start:stop], offset, use_z2)
        p, w = photoz[start:stop], weight[start:stop]
        valid = (
            detected(m).all(1)
            & np.isfinite(p)
            & (p >= edges[0])
            & (p < edges[-1])
            & np.isfinite(w)
            & (w > 0)
        )
        rows = start + np.flatnonzero(valid)
        _, cell = tree.query(
            (features(m[valid]) - model["center"]) / model["scale"], workers=workers
        )
        cells[rows] = cell
        prelim[rows] = np.searchsorted(edges, p[valid], side="right") - 1
    return prelim, cells, channel


def gold_threshold(cal_z, cal_photoz, floor=0.4, nmad_factor=5):
    """Compute max(floor, factor × NMAD) of unnormalized spec-z − BPZ."""
    residual = cal_z - cal_photoz
    good = np.isfinite(residual) & (cal_photoz >= 0) & (cal_photoz < 5)
    residual = residual[good]
    nmad = 1.4826 * np.median(abs(residual - np.median(residual)))
    return float(max(floor, nmad_factor * nmad))


def gold_mask(source_weight, source_photoz_sum, cal_count, cal_z_sum, threshold):
    """Keep supported groups whose mean spec-z and source BPZ agree within QC."""
    spec = np.divide(
        cal_z_sum, cal_count, out=np.zeros_like(cal_z_sum), where=cal_count > 0
    )
    bpz = np.divide(
        source_photoz_sum,
        source_weight,
        out=np.zeros_like(source_photoz_sum),
        where=source_weight > 0,
    )
    return (cal_count > 0) & ((source_weight == 0) | (abs(spec - bpz) <= threshold))


def tom_bin_ids(prelim, group, gold):
    """Return 1-based gold bin IDs, with zero for every excluded row."""
    result = np.zeros(len(prelim), np.int8)
    rows = np.flatnonzero(prelim >= 0)
    keep = gold[prelim[rows], group[rows]]
    result[rows[keep]] = prelim[rows[keep]] + 1
    return result


def deposit(z, weight, grid):
    """Linearly deposit mass, conserving its zeroth and first trapezoid moments."""
    if np.any(~np.isfinite(z) | (z < grid[0]) | (z > grid[-1])):
        raise ValueError("calibration redshift outside output grid")
    lo = np.clip(np.searchsorted(grid, z, side="right") - 1, 0, len(grid) - 2)
    fraction = (z - grid[lo]) / (grid[lo + 1] - grid[lo])
    mass = np.bincount(lo, weights=weight * (1 - fraction), minlength=len(grid))
    mass += np.bincount(lo + 1, weights=weight * fraction, minlength=len(grid))
    quadrature = np.r_[grid[1] - grid[0], grid[2:] - grid[:-2], grid[-1] - grid[-2]] / 2
    return mass / quadrature


def calibrate(
    prelim,
    group,
    weight,
    photoz,
    cal_z,
    cal_photoz,
    cal_group,
    edges,
    n_groups,
    grid,
    *,
    gold=None,
    threshold=None,
):
    """Calibrate each bin with source shear weight / in-bin calibrator count.

    With ``gold=None``, compute QC masks from this source sample.
    Supplying gold freezes the selection while recalibrating the group weights.
    Histograms have unit integral per bin; the writer applies mixture fractions.
    """
    weights, masks, bins, hists = [], [], [], []
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        rows = np.flatnonzero(prelim == k)
        g, w = group[rows].astype(np.int64), weight[rows]
        source = np.bincount(g, weights=w, minlength=n_groups)
        cm = np.isfinite(cal_photoz) & (cal_photoz >= lo) & (cal_photoz < hi)
        count = np.bincount(cal_group[cm], minlength=n_groups)
        if gold is None:
            zsum = np.bincount(cal_group[cm], weights=cal_z[cm], minlength=n_groups)
            psum = np.bincount(g, weights=w * photoz[rows], minlength=n_groups)
            mask = gold_mask(source, psum, count, zsum, threshold)
        else:
            mask = gold[k] & (count > 0)
        ratio = np.divide(source, count, out=np.zeros(n_groups), where=mask)
        cw = np.where(cm, ratio[cal_group], 0.0)
        norm = cw.sum()
        if norm <= 0:
            raise ValueError(f"empty calibration support in bin {k + 1}")
        kept = mask[g]
        bins.append(
            dict(
                bin=k + 1,
                edges=[float(lo), float(hi)],
                mean=float(cal_z @ cw / norm),
                weight_sum=float(norm),
                pre_gold_weight_sum=float(w.sum()),
                gold_fraction=float(norm / w.sum()),
                n_rows=int(kept.sum()),
                calibrators_in_bpZ_bin=int(cm.sum()),
                calibrators_with_positive_weight=int((cw > 0).sum()),
            )
        )
        weights.append(cw)
        masks.append(mask)
        hists.append(deposit(cal_z, cw / norm, grid))
    total = sum(b["weight_sum"] for b in bins)
    for b in bins:
        b["weight_fraction"] = b["weight_sum"] / total
    return dict(
        bins=bins,
        cal_weights=np.array(weights),
        gold=np.array(masks),
        bin_hists=np.array(hists),
    )


def write_nz(path, grid, bin_hists, weight_sums):
    """Write z and bin densities integrating to retained shear-weight fractions."""
    fractions = np.asarray(weight_sums) / np.sum(weight_sums)
    np.savetxt(
        path,
        np.column_stack([grid, (fractions[:, None] * bin_hists).T]),
        fmt="%.12e",
        header="z bin1 bin2 ...; columns integrate to retained "
        "shear-weight fractions, total integral=1",
    )
