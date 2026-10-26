"""Weighted metacal response primitives.

Definitions: Martin Kilbinger's 6 October 2026 slide, p14 (S1), and
/n23data1/mkilbing/imsim_wdes_response/wdes_response.py, case d.
Spatial deletion follows his response_jackknife, with fixed-W
comparisons and retained-sample refits in calibration.metacal.
No detection response is inferred from aligned metacal branches.
"""

import numpy as np

BRANCHES = ("ns", "p1", "m1", "p2", "m2")


def branch_error_columns(data):
    """Carry available sheared G_ERRs through extraction without requiring them.

    Historical merged products omitted them. DES can still use those products;
    modern IV independently requires a complete set and never freezes weights.
    """
    return {
        key: data[key]
        for shear in ("1P", "1M", "2P", "2M")
        for i in (1, 2)
        if (key := f"NGMIX_G{i}_ERR_{shear}") in data.dtype.names
    }


def cell_index(x, edges):
    """Right-closed cells; out-of-range finite values use nearest edge cell."""
    return np.clip(np.searchsorted(edges, x, side="left") - 1, 0, len(edges) - 2)


def evaluate_w(table, snr, size):
    """Evaluate one fixed table, including rows absent from the training sample.

    Unpooled empty/invalid cells and nonfinite coordinates get zero weight.
    """
    es, er, values, _ = table
    w = values[cell_index(snr, es), cell_index(size, er)].copy()
    w[~np.isfinite(snr) | ~np.isfinite(size) | ~np.isfinite(w)] = 0
    return w


def fit_w(snr, size, e, response, edges_s, edges_r, min_count=0):
    """Fit W = mean(trace R/2)^2 / mean(|e|^2/2), unweighted per cell.

    Uses second moments, not centered variances. Fitting and evaluation share
    the same boundary convention. Below min_count, pool moments over the
    smallest clipped concentric square of cells reaching min_count; if the
    entire training set is smaller, use its moments. Populated cells are exact.
    The returned counts are unmerged training counts.
    """
    shape = (len(edges_s) - 1, len(edges_r) - 1)
    idx = cell_index(snr, edges_s) * shape[1] + cell_index(size, edges_r)
    valid = np.isfinite(snr) & np.isfinite(size)
    valid &= np.all(np.isfinite(e), axis=0) & np.isfinite(response)
    idx = idx[valid]
    count = np.bincount(idx, minlength=np.prod(shape))
    rsum = np.bincount(idx, weights=response[valid], minlength=len(count))
    esum = np.bincount(
        idx, weights=np.mean(e[:, valid] ** 2, axis=0), minlength=len(count)
    )
    w = np.divide(
        rsum**2, count * esum, out=np.zeros_like(rsum), where=(count * esum) > 0
    )
    if min_count < 0:
        raise ValueError("min_count must be nonnegative")
    # Smallest concentric square in cell-index space (clipped at the edges).
    # Pool sufficient statistics, never weights; retain the original counts.
    if min_count:
        counts, rsums, esums = (x.reshape(shape) for x in (count, rsum, esum))
        for i, j in np.argwhere(counts < min_count):
            for radius in range(1, max(shape)):
                window = (
                    slice(max(0, i - radius), min(shape[0], i + radius + 1)),
                    slice(max(0, j - radius), min(shape[1], j + radius + 1)),
                )
                n = counts[window].sum()
                if n >= min_count or radius == max(shape) - 1:
                    r, e2 = rsums[window].sum(), esums[window].sum()
                    w[i * shape[1] + j] = r**2 / (n * e2) if n * e2 > 0 else 0
                    break
    return edges_s, edges_r, w.reshape(shape), count.reshape(shape)


def responses(branches, masks, weights, rg, step, keep=None):
    """S1 split and joint derivatives, each mean separately normalized.

    rg is the per-object response on no-shear-selected rows, aligned to
    masks['ns']. All other arrays are aligned to the common parent rows.
    """
    selected = {b: idx if keep is None else idx[keep[idx]] for b, idx in masks.items()}

    def mean(data, b):
        idx = selected[b]
        w = weights[b][idx]
        if not len(idx) or not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError(f"Empty or invalid weighted population: {b}")
        return np.array([np.average(x[idx], weights=w) for x in data])

    ns = masks["ns"]
    k = np.ones(len(ns), bool) if keep is None else keep[ns]
    shear = np.average(rg[:, :, k], axis=2, weights=weights["ns"][ns][k])
    selection = np.empty((2, 2))
    joint = np.empty((2, 2))
    for j, (p, m) in enumerate((("p1", "m1"), ("p2", "m2"))):
        selection[:, j] = (mean(branches["ns"], p) - mean(branches["ns"], m)) / (
            2 * step
        )
        joint[:, j] = (mean(branches[p], p) - mean(branches[m], m)) / (2 * step)
    total = shear + selection
    ce = mean(branches["ns"], "ns")
    cg = np.linalg.solve(total, ce)
    return dict(
        shear=shear, selection=selection, total=total, joint=joint, c_e=ce, c_g=cg
    )


def jackknife(branches, masks, weights, rg, step, groups):
    """Delete common spatial groups in all branches; conditional on fixed W.

    Covariance order: shear, selection, total, joint (row-major), c_e, c_g.
    Extends Martin's response_jackknife to additive bias and full covariance.
    """
    if len(groups) != len(branches["ns"][0]):
        raise ValueError("Spatial groups must align to the common parent rows")
    labels, group_index = np.unique(groups, return_inverse=True)
    if len(labels) < 2:
        raise ValueError("Response jackknife requires at least two spatial groups")

    # Fixed-W sufficient sums: O(N) rather than rereading N rows K times.
    def deleted_sum(values, index):
        sums = np.bincount(index, weights=values, minlength=len(labels))
        return sums.sum() - sums

    den = {}
    for b in masks:
        idx = masks[b]
        den[b] = deleted_sum(weights[b][idx], group_index[idx])
        if np.any(den[b] <= 0):
            raise ValueError(f"Empty weighted population after spatial deletion: {b}")

    def means(data, b):
        idx = masks[b]
        return np.array(
            [
                deleted_sum(x[idx] * weights[b][idx], group_index[idx]) / den[b]
                for x in data
            ]
        ).T

    ns = masks["ns"]
    shear = np.empty((len(labels), 2, 2))
    selection = np.empty_like(shear)
    joint = np.empty_like(shear)
    for i in range(2):
        for j in range(2):
            shear[:, i, j] = (
                deleted_sum(rg[i, j] * weights["ns"][ns], group_index[ns]) / den["ns"]
            )
    for j, (p, m) in enumerate((("p1", "m1"), ("p2", "m2"))):
        selection[:, :, j] = (means(branches["ns"], p) - means(branches["ns"], m)) / (
            2 * step
        )
        joint[:, :, j] = (means(branches[p], p) - means(branches[m], m)) / (2 * step)
    total = shear + selection
    ce = means(branches["ns"], "ns")
    cg = np.linalg.solve(total, ce[..., None])[..., 0]
    estimates = dict(
        shear=shear, selection=selection, total=total, joint=joint, c_e=ce, c_g=cg
    )
    vectors = np.concatenate(
        [estimates[k].reshape(len(labels), -1) for k in estimates], axis=1
    )
    delta = vectors - vectors.mean(axis=0)
    cov = (len(labels) - 1) / len(labels) * delta.T @ delta
    return labels, estimates, cov


def validate_weight_scheme(scheme, w_col):
    """Reject statistic/calibration mismatches, rather than relabel a response."""
    if scheme not in ("des", "iv"):
        raise ValueError("weight_scheme must be des or iv")
    if w_col != f"w_{scheme}":
        raise ValueError(
            f"Calibration weight_scheme={scheme} disagrees with w_col={w_col}"
        )
