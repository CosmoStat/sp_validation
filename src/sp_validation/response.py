"""Weighted metacal response primitives.

Definitions: Martin Kilbinger's 6 October 2026 slide, p14 (S1), and
scripts/validation/wdes_response/wdes_response.py, case d.
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


def cell_moments(snr, size, e, response, edges_s, edges_r):
    """Per-cell training moments of the DES table, flat over (SNR, size) cells.

    Returns the count, the summed response and the summed ``|e|^2 / 2`` of the
    rows whose coordinates, shapes and response are all finite. Moments of
    disjoint row sets add, so a catalogue too large for memory is fitted from
    the sum of its chunks' moments (``table_from_moments``).
    """
    n_cells = (len(edges_s) - 1) * (len(edges_r) - 1)
    idx = cell_index(snr, edges_s) * (len(edges_r) - 1) + cell_index(size, edges_r)
    valid = np.isfinite(snr) & np.isfinite(size)
    valid &= np.all(np.isfinite(e), axis=0) & np.isfinite(response)
    idx = idx[valid]
    count = np.bincount(idx, minlength=n_cells)
    rsum = np.bincount(idx, weights=response[valid], minlength=n_cells)
    esum = np.bincount(
        idx, weights=np.mean(e[:, valid] ** 2, axis=0), minlength=n_cells
    )
    return count, rsum, esum


def table_from_moments(count, rsum, esum, edges_s, edges_r, min_count=0):
    """The DES table W = mean(trace R/2)^2 / mean(|e|^2/2) of ``cell_moments``.

    Uses second moments, not centered variances. Below min_count, pool moments
    over the smallest clipped concentric square of cells reaching min_count; if
    the entire training set is smaller, use its moments. Populated cells are
    exact. The returned counts are unmerged training counts.
    """
    shape = (len(edges_s) - 1, len(edges_r) - 1)
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


def fit_w(snr, size, e, response, edges_s, edges_r, min_count=0):
    """Fit the DES table, unweighted per cell, on rows held in memory.

    Fitting and evaluation share the same boundary convention
    (``cell_index``); ``table_from_moments`` gives the definition.
    """
    moments = cell_moments(snr, size, e, response, edges_s, edges_r)
    return table_from_moments(*moments, edges_s, edges_r, min_count)


def cell_sums(branches, masks, coordinates, rg, edges_s, edges_r):
    """Per-cell sums from which ``responses`` follows for any table on the edges.

    A table weight depends on a row only through its (SNR, size) cell, so every
    weighted mean of ``responses`` is a ratio of per-cell sums contracted with
    the table. For each branch, over its selection and in the cells of its own
    ``coordinates[b] = (snr, size)``: the row count ``n``, the summed no-shear
    shapes ``ns_e`` and the summed own shapes ``own_e``; and for the no-shear
    selection the summed per-object response ``rg``. Rows with a nonfinite
    coordinate carry zero table weight and are left out. Sums of disjoint row
    sets add (``responses_from_sums`` reads their total).
    """
    n_cells = (len(edges_s) - 1) * (len(edges_r) - 1)
    sums = {"n": {}, "ns_e": {}, "own_e": {}}
    for b, idx in masks.items():
        snr, size = (x[idx] for x in coordinates[b])
        finite = np.isfinite(snr) & np.isfinite(size)
        idx = idx[finite]
        cell = cell_index(snr[finite], edges_s) * (len(edges_r) - 1)
        cell += cell_index(size[finite], edges_r)

        def total(values, cell=cell):
            return np.bincount(cell, weights=values, minlength=n_cells)

        sums["n"][b] = np.bincount(cell, minlength=n_cells)
        sums["ns_e"][b] = np.array([total(x[idx]) for x in branches["ns"]])
        sums["own_e"][b] = np.array([total(x[idx]) for x in branches[b]])
        if b == "ns":
            sums["rg"] = np.array(
                [[total(rg[i, j, finite]) for j in range(2)] for i in range(2)]
            )
    return sums


def add_sums(a, b):
    """Add two results of ``cell_sums`` or ``cell_moments``, or any equally
    nested arrays."""
    if isinstance(a, dict):
        return {key: add_sums(a[key], b[key]) for key in a}
    if isinstance(a, tuple):
        return tuple(add_sums(x, y) for x, y in zip(a, b, strict=True))
    return a + b


def responses_from_sums(sums, table, step):
    """``responses`` from ``cell_sums`` and one table, with the branch
    populations: ``counts`` and ``weight_sums`` per branch."""
    w = table[2].ravel()
    den = {b: n @ w for b, n in sums["n"].items()}
    if not all(np.isfinite(d) and d > 0 for d in den.values()):
        raise ValueError(f"Empty or invalid weighted population: {den}")
    shear = sums["rg"] @ w / den["ns"]
    selection = np.empty((2, 2))
    joint = np.empty((2, 2))
    for j, (p, m) in enumerate((("p1", "m1"), ("p2", "m2"))):
        for name, out in (("ns_e", selection), ("own_e", joint)):
            plus, minus = (sums[name][b] @ w / den[b] for b in (p, m))
            out[:, j] = (plus - minus) / (2 * step)
    total = shear + selection
    ce = sums["ns_e"]["ns"] @ w / den["ns"]
    return dict(
        shear=shear,
        selection=selection,
        total=total,
        joint=joint,
        c_e=ce,
        c_g=np.linalg.solve(total, ce),
        counts={b: int(n.sum()) for b, n in sums["n"].items()},
        weight_sums={b: float(d) for b, d in den.items()},
    )


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
