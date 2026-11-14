"""Exact refit-DES deletions from fixed-cell sufficient statistics.

Storage is O(groups * cells), independent of parent catalogue length. Feed
selected branch chunks with add_branch; no parent-sized scratch arrays are kept.
"""

import numpy as np

from sp_validation import response


def pooled_weight(moments, shape, min_count):
    """Reapply the concentric-square rule to retained count, sum R and sum e².

    Integral images make each window query O(1). All sparse cells are evaluated
    together at each radius; with the fixed 20x20 production grid this is O(cells).
    Original counts, not pooled counts, determine which cells need pooling.
    """
    count, rsum, esum = moments
    weight = np.divide(
        rsum**2, count * esum, out=np.zeros_like(rsum), where=count * esum > 0
    )
    if min_count < 0:
        raise ValueError("min_count must be nonnegative")
    pending = np.flatnonzero(count < min_count)
    if not len(pending):
        return weight
    # Sparse windows subtract large prefix sums. Extended precision avoids
    # cancellation at the high-index edges (visible even on the small grids).
    image = np.pad(
        moments.astype(np.longdouble).reshape(3, *shape),
        ((0, 0), (1, 0), (1, 0)),
    )
    image = image.cumsum(axis=1).cumsum(axis=2)
    for radius in range(1, max(shape)):
        i, j = np.unravel_index(pending, shape)
        lo_i, hi_i = np.maximum(i - radius, 0), np.minimum(i + radius + 1, shape[0])
        lo_j, hi_j = np.maximum(j - radius, 0), np.minimum(j + radius + 1, shape[1])
        sums = (
            image[:, hi_i, hi_j]
            - image[:, lo_i, hi_j]
            - image[:, hi_i, lo_j]
            + image[:, lo_i, lo_j]
        )
        ready = (sums[0] >= min_count) | (radius == max(shape) - 1)
        n, r, e2 = sums[:, ready]
        weight[pending[ready]] = np.divide(
            r**2, n * e2, out=np.zeros_like(r), where=n * e2 > 0
        )
        pending = pending[~ready]
        if not len(pending):
            break
    return weight


class DESJackknife:
    """Accumulate raw moments, then apply each deletion's refitted cell weights.

    The caller supplies the full-sample edges, never deletion-dependent edges.
    Per branch/cell we need count, sum of no-shear shapes (selection response),
    and sum of that branch's shapes (joint response). The ns branch also stores
    sum R_g and the three valid-training moments used by fit_w.
    """

    def __init__(self, labels, edges_s, edges_r, step, min_count=3):
        self.labels = np.asarray(labels)
        if len(labels) < 2:
            raise ValueError("Response jackknife requires at least two spatial groups")
        self.edges = edges_s, edges_r
        self.shape = len(edges_s) - 1, len(edges_r) - 1
        self.cells = np.prod(self.shape)
        self.step, self.min_count = step, min_count
        shape = len(labels), self.cells
        self.sums = {b: np.zeros((5, *shape)) for b in response.BRANCHES}
        self.rg = np.zeros((4, *shape))
        self.training = np.zeros((3, *shape))

    def add_branch(self, b, groups, snr, size, ns_e, e, rg=None):
        """Add selected rows of one branch; rg aligns to selected ns rows only."""
        finite = np.isfinite(snr) & np.isfinite(size)
        cell = response.cell_index(snr, self.edges[0]) * self.shape[
            1
        ] + response.cell_index(size, self.edges[1])
        index = np.searchsorted(self.labels, groups) * self.cells + cell

        def add(target, values, valid=finite):
            ix = index[valid]
            if not len(ix):
                return
            v = None if values is None else values[valid]
            lo, hi = ix.min(), ix.max() + 1
            flat = target.reshape(-1)
            if hi - lo <= 2 * len(ix):
                # Sorted spatial catalogues usually touch only a few groups
                # per chunk. Never clear a groups*cells scratch per chunk:
                # cap the dense span at O(chunk rows), otherwise scatter.
                flat[lo:hi] += np.bincount(ix - lo, weights=v, minlength=hi - lo)
            else:
                np.add.at(flat, ix, 1 if v is None else v)

        add(self.sums[b][0], None)
        for i in range(2):
            add(self.sums[b][1 + i], ns_e[i])
            add(self.sums[b][3 + i], e[i])
        if b == "ns":
            for i, values in enumerate(rg.reshape(4, -1)):
                add(self.rg[i], values)
            r = 0.5 * (rg[0, 0] + rg[1, 1])
            valid = finite & np.isfinite(e).all(axis=0) & np.isfinite(r)
            add(self.training[0], None, valid)
            add(self.training[1], r, valid)
            add(self.training[2], np.mean(e**2, axis=0), valid)

    def deleted_weight(self, group):
        """Return the retained table, useful for reference comparisons."""
        moments = self.training.sum(axis=1) - self.training[:, group]
        return pooled_weight(moments, self.shape, self.min_count).reshape(self.shape)

    def jackknife(self):
        """Return the same response/c estimates and covariance as the row path."""
        totals = {b: x.sum(axis=1) for b, x in self.sums.items()}
        training_total = self.training.sum(axis=1)
        rg_total = self.rg.sum(axis=1)
        samples = []
        for k in range(len(self.labels)):
            w = pooled_weight(
                training_total - self.training[:, k], self.shape, self.min_count
            )
            means = {}
            den = {}
            for b in response.BRANCHES:
                weighted = (totals[b] - self.sums[b][:, k]) @ w
                den[b] = weighted[0]
                if den[b] <= 0:
                    raise ValueError(
                        f"Empty weighted population after spatial deletion: {b}"
                    )
                means[b] = weighted[1:] / den[b]
            shear = ((rg_total - self.rg[:, k]) @ w / den["ns"]).reshape(2, 2)
            selection, joint = np.empty((2, 2)), np.empty((2, 2))
            for j, (p, m) in enumerate((("p1", "m1"), ("p2", "m2"))):
                selection[:, j] = (means[p][:2] - means[m][:2]) / (2 * self.step)
                joint[:, j] = (means[p][2:] - means[m][2:]) / (2 * self.step)
            total = shear + selection
            ce = means["ns"][:2]
            samples.append(
                dict(
                    shear=shear,
                    selection=selection,
                    total=total,
                    joint=joint,
                    c_e=ce,
                    c_g=np.linalg.solve(total, ce),
                )
            )
        estimates = {key: np.array([s[key] for s in samples]) for key in samples[0]}
        vectors = np.concatenate(
            [x.reshape(len(self.labels), -1) for x in estimates.values()], axis=1
        )
        delta = vectors - vectors.mean(axis=0)
        cov = (len(self.labels) - 1) / len(self.labels) * delta.T @ delta
        return self.labels, estimates, cov


def accumulate(gm, groups, chunk_size=500_000):
    """Read an existing metacal parent in bounded chunks, without copying it."""
    groups = np.asarray(groups)
    if len(groups) != len(gm.ns["g1"]):
        raise ValueError("Spatial groups must align to the common parent rows")
    # np.unique(return_inverse=True) on 350M rows would itself require GBs.
    labels = np.empty(0, dtype=groups.dtype)
    for start in range(0, len(groups), chunk_size):
        labels = np.union1d(labels, np.unique(groups[start : start + chunk_size]))
    acc = DESJackknife(labels, gm.W[0], gm.W[1], gm._step, gm._des_min_count)
    for b in response.BRANCHES:
        selected = gm.mask_dict[b]
        d = getattr(gm, b)
        for start in range(0, len(selected), chunk_size):
            idx = selected[start : start + chunk_size]
            acc.add_branch(
                b,
                groups[idx],
                d["flux"][idx] / d["flux_err"][idx],
                d["T"][idx] / d["Tpsf"][idx],
                np.array([x[idx] for x in gm.branches["ns"]]),
                np.array([x[idx] for x in gm.branches[b]]),
                gm.R_shear[:, :, start : start + chunk_size] if b == "ns" else None,
            )
    return acc
