"""Metacalibration response, weights, additive bias and image-simulation bias.

The definitions are in ``docs/source/response_calibration.md`` (Martin
Kilbinger's slide S1, 6 October 2026, and case d of
``scripts/validation/wdes_response/wdes_response.py``).

Every weight is ``w = u * W[cell]``: ``cell`` is the object's (S/N, size) cell
on the branch being averaged, ``W`` a :class:`Table` on those cells, and ``u``
a per-object factor, 1 or the inverse variance of that branch's measured
ellipticity. Every estimate is therefore a table contracted with per-cell,
``u``-weighted sums over the selected rows of each branch. :class:`Sums` holds
those sums per spatial group; they add over catalogue chunks and over groups,
so one object serves an in-memory catalogue, a catalogue read in chunks, the
delete-one-group jackknife (with the table fixed or refitted per deletion) and
the image-simulation multiplicative bias.

Typical use::

    cuts = Cuts.from_config(config["metacal"])
    edges = table_edges(cuts, config["metacal"]["des_num_bins"])
    sums = sum(accumulate(chunk, cuts, edges, groups=tiles, labels=all_tiles)
               for chunk, tiles in chunks)
    table = fit_table(sums, min_count=3)
    result = estimate(sums, table)
    labels, deleted, covariance = jackknife(sums, refit(3))

No detection response is inferred from aligned metacal branches.
"""

from dataclasses import dataclass, replace
from functools import partial
from typing import NamedTuple

import numpy as np
from astropy.io import fits
from numpy.lib.stride_tricks import sliding_window_view

BRANCHES = ("ns", "p1", "m1", "p2", "m2")
SUFFIX = dict(ns="NOSHEAR", p1="1P", m1="1M", p2="2P", m2="2M")
# Sheared branch pairs (+, -) of shear component j = 0, 1.
PAIRS = (("p1", "m1"), ("p2", "m2"))
# Estimates in covariance order; matrices flatten row-major.
ESTIMATES = ("shear", "selection", "total", "joint", "c_e", "c_g")


# Columns and selection


def branch_columns(rows, branch):
    """Columns of one metacal ``branch`` (``ns``, ``p1``, ``m1``, ``p2``, ``m2``).

    ``rows`` is a structured array or a mapping of ``NGMIX_*`` columns. Returns
    the shapes ``g1``, ``g2``, the table coordinates ``snr = FLUX/FLUX_ERR``
    and ``size = T/T_PSF_RECONV``, ``flag``, and the measured ellipticity
    errors ``g1_err``, ``g2_err`` when the rows carry them.
    """
    suffix = SUFFIX[branch]
    names = rows.dtype.names if hasattr(rows, "dtype") else rows

    def column(name):
        return rows[f"NGMIX_{name}_{suffix}"]

    with np.errstate(divide="ignore", invalid="ignore"):
        columns = dict(
            g1=column("G1"),
            g2=column("G2"),
            snr=column("FLUX") / column("FLUX_ERR"),
            size=column("T") / column("T_PSF_RECONV"),
            flag=column("FLAGS"),
        )
    for i in (1, 2):
        if f"NGMIX_G{i}_ERR_{suffix}" in names:
            columns[f"g{i}_err"] = column(f"G{i}_ERR")
    return columns


class Cuts(NamedTuple):
    """The metacal selection of one branch; every inequality is strict.

    With ``size_corr_ell`` the size cut applies to
    ``size * (1 - |e|^2) / (1 + |e|^2)``; the table coordinate stays ``size``.
    """

    snr_min: float
    snr_max: float
    size_min: float
    size_max: float
    size_corr_ell: bool

    @classmethod
    def from_config(cls, metacal):
        """Cuts of a ``metacal`` config block (``gal_snr_min``, ...)."""
        return cls(
            metacal["gal_snr_min"],
            metacal["gal_snr_max"],
            metacal["gal_rel_size_min"],
            metacal["gal_rel_size_max"],
            metacal["gal_size_corr_ell"],
        )


def select(columns, cuts):
    """Boolean selection of :func:`branch_columns` under ``cuts``."""
    size = columns["size"]
    if cuts.size_corr_ell:
        e2 = columns["g1"] ** 2 + columns["g2"] ** 2
        size = size * (1 - e2) / (1 + e2)
    snr = columns["snr"]
    with np.errstate(invalid="ignore"):
        return (
            (columns["flag"] == 0)
            & (snr > cuts.snr_min)
            & (snr < cuts.snr_max)
            & (size > cuts.size_min)
            & (size < cuts.size_max)
        )


def inverse_variance(columns, sigma_e):
    """``u = 1 / (2 sigma_e^2 + g1_err^2 + g2_err^2)`` from a branch's errors."""
    return 1 / (2 * sigma_e**2 + columns["g1_err"] ** 2 + columns["g2_err"] ** 2)


def shear_response(columns, step):
    """Per-object ``R_g[i, j] = (e_i^{j+} - e_i^{j-}) / 2h``, shape (2, 2, n).

    ``columns`` maps each branch to its :func:`branch_columns`.
    """
    return np.array(
        [
            [
                (columns[p][f"g{i + 1}"] - columns[m][f"g{i + 1}"]) / (2 * step)
                for p, m in PAIRS
            ]
            for i in range(2)
        ]
    )


# The weight table


def cell_index(x, edges):
    """Right-closed cells; out-of-range finite values use nearest edge cell."""
    return np.clip(np.searchsorted(edges, x, side="left") - 1, 0, len(edges) - 2)


def table_edges(cuts, num_bins=20):
    """Logarithmic (S/N, size) cell edges between the cuts, ``num_bins`` a side."""
    return (
        np.logspace(np.log10(cuts.snr_min), np.log10(cuts.snr_max), num_bins + 1),
        np.logspace(np.log10(cuts.size_min), np.log10(cuts.size_max), num_bins + 1),
    )


class Table(NamedTuple):
    """A weight table ``w`` on (S/N, size) cells, with its training counts.

    ``count`` is the unpooled number of training objects per cell and
    ``min_count`` the pooling threshold the table was fitted with.
    """

    snr_edges: np.ndarray
    size_edges: np.ndarray
    w: np.ndarray
    count: np.ndarray
    min_count: int

    def evaluate(self, snr, size):
        """Table weight at (snr, size); zero where a coordinate is nonfinite."""
        snr, size = np.asarray(snr), np.asarray(size)
        w = self.w[cell_index(snr, self.snr_edges), cell_index(size, self.size_edges)]
        return np.where(np.isfinite(snr) & np.isfinite(size), w, 0.0)


def unit_table(edges):
    """The all-ones table: unit weights, or inverse-variance weights with ``u``."""
    shape = (len(edges[0]) - 1, len(edges[1]) - 1)
    return Table(*edges, np.ones(shape), np.zeros(shape, dtype=np.int64), 0)


def _ratio(n, r, e2):
    return np.divide(r**2, n * e2, out=np.zeros_like(r), where=n * e2 > 0)


def _box_sums(moments, radius):
    """Sums of each cell's clipped square window of half-width ``radius``."""
    padded = np.pad(moments, ((0, 0), (radius, radius), (radius, radius)))
    rows = sliding_window_view(padded, 2 * radius + 1, axis=1).sum(axis=-1)
    return sliding_window_view(rows, 2 * radius + 1, axis=2).sum(axis=-1)


def pool(count, rsum, esum, min_count):
    """The DES weight per cell, ``W = (sum r)^2 / (n sum |e|^2/2)``.

    ``count``, ``rsum`` and ``esum`` are per-cell training moments on the
    (S/N, size) grid. A cell with fewer than ``min_count`` training objects
    takes the moments of the smallest square of cells centred on it (clipped
    at the table edges) that holds ``min_count``, or of the whole table. A
    cell whose (pooled) ``n sum |e|^2`` is zero gets zero weight. Window sums
    add moments directly, never by differencing cumulative sums, so a sparse
    window carries no cancellation error from dense neighbours.
    """
    moments = np.array([count, rsum, esum], dtype=float)
    w = _ratio(*moments)
    pending = count < min_count
    largest = max(count.shape) - 1
    for radius in range(1, largest + 1):
        if not pending.any():
            break
        window = _box_sums(moments, radius)
        ready = pending & ((window[0] >= min_count) | (radius == largest))
        w[ready] = _ratio(*window)[ready]
        pending &= ~ready
    return w


def fit_table(sums, min_count):
    """Fit the DES table on the no-shear training moments of ``sums``.

    The moments are unweighted, over no-shear-selected rows whose coordinates,
    shapes and response are all finite; :func:`pool` gives ``W``.
    """
    total = sums.total().arrays
    shape = (len(sums.edges[0]) - 1, len(sums.edges[1]) - 1)
    count = total["train_count"][0].reshape(shape)
    rsum, esum = (x[0].reshape(shape) for x in total["train"])
    return Table(*sums.edges, pool(count, rsum, esum, min_count), count, min_count)


def refit(min_count):
    """A table rule for :func:`estimate` and :func:`jackknife`: fit the table on
    whichever sums are being estimated (every deletion refits it)."""
    return partial(fit_table, min_count=min_count)


def write_table(path, table):
    """Write ``table`` as the DES weight product :func:`read_table` reads."""
    header = fits.Header()
    header["WGHTSCH"] = "des"
    header["WSNRCOL"] = "FLUX/FLUX_ERR"
    header["WSIZDEF"] = "T/T_PSF_RECONV"
    header["DESNMIN"] = (table.min_count, "Minimum DES training count")
    header["DESMERGE"] = "concentric square; pooled moments"
    fits.HDUList(
        [
            fits.PrimaryHDU(header=header),
            fits.ImageHDU(table.w, name="W_DES"),
            fits.ImageHDU(table.count, name="W_COUNT"),
            fits.ImageHDU(
                np.array([table.snr_edges, table.size_edges]), name="W_EDGES"
            ),
        ]
    ).writeto(path)


def read_table(path):
    """Read a DES weight table from a FITS product with ``W_DES``, ``W_COUNT``
    and ``W_EDGES`` extensions, whose header declares its coordinates."""
    with fits.open(path) as hdus:
        header = hdus[0].header
        if (
            header.get("WGHTSCH") != "des"
            or header.get("WSNRCOL") != "FLUX/FLUX_ERR"
            or header.get("WSIZDEF") != "T/T_PSF_RECONV"
        ):
            raise ValueError(f"{path} does not declare the DES table coordinates")
        snr_edges, size_edges = np.array(hdus["W_EDGES"].data, dtype=float)
        return Table(
            snr_edges,
            size_edges,
            np.array(hdus["W_DES"].data, dtype=float),
            np.array(hdus["W_COUNT"].data),
            header["DESNMIN"],
        )


# Sufficient statistics


@dataclass(frozen=True)
class Sums:
    """Per-(group, cell) sums of the selected rows of every metacal branch.

    ``arrays`` holds, with the group and cell axes last:

    - ``count`` (5, G, C), ``weight`` (5, G, C): rows and ``sum u`` per branch
      (in :data:`BRANCHES` order), over that branch's selection, in the cells
      of that branch's own coordinates;
    - ``ns_e`` (5, 2, G, C), ``own_e`` (5, 2, G, C): ``sum u e`` of the
      no-shear shapes and of the branch's own shapes over the same rows;
    - ``rg`` (2, 2, G, C): ``sum u R_g`` over the no-shear selection;
    - ``train_count`` (G, C), ``train`` (2, G, C): the table's unweighted
      training moments, ``n``, ``sum r`` with ``r = (R_g11 + R_g22) / 2`` and
      ``sum |e|^2 / 2``.

    ``labels`` names the G groups (``None``: one group), ``edges`` the cells,
    ``step`` the metacal shear step and ``sigma_e`` the inverse-variance
    ``u`` (``None``: ``u = 1``). Sums of disjoint row sets add with ``+``.
    """

    arrays: dict
    edges: tuple
    labels: np.ndarray = None
    step: float = 0.01
    sigma_e: float = None

    def _check(self, other):
        same = (
            self.step == other.step
            and self.sigma_e == other.sigma_e
            and all(map(np.array_equal, self.edges, other.edges))
            and (
                (self.labels is None and other.labels is None)
                or (
                    self.labels is not None
                    and other.labels is not None
                    and np.array_equal(self.labels, other.labels)
                )
            )
        )
        if not same:
            raise ValueError("Sums differ in groups, cells, step or weights")

    def __add__(self, other):
        if isinstance(other, int) and other == 0:  # the start value of sum()
            return self
        self._check(other)
        return replace(
            self, arrays={k: v + other.arrays[k] for k, v in self.arrays.items()}
        )

    __radd__ = __add__

    def total(self):
        """The sums over all groups, as one group."""
        return replace(
            self,
            arrays={k: v.sum(axis=-2, keepdims=True) for k, v in self.arrays.items()},
            labels=None,
        )

    def deletions(self):
        """Yield, for each group in turn, the sums of all the other groups."""
        total = self.total().arrays
        for k in range(len(self.labels)):
            yield replace(
                self,
                arrays={
                    key: total[key] - v[..., k : k + 1, :]
                    for key, v in self.arrays.items()
                },
                labels=None,
            )


def _empty(groups, cells):
    def zeros(*shape, dtype=float):
        return np.zeros((*shape, groups, cells), dtype=dtype)

    return dict(
        count=zeros(5, dtype=np.int64),
        weight=zeros(5),
        ns_e=zeros(5, 2),
        own_e=zeros(5, 2),
        rg=zeros(2, 2),
        train_count=zeros(dtype=np.int64),
        train=zeros(2),
    )


def _add(target, index, values=None):
    """Add ``values`` (or 1) at flat ``index`` of the contiguous ``target``."""
    if not len(index):
        return
    lo, hi = index.min(), index.max() + 1
    target.reshape(-1)[lo:hi] += np.bincount(
        index - lo, weights=values, minlength=hi - lo
    ).astype(target.dtype, copy=False)


def accumulate(
    rows,
    cuts,
    edges,
    *,
    preselection=None,
    groups=None,
    labels=None,
    sigma_e=None,
    step=0.01,
):
    """Reduce catalogue rows to :class:`Sums`.

    Parameters
    ----------
    rows : numpy.ndarray or mapping
        catalogue rows with the ``NGMIX_*`` columns of all five branches
    cuts : Cuts
        the metacal selection, made on each branch separately
    edges : tuple of numpy.ndarray
        (S/N, size) cell edges, the same for every chunk; those of the table
        the sums will be contracted with
    preselection : numpy.ndarray of bool, optional
        rows that enter the metacal selection at all
    groups : numpy.ndarray, optional
        spatial group (e.g. tile) of each row; ``None`` makes one group
    labels : numpy.ndarray, optional
        the sorted labels of all groups, shared by every chunk; defaults to the
        labels present in ``groups``
    sigma_e : float, optional
        ``u = 1 / (2 sigma_e^2 + g1_err^2 + g2_err^2)`` from each branch's own
        errors; ``None`` makes ``u = 1`` (DES and unit weights)
    step : float
        metacal shear step h

    Rows with a nonfinite coordinate on a branch are left out of that branch.
    """
    columns = {b: branch_columns(rows, b) for b in BRANCHES}
    if preselection is not None:
        columns = {
            b: {k: v[preselection] for k, v in c.items()} for b, c in columns.items()
        }
        groups = None if groups is None else np.asarray(groups)[preselection]
    n = len(columns["ns"]["g1"])
    if groups is None:
        labels, group = None, np.zeros(n, dtype=np.int64)
    else:
        groups = np.asarray(groups)
        labels = np.unique(groups) if labels is None else np.asarray(labels)
        group = np.searchsorted(labels, groups)
        if np.any(labels[np.minimum(group, len(labels) - 1)] != groups):
            raise ValueError("groups hold a label that labels lacks")
    n_size = len(edges[1]) - 1
    cells = (len(edges[0]) - 1) * n_size
    arrays = _empty(1 if labels is None else len(labels), cells)
    e_ns = np.array([columns["ns"]["g1"], columns["ns"]["g2"]])
    for k, b in enumerate(BRANCHES):
        c = columns[b]
        rows_b = np.flatnonzero(
            select(c, cuts) & np.isfinite(c["snr"]) & np.isfinite(c["size"])
        )
        index = group[rows_b] * cells + (
            cell_index(c["snr"][rows_b], edges[0]) * n_size
            + cell_index(c["size"][rows_b], edges[1])
        )
        u = (
            np.ones(len(rows_b))
            if sigma_e is None
            else inverse_variance(c, sigma_e)[rows_b]
        )
        _add(arrays["count"][k], index)
        _add(arrays["weight"][k], index, u)
        for i in range(2):
            _add(arrays["ns_e"][k, i], index, u * e_ns[i, rows_b])
            _add(arrays["own_e"][k, i], index, u * c[f"g{i + 1}"][rows_b])
        if b == "ns":
            rg = shear_response(columns, step)[:, :, rows_b]
            for i in range(2):
                for j in range(2):
                    _add(arrays["rg"][i, j], index, u * rg[i, j])
            r = (rg[0, 0] + rg[1, 1]) / 2
            e = e_ns[:, rows_b]
            valid = np.isfinite(r) & np.isfinite(e).all(axis=0)
            _add(arrays["train_count"], index[valid])
            _add(arrays["train"][0], index[valid], r[valid])
            _add(arrays["train"][1], index[valid], np.mean(e[:, valid] ** 2, axis=0))
    return Sums(arrays, tuple(edges), labels, step, sigma_e)


# Estimates


def _table(sums, table):
    """``table`` itself, or the table its rule makes from ``sums``."""
    if not isinstance(table, Table):
        table = table(sums)
    if not all(map(np.array_equal, sums.edges, table[:2])):
        raise ValueError("The table's cells are not those of the sums")
    return table


def estimate(sums, table):
    """Response, additive bias and branch populations of ``sums``.

    ``table`` is a :class:`Table` or a rule making one from the sums
    (:func:`refit`). Every branch mean is ``sum u W e / sum u W`` over that
    branch's own selection, weights and cells. Returns ``shear`` (``R_g``
    averaged on no-shear), ``selection``, ``total = shear + selection``,
    ``joint``, ``c_e`` (raw no-shear mean), ``c_g = total^-1 c_e``, and per
    branch the selected row ``counts`` and ``weight_sums``.
    """
    w = _table(sums, table).w.ravel()
    a = {k: v.sum(axis=-2) for k, v in sums.arrays.items()}
    den = a["weight"] @ w
    if not np.all(np.isfinite(den) & (den > 0)):
        raise ValueError(f"Empty or invalid weighted population: {den}")
    ns_e = a["ns_e"] @ w / den[:, None]
    own_e = a["own_e"] @ w / den[:, None]
    shear = a["rg"] @ w / den[0]
    selection, joint = np.empty((2, 2)), np.empty((2, 2))
    for j, (p, m) in enumerate(PAIRS):
        p, m = BRANCHES.index(p), BRANCHES.index(m)
        selection[:, j] = (ns_e[p] - ns_e[m]) / (2 * sums.step)
        joint[:, j] = (own_e[p] - own_e[m]) / (2 * sums.step)
    total = shear + selection
    return dict(
        shear=shear,
        selection=selection,
        total=total,
        joint=joint,
        c_e=ns_e[0],
        c_g=np.linalg.solve(total, ns_e[0]),
        counts=dict(zip(BRANCHES, a["count"].sum(axis=-1).tolist(), strict=True)),
        weight_sums=dict(zip(BRANCHES, den.tolist(), strict=True)),
    )


def _covariance(vectors):
    """Delete-one jackknife covariance ``(K - 1)/K sum delta delta^T``."""
    delta = vectors - vectors.mean(axis=0)
    return (len(vectors) - 1) / len(vectors) * delta.T @ delta


def jackknife(sums, table):
    """Delete each group of ``sums`` in turn and re-estimate.

    A :class:`Table` stays fixed; a rule (:func:`refit`) refits the table on
    each deletion's retained training moments. Returns the group labels, the
    deleted :data:`ESTIMATES` stacked along a first axis, and their covariance
    in :data:`ESTIMATES` order, matrices row-major (20 elements).
    """
    if sums.labels is None or len(sums.labels) < 2:
        raise ValueError("A jackknife needs at least two groups")
    samples = [estimate(retained, table) for retained in sums.deletions()]
    deleted = {key: np.array([s[key] for s in samples]) for key in ESTIMATES}
    vectors = np.concatenate(
        [deleted[key].reshape(len(samples), -1) for key in ESTIMATES], axis=1
    )
    return sums.labels, deleted, _covariance(vectors)


# Image-simulation shear bias


def shear_bias(pairs, g_in, table, response="total"):
    """Multiplicative and additive bias from simulations sheared by +-g_in.

    ``pairs`` maps shear component j (0, 1) to the :class:`Sums` of its ``+g``
    and ``-g`` simulations. Each simulation's mean shear is
    ``g = R^-1 <e>_ns`` with its own response ``R = estimate[response]``
    (``total``, or ``shear`` for the shear part alone) and no additive
    subtraction; then ``m_j = (g_j^+ - g_j^-) / 2 g_in - 1`` and
    ``c_j = (g_j^+ + g_j^-) / 2``. ``table`` is as in :func:`estimate`, used
    by every simulation. Returns ``{m1, c1, m2, c2}`` for the given components.
    """
    result = {}
    for j, sims in pairs.items():
        g = []
        for sums in sims:
            est = estimate(sums, table)
            g.append(np.linalg.solve(est[response], est["c_e"])[j])
        result[f"m{j + 1}"] = (g[0] - g[1]) / (2 * g_in) - 1
        result[f"c{j + 1}"] = (g[0] + g[1]) / 2
    return result


def shear_bias_jackknife(pairs, g_in, table, response="total"):
    """Delete each group from every simulation at once and re-measure.

    All simulations' sums must share their group labels (the tiles the
    simulations share). Returns the labels, the deleted values of
    :func:`shear_bias` keyed by name, and their covariance in that key order.
    """
    sims = [sums for pair in pairs.values() for sums in pair]
    labels = sims[0].labels
    if labels is None or len(labels) < 2:
        raise ValueError("A jackknife needs at least two groups")
    if not all(np.array_equal(s.labels, labels) for s in sims):
        raise ValueError("The simulations' groups differ")
    samples = []
    for retained in zip(*(s.deletions() for s in sims), strict=True):
        it = iter(retained)
        deleted = {j: (next(it), next(it)) for j in pairs}
        samples.append(shear_bias(deleted, g_in, table, response))
    deleted = {key: np.array([s[key] for s in samples]) for key in samples[0]}
    return labels, deleted, _covariance(np.array([list(s.values()) for s in samples]))


# Per-object products


def weights(table, columns, sigma_e=None):
    """Per-object weights ``u * W`` of one branch's :func:`branch_columns`."""
    w = table.evaluate(columns["snr"], columns["size"])
    return w if sigma_e is None else w * inverse_variance(columns, sigma_e)


def calibrate(g, result, additive=True):
    """Calibrated shear ``R^-1 (g - c_e)`` of shapes ``g`` (2, n), with the
    ``total`` response and ``c_e`` of an :func:`estimate`; ``R^-1 g`` without
    the ``additive`` correction (constant-shear simulations)."""
    g = np.asarray(g)
    if additive:
        g = g - result["c_e"][:, None]
    return np.linalg.solve(result["total"], g)


# Row-array implementations, checked against the core in test_response.py


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


def row_jackknife(branches, masks, weights, rg, step, groups):
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
