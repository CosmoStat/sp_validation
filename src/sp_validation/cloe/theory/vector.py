"""A Euclid data vector as the theory sees it: retained spectra, operators, n(z), CLOE grid.

Each retained spectrum's bandpowers are predicted as ``operator @ C_ell(nodes)``, one linear
map per spectrum folding three steps (dr1_cmbx's ``TheoryVector`` convention, so the
bands are those production computes):

1. C_ell is computed on sparse multipole nodes (:func:`sparse_ells`: every ell from 2 to
   50, then 40 per decade to the top of the windows) and spline-interpolated to every ell
   (:func:`interpolation_matrix`: a cubic spline of ell (ell + 1) C_ell in ln ell);
2. each leg's harmonic response R(ell) multiplies it: the release's
   ``metadata["harmonic_response"]`` per tracer family (``kappa``, ``g``, ``l``) or tracer
   (``g0``), unless a spectrum carries its own total ``harmonic_response``;
3. the spectrum's bandpower window ``bpws`` integrates it over the window's support.

Each delivered n(z) is normalised to unit integral (Simpson's rule on its own grid); it is
not otherwise touched.
"""
import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline

from .cloe import Grid


def sparse_ells(lmax, *, dense_below=50, per_decade=40):
    """Evaluation multipoles: every ell in ``2..dense_below``, then log-spaced to ``lmax``."""
    if lmax <= dense_below:
        return np.arange(2, lmax + 1)
    ndec = np.log10(lmax / dense_below)
    tail = np.geomspace(dense_below, lmax, max(int(np.ceil(per_decade * ndec)), 2) + 1)
    return np.unique(np.concatenate([np.arange(2, dense_below + 1), np.round(tail).astype(int)]))


def interpolation_matrix(nodes, lmax):
    """Linear operator ``(lmax+1, len(nodes))`` from C_ell at ``nodes`` to every ell: a cubic
    spline of ``ell (ell + 1) C_ell`` in ``ln ell``; the ell = 0, 1 rows are zero."""
    nodes = np.asarray(nodes)
    matrix = np.zeros((lmax + 1, len(nodes)))
    ells = np.arange(2, lmax + 1)
    spline = CubicSpline(np.log(nodes), np.diag(nodes * (nodes + 1.0)), axis=0)
    matrix[2:] = spline(np.log(ells)) / (ells * (ells + 1.0))[:, None]
    return matrix


def window_support(window):
    """Number of multipoles ``ell = 0..n-1`` the window reaches (last nonzero column + 1)."""
    columns = np.flatnonzero(np.any(window != 0, axis=0))
    return int(columns[-1]) + 1


def response(product, tracer, nell):
    """The harmonic response R(ell) of one tracer (``K``, ``G0``, ``L3``) over ``nell``
    multipoles: by tracer token, else family, else unity. A key that names no tracer of the
    release, or a response that is short or not finite, is an error."""
    table = (product.get("metadata") or {}).get("harmonic_response") or {}
    known = {"kappa", "g", "l"} | set(product["dndz"])
    if set(table) - known:
        raise ValueError(f"harmonic_response keys {sorted(set(table) - known)} name no tracer")
    token = "kappa" if tracer == "K" else tracer.lower()
    family = "kappa" if tracer == "K" else tracer[0].lower()
    value = table.get(token, table.get(family))
    if value is None:
        return np.ones(nell)
    value = np.asarray(value, float)
    if value.ndim != 1 or len(value) < nell or not np.all(np.isfinite(value)):
        raise ValueError(f"harmonic response of {token} must be finite and cover ell < {nell}")
    return value[:nell]


class EuclidVector:
    """The theory side of one :class:`~euclid_spt_like.selection.Selection` of a release.

    Attributes: ``keys``, ``pairs``, ``indices``, ``data``, ``icov`` (from the selection),
    ``ells`` (the multipole nodes), ``operators`` (n_spectra, n_bands, n_nodes),
    ``gc_bins``/``wl_bins`` (galaxy and shear bin numbers present), ``nz`` (tracer label ->
    (z, normalised n(z))), ``include_mag_bias``, ``ia_z_pivot`` and ``grid`` (the
    :class:`~euclid_spt_like.cloe.Grid`).
    """

    def __init__(self, product, selection, *, engine, ia_z_pivot, include_mag_bias):
        self.keys, self.pairs = selection.keys, selection.pairs
        self.indices, self.data, self.icov = selection.indices, selection.data, selection.icov
        self.ia_z_pivot, self.include_mag_bias = ia_z_pivot, include_mag_bias
        windows = {k: np.asarray(product["cls"][k]["bpws"], float) for k in self.keys}
        lmax = max(window_support(w) for w in windows.values()) - 1
        self.ells = sparse_ells(lmax)
        interp = interpolation_matrix(self.ells, lmax)
        operators = []
        for key, (a, b) in zip(self.keys, self.pairs):
            nell = window_support(windows[key])
            own = product["cls"][key].get("harmonic_response")
            pair_response = (np.asarray(own, float)[:nell] if own is not None else
                             response(product, a, nell) * response(product, b, nell))
            operators.append((windows[key][:, :nell] * pair_response) @ interp[:nell])
        self.operators = np.stack(operators)
        labels = sorted({t for pair in self.pairs for t in pair if t != "K"},
                        key=lambda t: (t[0] != "G", int(t[1:])))
        self.gc_bins = [int(t[1:]) for t in labels if t[0] == "G"]
        self.wl_bins = [int(t[1:]) for t in labels if t[0] == "L"]
        self.nz = {}
        for t in labels:
            z, nz = (np.asarray(v, float) for v in product["dndz"][t.lower()])
            self.nz[t] = (z, nz / simpson(nz, x=z))
        self.grid = Grid(max(z[-1] for z, _ in self.nz.values()), engine)
