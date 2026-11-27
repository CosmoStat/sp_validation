"""UNIONS data vectors on the CLOE JAX theory.

Every UNIONS two-point statistic is a fixed linear map of the angular power
spectra,

    model(statistic, pair, band) = operator(band, ell) @ C_ell(theta),

so one theory, :class:`Theory` (:class:`~.theory.jax_vector.JaxVector`: CLOE's
background, NLA, Limber and the CosmoPower-JAX HMcode-2020 + feedback P(k)),
serves them all, and each is differentiable. The operators are numpy arrays
computed once:

- harmonic bandpowers: a SACC's bandpower windows (:meth:`Vector.from_sacc`), e.g.
  a pseudo-C_ell coupling times binning;
- xi+, xi-, gamma_t, w(theta): :func:`angular` (full sky, Wigner d functions,
  bin-averaged), :func:`points` (full sky, at points) and :func:`hankel` (flat
  sky, at points, as CosmoSIS evaluates them);
- COSEBIs and pure E/B: :func:`compose` of sp_validation's own operators on binned
  xi+- (``b_modes.cosebis_operator``, ``pure_eb_operator``) with :func:`angular`.

Tracers: the theory names shear bins ``L<i>`` and galaxy (lens) bins ``G<i>``; in a
SACC they are ``source_<i>`` and ``lens_<i>`` (:mod:`sp_validation.sacc_io`).
"""

import re
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import simpson
from scipy.special import jv as bessel

from .. import sacc_io
from .theory.cloe import Grid, camb_background
from .theory.jax_vector import LINEAR_LOG_T_AGN, JaxVector
from .theory.vector import interpolation_matrix, sparse_ells

ARCMIN = np.pi / 180 / 60
#: SACC harmonic data type -> the theory's spectrum family.
FAMILY = {sacc_io.CL_GG: "GG", sacc_io.CL_GE: "GL", sacc_io.CL_EE: "LL"}


def tracer(name):
    """The theory's label of a SACC tracer: ``source_2 -> L2``, ``lens_0 -> G0``."""
    kind, i = name.rsplit("_", 1)
    return {"source": "L", "lens": "G"}[kind] + i


# --------------------------------------------------------------------------- #
# operators: (n_band, lmax + 1) maps from C_ell at every multipole
# --------------------------------------------------------------------------- #
#: statistic -> (m, m') of the Wigner d^l_{m m'}; shear spectra are E modes.
WIGNER = {"xip": (2, 2), "xim": (2, -2), "gammat": (2, 0), "w": (0, 0)}
#: statistic -> order of the Bessel function of the flat-sky transform.
HANKEL = {"xip": 0, "xim": 4, "gammat": 2, "w": 0}


def wigner_d(lmax, m, n, theta):
    """d^l_{m n}(theta) for l = 0..lmax (rows), m >= |n|, by the three-term
    recursion in l from l = m."""
    from math import lgamma

    theta = np.atleast_1d(np.asarray(theta, float))
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    out = np.zeros((lmax + 1, len(theta)))
    ln = 0.5 * (lgamma(2 * m + 1) - lgamma(m + n + 1) - lgamma(m - n + 1))
    out[m] = np.exp(ln) * c ** (m + n) * s ** (m - n)
    x = np.cos(theta)
    for ell in range(m, lmax):
        if ell == 0:
            out[1] = x * out[0]
            continue
        a = np.sqrt(((ell + 1.0) ** 2 - m * m) * ((ell + 1.0) ** 2 - n * n)) * ell
        b = (2 * ell + 1.0) * (ell * (ell + 1.0) * x - m * n)
        e = (ell + 1.0) * np.sqrt((ell * ell - m * m) * (ell * ell - n * n + 0.0))
        out[ell + 1] = (b * out[ell] - e * (out[ell - 1] if ell > m else 0.0)) / a
    return out


def angular(stat, theta_edges, lmax, nodes=48):
    """Full-sky averages of sum_l (2l+1)/4pi d^l C_l over each annulus of
    ``theta_edges`` (radians), weighted by area."""
    m, n = WIGNER[stat]
    x, w = np.polynomial.legendre.leggauss(nodes)
    lo, hi = np.asarray(theta_edges[:-1]), np.asarray(theta_edges[1:])
    t = 0.5 * (hi - lo)[:, None] * x + 0.5 * (hi + lo)[:, None]
    wt = w * np.sin(t)
    wt = wt / wt.sum(1, keepdims=True)
    d = wigner_d(lmax, m, n, t.ravel()).reshape(lmax + 1, *t.shape)
    ell = np.arange(lmax + 1)
    return ((2 * ell + 1) / (4 * np.pi))[None, :] * np.einsum("bq,lbq->bl", wt, d)


def points(stat, theta, lmax):
    """Full-sky values at ``theta`` (radians)."""
    m, n = WIGNER[stat]
    ell = np.arange(lmax + 1)
    return ((2 * ell + 1) / (4 * np.pi))[None, :] * wigner_d(lmax, m, n, theta).T


def hankel(stat, theta, lmax):
    """Flat-sky values at ``theta`` (radians): sum_l l/2pi J_n(l theta) C_l."""
    ell = np.arange(lmax + 1)
    return ell / (2 * np.pi) * bessel(HANKEL[stat], np.outer(theta, ell))


def compose(operator, theta_edges, lmax):
    """A statistic linear in binned [xi+; xi-] (an (n, 2 n_theta) operator on the
    bins of ``theta_edges``) as a map from C_ell."""
    n = len(theta_edges) - 1
    return operator[:, :n] @ angular("xip", theta_edges, lmax) + operator[
        :, n:
    ] @ angular("xim", theta_edges, lmax)


# --------------------------------------------------------------------------- #
# the data vector
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    """One statistic: its tracer pairs (``("G0", "L2")``), its operator per pair
    (n_pair, n_band, lmax + 1) or one (n_band, lmax + 1) for all, and the bands
    kept (a mask (n_pair, n_band); default all)."""

    name: str
    pairs: list
    operator: np.ndarray
    keep: np.ndarray = None


class Vector:
    """What :class:`Theory` reads: spectra, operators on the multipole nodes, n(z),
    data and covariance.

    ``nz``: {"L0": (z, n), ...}, every tracer on one uniform grid. ``data`` and
    ``cov`` are over the kept bands in block, pair, band order (:attr:`labels`); a
    None ``data`` is zeros until :meth:`Theory.fill`. Blocks may have different
    numbers of bands.
    """

    def __init__(
        self,
        blocks,
        nz,
        *,
        data=None,
        cov=None,
        ia_z_pivot=0.62,
        magnification=False,
        engine="cosmopower_mead2020_feedback",
    ):
        self.pairs = [p for b in blocks for p in b.pairs]
        self.keys = [f"{b.name}:{x}x{y}" for b in blocks for x, y in b.pairs]
        lmax = max(b.operator.shape[-1] for b in blocks) - 1
        self.ells = sparse_ells(lmax)
        interp = interpolation_matrix(self.ells, lmax)
        nband = max(b.operator.shape[-2] for b in blocks)
        ops, keep = [], []
        for b in blocks:
            op = b.operator
            if op.ndim == 2:
                op = np.broadcast_to(op, (len(b.pairs), *op.shape))
            k = np.ones(op.shape[:2], bool) if b.keep is None else np.asarray(b.keep)
            for o, kk in zip(op, k):
                full = np.zeros((nband, len(self.ells)))
                full[: o.shape[0]] = o @ interp[: o.shape[1]]
                ops.append(full)
                keep.append(np.r_[kk, np.zeros(nband - len(kk), bool)])
        self.operators = np.stack(ops)
        self.indices = np.flatnonzero(np.stack(keep).ravel())
        self.labels = [(self.keys[i // nband], int(i % nband)) for i in self.indices]
        n = len(self.indices)
        self.data = np.zeros(n) if data is None else np.asarray(data, float)
        self.cov = np.eye(n) if cov is None else np.asarray(cov, float)
        s = np.sqrt(np.diag(self.cov))
        self.icov = np.linalg.inv(self.cov / np.outer(s, s)) / np.outer(s, s)
        seen = sorted(
            {t for p in self.pairs for t in p}, key=lambda t: (t[0] != "G", int(t[1:]))
        )
        self.gc_bins = [int(t[1:]) for t in seen if t[0] == "G"]
        self.wl_bins = [int(t[1:]) for t in seen if t[0] == "L"]
        self.nz = {}
        for t in seen:
            z, p = (np.asarray(v, float) for v in nz[t])
            self.nz[t] = (z, p / simpson(p, x=z))
        self.include_mag_bias = magnification and bool(self.gc_bins)
        self.ia_z_pivot = ia_z_pivot
        self.grid = Grid(max(z[-1] for z, _ in self.nz.values()), engine)

    @classmethod
    def from_sacc(cls, s, **kw):
        """The harmonic spectra of SACC ``s`` (:data:`FAMILY`): each spectrum's
        bandpower windows are its operator, its means the data, and ``s``'s
        covariance on them the covariance."""
        blocks, index = [], []
        for dtype in s.get_data_types():
            for pair in s.get_tracer_combinations(dtype):
                idx = s.indices(dtype, pair)
                window = s.get_bandpower_windows(idx)
                op = np.zeros((len(idx), int(window.values.max()) + 1))
                op[:, window.values.astype(int)] = window.weight.T
                blocks.append(Block(dtype, [tuple(map(tracer, pair))], op[None]))
                index.append(idx)
        index = np.concatenate(index)
        nz = {tracer(n): (t.z, t.nz) for n, t in s.tracers.items() if n != "psf_stars"}
        cov = s.covariance.dense[np.ix_(index, index)]
        return cls(blocks, nz, data=s.mean[index], cov=cov, **kw)


#: A per-bin parameter and the tracer it belongs to (``bias_2 -> G2``).
TRACER_PARAMETER = re.compile(r"(bias|mag_slope|m|delta_z_[GL]|stretch_[GL])_(\d+)")


def reads(vector, name):
    """Whether the theory of ``vector`` reads parameter ``name``: a per-bin
    parameter of a tracer it has, IA with a shear bin, or anything else."""
    match = TRACER_PARAMETER.fullmatch(name)
    if match:
        stem, i = match.groups()
        galaxy = stem in ("bias", "mag_slope", "delta_z_G", "stretch_G")
        return int(i) in (vector.gc_bins if galaxy else vector.wl_bins)
    return name not in ("A_IA", "eta_IA") or bool(vector.wl_bins)


class Theory(JaxVector):
    """:class:`~.theory.jax_vector.JaxVector` on a :class:`Vector`, sampling either
    ln 10^10 A_s (``logA``) or S8 = sigma8 sqrt(Omega_m / 0.3), whichever ``names``
    or ``fixed`` holds.

    In the S8 basis ln A_s follows from the sigma8 network, sigma8 being
    proportional to sqrt(A_s); that network is also the derived sigma8's, so the
    derived S8 is the sampled one to the network's departure from that
    proportionality (about 1e-5).
    """

    #: ln 10^10 A_s at which the S8 basis evaluates the sigma8 network.
    REFERENCE_LOGA = 3.044

    def __init__(self, vector, names, *, fixed, **kw):
        if "S8" in names or "S8" in fixed:
            self.COSMOLOGY = ("S8",) + JaxVector.COSMOLOGY[1:]
        super().__init__(vector, names, fixed=fixed, **kw)

    def cosmology(self, theta):
        c = super().cosmology(theta)
        if self.COSMOLOGY[0] != "S8":
            return c
        ref = {**c, "logA": jnp.asarray(self.REFERENCE_LOGA)}
        log_t = ref["logT"] if self.feedback else LINEAR_LOG_T_AGN
        row = self._emulator_row(ref, log_t)[None, :]
        sigma = self.network(self._sigma, row)[0, 0]
        omega_m = (c["omch2"] + c["ombh2"] + self._nu_today) / c["h"] ** 2
        sigma8 = self._get(theta, "S8", np.nan) / jnp.sqrt(omega_m / 0.3)
        return {**c, "logA": self.REFERENCE_LOGA + 2 * jnp.log(sigma8 / sigma)}

    def model(self, theta):
        """The model of the kept bands, in :attr:`Vector.labels` order."""
        return self._bands_with_status(theta)[0][self.selection]

    def fill(self, theta):
        """Set the data to the model at ``theta`` (a noiseless vector); returns it."""
        self.data = np.asarray(jax.jit(self.model)(jnp.asarray(theta, float)))
        self._jit_chi2 = jax.jit(self._chi2_with_status)
        return self.data


# --------------------------------------------------------------------------- #
# scale cuts
# --------------------------------------------------------------------------- #
def cut(
    s,
    *,
    families=("GG", "GL", "LL"),
    shear=None,
    lens_ell_min=None,
    lens_k_max=None,
    lens_source_pairs=None,
    reference=None,
):
    """A copy of SACC ``s`` with the bands of a scale-cut configuration.

    Kept: spectra of ``families``; in shear spectra the bands with ``shear[0] <
    ell < shear[1]``; in spectra with a lens leg, galaxy autos only, galaxy-shear
    pairs ``(lens, source)`` of ``lens_source_pairs`` only, and bands with
    ``lens_ell_min < ell < k_max h chi(<z>)``: the linear-bias cut, ``k_max`` of
    ``lens_k_max`` per family (h/Mpc), <z> the lens bin's mean redshift and chi its
    comoving distance at the ``reference`` cosmology (``Omega_c, Omega_b, h, n_s,
    A_s, m_nu``).
    """
    if lens_k_max:
        background = camb_background(reference)
        chi = {}
        for name, t in s.tracers.items():
            if name.startswith("lens_"):
                zbar = np.trapezoid(t.z * t.nz, t.z) / np.trapezoid(t.nz, t.z)
                chi[tracer(name)] = float(background.comoving_distance(zbar))
    pairs = {tuple(p) for p in lens_source_pairs or ()}
    keep = []
    for i, point in enumerate(s.data):
        family = FAMILY[point.data_type]
        if family not in families:
            continue
        a, b = sorted(map(tracer, point.tracers))
        ell = point.get_tag("ell")
        if family == "LL":
            lo, hi = shear
        elif family == "GG" and a != b:
            continue
        elif family == "GL" and (int(a[1:]), int(b[1:])) not in pairs:
            continue
        else:
            lo = lens_ell_min
            legs = [t for t in (a, b) if t[0] == "G"]
            hi = min(lens_k_max[family] * reference["h"] * chi[t] for t in legs)
        if lo < ell < hi:
            keep.append(i)
    out = s.copy()
    out.keep_indices(np.array(keep))
    return out
