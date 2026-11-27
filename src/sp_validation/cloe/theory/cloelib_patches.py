"""Patches to cloelib v0.11 for jit/vmap/grad at production grid sizes.

Each patch is one small unit (a function and the subclass or mixin that applies it), written
against cloelib's own classes and nothing of dr1_cmbx, so it can become an upstream merge
request as it stands. Nothing here modifies cloelib: a caller opts in through the subclass,
or through :func:`compose`, which stacks the mixins of the patches it is given onto cloelib's
two-point, shear and positions classes. Importing this module checks that the installed
cloelib is of the pinned release series (0.11.x).

Effects are measured on the tr1_v2 full vector (90 spectra, 133 multipoles, 1199 Limber
nodes, Delta z = 0.005) through :mod:`.cloe_jax`, on 8 CPUs at one point, each patch off
with the others on: "value" is one log-likelihood, "gradient" one value-and-gradient, band
shifts are the largest over five points, in Gaussian sigma.

- P1 :class:`GLComovingDistance`: chi(z) by fixed Gauss-Legendre quadrature, vectorised over z.
- P3 :class:`UniformLnkTwoPoint`: the Limber P(k) gather on a uniform ln k table, with no
  binary search.
- P4 :class:`PowerLawTwoPoint`: no cubic extrapolation of log P beyond the k table.
- P5 :class:`LinearEfficiencyShear`, :class:`LinearEfficiencyPositions`: the lensing
  (magnification) efficiency in O(N).
- P6 :class:`SharedLimberPk`: one Limber P(k) for every pair of one evaluation.
- P8 :class:`LogTableTwoPoint`: the gather reads ln P directly from a backend that holds it.

P1 does not arise in :mod:`.cloe_jax`: its background holds chi as numbers, so cloelib's
per-redshift quadgk never runs. It is here for a caller of cloelib's own background.

P4 does not arise in :mod:`.cloe_jax`: its P(k) table reaches beyond every Limber
wavenumber of any cosmology in the emulator box (checked when the adapter is built), so
nothing extrapolates. It is here for a caller with a table that ends inside the Limber range.
"""

from functools import lru_cache

import numpy as np

import jax
import jax.numpy as jnp
from cloelib._version import __version__ as _cloelib_version
from cloelib.auxiliary.math_utils import stacked_simpson
from cloelib.cosmology.jax_cosmology import JAXBackground
from cloelib.observables.photo import PositionsTracer, ShearTracer
from cloelib.observables.photo.contributions import MATTER
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
import interpax

#: The cloelib release series the patches are written against (any 0.11.x).
CLOELIB_VERSION = "0.11"
if _cloelib_version != CLOELIB_VERSION and not _cloelib_version.startswith(CLOELIB_VERSION + "."):
    raise ImportError(f"cloelib_patches is pinned to cloelib {CLOELIB_VERSION}.x, "
                      f"found {_cloelib_version}")

C_KMS = 299792.458

#: patch -> [(role, rank, mixin)]: the classes that apply the patch to cloelib's "two_point",
#: "shear" or "positions" class; a lower rank sits earlier in the method resolution order.
LAYERS = {}


def layer(patch, role, rank=0):
    def register(cls):
        LAYERS.setdefault(patch, []).append((role, rank, cls))
        return cls
    return register


def compose(patches, bases=None):
    """The (two_point, shear, positions) classes with ``patches`` applied, on top of cloelib's;
    ``bases`` maps a role to extra classes placed first."""
    bases = bases or {}
    unknown = set(patches) - set(LAYERS)
    if unknown:
        raise ValueError(f"no two-point, shear or positions layer for {sorted(unknown)}")
    out = []
    for role, base in (("two_point", AngularTwoPoint), ("shear", ShearTracer), ("positions", PositionsTracer)):
        mixins = sorted(((rank, cls) for p in patches for r, rank, cls in LAYERS[p] if r == role),
                        key=lambda m: m[0])
        out.append(type(role, tuple(bases.get(role, ())) + tuple(c for _, c in mixins) + (base,), {}))
    return tuple(out)


# --------------------------------------------------------------------------- #
# P3: the Limber P(k) on a uniform ln k table
# --------------------------------------------------------------------------- #
def _spline_matrix(x_nodes, x_eval):
    from scipy.interpolate import CubicSpline
    return CubicSpline(np.asarray(x_nodes, float), np.eye(len(x_nodes)), axis=0)(np.asarray(x_eval, float))


def uniform_lnk_pkl_interp(k_lz, z_l, ks, zs, Pk):
    """Drop-in for ``Pkl_interp_vmap`` (same arguments, ``Pk`` (len(ks), len(zs)), output
    (n_ell, len(z_l))) for a table uniform in ln k: :func:`uniform_lnk_interp_ln` of ln P."""
    return uniform_lnk_interp_ln(k_lz, z_l, ks, zs, jnp.log(Pk).T)


def uniform_lnk_interp_ln(k_lz, z_l, ks, zs, lnp):
    """P(k_lz, z_l), (n_ell, len(z_l)), from ln P on (zs, ks), ks uniform in ln k.

    ln P is carried to the Limber redshifts by a fixed not-a-knot cubic-spline matrix (none
    when ``zs`` is ``z_l``), then evaluated at ln k by 4-point Lagrange with the interval
    found arithmetically. Beyond the table ln P continues linearly in ln k with the edge slope.
    """
    ks, zs, z_l = np.asarray(ks, float), np.asarray(zs, float), np.asarray(z_l, float)
    lnk_u = np.log(ks)
    step = np.diff(lnk_u)
    if not np.allclose(step, step[0], rtol=1e-8, atol=0):
        raise ValueError("uniform_lnk_pkl_interp needs a table uniform in ln k")
    dlnk = step[0]
    if zs.shape != z_l.shape or not np.array_equal(zs, z_l):
        lnp = jnp.asarray(_spline_matrix(zs, z_l)) @ lnp
    lnk = jnp.log(k_lz).T                                    # (len(z_l), n_ell)
    n = len(lnk_u)
    x = (jnp.clip(lnk, lnk_u[0], lnk_u[-1]) - lnk_u[0]) / dlnk
    i = jnp.clip(jnp.floor(x).astype(int), 1, n - 3)
    t = x - i
    rows = jnp.arange(lnp.shape[0])[:, None]
    p0, p1, p2, p3 = lnp[rows, i - 1], lnp[rows, i], lnp[rows, i + 1], lnp[rows, i + 2]
    value = (-t * (t - 1) * (t - 2) / 6 * p0 + (t + 1) * (t - 1) * (t - 2) / 2 * p1
             - (t + 1) * t * (t - 2) / 2 * p2 + (t + 1) * t * (t - 1) / 6 * p3)
    hi = (lnp[:, -1] - lnp[:, -2])[:, None] / dlnk * jnp.maximum(lnk - lnk_u[-1], 0.0)
    lo = (lnp[:, 1] - lnp[:, 0])[:, None] / dlnk * jnp.minimum(lnk - lnk_u[0], 0.0)
    return jnp.exp(value + hi + lo).T


class _PklTwoPoint(AngularTwoPoint):
    """v0.11's ``_limber_base_spectra`` with the interpolation as a class attribute."""

    pkl_interp = None

    def _limber_base_spectra(self, bank, k_lz, z_l):
        cache = {}
        interp = type(self).pkl_interp

        def limber_pk(field1, field2):
            key = tuple(sorted((field1, field2)))
            if key not in cache:
                cache[key] = interp(k_lz, z_l, bank.ks, bank.zs, bank.base(field1, field2).T)
            return cache[key]
        return limber_pk


@layer("P3", "two_point", rank=2)
class UniformLnkTwoPoint(_PklTwoPoint):
    """P3. AngularTwoPoint whose Limber P(k) is :func:`uniform_lnk_pkl_interp`.

    ``Pkl_interp`` (interpax 2D Akima) searches every query in the table (searchsorted): 21
    of cloelib's 27 ms per value on the prototype grid, and most of the gradient. Against it:
    30.2 -> 2.9 ms per value, 118 -> 15.1 ms per gradient; bands move by <= 6e-5 sigma.
    """

    pkl_interp = staticmethod(uniform_lnk_pkl_interp)


# --------------------------------------------------------------------------- #
# P5: lensing efficiencies in O(N)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def efficiency_correction(n, tail=6):
    """``cached_stacked_simpson(n)`` minus the upper triangle of ones, in banded form.

    Row j (m = n - j nodes) differs from one at its first three nodes j..j+2 and the grid's
    last three, (A[j], B[j]); the last ``tail`` rows, where the two overlap, are a dense block D
    on the last ``tail`` nodes. Applied with shifted slices only, so neither direction of
    the gradient needs a scatter."""
    with jax.ensure_compile_time_eval():  # a constant, also when first asked for under jit
        c = np.asarray(stacked_simpson(n)) - np.triu(np.ones((n, n)))
    tail = min(tail, n)
    a, b = np.zeros((n, 3)), np.zeros((n, 3))
    for j in range(n - tail):
        a[j] = c[j, j:j + 3]
        b[j] = c[j, n - 3:]
        rest = c[j].copy()
        rest[j:j + 3] = 0.0
        rest[n - 3:] = 0.0
        if np.any(rest):
            raise ValueError("stacked Simpson weights are not banded as assumed")
    return a, b, c[n - tail:, n - tail:]


def linear_lensing_efficiency(dndz, chi, dz, correction=None):
    """cloelib's efficiency ``einsum("ik,jk,jk->ij", dndz, 1 - chi_j / chi_k, W) dz`` with
    W = ``cached_stacked_simpson``, as A_j - chi_j B_j: A, B are suffix sums of dndz and
    dndz / chi plus W's banded end correction. The same quadrature, to rounding."""
    n = dndz.shape[-1]
    a, b, d = efficiency_correction(n) if correction is None else correction
    tail = d.shape[0]

    def weighted_tail(f):
        pad = jnp.pad(f, ((0, 0), (0, 2)))
        band = a[:, 0] * f + a[:, 1] * pad[:, 1:n + 1] + a[:, 2] * pad[:, 2:n + 2] + f[:, n - 3:] @ b.T
        end = jnp.pad(f[:, n - tail:] @ d.T, ((0, 0), (n - tail, 0)))
        return jnp.cumsum(f[:, ::-1], axis=1)[:, ::-1] + band + end
    return (weighted_tail(dndz) - chi * weighted_tail(dndz / chi)) * dz


class _LinearEfficiency:
    """P5. cloelib forms each efficiency through the O(N^2) distance-ratio matrix (1199^2 per
    bin, and its transpose in the gradient); :func:`linear_lensing_efficiency` is the same
    quadrature in O(N): 3.7 -> 2.9 ms per value, 27.6 -> 15.1 ms per gradient; bands move by
    2e-14 sigma."""

    _correction = None

    def _efficiency(self, z):
        if self._correction is None or self._correction[0].shape[0] != len(z):
            self._correction = efficiency_correction(len(z))
        return linear_lensing_efficiency(self.dndz_shifted, self.background.comoving_distance(z),
                                         z[1] - z[0], self._correction)


@layer("P5", "shear")
class LinearEfficiencyShear(_LinearEfficiency, ShearTracer):
    """P5: ShearTracer with :func:`linear_lensing_efficiency`."""

    def get_lensing_efficiency(self, z):
        return self._efficiency(z)


@layer("P5", "positions")
class LinearEfficiencyPositions(_LinearEfficiency, PositionsTracer):
    """P5: PositionsTracer with :func:`linear_lensing_efficiency` as its magnification efficiency."""

    def get_magnification_efficiency(self, z):
        return self._efficiency(z)



# --------------------------------------------------------------------------- #
# P6: one Limber P(k) for the pairs of one evaluation
# --------------------------------------------------------------------------- #
@layer("P6", "two_point", rank=0)
class SharedLimberPk(AngularTwoPoint):
    """P6. Each ``get_Cl_tensor`` interpolates P(k = (l+1/2)/chi, z) anew; for the 5 tracer-pair
    families of a 3x2pt + kappa vector that is 5 identical interpolations, which XLA merges in
    the value but not in the gradient (5 scatter-adds). Pairs given one dict by :meth:`share`
    reuse one interpolation per (perturbations, multipoles, k and Limber grids) while the dict lives (one
    evaluation): 28.5 -> 15.1 ms per gradient, the value unchanged, the bands identical."""

    _shared = None

    def share(self, shared):
        self._shared = shared
        return self

    def get_Cl_tensor(self, ells, nl, ks):
        self._ells = ells
        return super().get_Cl_tensor(ells, nl, ks)

    def _limber_base_spectra(self, bank, k_lz, z_l):
        if self._shared is None:
            return super()._limber_base_spectra(bank, k_lz, z_l)
        # One interpolation per table, multipoles, k grid and Limber grid: k_lz is (ells + 1/2) / chi
        # with chi from the perturbations' background on z_l.
        key = ("pk", id(self.tracer1.perturbations), id(self._ells), np.asarray(bank.ks).tobytes(),
               np.asarray(z_l).tobytes())
        if key not in self._shared:
            self._shared[key] = super()._limber_base_spectra(bank, k_lz, z_l)
        return self._shared[key]


# --------------------------------------------------------------------------- #
# P8: the gather from ln P itself
# --------------------------------------------------------------------------- #
@layer("P8", "two_point", rank=1)
class LogTableTwoPoint(UniformLnkTwoPoint):
    """P8. cloelib's Perturbations hand out P and the P3 gather works on ln P, so a backend that
    holds ln P (a table) pays an exp and a log per entry, and both again in the gradient. When
    ``perturbations`` has ``log_matter_power_spectrum(zs, ks)``, the total-matter Limber P is
    gathered from it directly; anything else falls back to P3. 15.7 -> 15.1 ms per gradient;
    the bands identical."""

    def _limber_base_spectra(self, bank, k_lz, z_l):
        fallback = super()._limber_base_spectra(bank, k_lz, z_l)
        log_pk = getattr(bank.perturbations, "log_matter_power_spectrum", None)
        if log_pk is None:
            return fallback
        cache = {}

        def limber_pk(field1, field2):
            if {field1, field2} != {MATTER}:
                return fallback(field1, field2)
            if "pk" not in cache:
                cache["pk"] = uniform_lnk_interp_ln(k_lz, z_l, bank.ks, bank.zs, log_pk(bank.zs, bank.ks))
            return cache["pk"]
        return limber_pk


# --------------------------------------------------------------------------- #
# P4: no Akima extrapolation beyond the k table
# --------------------------------------------------------------------------- #
def pkl_interp_powerlaw(k_l, z_l, ks, zs, Pk):
    """``Pkl_interp`` (interpax 2D Akima on log10 P in (log10 k, z)) with the query held to the
    table in log10 k and log10 P continued linearly beyond it with the edge slope. Inside the
    table it is ``Pkl_interp`` unchanged."""
    log_ks = jnp.log10(ks)
    log_pk = jnp.log10(Pk)
    log_k = jnp.log10(k_l)
    inside = 10 ** interpax.interp2d(jnp.clip(log_k, log_ks[0], log_ks[-1]), z_l, log_ks, zs, log_pk,
                                     method="akima", extrap=True)
    slope_hi = (log_pk[-1] - log_pk[-2]) / (log_ks[-1] - log_ks[-2])
    slope_lo = (log_pk[1] - log_pk[0]) / (log_ks[1] - log_ks[0])
    zq = jnp.clip(z_l, zs[0], zs[-1])
    beyond = (jnp.interp(zq, zs, slope_hi) * jnp.maximum(log_k - log_ks[-1], 0.0)
              + jnp.interp(zq, zs, slope_lo) * jnp.minimum(log_k - log_ks[0], 0.0))
    return inside * 10 ** beyond


@layer("P4", "two_point", rank=3)
class PowerLawTwoPoint(_PklTwoPoint):
    """P4. AngularTwoPoint whose Limber P(k) is :func:`pkl_interp_powerlaw`.

    ``Pkl_interp``'s ``extrap=True`` continues Akima's edge cubic. With the CosmoPower rows on
    their own k modes (ending at 49.2 Mpc^-1), where the Limber sum reaches k = 136 Mpc^-1, a
    band of tr1_v2 moves by 1877 sigma at a high-sigma_8 point; with P4, by <= 0.016 sigma
    against the table extended in k. It costs what ``Pkl_interp`` costs."""

    pkl_interp = staticmethod(jax.vmap(pkl_interp_powerlaw, in_axes=(0, None, None, None, None)))


# --------------------------------------------------------------------------- #
# P1: a vectorised comoving distance
# --------------------------------------------------------------------------- #
def comoving_distance_gl(hubble, zs, n=64):
    """chi(z) = c int_0^z dz' / H(z'), by ``n``-node Gauss-Legendre in ln(1 + z'), for every z
    at once; ``hubble(z)`` in km/s/Mpc, ``zs`` a scalar or any array."""
    x, w = np.polynomial.legendre.leggauss(n)
    zs = jnp.asarray(zs, dtype=float)
    top = jnp.log1p(zs)[..., None]
    u = 0.5 * top * (x + 1.0)
    z = jnp.expm1(u)
    return C_KMS * 0.5 * top[..., 0] * jnp.sum(w * (1.0 + z) / hubble(z), axis=-1)


class GLComovingDistance(JAXBackground):
    """P1. ``JAXBackground.comoving_distance`` builds one ``quadgk`` per redshift in a Python
    list, so its jaxpr grows with the grid: on 1199 nodes it takes 770 s to compile and 57 ms
    per call, and a scalar z (z*) fails. Fixed 64-node Gauss-Legendre in ln(1 + z) is one
    vectorised expression and takes scalars; it agrees with quadgk to < 1e-9 relative on
    0 < z <= 1090. On the 1199 nodes: 0.10 ms per value, 0.38 ms per gradient, 0.1 s to
    compile. Used for cloelib's background in place of JaxVector's, it moves the tr1_v2 bands
    by <= 0.009 sigma (that background has no radiation)."""

    def comoving_distance(self, zs):
        return comoving_distance_gl(self.hubble_parameter, zs)
