"""CLOE's theory of a Euclid data vector as one pure-JAX, batched, differentiable function.

:class:`JaxVector` is CLOE made JAX-compatible: the theory cloelib computes for a data
vector (one massive neutrino species, NLA, constant per-bin linear galaxy bias, optional
lens magnification, with the P4 power-law P(k) extrapolation), with the two slow pieces of
that computation ported to JAX: the CosmoPower-JAX emulator call (and the P(k) table cloelib
builds from it), and the Limber projection, which is cloelib's own ``AngularTwoPoint``
traced under ``jit``, ``vmap`` and ``grad`` (:mod:`.cloe_jax`). It takes from an
:class:`~euclid_spt_like.vector.EuclidVector`, never re-deriving them, the bandpower
operators, the multipole nodes, the delivered n(z), the NLA pivot, the scale cuts, the data
vector and the inverse covariance; from its :class:`~euclid_spt_like.cloe.Grid` the Limber
redshift grid, the P(k) redshift and wavenumber tables and the NLA normalisation C1 rho_crit;
from cloelib's CAMB background the radiation and neutrino densities.

This is the fast path of the production cmbx chains (dr1_cmbx's ``eDR1like.jax_vector``),
carried with its numerics unchanged; ``docs/gate.md`` shows it reproduces that path.

Two engines (:data:`~euclid_spt_like.cloe.ENGINES`) are ported:

- ``cosmopower_mead2020_feedback`` (the data baseline): P(k) is the emulator's own
  definition;
- ``camb_mead2020`` (dark-matter-only HMcode-2020, which no emulator covers, as the GLASS
  mocks): ln P is the emulator at a fixed log T_AGN, less HMcode-2020's feedback response
  ln S(k, z; theta) (:mod:`.jax_hmcode`), plus the fixed correction ln[P_CAMB / P_model](k, z)
  tabulated once from cloelib's CAMB engine at ``correction_reference``. Away from the
  reference this approximates CAMB.

Each piece, as pure CLOE (cloelib + ``AngularTwoPoint`` with P4) defines it:

- **Background** (float64): flat LCDM; the radiation (photons, massless neutrinos) and
  massive-neutrino physical densities omega(a) depend only on T_CMB, N_eff and m_nu, so they
  are tabulated once from cloelib's ``CAMBBackground`` (``get_background_densities``);
  H(z) is then closed form, chi(z) a quadrature, chi* = chi(1089).
  Omega_m = (omega_c + omega_b + omega_nu) / h^2, CAMB's ``Omega_m(0)``. The speed of light is
  cloelib's.
- **P(k, z)**: the CosmoPower-JAX HMcode-2020 + feedback network
  (``lcdm-1mass-nonlinear``, one massive species) on the grid's P(k) redshift nodes
  ``z_pk`` up to z = 5. Above its top mode (49.2 Mpc^-1) P continues as cloelib's ``logk2``
  ansatz (``extend_spectra``: k^-(4-n_s) A (ln k - ln k*)^2 through the last two modes, on
  extend_spectra's wavenumber nodes), and the table is interpolated linearly in k onto the
  grid's ``ks`` (to 50 Mpc^-1), as cloelib's ``RectBivariateSpline(kx=1, ky=1)`` does.
  Beyond ``ks`` ln P continues linearly in ln k with the table's edge slope: P4.
- **Above z = 5**: P(k, z) = P(k, 5) [D(z)/D(5)]^2 with D the cold-matter (Omega_cb source)
  linear growth on this background from D = a at z = 200, here by RK4 in ln a. The P(k)
  table gains a node at z = 5 when the grid reaches above it.
- **NLA growth**: cloelib's ``growth_factor(z, k)[:, 1]``, sqrt[P(z, k_1) / P(0, k_1)] at the
  second node of the P(k) state's k table (the emulator's k_1 = 1.1e-5 Mpc^-1), linear in
  P between the table's redshift nodes; above z = 5, D(5) D(z)/D(5).
- **n(z)**: each delivered n(z) shifted and stretched about its mean so that its realized
  mean is the delivered mean plus delta_z (:meth:`JaxVector.shift_nz`), then regridded onto
  the grid's ``z`` (scipy's Akima, zero outside the delivered grid); cloelib's tracers apply
  their own zero shift, unit stretch and trapezoid renormalisation, as in CLOE.
- **Projection**: cloelib's tracers and ``AngularTwoPoint.get_Cl_tensor`` with the patches of
  :mod:`.cloelib_patches` (:mod:`.cloe_jax`); bands = operator (15 x n_ell) @ C_ell per
  spectrum; chi^2 = d^T icov d.
- **Derived**: sigma8 from the CosmoPower-JAX ``lcdm-1mass-s8-fs8`` network at z = 0, with
  the inputs of cloelib's ``_CosmoPowerPerturbations`` (the point's log T_AGN), S8 =
  sigma8 sqrt(Omega_m / 0.3), Omega_m. Under mead2020 CLOE's sigma8 is CAMB's; the network
  stands in for it at log T_AGN 7.6 (cloelib's linear-sector value).

Where the numerics differ from pure CLOE (measured in production on the TR1 data profile, in
Gaussian sigma of the retained bands: at most 2.4e-4 sigma per band, sigma8 within 6e-7
relative):

- the gather of P at the Limber wavenumbers: ln P cubic-spline in z from the table's nodes
  and four-point Lagrange in ln k on ``ks`` (cloelib_patches P3/P8), where CLOE takes
  interpax's 2-D Akima of log10 P; above ``ks`` the edge slope is splined in z, where P4
  interpolates it linearly in z;
- the background: chi by Gauss-Legendre to the first Limber node then the quadratic-exact
  interval rule, chi* by Simpson in ln a, where CLOE calls CAMB's integrator; z* = 1089
  where CLOE takes CAMB's z_star (1089.90 at the Planck fiducial; chi* moves by ~1e-5
  relative and the CMB-lensing kernel by less);
- the emulator is clamped to its training box (CLOE raises outside it; see :meth:`in_box`).

Importing this module enables JAX's 64-bit mode (``jax_enable_x64``) process-wide; the
networks run in float32, as they were trained. Points whose n(z) shift cannot reach the
target mean, or whose stretch is not positive, get log L = -1e300.
"""

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

#: The CMB source redshift of the CMB-lensing kernel (the TheoryVector's ``cmb_lensing_z_source``).
Z_STAR = 1089.0
#: The emulator's trained redshift range, networks and parameter box (cloelib CP_EMULATOR_BOUNDS).
EMULATOR_Z_MAX = 5.0
EMULATOR_FILE = "lcdm-1mass-nonlinear.npz"
SIGMA_FILE = "lcdm-1mass-s8-fs8.npz"
EMULATOR_INPUTS = ["ombh2", "omch2", "H0", "ns", "lnAs", "z", "mnu", "logT_AGN"]
EMULATOR_BOUNDS = {"ombh2": (0.001, 0.1), "omch2": (0.05, 0.9), "H0": (20.0, 100.0),
                   "ns": (0.6, 1.3), "lnAs": (1.61, 5.0), "mnu": (0.0, 1.0),
                   "logT_AGN": (7.3, 8.5)}
#: cloelib ``extend_spectra``'s default wavenumber nodes (Mpc^-1): geomspace(kmin, kmax, n).
EXTEND_K = (1e-4, 500.0, 500)
#: The log T_AGN at which cloelib's linear sector evaluates the sigma8 network.
LINEAR_LOG_T_AGN = 7.6
#: CloelibBackend._growth starts the growing mode D = a at this redshift.
GROWTH_Z_START = 200.0
#: The linear total-matter and cold networks of the mead2020 feedback response.
LINEAR_EMULATOR_FILES = ("lcdm-1mass-linear.npz", "lcdm-1mass-cb-linear.npz")
#: The response's lowest k node and the top of its linear-spectrum grid (Mpc^-1).
RESPONSE_K_MIN = 0.01
RESPONSE_K_LINEAR_MAX = 1e4
#: Default cosmology of the mead2020 correction table: the production chains' provider cosmology.
CORRECTION_REFERENCE = {"Omega_c": 0.26066675992680977, "Omega_b": 0.04897468161869667,
                        "h": 0.6766, "n_s": 0.9665, "A_s": 2.1049988209300684e-09}


# --------------------------------------------------------------------------- #
# Fixed-node numerics
# --------------------------------------------------------------------------- #
def simpson_weights(n, h):
    """Composite Simpson weights for ``n`` (odd) uniformly spaced nodes."""
    if n % 2 == 0 or n < 3:
        raise ValueError("Simpson needs an odd number (>= 3) of nodes")
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w * h / 3.0


def scipy_simpson_weights(x):
    """The weight vector scipy.integrate.simpson applies on the nodes ``x`` (it is linear in y)."""
    from scipy.integrate import simpson
    eye = np.eye(len(x))
    return np.array([simpson(row, x=x) for row in eye])


def interval_integrals(f, h):
    """Integrals of ``f`` over each interval of a uniform grid, exact for quadratics.

    ``f`` (..., n) at nodes spaced ``h``; returns (..., n-1). Interval j uses the
    quadratic through nodes j-1, j, j+1 (j+2 for the first interval).
    """
    first = h / 12.0 * (5.0 * f[..., :1] + 8.0 * f[..., 1:2] - f[..., 2:3])
    rest = h / 12.0 * (-f[..., :-2] + 8.0 * f[..., 1:-1] + 5.0 * f[..., 2:])
    return jnp.concatenate([first, rest], axis=-1)


def akima_slopes(x, y):
    """The node slopes of scipy's ``Akima1DInterpolator`` (``method="akima"``) through
    ``(x, y)``, ``y`` (..., n): its end extension, and its fill value where both weights
    vanish (below 1e-9 of their largest along the last axis)."""
    m = jnp.diff(y, axis=-1) / jnp.diff(x)
    left1 = 2.0 * m[..., :1] - m[..., 1:2]
    left0 = 2.0 * left1 - m[..., :1]
    right1 = 2.0 * m[..., -1:] - m[..., -2:-1]
    right2 = 2.0 * right1 - m[..., -1:]
    m = jnp.concatenate([left0, left1, m, right1, right2], axis=-1)
    dm = jnp.abs(jnp.diff(m, axis=-1))
    f1, f2 = dm[..., 2:], dm[..., :-2]
    f12 = f1 + f2
    defined = f12 > 1e-9 * jnp.max(f12, axis=-1, keepdims=True)
    safe = jnp.where(defined, f12, 1.0)
    return jnp.where(defined, (f1 * m[..., 1:-2] + f2 * m[..., 2:-1]) / safe,
                     0.5 * (m[..., 3:] + m[..., :-3]))


def uniform_interp(x, xp, fp):
    """``jnp.interp(x, xp, fp, left=0, right=0)`` on a uniform grid ``xp`` (numpy), with the
    interval found arithmetically instead of by binary search; the same formula otherwise."""
    step = (xp[-1] - xp[0]) / (len(xp) - 1)
    i = jnp.clip(jnp.floor((x - xp[0]) / step).astype(int) + 1, 1, len(xp) - 1)
    xp = jnp.asarray(xp)
    lo, hi = xp[i - 1], xp[i]
    f = fp[i - 1] + ((x - lo) / (hi - lo)) * (fp[i] - fp[i - 1])
    return jnp.where((x < xp[0]) | (x > xp[-1]), 0.0, f)


def spline_matrix(x_nodes, x_eval, derivative=0):
    """Matrix of scipy's not-a-knot cubic spline through ``x_nodes``, evaluated at ``x_eval``."""
    from scipy.interpolate import CubicSpline
    spline = CubicSpline(np.asarray(x_nodes, float), np.eye(len(x_nodes)), axis=0)
    if derivative:
        spline = spline.derivative(derivative)
    return np.asarray(spline(np.asarray(x_eval, float)))


def linear_weights(nodes, x):
    """``(i, t)`` with ``x = (1 - t) nodes[i] + t nodes[i+1]``: linear interpolation on ``nodes``."""
    nodes, x = np.asarray(nodes, float), np.asarray(x, float)
    i = np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, len(nodes) - 2)
    return i, (x - nodes[i]) / (nodes[i + 1] - nodes[i])


def logk2(lnk, lnp, n_s):
    """cloelib's ``logk2`` continuation beyond the last two nodes ``(lnk, lnp)`` of ln P.

    ``lnk`` (2,), ``lnp`` (..., 2); returns the function of ``lnk_out`` giving ln P there. With
    K_i = k_i^(4-n_s) P_i and D = dK/dln k between the two nodes, P = k^-(4-n_s) A
    (ln k - ln k*)^2, A = D^2 / (4 K_1), ln k* = ln k_1 - 2 K_1 / D (continuous at k_1).
    """
    a = 4.0 - n_s
    k0, k1 = (jnp.exp(a * lnk[j] + lnp[..., j]) for j in (0, 1))
    slope = (k1 - k0) / (lnk[1] - lnk[0])
    lnk_star = lnk[1] - 2.0 * k1 / slope
    ln_amplitude = jnp.log(0.25 * slope ** 2 / k1)

    def at(lnk_out):
        return (-a * lnk_out + ln_amplitude[..., None]
                + 2.0 * jnp.log(jnp.abs(lnk_out - lnk_star[..., None])))
    return at


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class JaxVector:
    """Theory bands, log-likelihood and derived parameters of a Euclid data vector under CLOE,
    as pure JAX functions.

    Parameters are a flat array in :attr:`names` order (a batch is a leading axis);
    :meth:`params` turns a dict into one.

    Args:
        vector: the :class:`~euclid_spt_like.vector.EuclidVector`: the retained spectra, their
            bandpower operators on the multipole nodes, the delivered n(z), the data and
            inverse covariance, and the :class:`~euclid_spt_like.cloe.Grid` CLOE runs on.
        names: parameter order of the array the functions take. Each name is a nuisance or bias
            of the vector's tracers, ``A_cmb_lensing_cross`` (an amplitude on every pair with
            one CMB-lensing leg), or a sampling-basis cosmology name (``logA``, ``omch2``,
            ``ombh2``, ``h``, ``n_s``, ``HMCode_logT_AGN``). m_nu cannot be sampled.
        fixed: values of parameters not in ``names``; must hold ``m_nu``, every sampling-basis
            cosmology name except ``HMCode_logT_AGN`` (default 7.8, CAMB's), and every galaxy
            bias (and magnification slope, with magnification). Missing n(z) shifts are zero,
            stretches one, IA and m zero and ``A_cmb_lensing_cross`` one.
        newton_steps: bracketed Newton iterations of the n(z) realized-mean solve.
        n_star: Simpson nodes in ln a from the last Limber node to z* (chi*).
        n_growth: RK4 steps of the growth ODE of the z > 5 continuation (used only when a
            table or Limber node lies above z = 5).
        shift_bracket: half-width of the bracket about delta_z the effective shift is sought
            in (default: the n(z) grid's span plus one); a root outside it, or one the
            iterations do not reach, leaves a nonzero residual and an invalid point.
        correction_reference: under ``camb_mead2020``, the cosmology (``Omega_c, Omega_b, h,
            n_s, A_s``) at which ln[P_CAMB / P_model] is tabulated from cloelib's CAMB engine.
            Unused under the feedback emulator, which is the definition there.
        n_response_k, n_response_mass: the ``mead2020`` feedback response's k nodes (uniform in
            ln k on [0.01 Mpc^-1, the emulator's k_max]) and halo masses; it is evaluated on the
            emulator's redshift nodes.
        patches: the :mod:`.cloelib_patches` the projection composes (default
            :data:`.cloe_jax.PATCHES`).
    """

    COSMOLOGY = ("logA", "omch2", "ombh2", "h", "n_s", "HMCode_logT_AGN")
    #: The derived parameters of :meth:`derived` and :meth:`batched_derived`, in order.
    DERIVED = ("sigma8", "S8", "Omega_m")

    def __init__(self, vector, names, *, fixed, newton_steps=16, n_star=2049, n_growth=400,
                 correction_reference=CORRECTION_REFERENCE, n_response_k=32, n_response_mass=48,
                 shift_bracket=None, patches=None):
        from cloelib.auxiliary.units import SPEED_OF_LIGHT
        from .cloe import camb_background
        from .cloe_jax import PATCHES, CloelibProjection

        self.names = list(names)
        self.fixed = dict(fixed)
        grid = vector.grid
        self.feedback = grid.nonlinear_model == "mead2020_feedback"
        self.include_mag_bias = bool(vector.include_mag_bias)
        self.gc_bins, self.wl_bins = list(vector.gc_bins), list(vector.wl_bins)
        self._check_parameters()
        self.keys = list(vector.keys)
        self.ells = np.asarray(vector.ells, dtype=float)
        self.ia_z_pivot = float(vector.ia_z_pivot)
        self.c1_rho_crit = float(grid.c1_rho_crit)
        self.clight_hmpc = SPEED_OF_LIGHT / 1000.0 / 100.0
        self.newton_steps = int(newton_steps)
        self.shift_bracket = shift_bracket
        self.m_nu = float(self.fixed["m_nu"])

        # ---- tracers and spectra ----------------------------------------- #
        pairs = [tuple(p) for p in vector.pairs]
        self.tracers = ["K"] * any("K" in p for p in pairs) \
            + [f"G{i}" for i in self.gc_bins] + [f"L{i}" for i in self.wl_bins]
        self._operators = np.asarray(vector.operators, float)
        if self._operators.shape[2] != len(self.ells):
            raise ValueError("operators do not act on the multipole nodes")
        self.nbands = self._operators.shape[1]
        self._one_kappa = np.array([a.count("K") + b.count("K") == 1 for a, b in pairs])

        # ---- likelihood: retained bands, data, icov ---------------------- #
        self.selection = np.asarray(vector.indices)
        self.data = np.asarray(vector.data, float)
        self.icov = np.asarray(vector.icov, float)

        # ---- n(z): the delivered grid, and the regrid onto the Limber grid - #
        grids = [np.asarray(vector.nz[t][0], float) for t in self._nz_names]
        z_n = grids[0]
        if not all(np.array_equal(g, z_n) for g in grids):
            raise ValueError("every n(z) must share one redshift grid")
        step = np.diff(z_n)
        if not np.allclose(step, step[0], rtol=0, atol=1e-9 * step[0]):
            raise ValueError("the n(z) grid must be uniform")
        self.z_n = z_n
        self.nz = np.stack([np.asarray(vector.nz[t][1], float) for t in self._nz_names])
        self._simpson_n = scipy_simpson_weights(z_n)
        self.nz_mean = (self.nz @ (self._simpson_n * z_n)) / (self.nz @ self._simpson_n)
        self._source_z = np.r_[z_n[0] - (z_n[1] - z_n[0]), z_n, z_n[-1] + (z_n[-1] - z_n[-2])]
        z = self.z = np.asarray(grid.z, float)
        self.dz = float(z[1] - z[0])
        self._regrid_inside = (z >= z_n[0]) & (z <= z_n[-1])
        self._regrid_i = np.clip(np.searchsorted(z_n, z, side="right") - 1, 0, len(z_n) - 2)
        self._regrid_s = z - z_n[self._regrid_i]

        # ---- redshift nodes: chi quadratures, P(k) table, z > 5 growth ---- #
        gl_x, gl_w = np.polynomial.legendre.leggauss(8)
        self._a_sub = 1.0 / (1.0 + 0.5 * z[0] * (gl_x + 1.0))
        self._w_sub = 0.5 * z[0] * gl_w
        lna_star = np.linspace(-np.log1p(Z_STAR), -np.log1p(z[-1]), n_star)
        self._w_star = simpson_weights(n_star, lna_star[1] - lna_star[0])
        self._a_star = np.exp(lna_star)
        z_tab = np.asarray(grid.z_pk, float)
        if z_tab[-1] > EMULATOR_Z_MAX and not np.any(z_tab == EMULATOR_Z_MAX):
            z_tab = np.sort(np.r_[z_tab, EMULATOR_Z_MAX])
        self.z_tab = z_tab
        self.z_emu = z_tab[z_tab <= EMULATOR_Z_MAX]
        self._n_emu = len(self.z_emu)
        self._n_tail = len(z_tab) - self._n_emu
        self._ia_tail = np.flatnonzero(z > EMULATOR_Z_MAX)
        # D(z)/D(5) on the table rows above 5, then on the Limber nodes above 5.
        z_tail = np.r_[z_tab[self._n_emu:], z[self._ia_tail]]
        self._a_growth = np.ones(1)
        if len(z_tail):
            lna_growth = np.linspace(-np.log1p(GROWTH_Z_START), 0.0, 2 * n_growth + 1)
            self._dlna_growth = 2 * (lna_growth[1] - lna_growth[0])
            lna_nodes = -np.log1p(np.r_[EMULATOR_Z_MAX, z_tail])
            coarse = lna_growth[::2]
            idx = np.clip(np.searchsorted(coarse, lna_nodes) - 1, 0, len(coarse) - 2)
            self._growth_idx, self._growth_t = idx, (lna_nodes - coarse[idx]) / self._dlna_growth
            self._a_growth = np.exp(lna_growth)
        self._tail = len(z_tail) > 0

        # ---- background: CAMB's radiation and massive-neutrino tables ---- #
        background = camb_background(dict(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.96,
                                          A_s=2e-9, m_nu=self.m_nu))
        (r0, nu0), (self._r_z, self._nu_z), (self._r_sub, self._nu_sub), \
            (self._r_star, self._nu_star), (self._r_growth, self._nu_growth) = self._density_tables(
                background, [np.ones(1), 1.0 / (1.0 + z), self._a_sub, self._a_star, self._a_growth])
        self._r_today, self._nu_today = float(r0[0]), float(nu0[0])

        # ---- P(k): the networks, the logk2 nodes and the k map onto ks ---- #
        self._emulator = self._network(EMULATOR_FILE, EMULATOR_INPUTS)
        self._sigma = self._network(SIGMA_FILE, EMULATOR_INPUTS)
        self.k_emu = self._emulator["modes"]
        self.ks = np.asarray(grid.ks, float)
        base = np.geomspace(*EXTEND_K)
        if np.any(base < self.k_emu[0]):
            raise ValueError("the emulator starts above extend_spectra's k_min: its low-k "
                             "power law is not ported")
        self.correction_reference = None
        self._response = None
        self.emulator_log_t_agn = 7.8
        if self.feedback:
            self.ia_k = float(self.k_emu[1])
        else:
            if "HMCode_logT_AGN" in self.names or "HMCode_logT_AGN" in self.fixed:
                raise ValueError("HMCode_logT_AGN is not a parameter of mead2020")
            self.correction_reference = dict(correction_reference)
            reference_state = grid.camb_power(self._backend_params(self.correction_reference))
            self.ia_k = float(np.asarray(reference_state.k)[1])
        query = np.r_[self.ks, self.ia_k]
        plus = base[base > self.k_emu[-1]]
        self._k_plus = plus[:int(np.searchsorted(plus, query.max())) + 1]
        extended = np.r_[self.k_emu, self._k_plus]
        if query.min() < extended[0] or query.max() > extended[-1]:
            raise ValueError("the P(k) table does not cover the grid's ks")
        self._k_i, self._k_t = linear_weights(extended, query)
        self._ln_correction = np.zeros((len(z_tab), len(query)))
        if not self.feedback:
            self._build_response(n_response_k, n_response_mass, query)
            model = np.asarray(self.lnp_table(self._reference_cosmology()))
            camb = np.column_stack([reference_state.matter_power_spectrum(z_tab, self.ks),
                                    reference_state.matter_power_spectrum(z_tab, [self.ia_k])])
            self._ln_correction = np.log(camb) - model
        # NLA growth: P(z, k_1) linear in z between the table's nodes, onto the Limber grid.
        i, t = linear_weights(z_tab, z)
        self._ia_z = np.zeros((len(z), len(z_tab)))
        self._ia_z[np.arange(len(z)), i] = 1.0 - t
        self._ia_z[np.arange(len(z)), i + 1] += t

        # ---- parameter routing ------------------------------------------- #
        self._index = {n: i for i, n in enumerate(self.names)}
        self._projection = CloelibProjection(self, pairs, PATCHES if patches is None else patches)
        self._jit_bands = jax.jit(self._bands_with_status)
        self._jit_chi2 = jax.jit(self._chi2_with_status)
        self._jit_derived = jax.jit(self.derived)

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    def _check_parameters(self):
        """Every name and fixed value is one the model evaluates; the cosmology, the galaxy
        biases and (with magnification) the slopes are complete."""
        consumed = set(self.COSMOLOGY) | {"m_nu", "A_cmb_lensing_cross"}
        required = [f"bias_{i}" for i in self.gc_bins]
        if self.include_mag_bias:
            required += [f"mag_slope_{i}" for i in self.gc_bins]
        consumed |= set(required)
        consumed |= {f"{p}_{f}_{i}" for p in ("delta_z", "stretch")
                     for f, bins in (("G", self.gc_bins), ("L", self.wl_bins)) for i in bins}
        if self.wl_bins:
            consumed |= {"A_IA", "eta_IA"} | {f"m_{i}" for i in self.wl_bins}
        if len(set(self.names)) != len(self.names):
            raise ValueError("repeated parameter names")
        unknown = sorted((set(self.names) | set(self.fixed)) - consumed)
        if unknown:
            raise ValueError(f"JaxVector does not evaluate {unknown}")
        if "m_nu" in self.names:
            raise ValueError("m_nu cannot be sampled: the neutrino tables and the emulator's "
                             "m_nu are fixed at build")
        both = sorted(set(self.names) & set(self.fixed))
        if both:
            raise ValueError(f"{both} are both sampled and fixed")
        missing = [n for n in ["m_nu", *self.COSMOLOGY[:-1], *required]
                   if n not in self.names and n not in self.fixed]
        if missing:
            raise ValueError(f"{missing} neither sampled nor fixed")

    def _backend_params(self, reference):
        """The physical cosmology dict of a ``correction_reference``."""
        out = {k: float(reference[k]) for k in ("Omega_c", "Omega_b", "h", "n_s", "A_s")}
        return {**out, "m_nu": self.m_nu}

    def _reference_cosmology(self):
        """:attr:`correction_reference` in the sampling basis, at the emulator's log T."""
        ref = self.correction_reference
        h = float(ref["h"])
        return dict(logA=np.log(1e10 * ref["A_s"]), h=h, omch2=ref["Omega_c"] * h ** 2,
                    ombh2=ref["Omega_b"] * h ** 2, n_s=float(ref["n_s"]), logT=self.emulator_log_t_agn)

    def _density_tables(self, background, a_sets):
        """(radiation, massive-neutrino) physical densities omega(a) on each node set, from
        cloelib's ``CAMBBackground`` (``get_background_densities``: 8 pi G a^4 rho in Mpc^-2).

        omega_x(a) = 8 pi G rho_x / 3 / (100 km/s/Mpc / c)^2 depends only on T_CMB, N_eff and
        the neutrino mass, never on the sampled parameters, so one background fixes it.
        """
        results = background.results
        out = []
        for a in a_sets:
            a = np.asarray(a, float)
            d = results.get_background_densities(a, vars=["photon", "neutrino", "nu"])
            scale = 3.0 * a ** 4 / self.clight_hmpc ** 2
            out.append((np.asarray(d["photon"] + d["neutrino"]) / scale, np.asarray(d["nu"]) / scale))
        return out

    @staticmethod
    def _network(filename, expected):
        """A CosmoPower-JAX network (P(k) or sigma8) as float32 arrays, its inputs checked."""
        import warnings
        from cloelib.cosmology.cosmopower_jax_cosmology import emulator_data, load_pk_emulator
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            emu = load_pk_emulator(emulator_data(filename))
        if list(emu.parameters) != expected:
            raise ValueError(f"{filename} inputs {emu.parameters} != {expected}")
        f32 = lambda v: jnp.asarray(np.asarray(v, np.float32))
        return dict(weights=[(f32(w), f32(b)) for w, b in emu.weights],
                    hyper=[(f32(a), f32(b)) for a, b in emu.hyper_params],
                    inputs=(f32(emu.param_train_mean), f32(emu.param_train_std)),
                    outputs=(f32(emu.feature_train_mean), f32(emu.feature_train_std)),
                    modes=np.asarray(emu.modes, float))

    def _build_response(self, n_k, n_mass, query):
        """The mead2020 feedback response: linear networks, the response nodes and the map from
        them to the P(k) query wavenumbers (zero below 0.01 Mpc^-1, held above the top node)."""
        from .jax_hmcode import FeedbackResponse
        expected = [n for n in EMULATOR_INPUTS if n != "logT_AGN"]
        self._linear = [self._network(f, expected) for f in LINEAR_EMULATOR_FILES]
        for net in self._linear:
            if not np.array_equal(net["modes"], self.k_emu):
                raise ValueError("the linear and nonlinear emulators' k modes differ")
        lnk = np.log(self.k_emu)
        # the linear spectra on a uniform ln k grid, continued above the emulator as a power law
        lnk_linear = np.arange(np.log(1e-5), np.log(RESPONSE_K_LINEAR_MAX), 0.01)
        inside = lnk_linear <= lnk[-1]
        linear_map = np.zeros((len(lnk_linear), len(lnk)))
        linear_map[inside] = spline_matrix(lnk, lnk_linear[inside])
        t = (lnk_linear[~inside] - lnk[-1]) / (lnk[-1] - lnk[-2])
        linear_map[~inside, -1] = 1.0 + t
        linear_map[~inside, -2] = -t
        self._linear_map = linear_map
        lnk_r = np.linspace(np.log(RESPONSE_K_MIN), lnk[-1], n_k)
        self._response = FeedbackResponse(self.z_emu, np.exp(lnk_r), lnk_linear, n_mass=n_mass)
        lnq = np.log(query)
        above = lnq >= lnk_r[0]
        self._response_k = np.zeros((len(query), n_k))
        self._response_k[above] = spline_matrix(lnk_r, np.minimum(lnq[above], lnk_r[-1]))

    def response_rows(self, cosmo):
        """ln S of HMcode-2020 at the emulator's log T on (:attr:`z_emu`, ``ks`` then the NLA k_1)."""
        h = cosmo["h"]
        row = self._emulator_row(cosmo, 0.0)[:7]
        tot, cb = (np.log(10.0) * self._linear_map @ self.network(net, row[None, :])[0]
                   for net in self._linear)
        ln_s = self._response.ln_response(tot, cb, cosmo["omch2"] / h ** 2, cosmo["ombh2"] / h ** 2,
                                          self._nu_today / h ** 2, h, cosmo["logT"])
        return ln_s @ self._response_k.T

    # ------------------------------------------------------------------ #
    # parameter access
    # ------------------------------------------------------------------ #
    def params(self, values):
        """The parameter array (in :attr:`names` order) of a dict."""
        return np.array([values[n] for n in self.names], dtype=float)

    def _get(self, theta, name, default):
        if name in self._index:
            return theta[self._index[name]]
        return jnp.asarray(self.fixed.get(name, default), dtype=float)

    def cosmology(self, theta):
        h = self._get(theta, "h", np.nan)
        return dict(logA=self._get(theta, "logA", np.nan), omch2=self._get(theta, "omch2", np.nan),
                    ombh2=self._get(theta, "ombh2", np.nan), h=h, n_s=self._get(theta, "n_s", np.nan),
                    logT=self._get(theta, "HMCode_logT_AGN", self.emulator_log_t_agn))

    # ------------------------------------------------------------------ #
    # background and growth
    # ------------------------------------------------------------------ #
    def _e(self, a, omega_cb, rad, nu, h):
        omega_l = h ** 2 - omega_cb - self._r_today - self._nu_today
        return jnp.sqrt(omega_cb / a ** 3 + rad + nu + omega_l) / h

    def background(self, cosmo):
        """E(z) and chi(z) [Mpc] on the Limber grid, chi*, Omega_m and h."""
        omega_cb, h = cosmo["omch2"] + cosmo["ombh2"], cosmo["h"]
        e_z = self._e(1.0 / (1.0 + self.z), omega_cb, self._r_z, self._nu_z, h)
        e_sub = self._e(self._a_sub, omega_cb, self._r_sub, self._nu_sub, h)
        hubble_dist = self.clight_hmpc / h
        chi0 = hubble_dist * jnp.sum(self._w_sub / e_sub)
        steps = interval_integrals(1.0 / e_z, self.dz)
        chi = chi0 + hubble_dist * jnp.concatenate([jnp.zeros(1), jnp.cumsum(steps)])
        e_star = self._e(self._a_star, omega_cb, self._r_star, self._nu_star, h)
        chi_star = chi[-1] + hubble_dist * jnp.sum(self._w_star / (self._a_star * e_star))
        omega_m = (omega_cb + self._nu_today) / h ** 2
        return dict(e=e_z, chi=chi, chi_star=chi_star, omega_m=omega_m, h=h)

    def growth_ratio(self, cosmo):
        """D(z)/D(5) on the table rows above z = 5, then on the Limber nodes above z = 5: the
        cold-matter linear growth (D = a at z = 200), RK4 in ln a."""
        omega_cb, h = cosmo["omch2"] + cosmo["ombh2"], cosmo["h"]
        a = self._a_growth
        e = self._e(a, omega_cb, self._r_growth, self._nu_growth, h)
        om_cb = omega_cb / a ** 3 / (h ** 2 * e ** 2)
        # y0 = D, y1 = a^3 E dD/da; d/dln a: D' = y1 / (a^2 E), y1' = 1.5 E a^2 Omega_cb D.
        coef_d = 1.0 / (a ** 2 * e)
        coef_y = 1.5 * e * a ** 2 * om_cb
        step = self._dlna_growth

        def body(state, i):
            d, y = state
            k0, k1, k2 = 2 * i, 2 * i + 1, 2 * i + 2
            dd1, dy1 = coef_d[k0] * y, coef_y[k0] * d
            dd2, dy2 = coef_d[k1] * (y + 0.5 * step * dy1), coef_y[k1] * (d + 0.5 * step * dd1)
            dd3, dy3 = coef_d[k1] * (y + 0.5 * step * dy2), coef_y[k1] * (d + 0.5 * step * dd2)
            dd4, dy4 = coef_d[k2] * (y + step * dy3), coef_y[k2] * (d + step * dd3)
            d = d + step / 6 * (dd1 + 2 * dd2 + 2 * dd3 + dd4)
            y = y + step / 6 * (dy1 + 2 * dy2 + 2 * dy3 + dy4)
            return (d, y), (d, y)

        init = (a[0], a[0] ** 3 * e[0])
        _, (ds, ys) = jax.lax.scan(body, init, jnp.arange((len(a) - 1) // 2))
        ds = jnp.concatenate([jnp.array([init[0]]), ds])
        slope = jnp.concatenate([jnp.array([init[1]]), ys]) * coef_d[::2]  # dD/dln a
        i, t = self._growth_idx, self._growth_t
        hh = self._dlna_growth
        h00, h10 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t
        h01, h11 = -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
        nodes = h00 * ds[i] + h10 * hh * slope[i] + h01 * ds[i + 1] + h11 * hh * slope[i + 1]
        return nodes[1:] / nodes[0]

    # ------------------------------------------------------------------ #
    # P(k, z)
    # ------------------------------------------------------------------ #
    @staticmethod
    def network(net, inputs):
        """The output of a network from :meth:`_network` (log10 P on its modes for a P(k)
        network; sigma8 and f sigma8 for the sigma8 network), computed in float32."""
        x = (inputs.astype(jnp.float32) - net["inputs"][0]) / net["inputs"][1]
        for (w, b), (alpha, beta) in zip(net["weights"][:-1], net["hyper"]):
            x = x @ w.T + b
            x = (beta + jax.nn.sigmoid(alpha * x) * (1.0 - beta)) * x
        w, b = net["weights"][-1]
        out = (x @ w.T + b) * net["outputs"][1] + net["outputs"][0]
        return out.astype(jnp.float64)

    def _emulator_row(self, cosmo, log_t):
        """The networks' input row at z = 0, clamped to the training box (see :meth:`in_box`)."""
        clamp = lambda v, name: jnp.clip(v, *EMULATOR_BOUNDS[name])
        return jnp.stack([clamp(cosmo["ombh2"], "ombh2"), clamp(cosmo["omch2"], "omch2"),
                          clamp(100.0 * cosmo["h"], "H0"), clamp(cosmo["n_s"], "ns"),
                          clamp(cosmo["logA"], "lnAs"), 0.0, self.m_nu, clamp(log_t, "logT_AGN")])

    def emulator_inputs(self, cosmo):
        """The emulator's input rows on :attr:`z_emu`."""
        inputs = jnp.tile(self._emulator_row(cosmo, cosmo["logT"]), (self._n_emu, 1))
        return inputs.at[:, 5].set(self.z_emu)

    def in_box(self, cosmo):
        """True when the cosmology lies inside the emulator's training box.

        Outside it the emulator is evaluated at the nearest point of the box (the inputs are
        clamped): the point stays valid, its P(k) is that of the box face, and the importance
        reweight to the exact model corrects its weight. Refusing it instead (log L = -inf)
        would truncate the explored measure, at the logA prior edge 1.609 below the box's
        1.61, and bias the reweighted posterior there.
        """
        values = {"ombh2": cosmo["ombh2"], "omch2": cosmo["omch2"], "H0": 100 * cosmo["h"],
                  "ns": cosmo["n_s"], "lnAs": cosmo["logA"], "mnu": self.m_nu,
                  "logT_AGN": cosmo["logT"]}
        ok = jnp.array(True)
        for name, (lo, hi) in EMULATOR_BOUNDS.items():
            ok = ok & (values[name] >= lo) & (values[name] <= hi)
        return ok

    def emulator_lnp(self, cosmo):
        """ln P of the emulator's definition on (:attr:`z_emu`, ``ks`` then the NLA k_1): the
        network on its modes, cloelib's logk2 nodes above them, linear in k in between."""
        lnp = np.log(10.0) * self.network(self._emulator, self.emulator_inputs(cosmo))
        plus = logk2(np.log(self.k_emu[-2:]), lnp[:, -2:], cosmo["n_s"])(jnp.asarray(np.log(self._k_plus)))
        power = jnp.exp(jnp.concatenate([lnp, plus], axis=1))
        i, t = self._k_i, self._k_t
        return jnp.log((1.0 - t) * power[:, i] + t * power[:, i + 1])

    def lnp_table(self, cosmo, growth=None):
        """ln P on (:attr:`z_tab`, ``ks`` then the NLA k_1): the emulator rows (under mead2020
        less the feedback response, plus the correction), above z = 5 the row at 5 grown by
        [D/D(5)]^2 (``growth``: :meth:`growth_ratio`, computed when not given)."""
        rows = self.emulator_lnp(cosmo)
        if self._response is not None:
            rows = rows - self.response_rows(cosmo)
        if self._n_tail:
            growth = self.growth_ratio(cosmo) if growth is None else growth
            tail = rows[self._n_emu - 1] + 2.0 * jnp.log(growth[:self._n_tail])[:, None]
            rows = jnp.concatenate([rows, tail])
        return rows + self._ln_correction

    def ia_growth(self, lnp_ia, growth):
        """cloelib's NLA D(z) on the Limber grid: sqrt[P(z, k_1)/P(0, k_1)], linear in P
        between the table's nodes; above z = 5, D(5) D(z)/D(5)."""
        p = jnp.exp(lnp_ia - lnp_ia[0])
        d = jnp.sqrt(jnp.asarray(self._ia_z) @ p)
        if len(self._ia_tail):
            d = d.at[self._ia_tail].set(jnp.sqrt(p[self._n_emu - 1]) * growth[self._n_tail:])
        return d

    # ------------------------------------------------------------------ #
    # n(z)
    # ------------------------------------------------------------------ #
    def _shifted_copy(self, nz, mean, shift, stretch):
        # The reference special-cases stretch == 1 against cancellation on nonuniform grids;
        # on this uniform grid the general form differs from it by rounding only, and keeps
        # d/d stretch at stretch = 1. A nonpositive stretch is invalid (shift_nz flags it);
        # it is evaluated at stretch 1 so that its gradient stays finite.
        z = self.z_n
        stretch = jnp.where(stretch > 0, stretch, 1.0)
        sample = mean + (z - mean - shift) / stretch
        source = jnp.concatenate([jnp.zeros(1), nz, jnp.zeros(1)])
        copy = uniform_interp(sample, self._source_z, source) / stretch
        norm = jnp.dot(copy, self._simpson_n)
        copy = copy / jnp.where(norm > 0, norm, 1.0)
        return copy, jnp.dot(copy, self._simpson_n * z), norm

    def shift_nz(self, nz, mean, delta_z, stretch):
        """The realized-mean shift + stretch of one normalized n(z); returns (n, residual).

        The effective shift solves realized mean = mean + delta_z by Newton steps kept inside
        the bracket delta_z +- ``shift_bracket`` (bisection whenever a step leaves it), as the
        reference's bracketed brentq does; a copy with no mass left counts as moved off the grid on its shift's side.
        The iterations carry no derivative: the root's derivative in (delta_z, stretch) is the
        implicit-function one, -(d excess / d theta) / (d excess / d shift) at the root.
        """
        span = (float(self.z_n[-1] - self.z_n[0]) + 1.0 if self.shift_bracket is None
                else float(self.shift_bracket))

        def excess(d, delta_z, stretch):
            _, realized, norm = self._shifted_copy(nz, mean, d, stretch)
            return jnp.where(norm > 0, realized - (mean + delta_z), jnp.sign(d))

        fixed = jax.lax.stop_gradient((delta_z, stretch))
        d = jax.lax.stop_gradient(self._solve_shift(lambda d: excess(d, *fixed), fixed[0], span))
        r, slope = jax.jvp(lambda x: excess(x, delta_z, stretch), (d,), (jnp.ones_like(d),))
        newton = -r / jax.lax.stop_gradient(jnp.where(slope > 0, slope, 1.0))
        d = d + (newton - jax.lax.stop_gradient(newton))  # value: the root; derivative: implicit
        copy, realized, norm = self._shifted_copy(nz, mean, d, stretch)
        residual = jnp.abs(realized - (mean + delta_z))
        return copy, jnp.where((norm > 0) & (stretch > 0), residual, jnp.inf)

    #: |excess| at which the shift solve stops early (the validity threshold is 1e-9).
    SHIFT_TOLERANCE = 1e-15

    def _solve_shift(self, excess, delta_z, span):
        """Bracketed Newton on ``excess(d) = 0`` from d = delta_z, within delta_z +- span, for
        at most :attr:`newton_steps` steps, stopping once |excess| <= SHIFT_TOLERANCE."""
        def step(state):
            k, d, lo, hi, _ = state
            r, slope = jax.jvp(excess, (d,), (jnp.ones_like(d),))
            lo = jnp.where(r < 0, d, lo)
            hi = jnp.where(r > 0, d, hi)
            newton = d - r / jnp.where(slope > 0, slope, 1.0)
            inside = (slope > 0) & (newton > lo) & (newton < hi)
            done = jnp.abs(r) <= self.SHIFT_TOLERANCE
            d = jnp.where(done, d, jnp.where(inside, newton, 0.5 * (lo + hi)))
            return k + 1, d, lo, hi, done

        def going(state):
            k, _, _, _, done = state
            return (k < self.newton_steps) & ~done

        init = (0, delta_z * jnp.ones(()), delta_z - span, delta_z + span, jnp.array(False))
        return jax.lax.while_loop(going, step, init)[1]

    def nz_on_grid(self, theta):
        """The shifted, stretched n(z) of every n(z) tracer on the Limber grid (cloelib's
        regrid: scipy's Akima through the delivered nodes, zero outside them; unnormalised,
        as cloelib's tracers renormalise), and the worst n(z) solve residual."""
        delta = jnp.stack([self._get(theta, f"delta_z_{t[0]}_{t[1:]}", 0.0) for t in self._nz_names])
        stretch = jnp.stack([self._get(theta, f"stretch_{t[0]}_{t[1:]}", 1.0) for t in self._nz_names])
        nz, residual = jax.vmap(self.shift_nz)(jnp.asarray(self.nz), jnp.asarray(self.nz_mean),
                                               delta, stretch)
        x = self.z_n
        slopes = akima_slopes(x, nz)
        i, s = self._regrid_i, self._regrid_s
        h = x[i + 1] - x[i]
        m = (nz[:, i + 1] - nz[:, i]) / h
        t0, t1 = slopes[:, i], slopes[:, i + 1]
        value = nz[:, i] + t0 * s + (3 * m - 2 * t0 - t1) / h * s ** 2 + (t0 + t1 - 2 * m) / h ** 2 * s ** 3
        return jnp.where(self._regrid_inside, value, 0.0), jnp.max(residual)

    @property
    def _nz_names(self):
        return [t for t in self.tracers if t != "K"]

    # ------------------------------------------------------------------ #
    # spectra, bands, likelihood, derived parameters
    # ------------------------------------------------------------------ #
    def spectra(self, theta, lnp=None):
        """C_ell (n_spectra, n_ell) with every amplitude applied, and the status.

        ``lnp`` replaces the model's P(k) by a ln P table on (:attr:`z_tab`, ``ks`` then the
        NLA k_1), the layout of :meth:`lnp_table`: the projection-only path.
        """
        cosmo = self.cosmology(theta)
        bg = self.background(cosmo)
        growth = self.growth_ratio(cosmo) if self._tail else None
        if lnp is None:
            lnp = self.lnp_table(cosmo, growth)
        d_ia = self.ia_growth(lnp[:, -1], growth)
        nz, residual = self.nz_on_grid(theta)
        cls = self._projection.project(theta, bg, lnp[:, :-1], d_ia, nz)
        amplitude = self._get(theta, "A_cmb_lensing_cross", 1.0)
        cls = cls * jnp.where(self._one_kappa, amplitude, 1.0)[:, None]
        return cls, dict(residual=residual, in_box=self.in_box(cosmo))

    def _bands_with_status(self, theta):
        cls, status = self.spectra(theta)
        bands = jnp.einsum("pbl,pl->pb", self._operators, cls).reshape(-1)
        return bands, status

    def bands(self, theta):
        """The full theory vector (n_spectra * n_bands) in TheoryVector key order."""
        return self._jit_bands(jnp.asarray(theta, dtype=float))[0]

    def _chi2_with_status(self, theta):
        bands, status = self._bands_with_status(theta)
        delta = bands[self.selection] - self.data
        chi2 = delta @ (self.icov @ delta)
        valid = (status["residual"] < 1e-9) & jnp.isfinite(chi2)
        return chi2, valid, status["in_box"]

    def chi2(self, theta):
        return self._jit_chi2(jnp.asarray(theta, dtype=float))[0]

    def loglike(self, theta):
        """log L = -chi^2 / 2 (no jit, no vmap: compose with them).

        :meth:`batched_loglike` also returns the emulator-box flag and the derived parameters."""
        chi2, valid, _ = self._chi2_with_status(theta)
        return jnp.where(valid, -0.5 * chi2, -1e300)

    def derived(self, theta):
        """(sigma8, S8, Omega_m), :attr:`DERIVED`: sigma8 of the CosmoPower-JAX sigma8 network at
        z = 0 (log T_AGN the point's under feedback, 7.6 under mead2020), Omega_m with the
        massive neutrinos (CAMB's Omega_m(0)), S8 = sigma8 sqrt(Omega_m / 0.3). No jit."""
        cosmo = self.cosmology(theta)
        log_t = cosmo["logT"] if self.feedback else LINEAR_LOG_T_AGN
        sigma8 = self.network(self._sigma, self._emulator_row(cosmo, log_t)[None, :])[0, 0]
        omega_m = (cosmo["omch2"] + cosmo["ombh2"] + self._nu_today) / cosmo["h"] ** 2
        return jnp.stack([sigma8, sigma8 * jnp.sqrt(omega_m / 0.3), omega_m])

    @staticmethod
    def _batched(single, chunk):
        """jit(vmap(single)) over a (n, n_params) batch; ``chunk`` evaluates it in sequential
        chunks of that size (bounded memory)."""
        vectorized = jax.vmap(single)
        if chunk is None:
            return jax.jit(vectorized)

        @jax.jit
        def chunked(thetas):
            n = thetas.shape[0]
            pad = (-n) % chunk
            padded = jnp.concatenate([thetas, jnp.repeat(thetas[-1:], pad, axis=0)])
            out = jax.lax.map(vectorized, padded.reshape(-1, chunk, thetas.shape[1]))
            return jax.tree_util.tree_map(lambda o: o.reshape((-1,) + o.shape[2:])[:n], out)
        return chunked

    def batched_loglike(self, *, chunk=None):
        """A jitted log-likelihood of a (n, n_params) batch, returning ``(log L, in_box)``.
        ``chunk`` evaluates the batch in sequential chunks of that size."""
        def single(theta):
            chi2, valid, in_box = self._chi2_with_status(theta)
            return jnp.where(valid, -0.5 * chi2, -1e300), in_box
        return self._batched(single, chunk)

    def batched_derived(self, *, chunk=None):
        """A jitted :meth:`derived` of a (n, n_params) batch: (n, 3) in :attr:`DERIVED` order,
        no projection evaluated. ``chunk`` as in :meth:`batched_loglike`."""
        return self._batched(self.derived, chunk)
