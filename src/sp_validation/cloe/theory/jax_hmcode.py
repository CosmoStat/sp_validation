"""HMcode-2020's baryonic-feedback response in JAX.

CAMB's HMcode-2020 with feedback (``mead2020_feedback``, Mead et al. 2021) is the gravity-only
HMcode-2020 prediction times a response S(k, z) = P_feedback / P_gravity, the ratio of two
*vanilla* halo models (CAMB ``halofit.f90``, imead 5 over imead 4; HMcode-python
``_get_feedback_suppression``): P = P_lin + P_1h, the one-halo term damped by
(k/k*)^4 / (1 + (k/k*)^4) with k* = 0.05618 sigma_8,cold(z)^-1.013. Both use a Sheth-Tormen
mass function in the cold peak height nu = delta_c / sigma_cold(M, z), Mead (2017) delta_c and
Delta_v, Bullock et al. (2001) formation redshifts (f = 0.01) and truncated NFW haloes:

- gravity: concentration 4 (1 + z_f)/(1 + z), window (1 - f_nu) U_NFW;
- feedback: concentration B(z) (1 + z_f)/(1 + z), window (Omega_c/Omega_m + f_g(M)) U_NFW + f_*,
  with f_g = (Omega_b/Omega_m - f_*) (M/M_b)^2 / (1 + (M/M_b)^2) and the one-parameter
  log T_AGN relations for B, M_b and f_* (f_* capped at Omega_b/Omega_m).

:class:`FeedbackResponse` evaluates ln S on fixed (z, k) nodes from the linear total and cold
spectra at z = 0, scaled by HMcode's growth (flat LCDM without radiation, massive neutrinos as
matter). Against CAMB's own response (through CCL), over logA 1.8-3.6, omch2 0.085-0.2,
h 0.47-0.84 and n_s 0.83-1.09, it agrees to ~1e-3 in ln P at k < 5 Mpc^-1 and z <= 1 (to 5e-3
at 10 Mpc^-1), and the change between two cosmologies to ~1e-3 below z = 2 and ~5e-3 at z = 2,
k > 2 h/Mpc, where f_* nears Omega_b/Omega_m. Internally h units (k in h/Mpc, M in Msun/h), as
HMcode.
"""

from math import gamma

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

#: Critical density today in (Msun/h) (Mpc/h)^-3 (HMcode's constants).
RHO_CRITICAL = 3.0 * 100.0 ** 2 / (8.0 * np.pi * (6.6743e-11 * 1.9884e30 / (3.0857e22 * 1e3 ** 2)))
DELTA_C0 = (3.0 / 20.0) * (12.0 * np.pi) ** (2.0 / 3.0)
DELTA_V0 = 18.0 * np.pi ** 2
EULER_GAMMA = 0.5772156649015329
ST_P, ST_Q = 0.3, 0.707
#: Sheth-Tormen normalisation: g(nu) integrates to one.
ST_A = float(np.sqrt(2.0 * ST_Q) / (np.sqrt(np.pi) + gamma(0.5 - ST_P) / 2 ** ST_P))


@jax.custom_jvp
def sici(x):
    """(Si(x), Ci(x)) for x > 0.

    Power series up to x = 20 (cancellation error ~ e^20 eps ~ 5e-9), the asymptotic auxiliary
    functions above (truncation error ~ 20!/20^20 ~ 2e-8). Both branches see clamped arguments,
    so neither overflows. The derivative is the closed form (sin x / x, cos x / x), so reverse
    mode stores none of the series.
    """
    x0 = 20.0
    xs = jnp.minimum(x, x0)
    xa = jnp.maximum(x, x0)
    si, ci = jnp.zeros_like(xs), jnp.zeros_like(xs)
    term = xs  # (-1)^n x^(2n+1) / (2n+1)!
    for n in range(40):
        si = si + term / (2 * n + 1)
        term_c = term * xs / (2 * n + 2)  # (-1)^n x^(2n+2) / (2n+2)!
        ci = ci - term_c / (2 * n + 2)
        term = -term_c * xs / (2 * n + 3)
    ci = ci + EULER_GAMMA + jnp.log(xs)
    ia2 = 1.0 / (xa * xa)
    f, g = jnp.zeros_like(xa), jnp.zeros_like(xa)
    tf, tg = jnp.ones_like(xa), jnp.ones_like(xa)
    for n in range(10):
        f, g = f + tf, g + tg
        tf = -tf * (2 * n + 1) * (2 * n + 2) * ia2
        tg = -tg * (2 * n + 2) * (2 * n + 3) * ia2
    f, g = f / xa, g * ia2
    si_a = 0.5 * np.pi - f * jnp.cos(xa) - g * jnp.sin(xa)
    ci_a = f * jnp.sin(xa) - g * jnp.cos(xa)
    big = x > x0
    return jnp.where(big, si_a, si), jnp.where(big, ci_a, ci)


@sici.defjvp
def _sici_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    si, ci = sici(x)
    return (si, ci), (jnp.sin(x) / x * dx, jnp.cos(x) / x * dx)


def nfw_window(k, rv, c):
    """Normalised Fourier transform of an NFW profile truncated at ``rv``: (nk,), (nM,) -> (nk, nM)."""
    rs = rv / c
    ks, kv = k[:, None] * rs[None, :], k[:, None] * rv[None, :]
    si_sv, ci_sv = sici(ks + kv)
    si_s, ci_s = sici(ks)
    f = jnp.cos(ks) * (ci_sv - ci_s) + jnp.sin(ks) * (si_sv - si_s) - jnp.sin(kv) / (ks + kv)
    return f / (jnp.log1p(c) - c / (1.0 + c))


def tophat(x):
    """Fourier transform of a real-space top hat, 3 (sin x - x cos x) / x^3."""
    small = jnp.abs(x) < 1e-3
    xs = jnp.where(small, 1.0, x)
    return jnp.where(small, 1.0 - x ** 2 / 10.0, 3.0 * (jnp.sin(xs) - xs * jnp.cos(xs)) / xs ** 3)


class FeedbackResponse:
    """ln S(k, z) of HMcode-2020 at fixed (z, k) nodes; see the module docstring.

    Args:
        z: redshift nodes of the response.
        k: wavenumber nodes of the response, Mpc^-1.
        lnk_linear: uniform ln k grid (Mpc^-1) of the z = 0 linear spectra passed to
            :meth:`ln_response`; it must reach the top-hat scales of the mass range (the
            smallest Lagrangian radius is ~0.014 Mpc/h at 1e6 Msun/h).
        n_mass, mass_range: log-uniform halo masses, Msun/h.
        n_a: uniform scale-factor nodes on [0, 1] of the growth tables.
    """

    def __init__(self, z, k, lnk_linear, *, n_mass=128, mass_range=(1e6, 1e17), n_a=2049):
        self.z = np.asarray(z, float)
        self.k = np.asarray(k, float)
        self.lnk_linear = np.asarray(lnk_linear, float)
        step = np.diff(self.lnk_linear)
        if not np.allclose(step, step[0]):
            raise ValueError("lnk_linear must be uniform")
        self.dlnk = float(step[0])
        self.mass = np.geomspace(*mass_range, n_mass)
        self.a_grid = np.linspace(0.0, 1.0, n_a)

    def growth(self, om_m):
        """HMcode's growth g(a) (-> a for a << 1) and accumulated growth G(a) = int_0^a g/a' da'
        on :attr:`a_grid`, for flat LCDM without radiation: g = (5/2) Om E int_0^a da'/(a' E)^3."""
        a = self.a_grid
        e = jnp.sqrt(om_m * a[1:] ** -3 + 1.0 - om_m)
        inner = jnp.concatenate([jnp.zeros(1), 1.0 / (a[1:] * e) ** 3])
        da = a[1] - a[0]
        cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (inner[1:] + inner[:-1]) * da)])
        g = jnp.concatenate([jnp.zeros(1), 2.5 * om_m * e * cum[1:]])
        ratio = jnp.concatenate([jnp.ones(1), g[1:] / a[1:]])
        big_g = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (ratio[1:] + ratio[:-1]) * da)])
        return g, big_g

    def sigma0(self, radius_h, lnp_cb0, h):
        """Cold sigma(R) at z = 0 for R in Mpc/h, from ln P_cb (Mpc^3) on :attr:`lnk_linear`."""
        k_h = np.exp(self.lnk_linear) / h
        power_h = jnp.exp(lnp_cb0) * h ** 3
        w = tophat(k_h[None, :] * radius_h[:, None]) ** 2
        integrand = k_h ** 3 * power_h * w / (2.0 * np.pi ** 2)
        return jnp.sqrt(jnp.sum(0.5 * (integrand[:, 1:] + integrand[:, :-1]), axis=1) * self.dlnk)

    def ln_response(self, lnp_tot0, lnp_cb0, om_c, om_b, om_nu, h, log_t):
        """ln S on (:attr:`z`, :attr:`k`).

        ``lnp_tot0``, ``lnp_cb0``: ln of the z = 0 linear total-matter and cold spectra (Mpc^3) on
        :attr:`lnk_linear`; ``om_*`` the density parameters today; ``log_t`` = log10 T_AGN/K.
        """
        om_m = om_c + om_b + om_nu
        f_nu = om_nu / om_m
        rhom = RHO_CRITICAL * om_m
        g_tab, big_g_tab = self.growth(om_m)
        z = self.z
        a = 1.0 / (1.0 + z)
        g = jnp.interp(a, self.a_grid, g_tab)
        big_g = jnp.interp(a, self.a_grid, big_g_tab)
        grow = g / g_tab[-1]
        om_mz = om_m * a ** -3 / (om_m * a ** -3 + 1.0 - om_m)
        x, y = 1.0 - g / a, 1.0 - big_g / a
        mead = lambda p0, p1, p2, p3: p0 + p1 * x + p2 * x ** 2 + p3 * y
        lo = jnp.log10(om_mz)
        dc = ((1.0 + mead(-0.0069, -0.0208, 0.0312, 0.0021) * lo + mead(0.0001, -0.0647, -0.0417, 0.0646))
              * DELTA_C0 * (1.0 - 0.041 * f_nu))
        dv = ((1.0 + mead(-0.79, -10.17, 2.51, 6.51) * lo + mead(-1.89, 0.38, 18.8, -15.87) * lo ** 2)
              * DELTA_V0 * (1.0 + 0.763 * f_nu))

        m = jnp.asarray(self.mass)
        r_lag = jnp.cbrt(3.0 * m / (4.0 * np.pi * rhom))
        n_m = len(self.mass)
        sig = self.sigma0(jnp.concatenate([r_lag, np.cbrt(0.01) * r_lag, jnp.array([8.0])]), lnp_cb0, h)
        sig_m, sig_f, sig8 = sig[:n_m], sig[n_m:2 * n_m], sig[-1]
        k_h = jnp.asarray(self.k) / h
        lnk_h = jnp.asarray(self.lnk_linear) - jnp.log(h)
        p_lin0 = jnp.exp(jnp.interp(jnp.log(k_h), lnk_h, lnp_tot0)) * h ** 3
        p_lin = p_lin0[None, :] * grow[:, None] ** 2

        theta = log_t - 7.8
        b0, bz = 3.44 - 0.496 * theta, -0.0671 - 0.0371 * theta
        mb0, mbz = 10.0 ** (13.87 + 1.81 * theta), -0.108 + 0.195 * theta
        f0, fz = 0.0201 - 0.003 * theta, 0.409 + 0.0224 * theta

        def one_halo(zz, aa, gg, dcz, dvz, grz):
            nu = dcz / (sig_m * grz)
            # Bullock et al. (2001): g(a_f) = g(a) delta_c / sigma(0.01 M, z); haloes that would
            # form "in the future" form now.
            target = gg * dcz / (sig_f * grz)
            af = jnp.where(target >= gg, aa, jnp.interp(target, g_tab, self.a_grid))
            zf = 1.0 / jnp.maximum(af, 1e-3) - 1.0
            rv = r_lag / jnp.cbrt(dvz)
            kstar = 0.05618 * (sig8 * grz) ** -1.013
            damp = (k_h / kstar) ** 4 / (1.0 + (k_h / kstar) ** 4)
            gnu = ST_A * (1.0 + (ST_Q * nu ** 2) ** -ST_P) * jnp.exp(-ST_Q * nu ** 2 / 2.0)
            weight = 0.5 * gnu * m * jnp.concatenate([nu[1:2] - nu[:1], nu[2:] - nu[:-2],
                                                       nu[-1:] - nu[-2:-1]]) / rhom

            def p1h(w):  # trapezium rule over nu of g(nu) M W^2 / rhom
                return damp * (w ** 2 @ weight)
            gravity = (1.0 - f_nu) * nfw_window(k_h, rv, 4.0 * (1.0 + zf) / (1.0 + zz))
            fstar = jnp.minimum(f0 * 10.0 ** (zz * fz), om_b / om_m)
            ratio = (m / (mb0 * 10.0 ** (zz * mbz))) ** 2
            gas = (om_b / om_m - fstar) * ratio / (1.0 + ratio)
            concentration = b0 * 10.0 ** (zz * bz) * (1.0 + zf) / (1.0 + zz)
            feedback = (om_c / om_m + gas)[None, :] * nfw_window(k_h, rv, concentration) + fstar
            return p1h(gravity), p1h(feedback)

        p_gravity, p_feedback = jax.vmap(one_halo)(z, a, g, dc, dv, grow)
        return jnp.log(p_lin + p_feedback) - jnp.log(p_lin + p_gravity)
