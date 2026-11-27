"""cloelib's Limber projection under JAX: the projection of :class:`~.jax_vector.JaxVector`.

Every kernel and every Limber integral is cloelib's own: ``PositionsTracer``,
``ShearTracer`` (NLA), ``CMBLensingTracer`` and ``AngularTwoPoint.get_Cl_tensor``, which trace
under ``jit``, ``vmap`` and ``grad``. JaxVector hands them, as duck-typed cloelib objects on
the Limber grid (:class:`~.cloe.Grid`), what cloelib itself would hand them from CAMB and
the emulator:

- :class:`GridBackground`: H(z), chi(z), chi* and Omega_m(0) on the grid;
- :class:`TablePerturbations`: ln P on the grid's P(k) table (``z_pk``, ``ks``), read
  directly by the gather (P8), and the NLA growth D(z) on the grid;
- the n(z) regridded onto the grid (Akima, as CCL and cloelib's own regrid); the tracers apply cloelib's own zero
  shift, unit stretch and trapezoid renormalisation, as in CLOE.

Parameter mapping (ours to cloelib's): ``b1_photo_bin{i}`` (0-based) = ``bias_<bin>``,
``multiplicative_bias_{i+1}`` = ``m_<bin>``, ``AIA = A_IA (1 + z_pivot)^-eta_IA`` (cloelib's
NLA has no pivot), ``CIA`` = C1 rho_crit (:data:`~.cloe.C1_RHO_CRIT`), ``EtaIA = eta_IA``. Magnification is
a positions tracer of its own (``b1 = 0``, ``magnification_bias = 5s - 2``), so that the
convergence angular response l(l+1)/(l+1/2)^2 multiplies each magnification leg, as on the
separate M term of CLOE's magnification contribution; cloelib applies the spin-2 prefactor to shear legs, the
l(l+1)/(l+1/2)^2 of a CMB-lensing leg is applied here, and ``A_cmb_lensing_cross`` by
JaxVector.

The cloelib classes are composed (:func:`~.cloelib_patches.compose`) with the patches named in
``patches``, default :data:`PATCHES`. P(k) beyond the ``ks`` table is P4's power law: P3's
gather (under P8) continues ln P linearly in ln k with the table's edge slope, which on a table
uniform in ln k is P4's definition, so P4's own (interpax) class is not composed here. P1
applies to cloelib's background, which is not used.

Not covered: the CMB-lensing auto-spectrum (cloelib's kappa kernel ends at the grid's z_max;
CLOE refuses it).
"""

import numpy as np

import jax.numpy as jnp
from cloelib.observables.cmb import CMBLensingTracer

from . import cloelib_patches as P

C_KMS = P.C_KMS
#: The patches the projection applies by default: every speed-up. P4 is the gather's own edge
#: continuation (see the module docstring); P1 is a background patch.
PATCHES = ('P3', 'P5', 'P6', 'P8')


class GridBackground:
    """Duck-typed cloelib Background whose quantities are traced arrays on one fixed grid.

    cloelib asks for H and chi on the tracer grid only, chi at :attr:`z_star` (a marker
    object), Omega_m(0) and H0.
    """

    z_star = object()

    def __init__(self, z, h, e, chi, chi_star, omega_m):
        self.z, self.h, self.H0 = z, h, 100.0 * h
        self._e, self._chi, self._chi_star, self._omega_m = e, chi, chi_star, omega_m

    def _on_grid(self, zs):
        if np.shape(zs) != np.shape(self.z):
            raise ValueError("GridBackground is evaluated on its grid only")

    def hubble_parameter(self, zs, units="km/s/Mpc"):
        self._on_grid(zs)
        hz = self.H0 * self._e
        return hz if units == "km/s/Mpc" else hz / C_KMS

    def comoving_distance(self, zs):
        if zs is self.z_star:
            return self._chi_star
        self._on_grid(zs)
        return self._chi

    def Omega_m(self, zs):
        if np.ndim(zs) or zs != 0:
            raise ValueError("GridBackground holds Omega_m today only")
        return self._omega_m


class TablePerturbations:
    """Duck-typed cloelib Perturbations: ln P on (:attr:`z`, :attr:`k`), and D on the
    background's grid."""

    def __init__(self, background, z, k, lnp, growth):
        self.background, self.z, self.k = background, z, k
        self.lnp, self._growth = lnp, growth

    def matter_power_spectrum(self, zs, ks):
        if np.shape(zs) != np.shape(self.z) or np.shape(ks) != np.shape(self.k):
            raise ValueError("TablePerturbations is evaluated on its table only")
        return jnp.exp(self.lnp)

    def log_matter_power_spectrum(self, zs, ks):
        """ln P itself, which :class:`~.cloelib_patches.LogTableTwoPoint` (P8) gathers from."""
        self.matter_power_spectrum(zs, ks)
        return self.lnp

    def growth_factor(self, zs, ks=None):
        if np.shape(zs) != np.shape(self.background.z):
            raise ValueError("TablePerturbations' growth is evaluated on the grid only")
        return self._growth


class CloelibProjection:
    """The tracers and Limber sum of a JaxVector's spectra, through cloelib (see the module)."""

    def __init__(self, jv, pairs, patches=PATCHES):
        self.jv = jv
        self.patches = tuple(patches)
        self.two_point, self.shear, self.positions = P.compose(self.patches)
        names = jv._nz_names
        self.g = [i for i, t in enumerate(names) if t[0] == "G"]
        self.l = [i for i, t in enumerate(names) if t[0] == "L"]
        column = {t: i for i, t in enumerate(names[j] for j in self.g)}
        column.update({t: i for i, t in enumerate(names[j] for j in self.l)})
        self.slots = []
        for a, b in pairs:
            if a[0] + b[0] in ("GK", "LK", "LG"):
                a, b = b, a
            self.slots.append((a[0] + b[0], 0 if a == "K" else column[a], column[b]))
        ell = jv.ells
        kappa = ell * (ell + 1) / (ell + 0.5) ** 2
        self.prefactor = np.stack([kappa ** (a.count("K") + b.count("K")) for a, b in pairs])

    def project(self, theta, bg, lnp, growth, nz):
        """C_ell (n_spectra, n_ell) with the multipole prefactors.

        ``bg`` is :meth:`~.jax_vector.JaxVector.background`'s, ``lnp`` ln P on the grid's
        (``z_pk``, ``ks``) table, ``growth`` the NLA D(z) and ``nz`` (n(z) tracers, grid) the
        regridded n(z)."""
        jv = self.jv
        get = jv._get
        z = jv.z
        background = GridBackground(z, bg["h"], bg["e"], bg["chi"], bg["chi_star"], bg["omega_m"])
        table = TablePerturbations(background, jv.z_tab, jv.ks, lnp, growth)
        names = jv._nz_names
        pos = {f"b1_photo_bin{i}": get(theta, f"bias_{names[j][1:]}", np.nan)
               for i, j in enumerate(self.g)}
        for i in range(len(self.g)):
            pos.update({f"dz_pos_{i + 1}": 0.0, f"width_pos_{i + 1}": 1.0,
                        f"magnification_bias_{i + 1}": 0.0})
        eta = get(theta, "eta_IA", 0.0)
        she = {"AIA": get(theta, "A_IA", 0.0) * (1.0 + jv.ia_z_pivot) ** (-eta),
               "CIA": jv.c1_rho_crit, "EtaIA": eta}
        for i, j in enumerate(self.l):
            she.update({f"multiplicative_bias_{i + 1}": get(theta, f"m_{names[j][1:]}", 0.0),
                        f"dz_shear_{i + 1}": 0.0, f"width_shear_{i + 1}": 1.0})
        tracers = {"K": CMBLensingTracer(table, z)}
        if self.g:
            galaxies = nz[jnp.asarray(self.g)]
            tracers["G"] = self.positions(table, galaxies, z, "per_bin", pos)
            if jv.include_mag_bias:
                mag = dict(pos)
                for i, j in enumerate(self.g):
                    mag[f"b1_photo_bin{i}"] = 0.0
                    mag[f"magnification_bias_{i + 1}"] = 5.0 * get(theta, f"mag_slope_{names[j][1:]}", np.nan) - 2.0
                tracers["M"] = self.positions(table, galaxies, z, "per_bin", mag)
        if self.l:
            tracers["L"] = self.shear(table, nz[jnp.asarray(self.l)], z, she, ia_model="NLA")
        ells = jnp.asarray(jv.ells)
        shared, tensors = {}, {}

        def tensor(family):
            if family not in tensors:
                pair = self.two_point(tracers[family[0]], tracers[family[1]])
                if hasattr(pair, "share"):
                    pair.share(shared)
                tensors[family] = pair.get_Cl_tensor(ells, 0, jv.ks)
            return tensors[family]

        cls = jnp.stack([tensor(f)[:, i, j] for f, i, j in self.slots]) * self.prefactor
        if jv.include_mag_bias:
            kappa = ells * (ells + 1) / (ells + 0.5) ** 2
            extra = []
            for p, (family, i, j) in enumerate(self.slots):
                a, b = family
                value = jnp.zeros_like(ells)
                if a == "G":
                    value = value + tensor("M" + b)[:, i, j] * kappa
                if b == "G":
                    value = value + tensor(a + "M")[:, i, j] * kappa
                if a == "G" and b == "G":
                    value = value + tensor("MM")[:, i, j] * kappa ** 2
                extra.append(value * self.prefactor[p])
            cls = cls + jnp.stack(extra)
        return cls
