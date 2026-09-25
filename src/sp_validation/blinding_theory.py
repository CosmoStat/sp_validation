"""Blinding theory: the fiducial configuration and CCL's shear two-point prediction.

:Name: blinding_theory.py

:Description: :class:`TheoryConfig` is the fiducial cosmology and model
    recipe; :func:`xi_ccl` and :func:`cl_ee` are the tomographic shear ξ± and
    Cℓ_EE between two bins' n(z). CCL builds the nonlinear P(k) through its
    Boltzmann-CAMB HMCode2020 route and projects with its own Limber
    (``angular_cl``) and FFTLog (``correlation``). ``test_camb_ccl_crosscheck``
    compares this path with a direct CAMB run.

    The generic cosmology machinery here is destined for ``cs_util.cosmo``
    (cs_util#80).

    Only ``numpy`` is imported at module level; CCL is imported inside the
    functions that need it, so importing :class:`TheoryConfig` never drags in a
    theory backend.
"""

import dataclasses

import numpy as np

# Fixed constants of the fiducial, passed explicitly to CCL (and to the CAMB
# oracle in the tests) rather than left to either stack's default.
NEFF = 3.046
T_CMB = 2.7255


# --------------------------------------------------------------------------- #
# Configuration surface — the ONE place fiducial cosmology + model choices live
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class TheoryConfig:
    """Fiducial cosmology and model configuration for the theory paths.

    Every field is a deliberate, configurable choice. The defaults mirror the
    ``cosmo_inference`` CosmoSIS fiducial (the ``SP_v1.4.6.3_A_cell`` pipeline
    + ``values_ia.ini`` central values), so the CCL theory computed here and
    the CAMB theory CosmoSIS computes agree to the level the CAMB↔CCL
    cross-check test asserts. Adopting a different named group fiducial is a
    change to these *values*, not to any code.

    Cosmology is parametrised by the blind axes ``S8`` and ``Omega_m`` and
    converted to CCL's native ``sigma8``/``Omega_c`` by :meth:`sigma8` /
    :meth:`omega_c`.

    ``halofit_version`` names the nonlinear recipe CAMB runs, whether CCL
    calls it or CosmoSIS does.
    """

    # Cosmological parameters (blind axes S8, Omega_m + the rest).
    S8: float = 0.80  # values_ia.ini S_8_input central
    Omega_m: float = 0.30
    Omega_b: float = 0.0469  # ombh2=0.023 at h=0.7  ->  0.023/0.7^2
    h: float = 0.70
    n_s: float = 0.96
    m_nu: float = 0.06  # Σm_ν in eV, distributed under `mass_split`
    w0: float = -1.0
    wa: float = 0.0

    # Neutrino mass split: normal hierarchy (CosmoSIS `neutrino_hierarchy=normal`).
    mass_split: str = "normal"

    # Boltzmann backend for the CCL path (#280). `boltzmann_camb` shares one
    # power-spectrum path with the CosmoSIS+CAMB inference stack; any other
    # backend falls back to CCL's own halofit (see `ccl_cosmology`), a
    # deliberate cross-check tool rather than a production setting.
    transfer_function: str = "boltzmann_camb"

    # CAMB HMCode2020 + baryonic feedback.
    halofit_version: str = "mead2020_feedback"
    hmcode_logT_AGN: float = 7.5  # values_ia.ini logT_AGN central

    # Intrinsic alignments: NLA. The fiducial defaults IA OFF (ia_bias=0) —
    # the blinding shift is a difference of two theory vectors at the same IA,
    # so IA nearly cancels there, and IA-off keeps the CAMB↔CCL cross-check a
    # clean test of the shear calculation. Set `ia_bias` nonzero (CosmoSIS
    # central A=1.0) to include NLA.
    ia_bias: float = 0.0
    ia_z_piv: float = 0.62
    ia_alphaz: float = 0.0

    def sigma8(self):
        """CCL ``sigma8`` implied by ``S8`` and ``Omega_m``.

        ``S8 ≡ σ8 √(Ωm / 0.3)`` — the standard weak-lensing definition — so
        ``σ8 = S8 / √(Ωm / 0.3)``. At the fiducial (S8=0.80, Ωm=0.30),
        σ8 = 0.80.
        """
        return self.S8 / np.sqrt(self.Omega_m / 0.3)

    def omega_c(self):
        """CCL cold-dark-matter density ``Omega_c = Omega_m − Omega_b − Ω_ν``.

        The neutrino density ``Ω_ν h² = Σm_ν / 93.14 eV`` is subtracted so
        the *total* matter density is exactly ``Omega_m`` (CCL treats massive
        neutrinos as a separate species, not part of ``Omega_c``).
        """
        omega_nu = self.m_nu / (93.14 * self.h**2)
        return self.Omega_m - self.Omega_b - omega_nu

    def ccl_params(self):
        """This point as a plain CCL-native parameter mapping.

        Exactly the keys ``Omega_c, Omega_b, h, n_s, sigma8, m_nu,
        mass_split, w0, wa, Neff, T_CMB`` and no others, so no CCL default
        rides along; ``Neff``/``T_CMB`` are the fixed module constants.
        """
        return {
            "Omega_c": self.omega_c(),
            "Omega_b": self.Omega_b,
            "h": self.h,
            "n_s": self.n_s,
            "sigma8": self.sigma8(),
            "m_nu": self.m_nu,
            "mass_split": self.mass_split,
            "w0": self.w0,
            "wa": self.wa,
            "Neff": NEFF,
            "T_CMB": T_CMB,
        }


# --------------------------------------------------------------------------- #
# CCL: cosmology construction, Cℓ_EE, ξ±
# --------------------------------------------------------------------------- #
# The two cosmologies of a blind (fiducial and hidden) are evaluated for every
# block of a SACC; caching the ccl.Cosmology per parameter point avoids
# re-running the CAMB P(k) computation for each block.
_COSMO_CACHE = {}


def ccl_cosmology(params, config):
    """A ``pyccl.Cosmology`` at ``params`` with ``config``'s nonlinear recipe.

    ``params`` is a plain CCL-native mapping (:meth:`TheoryConfig.ccl_params`,
    possibly with keys overlaid by the hidden draw); ``config`` supplies the
    non-sampled recipe tokens. Under ``boltzmann_camb`` the nonlinear P(k)
    runs through CAMB's HMCode2020; any other backend has no CAMB run to hand
    tokens to and takes CCL's own halofit. Cached per parameter point: CCL
    memoises P(k) on the object, so the cache saves repeated Boltzmann runs.
    """
    import pyccl as ccl

    key = (
        tuple(sorted(params.items())),
        config.transfer_function,
        config.halofit_version,
        config.hmcode_logT_AGN,
    )
    if key not in _COSMO_CACHE:
        nonlinear = (
            {
                "matter_power_spectrum": "camb",
                "extra_parameters": {
                    "camb": {
                        "halofit_version": config.halofit_version,
                        "HMCode_logT_AGN": config.hmcode_logT_AGN,
                    }
                },
            }
            if config.transfer_function == "boltzmann_camb"
            else {"matter_power_spectrum": "halofit"}
        )
        _COSMO_CACHE[key] = ccl.Cosmology(
            **params,
            transfer_function=config.transfer_function,
            **nonlinear,
        )
    return _COSMO_CACHE[key]


def xi_ell_grid():
    """The ℓ grid the ξ± Hankel projection integrates over.

    Integers 2…49, then 200 log-spaced multipoles up to 6·10⁴ — dense enough
    at low ℓ (where ξ± at large θ lives) and wide enough for the small-θ
    tail. ``ccl.correlation`` interpolates C(ℓ) internally, so this fixes the
    resolution of every ξ± this module produces.
    """
    return np.unique(
        np.concatenate([np.arange(2, 50), np.geomspace(50, 6e4, 200)]).astype(float)
    )


# Tracers are rebuilt for every pair of every block at both cosmologies of a
# blind, and each build runs CCL's lensing-kernel integral over the bin's n(z).
# Cached for the same reason as `_COSMO_CACHE`, and keyed on `id(cosmo)`
# because that cache pins every cosmology for the process lifetime, so an id
# can never be recycled onto a different object.
_TRACER_CACHE = {}


def _tracer(cosmo, z, nz, config):
    """A ``WeakLensingTracer`` for one bin's n(z), NLA from ``config``.

    With the fiducial ``ia_bias = 0`` the tracer is built bare — no IA term.
    A nonzero ``ia_bias`` enters as the NLA amplitude
    ``A(z) = ia_bias · ((1+z)/(1+z_piv))^alphaz``.
    """
    z = np.asarray(z)
    nz = np.asarray(nz)
    key = (
        id(cosmo),
        z.tobytes(),
        nz.tobytes(),
        config.ia_bias,
        config.ia_z_piv,
        config.ia_alphaz,
    )
    if key not in _TRACER_CACHE:
        _TRACER_CACHE[key] = _build_tracer(cosmo, z, nz, config)
    return _TRACER_CACHE[key]


def _build_tracer(cosmo, z, nz, config):
    import pyccl as ccl

    if config.ia_bias == 0.0:
        return ccl.WeakLensingTracer(cosmo, dndz=(z, nz))
    a_ia = config.ia_bias * ((1 + z) / (1 + config.ia_z_piv)) ** config.ia_alphaz
    return ccl.WeakLensingTracer(cosmo, dndz=(z, nz), ia_bias=(z, a_ia), use_A_ia=True)


def cl_ee(params, config, nz_i, nz_j, ell):
    """Cross Cℓ_EE at ``ell`` for the bin pair with n(z) ``nz_i``, ``nz_j``.

    Two-tracer: one :class:`~pyccl.WeakLensingTracer` per bin from that bin's
    own ``(z, nz)``, then ``angular_cl(cosmo, tracer_i, tracer_j, ell)`` —
    the cross-spectrum for i ≠ j, the auto-spectrum when the two n(z) are the
    same bin. The shear ``angular_cl`` is the E-mode spectrum; B and EB are
    zero in theory, which is why only Cℓ_EE ever receives a blinding shift.
    """
    import pyccl as ccl

    cosmo = ccl_cosmology(params, config)
    tracer_i = _tracer(cosmo, *nz_i, config)
    tracer_j = _tracer(cosmo, *nz_j, config)
    return ccl.angular_cl(cosmo, tracer_i, tracer_j, np.asarray(ell, dtype=float))


def xi_ccl(params, config, nz_i, nz_j, theta_arcmin, ell=None):
    """ξ± at ``theta_arcmin`` for one bin pair.

    Cross Cℓ_EE on :func:`xi_ell_grid` (or ``ell``), then ``ccl.correlation``
    (FFTLog Hankel transform) at θ in degrees, ``type="GG+"`` / ``"GG-"``.

    Returns
    -------
    (np.ndarray, np.ndarray)
        ``(xip, xim)`` aligned to ``theta_arcmin``.
    """
    import pyccl as ccl

    ell = xi_ell_grid() if ell is None else np.asarray(ell, dtype=float)
    cosmo = ccl_cosmology(params, config)
    cl = cl_ee(params, config, nz_i, nz_j, ell)
    theta_deg = np.asarray(theta_arcmin) / 60.0
    xip = ccl.correlation(cosmo, ell=ell, C_ell=cl, theta=theta_deg, type="GG+")
    xim = ccl.correlation(cosmo, ell=ell, C_ell=cl, theta=theta_deg, type="GG-")
    return xip, xim
