"""Blinding theory: fiducial configuration and the two ξ± theory paths.

:Name: blinding_theory.py

:Description: The blinding backend's theory surface — the fiducial
    configuration (:class:`TheoryConfig`) and two independent routes to the
    tomographic shear two-point prediction.

    The generic cosmology machinery here is destined for ``cs_util.cosmo``
    (cs_util#80).

    Two independent routes to the shear two-point prediction:

    - **CCL-native path** (:func:`xi_ccl`, :func:`cl_ee`): CCL builds the
      nonlinear P(k) through its Boltzmann-CAMB HMCode2020 route
      (``matter_power_spectrum='camb'`` + ``extra_parameters``) and projects
      to Cℓ/ξ± via its own Limber (``angular_cl``) + FFTLog
      (``correlation``). This is the recipe the blinding theory backends use.
    - **Independent-CAMB path** (:func:`xi_camb`): a direct ``pycamb`` run
      produces the HMCode2020 ``P(k, z)`` (σ8-matched via the closed-form
      A_s rescale of :func:`camb_As_for_sigma8`), wrapped in a ``ccl.Pk2D``
      and projected through the same CCL Limber + FFTLog machinery.

    Because both paths route their nonlinear P(k) through CAMB's HMCode2020
    and both project through CCL, a common Limber+FFTLog bug cancels between
    them: the CAMB↔CCL cross-check test built on these two paths validates
    the **P(k) recipe** and the **σ8/A_s amplitude convention**, not the
    projection machinery.

    This module imports only ``numpy`` at module level; CCL and CAMB are
    imported inside the functions that need them, so importing
    :class:`TheoryConfig` never drags in a theory backend.
"""

import dataclasses

import numpy as np

# Fixed constants of the fiducial — load-bearing for the CAMB↔CCL amplitude
# match, so they are emitted explicitly to both stacks rather than left to
# either stack's default. Not user-facing TheoryConfig fields.
NEFF = 3.046
T_CMB = 2.7255

# Matter-power redshift grid shared by `make_camb_params` and `xi_camb`, so the
# CAMB run and the Pk2D built from it sample the same redshifts.
PK_ZMAX = 3.0
PK_NZ = 48


def coerce_fields(cls, overrides):
    """Validate ``overrides`` against ``cls``'s fields, coercing floats.

    Unknown keys raise. Every ``float``-declared field goes through
    :func:`float`, so a YAML/CLI ``w0: -1`` (int) yields the same value — and
    the same config digest — as the float default ``-1.0``. Digest stability
    depends on this, so it is one helper rather than three copies.
    """
    by_name = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(overrides) - set(by_name)
    if unknown:
        raise ValueError(
            f"unknown {cls.__name__} fields {sorted(unknown)}; "
            f"valid fields are {sorted(by_name)}"
        )
    return {
        name: (float(v) if by_name[name].type in (float, "float") else v)
        for name, v in overrides.items()
    }


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

    One nonlinear recipe is named by two tokens (``ccl_halofit_version``,
    ``camb_halofit_version``) because CCL and CAMB could name it differently;
    each stack is fed its own so a rename cannot silently split the recipe.
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

    # CAMB HMCode2020 + baryonic feedback — see the class docstring.
    ccl_halofit_version: str = "mead2020_feedback"
    camb_halofit_version: str = "mead2020_feedback"
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
        """The fiducial point as a plain CCL-native parameter mapping.

        Exactly the keys ``Omega_c, Omega_b, h, n_s, sigma8, m_nu,
        mass_split, w0, wa, Neff, T_CMB`` and no others — no CCL default
        rides along. ``Neff``/``T_CMB`` are the fixed module constants. This
        mapping is what the Smokescreen fork receives as ``fiducial_params``
        and what every ``theory_fn`` receives back (possibly with
        ``sigma8``/``Omega_c`` overlaid by the hidden draw).
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

    @classmethod
    def from_overrides(cls, overrides):
        """Build from a mapping of field overrides (fail loud on unknown keys)."""
        return cls(**coerce_fields(cls, overrides))


# --------------------------------------------------------------------------- #
# CCL-native path: cosmology construction, Cℓ_EE, ξ±
# --------------------------------------------------------------------------- #
# The two cosmologies of a blind (fiducial + hidden) are evaluated by three
# theory backends over multiple blocks; caching the ccl.Cosmology per parameter
# point avoids re-running the CAMB P(k) computation for every block.
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
        config.ccl_halofit_version,
        config.hmcode_logT_AGN,
    )
    if key not in _COSMO_CACHE:
        nonlinear = (
            {
                "matter_power_spectrum": "camb",
                "extra_parameters": {
                    "camb": {
                        "halofit_version": config.ccl_halofit_version,
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
    """CCL-native ξ± at ``theta_arcmin`` for one bin pair (Path A).

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


# --------------------------------------------------------------------------- #
# Independent-CAMB path: A_s reconciliation + P(k) → Pk2D → CCL projection
# --------------------------------------------------------------------------- #
def make_camb_params(config, As, *, nonlinear, zmax=PK_ZMAX, n_z=PK_NZ, kmax=20.0):
    """A ``CAMBparams`` at ``config``'s background with amplitude ``As``.

    Every :class:`TheoryConfig` field CCL sees is fed to CAMB from the same
    source — ``w0``/``wa`` via ``set_dark_energy``, ``Neff``/``T_CMB`` as the
    module constants, ``m_nu``/``mass_split`` through ``set_cosmology`` — so
    the independent path differs from the CCL path only in who computes P(k),
    never in an unmatched background parameter.
    """
    import camb

    p = camb.CAMBparams()
    p.set_cosmology(
        H0=config.h * 100,
        ombh2=config.Omega_b * config.h**2,
        omch2=config.omega_c() * config.h**2,
        mnu=config.m_nu,
        num_massive_neutrinos=1,
        neutrino_hierarchy=config.mass_split,
        nnu=NEFF,
        TCMB=T_CMB,
    )
    p.set_dark_energy(w=config.w0, wa=config.wa, dark_energy_model="ppf")
    p.InitPower.set_params(As=As, ns=config.n_s)
    p.set_matter_power(redshifts=list(np.linspace(0.0, zmax, n_z)), kmax=kmax)
    if nonlinear:
        p.NonLinear = camb.model.NonLinear_both
        p.NonLinearModel.set_params(
            halofit_version=config.camb_halofit_version,
            HMCode_logT_AGN=config.hmcode_logT_AGN,
        )
    else:
        p.NonLinear = camb.model.NonLinear_none
    return p


def camb_linear_sigma8(config, As, **kwargs):
    """CAMB's linear σ8(z=0) at amplitude ``As``."""
    import camb

    results = camb.get_results(make_camb_params(config, As, nonlinear=False, **kwargs))
    return float(results.get_sigma8_0())


def camb_As_for_sigma8(config, sigma8_target, As_seed=2.1e-9, **kwargs):
    """The CAMB ``A_s`` whose linear σ8 equals ``sigma8_target``.

    Closed-form: linear σ8² ∝ A_s exactly, so one CAMB linear-σ8 evaluation
    at ``As_seed`` and one rescale ``As_seed · (σ8_target/σ8_seed)²`` land on
    the target — no iteration. This settles the convention subtlety that our
    fiducial fixes σ8 for CCL but A_s for CAMB: a nominal ``A_s = 2.1e-9``
    leaves CAMB's σ8 ≈3% off target, enough to blow a ξ± comparison to
    ~9–10%.
    """
    sigma8_seed = camb_linear_sigma8(config, As_seed, **kwargs)
    return As_seed * (sigma8_target / sigma8_seed) ** 2


def xi_camb(config, nz, theta_arcmin, *, n_ell=300, ell_max=60000, kmax=20.0, n_k=400):
    """Independent-CAMB ξ± for one bin (Path B): CAMB P(k) → Pk2D → CCL.

    A direct pycamb run produces the HMCode2020 nonlinear ``P(k, z)`` at a
    σ8-matched ``A_s`` (:func:`camb_As_for_sigma8`), wrapped in a ``Pk2D`` and
    projected by CCL's Limber + FFTLog with a bare tracer (IA off — this path
    exists for the cross-check). ``hubble_units=False, k_hunit=False`` already
    returns CCL's native units (k in 1/Mpc, P in Mpc³), so applying an
    ``·h``/``/h³`` conversion here would double-count an h³ amplitude error.

    Returns
    -------
    (np.ndarray, np.ndarray, float)
        ``(xip, xim, As)`` — the σ8-matched amplitude is returned for
        assertion by the cross-check test.
    """
    import camb
    import pyccl as ccl

    sigma8 = config.sigma8()
    As = camb_As_for_sigma8(config, sigma8, kmax=kmax)
    results = camb.get_results(make_camb_params(config, As, nonlinear=True, kmax=kmax))
    interp = results.get_matter_power_interpolator(
        nonlinear=True, hubble_units=False, k_hunit=False
    )
    k = np.geomspace(1e-4, kmax * config.h, n_k)  # 1/Mpc
    z = np.linspace(0.0, PK_ZMAX, PK_NZ)  # the grid make_camb_params computed
    pk = interp.P(z, k)  # (n_z, n_k), Mpc^3
    a = 1.0 / (1.0 + z)
    order = np.argsort(a)  # Pk2D wants ascending scale factor
    pk2d = ccl.Pk2D(
        a_arr=a[order], lk_arr=np.log(k), pk_arr=np.log(pk[order]), is_logp=True
    )

    cosmo = ccl_cosmology(config.ccl_params(), config)
    z_nz, nz_vals = nz
    lens = ccl.WeakLensingTracer(cosmo, dndz=(np.asarray(z_nz), np.asarray(nz_vals)))
    ells = np.unique(np.geomspace(2, ell_max, n_ell).astype(int)).astype(float)
    cl = ccl.angular_cl(cosmo, lens, lens, ells, p_of_k_a=pk2d)
    theta_deg = np.asarray(theta_arcmin) / 60.0
    xip = ccl.correlation(cosmo, ell=ells, C_ell=cl, theta=theta_deg, type="GG+")
    xim = ccl.correlation(cosmo, ell=ells, C_ell=cl, theta=theta_deg, type="GG-")
    return xip, xim, As
