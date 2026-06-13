"""GLASS mock generation core for UNIONS weak-lensing simulations.

The reproducibility surface of the UNIONS GLASS mocks lives here: the fixed
cosmology (Planck18 + HMCode AGN feedback, ``sigma8``-rescaled CAMB power
spectrum) and the lognormal matter / lensing-map generation that turns a seed
into a sky. Galaxy sampling and catalogue I/O are a thin layer downstream
(``glass_mock/make_unions_glass_sim.py``), so that the load-bearing
config can be characterized in isolation — see
``tests/test_glass_mock.py``.

The ``glass`` dependency is imported lazily inside the map-generation functions
so that this module — and therefore the import guard — resolves in any
environment with CAMB, even one without GLASS installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "GlassMockConfig",
    "build_camb_params",
    "camb_sigma8",
    "build_shells",
    "matter_shell_cls",
    "generate_matter_maps",
]


@dataclass(frozen=True)
class GlassMockConfig:
    """Fixed configuration for a UNIONS GLASS mock realization.

    Every field here is a value that must not silently drift: the cosmology
    (a Planck18 base with explicit ``ns``/``sigma8``/AGN-feedback choices),
    the shell construction (``dx``, ``zmax``), and the resolution/seed. The
    characterization test pins these against a committed reference.

    Cosmology defaults reproduce the fiducial UNIONS v1.4.6.3 mock. The
    ``Planck18`` matter/baryon densities are read at construction time so the
    dataclass carries concrete numbers (not an astropy object) and round-trips
    cleanly to a reference.
    """

    # --- resolution / sampling ---
    nside: int = 32
    seed: int = 42
    # --- shell construction ---
    dx: float = 200.0  # comoving shell thickness [Mpc]
    zmax: float = 3.0
    # --- cosmology (Planck18 base, UNIONS fiducial overrides) ---
    h: float = field(default=0.6766)
    Om: float = field(default=0.30966)
    Ob: float = field(default=0.04897)
    ns: float = 0.9665
    sigma8: float = 0.8102
    mnu: float = 0.06
    log_T_AGN: float = 7.8
    As_init: float = 2.1e-9  # seed As before sigma8 rescaling
    kmax: float = 20.0
    # --- galaxy population (downstream of map generation) ---
    n_arcmin2: float = 6.0905
    sigma_e: float = 0.2684
    bias: float = 1.2  # constant linear galaxy bias b(z)
    phz_sigma_0: float = 0.03
    nbins: int = 1  # number of tomographic bins
    ia_bias: float | None = None

    @classmethod
    def from_planck18(cls, **overrides) -> "GlassMockConfig":
        """Build a config with matter/baryon densities pulled from Planck18.

        Keeps the dataclass defaults and the production script in sync with the
        single source of truth (``astropy.cosmology.Planck18``) without making
        the dataclass depend on astropy at import time.
        """
        from astropy.cosmology import Planck18

        base = dict(
            h=Planck18.H0.value / 100,
            Om=Planck18.Om0,
            Ob=Planck18.Ob0,
        )
        base.update(overrides)
        return cls(**base)

    @property
    def lmax(self) -> int:
        """``lmax`` is tied to ``nside`` in the production mocks."""
        return self.nside

    @property
    def Oc(self) -> float:
        """Cold dark matter density (CDM = matter - baryons), pre-neutrino."""
        return self.Om - self.Ob


def build_camb_params(config: GlassMockConfig):
    """Build the CAMB parameters for a mock, with ``As`` rescaled to ``sigma8``.

    This is the cosmological heart of the mock and is GLASS-free: it reproduces
    the exact CAMB setup of the production script — Planck18 densities, HMCode
    ``mead2020_feedback`` non-linear model with ``HMCode_logT_AGN``, neutrino
    mass subtracted from CDM, and an iterative ``As`` rescaling so the realized
    ``sigma8(z=0)`` matches ``config.sigma8``.

    Returns the CAMB ``CAMBparams`` object; ``camb_sigma8`` reads the realized
    value back out.
    """
    import camb

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=100 * config.h,
        omch2=config.Oc * config.h**2,
        ombh2=config.Ob * config.h**2,
        mnu=config.mnu,
    )
    # Neutrino density steals from CDM; re-set with the corrected omch2.
    Oc = config.Oc - pars.omeganu
    pars.set_cosmology(
        H0=100 * config.h,
        omch2=Oc * config.h**2,
        ombh2=config.Ob * config.h**2,
        mnu=config.mnu,
    )

    pars.InitPower.set_params(As=config.As_init, ns=config.ns)
    pars.WantTransfer = True
    pars.NonLinear = camb.model.NonLinear_both
    pars.NonLinearModel.set_params(
        halofit_version="mead2020_feedback",
        HMCode_logT_AGN=config.log_T_AGN,
    )
    pars.set_matter_power(nonlinear=True, kmax=config.kmax)

    # Rescale As ∝ sigma8^2 so the realized sigma8(z=0) hits the target.
    sigma8_temp = camb.get_results(pars).get_sigma8_0()
    As_scaled = config.As_init * (config.sigma8 / sigma8_temp) ** 2
    init_power = camb.InitialPowerLaw()
    init_power.set_params(As=As_scaled, ns=config.ns)
    pars.InitPower = init_power

    realized = camb_sigma8(pars)
    assert np.isclose(config.sigma8, realized), (
        f"As rescaling missed target sigma8: wanted {config.sigma8}, "
        f"got {realized}"
    )
    return pars


def camb_sigma8(pars) -> float:
    """Realized ``sigma8(z=0)`` for a CAMB parameter set."""
    import camb

    return float(camb.get_results(pars).get_sigma8_0())


def build_shells(config: GlassMockConfig, pars):
    """Linear-window matter shells on a comoving distance grid.

    Returns the GLASS shell list (``(z, w, zeff)`` windows). Lazily imports
    GLASS; only callable where GLASS is installed.
    """
    import glass
    from cosmology import Cosmology

    cosmo = Cosmology.from_camb(pars)
    zb = glass.distance_grid(cosmo, 0.0, config.zmax, dx=config.dx)
    return glass.linear_windows(zb)


def matter_shell_cls(config: GlassMockConfig, pars, shells):
    """Matter angular power spectra for the shells, from CAMB via GLASS."""
    import glass.ext.camb

    return glass.ext.camb.matter_cls(pars, config.lmax, shells)


def generate_matter_maps(config: GlassMockConfig, pars, shells, cls):
    """Yield ``(delta, kappa, gamma1, gamma2)`` HEALPix maps, shell by shell.

    This is the deterministic core of the mock at the map level: a seeded RNG
    drives lognormal matter fields (``ncorr=3``), accumulated into convergence
    via ``MultiPlaneConvergence`` and shear via ``shear_from_convergence``. For
    a fixed config + seed the emitted maps are bit-for-bit reproducible within
    a GLASS/NumPy version, which is what the self-consistency test asserts.

    The galaxy sampling that consumes these maps lives in the production script;
    keeping it out of this generator is what makes the map level testable.
    """
    import glass

    rng = np.random.default_rng(config.seed)
    fields = glass.lognormal_fields(shells)
    gls = glass.solve_gaussian_spectra(fields, cls)
    matter = glass.generate(fields, gls, config.nside, ncorr=3, rng=rng)
    convergence = glass.MultiPlaneConvergence(Cosmology_from_camb(pars))

    for i, delta_i in enumerate(matter):
        convergence.add_window(delta_i, shells[i])
        kappa_i = convergence.kappa
        gamma1_i, gamma2_i = glass.shear_from_convergence(kappa_i)
        yield delta_i, kappa_i, gamma1_i, gamma2_i


def Cosmology_from_camb(pars):
    """Thin indirection so the convergence cosmology is built once, lazily."""
    from cosmology import Cosmology

    return Cosmology.from_camb(pars)
