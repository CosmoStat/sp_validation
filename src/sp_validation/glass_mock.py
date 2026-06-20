"""GLASS mock generation core for UNIONS weak-lensing simulations.

The reproducibility surface of the UNIONS GLASS mocks lives here: the fixed
cosmology (Planck18 + HMCode AGN feedback, ``sigma8``-rescaled CAMB power
spectrum) and the lognormal matter / lensing-map generation that turns a seed
into a sky. Galaxy sampling and catalogue I/O are a thin layer downstream
(``scripts/glass_mock/make_unions_glass_sim.py``), so that the load-bearing
config can be characterized in isolation — see
``tests/test_glass_mock.py``.

This module also collects the reusable two-point and PSF-leakage helpers the
mock post-processing runners in ``scripts/glass_mock/`` share (configuration-
space ξ±, NaMaster pseudo-Cℓ, harmonic rho/tau, footprint masks).

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
    "downgrade_mask",
    "growth_factor",
    "ia_convergence",
    "create_mask_from_catalogue",
    "compute_two_point_xi",
    "compute_two_point_cl",
    "compute_two_point_cl_map",
    "get_n_gal_map",
    "compute_leakage_harmony",
    "TREECORR_CONFIG",
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
        f"As rescaling missed target sigma8: wanted {config.sigma8}, got {realized}"
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


# --- mask / galaxy-sampling helpers (used by the generation runner) ---------


def downgrade_mask(mask, nside):
    """Downgrade a HEALPix mask to a lower ``nside`` (>=0.75 -> 1, else 0).

    Returns ``mask`` unchanged when it is already at ``nside``.
    """
    import healpy as hp

    nside_mask = hp.get_nside(mask)
    if nside_mask == nside:
        return mask
    print(f"[!] Downgrading mask from Nside {nside_mask} to Nside {nside}.")
    print("[!] Pixels with values <0.75 will be set to 0.0")
    print("[!] Pixels with values >=0.75 will be set to 1.0")
    mask_down = hp.ud_grade(mask, nside)
    mask_down[mask_down < 0.75] = 0.0
    mask_down[mask_down >= 0.75] = 1.0
    print("[!] Done.")
    return mask_down


def _growth_E_sq(z, om0):
    """Flat-LCDM ``E^2(z) = H^2(z)/H0^2``."""
    return om0 * (1 + z) ** 3 + 1 - om0


def _growth_integrand(z, om0):
    """Growth-factor redshift integrand."""
    return (z + 1) / (_growth_E_sq(z, om0)) ** 1.5


def _growth_factor_single(z, om0):
    """Normalised linear growth factor at a single redshift."""
    import scipy as sp

    first = sp.integrate.quad(_growth_integrand, z, np.inf, args=(om0,))[0]
    second = sp.integrate.quad(_growth_integrand, 0, np.inf, args=(om0,))[0]
    return (_growth_E_sq(z, om0) ** 0.5) * first / second


def growth_factor(z, om0):
    """Normalised linear growth factor ``D(z)`` (scalar or array ``z``)."""
    if isinstance(z, (float, int)):
        return _growth_factor_single(z, om0)
    return np.array([_growth_factor_single(zi, om0) for zi in list(z)])


def ia_convergence(delta, shell, config):
    """Intrinsic-alignment contribution to the convergence (NLA model).

    Mirrors the production NLA prefactor: ``-A_ia * rho_c1 * Om0 / D(z_eff)``
    applied to the matter overdensity ``delta`` of one shell.
    """
    import astropy.constants as const
    import astropy.units as u

    Om0 = config.Om
    A_ia = config.ia_bias
    _, _, z_eff = shell
    c1 = 5e-14 * (u.Mpc**3.0) / u.solMass
    c1_cgs = (c1 * (1.0 / config.h) ** 2.0).cgs
    H0 = (100 * config.h * u.km / u.s / u.Mpc).to(u.s**-1)
    G = (const.G * u.m**3 / (u.kg * u.s**2)).cgs
    rho_crit = 3 * H0**2 / (8 * np.pi * G)
    rho_c1 = (c1_cgs * rho_crit).value
    prefactor = -A_ia * rho_c1 * Om0
    return prefactor / growth_factor(z_eff, Om0) * delta


def create_mask_from_catalogue(nside, path, output, ra_col="RA", dec_col="DEC"):
    """Write a HEALPix footprint mask from the populated pixels of a catalogue.

    Pixels containing at least one object get 1.0, everything else 0.0.
    """
    import healpy as hp
    from astropy.io import fits

    cat_gal = fits.getdata(path)
    ra = cat_gal[ra_col]
    dec = cat_gal[dec_col]

    theta = (90.0 - dec) * np.pi / 180.0
    phi = ra * np.pi / 180.0
    pix = hp.ang2pix(nside, theta, phi)

    _unique_pix, _idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
    n_gal = np.zeros(hp.nside2npix(nside))
    n_gal[np.unique(pix)] = np.bincount(idx_rep)
    mask = n_gal != 0

    hp.write_map(output, mask, dtype=np.float32, overwrite=True)


# --- two-point statistics on mock catalogues --------------------------------

# Configuration-space treecorr defaults for the GLASS-mock 2PCF.
TREECORR_CONFIG = {
    "ra_units": "degrees",
    "dec_units": "degrees",
    "min_sep": 1.0,
    "max_sep": 250.0,
    "nbins": 20,
    "sep_units": "arcmin",
    "num_threads": 4,
}


def _mock_powspace_bin(nside, lmin, n_bins):
    """Powspace NaMaster binning for the mock harmonic stats, via the primitive.

    Returns ``(bin, ell_eff, lmax, b_lmax)`` — the 4-tuple the mock harmonic
    helpers consume. The binning itself is the shared
    ``pseudo_cl.make_namaster_bin`` powspace scheme (``power=0.5`` reproduces the
    old square-root spacing exactly); ``lmax = 2 * nside`` / ``b_lmax = lmax - 1``
    come from ``pseudo_cl.pseudo_cl_geometry``. Note the mock's ``lmin`` floor is
    a call argument here, whereas the geometry helper fixes ``lmin = 8``; the two
    coincide for the production default, and this wrapper keeps the override.
    """
    from sp_validation.pseudo_cl import make_namaster_bin, pseudo_cl_geometry

    _lmin, lmax, b_lmax = pseudo_cl_geometry(nside)
    b = make_namaster_bin(lmin, lmax, b_lmax, "powspace", n_ell_bins=n_bins, power=0.5)
    return b, b.get_effective_ells(), lmax, b_lmax


def compute_two_point_xi(cat, config=None):
    """Configuration-space shear 2PCF (treecorr ``GGCorrelation``) for a mock."""
    import treecorr

    treecorr_config = TREECORR_CONFIG if config is None else config
    treecorr_cat = treecorr.Catalog(
        ra=cat["ra"],
        dec=cat["dec"],
        g1=cat["e1"],
        g2=cat["e2"],
        ra_units="degrees",
        dec_units="degrees",
    )
    gg = treecorr.GGCorrelation(treecorr_config)
    gg.process(treecorr_cat)
    return gg


def get_n_gal_map(nside, ra, dec):
    """Galaxy-count HEALPix map plus the unique-pixel bookkeeping arrays.

    The unweighted twin of the pseudo-Cl ``get_n_gal_map`` primitive: delegates
    to it with ``weights=None`` (counts). Imported lazily so this module keeps
    resolving in CAMB-only environments without the harmonic stack.
    """
    from sp_validation.pseudo_cl import get_n_gal_map as _get_n_gal_map

    return _get_n_gal_map(nside, ra, dec)


def compute_two_point_cl(cat, nside=1024, lmin=8, n_bins=32):
    """Catalogue-based NaMaster pseudo-Cl for a mock shear field.

    Returns ``(cl_coupled, cl_all)``, each with the ell axis prepended.

    Deliberately keeps its own ``NmtFieldCatalog`` construction rather than
    delegating to the ``pseudo_cl.get_pseudo_cls_catalog`` primitive: the mock
    field is UNWEIGHTED (``weights = ones``, no per-object shear weight) and this
    function returns the COUPLED pseudo-Cl alongside the decoupled one (both with
    an ell axis prepended), which the primitive's ``(ell_eff, cl_all, wsp)``
    contract does not expose. The shared piece — the powspace binning — is taken
    from ``_mock_powspace_bin`` so the bandpower drift is gone.
    """
    import pymaster as nmt

    e1 = cat["e1"]
    e2 = cat["e2"]
    ra = cat["ra"]
    dec = cat["dec"]
    del cat

    ra[ra < 0] += 360

    b, ell_eff, lmax, b_lmax = _mock_powspace_bin(nside, lmin, n_bins)

    factor = -1
    f_all = nmt.NmtFieldCatalog(
        positions=[ra, dec],
        weights=np.ones_like(e1),
        field=[e1, factor * e2],
        lmax=b_lmax,
        lmax_mask=b_lmax,
        spin=2,
        lonlat=True,
    )

    wsp = nmt.NmtWorkspace.from_fields(f_all, f_all, b)
    cl_coupled = nmt.compute_coupled_cell(f_all, f_all)
    cl_all = wsp.decouple_cell(cl_coupled)

    cl_coupled = np.concatenate([np.arange(1, lmax + 1)[np.newaxis, :], cl_coupled])
    cl_all = np.concatenate([ell_eff[np.newaxis, ...], cl_all])
    return cl_coupled, cl_all


def compute_two_point_cl_map(cat, nside=1024, lmin=8, n_bins=32):
    """Map-based NaMaster pseudo-Cl for a mock shear field.

    Bins the galaxies onto a HEALPix shear map (count-weighted mean per pixel)
    and runs the MCM on the resulting masked field. Returns
    ``(cl_coupled, cl_all)`` with the ell axis prepended.

    Deliberately keeps its own ``NmtField`` construction rather than delegating
    to the ``pseudo_cl.get_pseudo_cls_map`` primitive: the mock shear map is
    built from UNWEIGHTED galaxy counts (the mask is the count map and the
    per-pixel mean divides by counts, not summed shear weights), and this
    function returns the COUPLED pseudo-Cl alongside the decoupled one (both with
    an ell axis prepended), which the primitive's ``(ell_eff, cl_all, wsp)``
    contract does not expose. The shared pieces — the powspace binning and the
    galaxy-count map — come from ``_mock_powspace_bin`` and ``get_n_gal_map``.
    """
    import healpy as hp
    import pymaster as nmt

    e1 = cat["e1"]
    e2 = cat["e2"]
    ra = cat["ra"]
    dec = cat["dec"]
    del cat

    ra[ra < 0] += 360

    b, ell_eff, lmax, b_lmax = _mock_powspace_bin(nside, lmin, n_bins)

    factor = -1
    n_gal_map, unique_pix, _idx, idx_rep = get_n_gal_map(nside, ra, dec)
    mask = n_gal_map > 0

    shear_map_e1 = np.zeros(hp.nside2npix(nside))
    shear_map_e2 = np.zeros(hp.nside2npix(nside))
    shear_map_e1[unique_pix] += np.bincount(idx_rep, weights=e1)
    shear_map_e2[unique_pix] += np.bincount(idx_rep, weights=e2)
    shear_map_e1[mask] /= n_gal_map[mask]
    shear_map_e2[mask] /= n_gal_map[mask]

    f_all = nmt.NmtField(
        mask=n_gal_map,
        maps=[shear_map_e1, factor * shear_map_e2],
        lmax=b_lmax,
    )

    wsp = nmt.NmtWorkspace.from_fields(f_all, f_all, b)
    cl_coupled = nmt.compute_coupled_cell(f_all, f_all)
    cl_all = wsp.decouple_cell(cl_coupled)

    cl_coupled = np.concatenate([np.arange(1, lmax + 1)[np.newaxis, :], cl_coupled])
    cl_all = np.concatenate([ell_eff[np.newaxis, ...], cl_all])
    return cl_coupled, cl_all


def compute_leakage_harmony(cat, cat_star, nside=1024, lmin=8, n_bins=32):
    """Harmonic-space PSF-leakage rho_0 / tau_0 from galaxy + star catalogues.

    Returns ``(rho_cl, tau_cl)``: the PSF-PSF (rho_0) and galaxy-PSF (tau_0)
    decoupled pseudo-Cl, each with the effective-ell axis prepended.
    """
    import pymaster as nmt

    e1 = cat["e1"]
    e2 = cat["e2"]
    e1_psf = cat_star["E1_PSF_HSM"]
    e2_psf = cat_star["E2_PSF_HSM"]

    ra = cat["ra"]
    dec = cat["dec"]
    ra_star = cat_star["RA"]
    dec_star = cat_star["DEC"]
    del cat

    ra[ra < 0] += 360
    ra_star[ra_star < 0] += 360

    b, ell_eff, _lmax, b_lmax = _mock_powspace_bin(nside, lmin, n_bins)

    factor = -1
    f_psf = nmt.NmtFieldCatalog(
        positions=[ra_star, dec_star],
        weights=np.ones_like(ra_star),
        field=[e1_psf, factor * e2_psf],
        lmax=b_lmax,
        lmax_mask=b_lmax,
        spin=2,
        lonlat=True,
    )
    f_gal = nmt.NmtFieldCatalog(
        positions=[ra, dec],
        weights=np.ones_like(ra),
        field=[e1, factor * e2],
        lmax=b_lmax,
        lmax_mask=b_lmax,
        spin=2,
        lonlat=True,
    )

    # rho_0: PSF-PSF
    wsp = nmt.NmtWorkspace.from_fields(f_psf, f_psf, b)
    rho_cl = wsp.decouple_cell(nmt.compute_coupled_cell(f_psf, f_psf))
    rho_cl = np.concatenate([ell_eff[np.newaxis, ...], rho_cl])

    # tau_0: galaxy-PSF
    wsp = nmt.NmtWorkspace.from_fields(f_gal, f_psf, b)
    tau_cl = wsp.decouple_cell(nmt.compute_coupled_cell(f_gal, f_psf))
    tau_cl = np.concatenate([ell_eff[np.newaxis, ...], tau_cl])

    return rho_cl, tau_cl
