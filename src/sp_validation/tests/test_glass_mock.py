"""Characterization tests for the GLASS mock generation core.

Cail's spec: *a strong test that reproduces the exact same mock so config
values can't shift under us.* The true map level needs GLASS, which is not in
the production container (CAMB/healpy/pyccl are, GLASS is not). So this file
characterizes the mock at two depths:

1. **Cosmology reference (CAMB, no GLASS).** ``build_camb_params`` is the
   cosmological heart of the mock — Planck18 densities, ``sigma8``-rescaled
   ``As``, HMCode AGN feedback. We pin its outputs (rescaled ``As``, realized
   ``sigma8``, ``H0``/``omch2``/``ombh2``/``ns``, and a small CAMB matter
   ``P(k)``) against a committed reference (``data/glass_mock_camb_reference``)
   via ``np.allclose``. *Any* drift in a config value — h, Om, Ob, ns, sigma8,
   mnu, log_T_AGN, kmax — moves the rescaled ``As`` or the spectrum and turns
   this red. This is the load-bearing "config can't shift" guarantee, and it
   runs wherever CAMB is installed.

2. **Map self-consistency (GLASS-gated).** Where GLASS is installed, generate
   the matter/lensing maps for a tiny config + fixed seed and assert
   same-seed → identical maps, plus that the config values flow through to the
   CAMB params unchanged. When a GLASS-capable image lands, promote this to a
   committed-map ``np.allclose`` check against a reference ``.npy``.

The reference was generated in the production container (CAMB 1.6.5); the
tolerance below absorbs cross-version CAMB float drift while still catching a
genuine config change, which moves these numbers by parts in 10⁰–10⁻³.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sp_validation.glass_mock import (
    GlassMockConfig,
    build_camb_params,
    camb_sigma8,
)

camb = pytest.importorskip("camb")

def _have(module):
    """True if ``module`` is importable. False if it or a parent is missing."""
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        # find_spec imports the parent package, which raises if it is absent.
        return False


HAVE_GLASS = _have("glass")

# The map path needs more than GLASS itself: build_shells and
# Cosmology_from_camb import ``cosmology.compat.camb``, which ships with the
# ``cosmology`` API package rather than with GLASS. Gate on it separately so a
# missing dependency reports as a skip instead of masquerading as the (real,
# but different) glass/cosmology API incompatibility tracked in the xfail below.
HAVE_COSMOLOGY_CAMB = _have("cosmology.compat.camb")

REFERENCE = Path(__file__).parent / "data" / "glass_mock_camb_reference.npz"

# Reference config: the small, fast realization the reference was built from.
# nside=16 keeps CAMB cheap; cosmology is config-default (UNIONS fiducial).
REF_CONFIG = GlassMockConfig(nside=16)

# Cross-version CAMB float tolerance. Scalars and P(k) reproduce to ~1e-12
# within a version; 1e-6 relative leaves headroom for a CAMB point release
# while a real config change moves these by >>1e-3.
RTOL = 1e-6
ATOL = 0.0


def _camb_pk(pars, config):
    """Small fixed-grid matter P(k) at z=0 — a cosmology-sensitive fingerprint."""
    pars.set_matter_power(redshifts=[0.0], kmax=config.kmax, nonlinear=True)
    kh, _z, pk = camb.get_results(pars).get_matter_power_spectrum(
        minkh=1e-3, maxkh=1.0, npoints=8
    )
    return kh, pk[0]


def test_reference_exists():
    """Guard the guard: a missing reference must fail loudly, not skip."""
    assert REFERENCE.exists(), f"committed reference missing: {REFERENCE}"


def test_config_defaults_are_planck18_fiducial():
    """The config surface holds the exact fiducial values; freeze them here."""
    cfg = GlassMockConfig()
    assert cfg.h == 0.6766
    assert cfg.Om == 0.30966
    assert cfg.Ob == 0.04897
    assert cfg.ns == 0.9665
    assert cfg.sigma8 == 0.8102
    assert cfg.mnu == 0.06
    assert cfg.log_T_AGN == 7.8
    assert cfg.dx == 200.0
    assert cfg.zmax == 3.0
    assert cfg.lmax == cfg.nside
    # CDM density is matter minus baryons (pre-neutrino).
    assert np.isclose(cfg.Oc, cfg.Om - cfg.Ob)


def test_camb_params_match_committed_reference():
    """The CAMB cosmology reproduces the committed reference to RTOL.

    This is the strong characterization: it reproduces the cosmological core of
    the mock and asserts every derived number against a fixed reference, so a
    config drift cannot pass unnoticed.
    """
    ref = np.load(REFERENCE)
    pars = build_camb_params(REF_CONFIG)
    kh, pk = _camb_pk(pars, REF_CONFIG)

    assert np.allclose(pars.InitPower.As, ref["As_scaled"], rtol=RTOL, atol=ATOL)
    assert np.allclose(camb_sigma8(pars), ref["sigma8"], rtol=RTOL, atol=ATOL)
    assert np.allclose(pars.H0, ref["H0"], rtol=RTOL, atol=ATOL)
    assert np.allclose(pars.omch2, ref["omch2"], rtol=RTOL, atol=ATOL)
    assert np.allclose(pars.ombh2, ref["ombh2"], rtol=RTOL, atol=ATOL)
    assert np.allclose(pars.InitPower.ns, ref["ns"], rtol=RTOL, atol=ATOL)
    assert np.allclose(kh, ref["kh"], rtol=RTOL, atol=ATOL)
    assert np.allclose(pk, ref["pk"], rtol=RTOL, atol=ATOL)


def test_sigma8_rescaling_hits_target():
    """``As`` is rescaled so realized ``sigma8(z=0)`` equals the target."""
    pars = build_camb_params(REF_CONFIG)
    assert np.isclose(camb_sigma8(pars), REF_CONFIG.sigma8, rtol=1e-8)


def test_config_change_breaks_reference():
    """A changed config value must NOT reproduce the reference (test has teeth).

    Bumping ``sigma8`` rescales ``As`` and moves the whole spectrum; if this
    still matched the reference, the characterization above would be vacuous.
    """
    ref = np.load(REFERENCE)
    shifted = GlassMockConfig(nside=16, sigma8=0.85)
    pars = build_camb_params(shifted)
    # atol=0: As is O(1e-9), so the np.allclose default atol=1e-8 would call
    # any two such values "close". The characterization above uses atol=0 too.
    assert not np.allclose(pars.InitPower.As, ref["As_scaled"], rtol=RTOL, atol=0.0)


# --- map-level self-consistency (GLASS-gated) -------------------------------


@pytest.mark.skipif(not HAVE_GLASS, reason="GLASS not installed in this image")
@pytest.mark.skipif(
    not HAVE_COSMOLOGY_CAMB,
    reason="cosmology.compat.camb not installed in this image",
)
@pytest.mark.xfail(
    reason=(
        "glass_mock map path is incompatible with the installed glass/cosmology "
        "API: cosmology.Cosmology.from_camb returns a CambCosmology lacking "
        "comoving_distance, which glass.distance_grid / MultiPlaneConvergence "
        "require. The map path was never exercised before GLASS was added to the "
        "image. Fix = pin a compatible glass+cosmology pair (or adapt the API "
        "calls) and verify in the fresh image; then drop this xfail. "
        "See fiber shapepipe/sp_validation glass-cosmology-api-pin."
    ),
    strict=False,
    raises=AttributeError,
)
def test_matter_maps_are_seed_deterministic():
    """Same config + seed → bit-identical matter/lensing maps.

    When GLASS lands in the image, replace/augment this with a committed-map
    ``np.allclose`` reference for a full map-level characterization.
    """
    from sp_validation.glass_mock import (
        build_shells,
        generate_matter_maps,
        matter_shell_cls,
    )

    cfg = GlassMockConfig(nside=16, dx=300.0, zmax=1.0, seed=7)
    pars = build_camb_params(cfg)
    shells = build_shells(cfg, pars)
    cls = matter_shell_cls(cfg, pars, shells)

    def run():
        return [
            tuple(m.copy() for m in maps)
            for maps in generate_matter_maps(cfg, pars, shells, cls)
        ]

    first = run()
    second = run()

    assert len(first) == len(second) > 0
    for (d1, k1, g1a, g2a), (d2, k2, g1b, g2b) in zip(first, second):
        assert np.array_equal(d1, d2)
        assert np.array_equal(k1, k2)
        assert np.array_equal(g1a, g1b)
        assert np.array_equal(g2a, g2b)
