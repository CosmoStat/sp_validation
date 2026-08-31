"""CAMB↔CCL theory cross-check (blinding PRD AC10–14).

The blinding shift is a difference of CCL theory vectors; downstream
inference runs CAMB (CosmoSIS). The shift only means what it is intended to
mean if CCL and CAMB predict the same ξ± at a fixed cosmology on our θ grid.
This module asserts that agreement between the two independent ξ± paths in
:mod:`sp_validation.blinding_theory`:

- **Path A** (:func:`~sp_validation.blinding_theory.xi_ccl`): CCL-native — CCL's
  Boltzmann-CAMB HMCode2020 P(k) route, projected by CCL Limber + FFTLog.
- **Path B** (:func:`~sp_validation.blinding_theory.xi_camb`): an independent
  pycamb run produces the HMCode2020 ``P(k, z)`` (σ8-matched ``A_s``),
  wrapped in a ``ccl.Pk2D`` and projected through the same CCL machinery.

Because both paths route their nonlinear P(k) through CAMB's HMCode2020 and
both project through CCL, a common Limber+FFTLog bug cancels: this test
validates the **P(k) recipe** and the **σ8/A_s amplitude convention**, not
the projection. The one convention subtlety it settles: the fiducial fixes
σ8 for CCL but A_s for CAMB; a nominal ``A_s = 2.1e-9`` leaves CAMB's σ8
≈3% off target — enough to blow a ξ± comparison to ~9–10%.
"""

import pathlib
import re

import numpy as np
import pytest

from sp_validation import blinding_theory as cm

# Tolerances (AC11/AC12). Observed floor on this fixture: see the printed
# numbers in the slow tests — the tolerances sit above the floor with
# headroom; version bumps move the floor and that is not a regression.
XIP_RTOL = 0.005  # 0.5 %
XIM_RTOL = 0.010  # 1.0 %
# ξ− crosses zero on this grid: the relative assertion applies only where
# |ξ−| exceeds an absolute floor set from the fixture's peak |ξ−|.
XIM_FLOOR_FRAC = 0.05


# --------------------------------------------------------------------------- #
# Deterministic fixture: one Gaussian source bin, 12-point θ grid
# --------------------------------------------------------------------------- #
def _gauss_nz(n=400):
    z = np.linspace(0.01, 3.0, n)
    nz = np.exp(-0.5 * ((z - 0.7) / 0.2) ** 2)
    return z, nz / np.trapezoid(nz, z)


THETA_ARCMIN = np.geomspace(5.0, 250.0, 12)


def _both_paths(config, **camb_kwargs):
    z, nz = _gauss_nz()
    xip_a, xim_a = cm.xi_ccl(
        config.ccl_params(), config, (z, nz), (z, nz), THETA_ARCMIN
    )
    xip_b, xim_b, As = cm.xi_camb(config, (z, nz), THETA_ARCMIN, **camb_kwargs)
    return (xip_a, xim_a), (xip_b, xim_b), As


def _assert_xi_agreement(a, b, label):
    (xip_a, xim_a), (xip_b, xim_b) = a, b
    assert np.all(xip_a > 0) and np.all(xip_b > 0)  # sensible cosmic shear
    rel_p = np.abs(xip_b - xip_a) / np.abs(xip_a)
    assert rel_p.max() < XIP_RTOL, (
        f"{label}: ξ+ max rel diff {rel_p.max():.3%} ≥ {XIP_RTOL:.1%}"
    )
    floor = XIM_FLOOR_FRAC * np.max(np.abs(xim_a))
    above = np.abs(xim_a) > floor
    rel_m = np.abs(xim_b - xim_a)[above] / np.abs(xim_a)[above]
    assert rel_m.max() < XIM_RTOL, (
        f"{label}: ξ− max rel diff {rel_m.max():.3%} ≥ {XIM_RTOL:.1%} (on |ξ−| > floor)"
    )
    # near the zero crossing: absolute agreement at the floor scale
    abs_m = np.abs(xim_b - xim_a)[~above]
    if len(abs_m):
        assert abs_m.max() < XIM_RTOL * floor, (
            f"{label}: ξ− absolute diff {abs_m.max():.3e} near zero crossing"
        )
    print(
        f"\n{label}: ξ+ max rel {rel_p.max():.3%}; "
        f"ξ− max rel {rel_m.max():.3%} (above floor, "
        f"{above.sum()}/{len(above)} points)"
    )


# --------------------------------------------------------------------------- #
# AC10: σ8/A_s reconciliation
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_ac10_sigma8_As_reconciliation():
    """(a) nominal A_s leaves CAMB's σ8 >2% off target — the convention
    offset is real; (b) the closed-form rescale lands on target to <1e-4."""
    cfg = cm.TheoryConfig()
    target = cfg.sigma8()

    nominal = cm.camb_linear_sigma8(cfg, 2.1e-9)
    offset = abs(nominal / target - 1)
    print(f"\nAC10 nominal-A_s σ8 offset: {offset:.4f}")
    assert offset > 0.02

    As = cm.camb_As_for_sigma8(cfg, target)
    matched = cm.camb_linear_sigma8(cfg, As)
    print(f"AC10 σ8-matched residual: {abs(matched - target):.2e} (A_s={As:.4e})")
    assert abs(matched - target) < 1e-4


# --------------------------------------------------------------------------- #
# AC11 + AC12: ξ± agreement at and off the fiducial
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_ac11_xi_agreement_at_fiducial():
    cfg = cm.TheoryConfig()
    a, b, _ = _both_paths(cfg)
    _assert_xi_agreement(a, b, "AC11 fiducial")


@pytest.mark.slow
def test_ac12_xi_agreement_off_fiducial():
    """A representative in-envelope offset — the *shift* (a difference of two
    theory vectors) must not inherit a stack-disagreement bias."""
    cfg = cm.TheoryConfig.from_overrides({"S8": 0.80 + 0.075, "Omega_m": 0.30 - 0.05})
    a, b, _ = _both_paths(cfg)
    _assert_xi_agreement(a, b, "AC12 off-fiducial")


# --------------------------------------------------------------------------- #
# AC13: halofit token pinned to the inference config (fast)
# --------------------------------------------------------------------------- #
def test_ac13_halofit_token_matches_inference_config():
    """The blinding fiducial's CCL halofit token equals the CosmoSIS
    inference config's ``halofit_version`` — asserted against the config
    file itself. All three blinding backends share one recipe by
    construction and would agree with each other while jointly diverging
    from the inference stack, so this cannot be caught by the cross-backend
    test and is asserted independently here."""
    ini = (
        pathlib.Path(__file__).resolve().parents[3]
        / "cosmo_inference"
        / "cosmosis_config"
        / "templates"
        / "cosmosis_pipeline_A_ia_cell.ini"
    )
    match = re.search(r"^halofit_version\s*=\s*(\S+)", ini.read_text(), re.MULTILINE)
    assert match, f"no halofit_version in {ini}"
    inference_token = match.group(1)
    cfg = cm.TheoryConfig()
    assert cfg.ccl_halofit_version == inference_token
    # the two stack tokens denote ONE recipe; a divergence is a config bug
    assert cfg.camb_halofit_version == cfg.ccl_halofit_version
    # #280: the shipped Boltzmann backend is CAMB-through-CCL, matching the
    # CosmoSIS+CAMB inference stack — one power-spectrum path. The cross-check
    # tests above (AC10–12, 14) all run at this default configuration.
    assert cfg.transfer_function == "boltzmann_camb"


# --------------------------------------------------------------------------- #
# AC14: fast smoke — broken wiring caught in the fast suite
# --------------------------------------------------------------------------- #
def test_ac14_crosscheck_smoke():
    """Both paths run at coarse resolution: finite, positive,
    few-percent-agreeing ξ+, and a σ8-matched A_s in a sane range."""
    cfg = cm.TheoryConfig()
    z, nz = _gauss_nz(n=150)
    theta = np.geomspace(10.0, 100.0, 4)
    xip_a, _ = cm.xi_ccl(cfg.ccl_params(), cfg, (z, nz), (z, nz), theta)
    xip_b, _, As = cm.xi_camb(
        cfg, (z, nz), theta, n_ell=120, ell_max=30000, kmax=10.0, n_k=200
    )
    assert np.all(np.isfinite(xip_a)) and np.all(np.isfinite(xip_b))
    assert np.all(xip_a > 0) and np.all(xip_b > 0)
    assert 1e-9 < As < 3e-9
    rel = np.abs(xip_b - xip_a) / np.abs(xip_a)
    assert rel.max() < 0.05, f"smoke ξ+ rel diff {rel.max():.3%} unexpectedly large"
