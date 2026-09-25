"""The blinding theory against an independent CAMB oracle (AC10–13).

The blinding shift is a difference of CCL theory vectors; inference runs CAMB
(CosmoSIS). The shift means what it is meant to only if CCL and CAMB predict the
same ξ± at a fixed cosmology on our θ grid. This module compares
:func:`sp_validation.blinding_theory.xi_ccl` (CCL's Boltzmann-CAMB HMCode2020
P(k), projected by CCL's Limber + FFTLog) with an oracle written here: a direct
pycamb run of the HMCode2020 ``P(k, z)`` at a σ8-matched ``A_s``, wrapped in a
``ccl.Pk2D`` and projected by the same CCL machinery.

Both paths route P(k) through CAMB and project through CCL, so a shared
projection bug cancels: the comparison validates the P(k) recipe and the σ8/A_s
amplitude convention. The fiducial fixes σ8 for CCL but A_s for CAMB; a nominal
``A_s = 2.1e-9`` leaves CAMB's σ8 ≈3% off target, enough to move ξ± by ~10%.
"""

import dataclasses
import pathlib
import re

import numpy as np

from sp_validation import blinding_theory as bt

XIP_RTOL = 0.005
XIM_RTOL = 0.010
# ξ− crosses zero on this grid: the relative bound applies only where |ξ−|
# exceeds this fraction of its peak.
XIM_FLOOR_FRAC = 0.05

THETA_ARCMIN = np.geomspace(5.0, 250.0, 12)
PK_ZMAX, PK_NZ = 3.0, 48


def _gauss_nz(n=400):
    z = np.linspace(0.01, 3.0, n)
    nz = np.exp(-0.5 * ((z - 0.7) / 0.2) ** 2)
    return z, nz / np.trapezoid(nz, z)


# --------------------------------------------------------------------------- #
# The CAMB oracle
# --------------------------------------------------------------------------- #
def camb_params(config, As, *, nonlinear, kmax=20.0):
    """``CAMBparams`` at ``config``'s background, every field fed from one source."""
    import camb

    p = camb.CAMBparams()
    p.set_cosmology(
        H0=config.h * 100,
        ombh2=config.Omega_b * config.h**2,
        omch2=config.omega_c() * config.h**2,
        mnu=config.m_nu,
        num_massive_neutrinos=1,
        neutrino_hierarchy=config.mass_split,
        nnu=bt.NEFF,
        TCMB=bt.T_CMB,
    )
    p.set_dark_energy(w=config.w0, wa=config.wa, dark_energy_model="ppf")
    p.InitPower.set_params(As=As, ns=config.n_s)
    p.set_matter_power(redshifts=list(np.linspace(0.0, PK_ZMAX, PK_NZ)), kmax=kmax)
    if nonlinear:
        p.NonLinear = camb.model.NonLinear_both
        p.NonLinearModel.set_params(
            halofit_version=config.halofit_version,
            HMCode_logT_AGN=config.hmcode_logT_AGN,
        )
    else:
        p.NonLinear = camb.model.NonLinear_none
    return p


def camb_sigma8(config, As):
    import camb

    return float(
        camb.get_results(camb_params(config, As, nonlinear=False)).get_sigma8_0()
    )


def camb_As_for_sigma8(config, target, As_seed=2.1e-9):
    """σ8² ∝ A_s exactly, so one evaluation and one rescale land on target."""
    return As_seed * (target / camb_sigma8(config, As_seed)) ** 2


def xi_camb(config, nz, theta_arcmin, *, n_ell=300, ell_max=60000, kmax=20.0, n_k=400):
    """ξ± from a direct CAMB P(k), projected by CCL; returns ``(xip, xim, As)``."""
    import camb
    import pyccl as ccl

    As = camb_As_for_sigma8(config, config.sigma8())
    results = camb.get_results(camb_params(config, As, nonlinear=True, kmax=kmax))
    # CCL's native units already: k in 1/Mpc, P in Mpc³.
    interp = results.get_matter_power_interpolator(
        nonlinear=True, hubble_units=False, k_hunit=False
    )
    k = np.geomspace(1e-4, kmax * config.h, n_k)
    z = np.linspace(0.0, PK_ZMAX, PK_NZ)
    a = 1.0 / (1.0 + z)
    order = np.argsort(a)
    pk2d = ccl.Pk2D(
        a_arr=a[order],
        lk_arr=np.log(k),
        pk_arr=np.log(interp.P(z, k)[order]),
        is_logp=True,
    )
    cosmo = bt.ccl_cosmology(config.ccl_params(), config)
    lens = ccl.WeakLensingTracer(cosmo, dndz=nz)
    ells = np.unique(np.geomspace(2, ell_max, n_ell).astype(int)).astype(float)
    cl = ccl.angular_cl(cosmo, lens, lens, ells, p_of_k_a=pk2d)
    theta_deg = np.asarray(theta_arcmin) / 60.0
    xip = ccl.correlation(cosmo, ell=ells, C_ell=cl, theta=theta_deg, type="GG+")
    xim = ccl.correlation(cosmo, ell=ells, C_ell=cl, theta=theta_deg, type="GG-")
    return xip, xim, As


# --------------------------------------------------------------------------- #
# AC10–12
# --------------------------------------------------------------------------- #
def _assert_agreement(config, label):
    nz = _gauss_nz()
    xip_a, xim_a = bt.xi_ccl(config.ccl_params(), config, nz, nz, THETA_ARCMIN)
    xip_b, xim_b, _ = xi_camb(config, nz, THETA_ARCMIN)
    assert np.all(xip_a > 0) and np.all(xip_b > 0)
    rel_p = np.abs(xip_b - xip_a) / np.abs(xip_a)
    assert rel_p.max() < XIP_RTOL, f"{label}: ξ+ max rel diff {rel_p.max():.3%}"
    floor = XIM_FLOOR_FRAC * np.max(np.abs(xim_a))
    above = np.abs(xim_a) > floor
    rel_m = np.abs(xim_b - xim_a)[above] / np.abs(xim_a)[above]
    assert rel_m.max() < XIM_RTOL, f"{label}: ξ− max rel diff {rel_m.max():.3%}"
    assert np.all(np.abs(xim_b - xim_a)[~above] < XIM_RTOL * floor), label


def test_ac10_sigma8_As_reconciliation():
    """Nominal A_s misses σ8 by >2%; the closed-form rescale lands within 1e-4."""
    cfg = bt.TheoryConfig()
    assert abs(camb_sigma8(cfg, 2.1e-9) / cfg.sigma8() - 1) > 0.02
    assert (
        abs(camb_sigma8(cfg, camb_As_for_sigma8(cfg, cfg.sigma8())) - cfg.sigma8())
        < 1e-4
    )


def test_ac11_xi_agreement_at_fiducial():
    _assert_agreement(bt.TheoryConfig(), "fiducial")


def test_ac12_xi_agreement_off_fiducial():
    """An in-envelope point: the shift must not inherit a stack disagreement."""
    cfg = dataclasses.replace(bt.TheoryConfig(), S8=0.80 + 0.075, Omega_m=0.30 - 0.05)
    _assert_agreement(cfg, "off-fiducial")


# --------------------------------------------------------------------------- #
# AC13: the nonlinear recipe is the inference config's
# --------------------------------------------------------------------------- #
def test_ac13_halofit_token_matches_inference_config():
    """The blinding recipe is the CosmoSIS pipeline's, read from its config file.

    The CCL path and the CAMB oracle share the recipe by construction, so they
    would agree while jointly diverging from inference; only this lineage check
    catches that.
    """
    ini = (
        pathlib.Path(__file__).resolve().parents[3]
        / "cosmo_inference"
        / "cosmosis_config"
        / "templates"
        / "cosmosis_pipeline_A_ia_cell.ini"
    )
    match = re.search(r"^halofit_version\s*=\s*(\S+)", ini.read_text(), re.MULTILINE)
    assert match, f"no halofit_version in {ini}"
    cfg = bt.TheoryConfig()
    assert cfg.halofit_version == match.group(1)
    assert cfg.transfer_function == "boltzmann_camb"
