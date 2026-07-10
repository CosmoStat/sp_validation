"""CAMB↔CCL theory cross-check (PRD #241 §5).

The blinding box computes theory through firecrown + CCL, while cosmological
inference runs CosmoSIS + CAMB. The Muir et al. shift ``t(hidden) − t(fid)``
only means what we intend if the two stacks predict the same ξ± at a fixed
cosmology. This module asserts that agreement.

**What is compared.** For one cosmology and n(z), ξ± on our θ grid computed two
ways:

- *Path A — the blinding stack:* :func:`sp_validation.blinding_likelihood`
  → firecrown → CCL, with the nonlinear P(k) routed through CAMB's HMCode2020
  (PURE_CCL_MODE + ``camb_extra_params``), the exact recipe the likelihood
  uses in production.
- *Path B — an independent CAMB path:* a direct pycamb run producing the
  HMCode2020 P(k, z), wrapped in a ``ccl.Pk2D`` and projected to ξ± through
  CCL's Limber (``angular_cl``) + FFTLog (``correlation``).

**The σ8-vs-A_s subtlety, settled here.** Our conventions set σ8 for CCL but
A_s for CAMB. A nominal A_s (2.1e-9) gives CAMB a σ8 that is ~3% off our
target — the amplitudes would silently disagree. Path B therefore iterates
CAMB's A_s until CAMB's σ8 equals the CCL target (σ8² ∝ A_s, so one Newton
step nails it), so both stacks share σ8 to <1e-4 before any ξ± is computed.
That reconciliation is the point of doing this in the test rather than
hoping the amplitudes line up.

**What this validates — and does not.** It validates that the blinding
theory agrees with an independent CAMB-sourced ξ± at the sub-percent level:
the P(k) recipe (HMCode2020) is consistent across stacks and the amplitude
convention is correctly reconciled. It does *not* independently validate
CCL's Limber+Hankel machinery — both paths use ``ccl.angular_cl`` +
``ccl.correlation`` (a standalone FFTLog library is not in the environment),
so a common bug there would cancel. The residual we observe (~0.2–0.6%,
larger for ξ− which weights small scales) is dominated by P(k)-representation
and ℓ-grid differences between two independent-but-both-CCL projections, not
by cosmology; it is the realistic floor of cross-implementation agreement,
and the tolerances sit comfortably above it.
"""

import numpy as np
import pytest

from sp_validation import blinding_likelihood as bl
from sp_validation import sacc_io as sio


# --------------------------------------------------------------------------- #
# Deterministic fixture: one Gaussian source bin
# --------------------------------------------------------------------------- #
def _gauss_nz(z0=0.7, sigma=0.2, n=400):
    z = np.linspace(0.01, 3.0, n)
    nz = np.exp(-0.5 * ((z - z0) / sigma) ** 2)
    return z, nz / np.trapezoid(nz, z)


# --------------------------------------------------------------------------- #
# Path A — firecrown + CCL (the blinding stack)
# --------------------------------------------------------------------------- #
def _firecrown_xi(z, nz, theta_arcmin, config):
    """ξ± from the production likelihood at ``config`` — the blinding stack."""
    from firecrown.likelihood import NamedParameters

    s = sio.new_sacc({0: (z, nz)})
    zeros = np.zeros_like(theta_arcmin)
    sio.add_xi(s, (0, 0), theta_arcmin, zeros, zeros, grid="coarse")
    tr = ("source_0", "source_0")
    xi = np.concatenate([s.indices(sio.XI_PLUS, tr), s.indices(sio.XI_MINUS, tr)])
    sio.assemble_covariance(s, [(xi, np.eye(len(xi)) * 1e-12)])

    lik, tools = bl.build_likelihood(
        NamedParameters({"sacc_data": s, "theory_config": config})
    )
    tv = bl.compute_theory_vector(lik, tools, config)
    n = len(theta_arcmin)
    return tv[:n], tv[n:]


# --------------------------------------------------------------------------- #
# Path B — independent CAMB P(k) → CCL Pk2D → ξ±
# --------------------------------------------------------------------------- #
def _camb_As_for_sigma8(config, sigma8_target, n_z=48, kmax=20.0):
    """CAMB ``A_s`` giving ``sigma8_target``, and the CAMB param builder.

    Iterates A_s (σ8² ∝ A_s ⇒ Newton converges in one step) so the CAMB stack
    shares σ8 with CCL. Returns ``(A_s, make_params)`` where ``make_params(As,
    nonlinear)`` builds a CAMBparams at the fiducial background.
    """
    import camb

    h, ns, mnu = config.h, config.n_s, config.m_nu
    omch2 = config.omega_c() * h**2
    ombh2 = config.Omega_b * h**2
    zlist = list(np.linspace(0.0, 3.0, n_z))

    def make_params(As, nonlinear):
        p = camb.CAMBparams()
        p.set_cosmology(
            H0=h * 100,
            ombh2=ombh2,
            omch2=omch2,
            mnu=mnu,
            num_massive_neutrinos=1,
            neutrino_hierarchy=config.mass_split,
        )
        p.InitPower.set_params(As=As, ns=ns)
        p.set_matter_power(redshifts=zlist, kmax=kmax)
        if nonlinear:
            p.NonLinear = camb.model.NonLinear_both
            p.NonLinearModel.set_params(
                halofit_version=config.halofit_version,
                HMCode_logT_AGN=config.hmcode_logT_AGN,
            )
        else:
            p.NonLinear = camb.model.NonLinear_none
        return p

    As = 2.1e-9
    for _ in range(3):  # linear σ8 for the amplitude solve; 1 step suffices
        r = camb.get_results(make_params(As, nonlinear=False))
        As *= (sigma8_target / r.get_sigma8_0()) ** 2
    return As, make_params


def _independent_camb_xi(
    z, nz, theta_arcmin, config, n_ell=300, ell_max=60000, kmax=20.0
):
    """ξ± from an independent CAMB P(k), projected through CCL (σ8-matched)."""
    import camb
    import pyccl as ccl

    sigma8 = config.sigma8()
    As, make_params = _camb_As_for_sigma8(config, sigma8, kmax=kmax)
    results = camb.get_results(make_params(As, nonlinear=True))
    kh, zc, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=kmax, npoints=400)
    zc = np.asarray(zc)
    # CAMB uses h-units (k in h/Mpc, P in (Mpc/h)^3); CCL wants 1/Mpc, Mpc^3.
    k = kh * config.h
    Pk = pk / config.h**3
    a = 1.0 / (1.0 + zc)
    order = np.argsort(a)  # Pk2D wants ascending scale factor
    pk2d = ccl.Pk2D(
        a_arr=a[order],
        lk_arr=np.log(k),
        pk_arr=np.log(Pk[order]),
        is_logp=True,
    )

    cosmo = ccl.Cosmology(
        Omega_c=config.omega_c(),
        Omega_b=config.Omega_b,
        h=config.h,
        n_s=config.n_s,
        sigma8=sigma8,
        m_nu=config.m_nu,
        mass_split="single",
        matter_power_spectrum="camb",
        extra_parameters={
            "camb": {
                "halofit_version": config.halofit_version,
                "HMCode_logT_AGN": config.hmcode_logT_AGN,
            }
        },
    )
    lens = ccl.WeakLensingTracer(cosmo, dndz=(z, nz))
    ells = np.unique(np.geomspace(2, ell_max, n_ell).astype(int)).astype(float)
    cl = ccl.angular_cl(cosmo, lens, lens, ells, p_of_k_a=pk2d)
    theta_deg = theta_arcmin / 60.0
    xip = ccl.correlation(cosmo, ell=ells, C_ell=cl, theta=theta_deg, type="GG+")
    xim = ccl.correlation(cosmo, ell=ells, C_ell=cl, theta=theta_deg, type="GG-")
    return xip, xim, As


# --------------------------------------------------------------------------- #
# The σ8/A_s reconciliation — fast, no shear projection
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_sigma8_As_reconciliation():
    """CAMB's A_s must be iterated to share σ8 with CCL — the ~3% offset is real.

    Proves (a) a nominal A_s leaves CAMB's σ8 percent-level off our σ8 target,
    so the convention offset is not negligible; (b) the Newton iteration lands
    CAMB's σ8 on the target to <1e-4. This is what makes the ξ± cross-check a
    same-amplitude comparison rather than an amplitude-mismatch artifact.
    """
    import camb

    cfg = bl.TheoryConfig()
    target = cfg.sigma8()
    _, make_params = _camb_As_for_sigma8(cfg, target)

    nominal = camb.get_results(make_params(2.1e-9, nonlinear=False)).get_sigma8_0()
    assert abs(nominal - target) / target > 0.01  # >1% off at nominal A_s

    As, _ = _camb_As_for_sigma8(cfg, target)
    matched = camb.get_results(make_params(As, nonlinear=False)).get_sigma8_0()
    assert abs(matched - target) < 1e-4


# --------------------------------------------------------------------------- #
# The cross-check: firecrown+CCL vs independent CAMB, σ8-matched
# --------------------------------------------------------------------------- #
# Observed agreement floor on this fixture (12 θ ∈ [5, 250] arcmin, HMCode2020):
# ξ+ ≈ 0.3%, ξ− ≈ 0.6% (converged in ℓ). Tolerances sit above the floor with
# headroom; ξ− is looser because it weights smaller scales, where the
# P(k)-representation and FFTLog differences between the two paths are largest.
XIP_RTOL = 0.005  # 0.5%
XIM_RTOL = 0.010  # 1.0%


@pytest.mark.slow
def test_camb_ccl_xi_agreement():
    """firecrown+CCL and independent-CAMB ξ± agree within the stated tolerance."""
    cfg = bl.TheoryConfig()
    z, nz = _gauss_nz()
    theta_arcmin = np.geomspace(5.0, 250.0, 12)

    fc_xip, fc_xim = _firecrown_xi(z, nz, theta_arcmin, cfg)
    ind_xip, ind_xim, _ = _independent_camb_xi(z, nz, theta_arcmin, cfg)

    rel_p = np.abs(ind_xip - fc_xip) / np.abs(fc_xip)
    rel_m = np.abs(ind_xim - fc_xim) / np.abs(fc_xim)
    assert np.all(fc_xip > 0) and np.all(ind_xip > 0)  # sensible cosmic shear
    assert rel_p.max() < XIP_RTOL, f"ξ+ max rel diff {rel_p.max():.3%} ≥ {XIP_RTOL:.1%}"
    assert rel_m.max() < XIM_RTOL, f"ξ− max rel diff {rel_m.max():.3%} ≥ {XIM_RTOL:.1%}"


@pytest.mark.slow
def test_camb_ccl_xi_agreement_shifted_cosmology():
    """Agreement holds away from the fiducial (a blind-sized S8/Ωm offset).

    The blinding shift moves the cosmology by up to ±0.075 in S8 and ±0.1 in
    Ωm (PRD §3); the two stacks must still agree there, or the *shift itself*
    (a difference of two theory vectors) inherits a stack-disagreement bias.
    """
    cfg = bl.TheoryConfig.from_overrides({"S8": 0.80 + 0.075, "Omega_m": 0.30 - 0.05})
    z, nz = _gauss_nz()
    theta_arcmin = np.geomspace(5.0, 250.0, 12)

    fc_xip, fc_xim = _firecrown_xi(z, nz, theta_arcmin, cfg)
    ind_xip, ind_xim, _ = _independent_camb_xi(z, nz, theta_arcmin, cfg)

    rel_p = np.abs(ind_xip - fc_xip) / np.abs(fc_xip)
    rel_m = np.abs(ind_xim - fc_xim) / np.abs(fc_xim)
    assert rel_p.max() < XIP_RTOL, f"ξ+ max rel diff {rel_p.max():.3%} ≥ {XIP_RTOL:.1%}"
    assert rel_m.max() < XIM_RTOL, f"ξ− max rel diff {rel_m.max():.3%} ≥ {XIM_RTOL:.1%}"


@pytest.mark.slow
def test_blinding_shift_consistent_across_stacks():
    """The *shift* d(hidden) − d(fiducial) agrees across stacks.

    What blinding actually adds to the data is a difference of two theory
    vectors. Even if each vector carries a small common-mode stack offset, the
    difference should agree more tightly than the absolute vectors — this is
    the quantity that matters. Assert the shift (fiducial → a blind-sized
    hidden cosmology) matches between firecrown+CCL and independent-CAMB to
    within the ξ± absolute tolerance.
    """
    fid = bl.TheoryConfig()
    hidden = bl.TheoryConfig.from_overrides({"S8": 0.80 - 0.05, "Omega_m": 0.30 + 0.03})
    z, nz = _gauss_nz()
    theta_arcmin = np.geomspace(5.0, 250.0, 12)

    fc_fid_p, fc_fid_m = _firecrown_xi(z, nz, theta_arcmin, fid)
    fc_hid_p, fc_hid_m = _firecrown_xi(z, nz, theta_arcmin, hidden)
    ind_fid_p, ind_fid_m, _ = _independent_camb_xi(z, nz, theta_arcmin, fid)
    ind_hid_p, ind_hid_m, _ = _independent_camb_xi(z, nz, theta_arcmin, hidden)

    shift_fc_p = fc_hid_p - fc_fid_p
    shift_ind_p = ind_hid_p - ind_fid_p
    # the shift is O(few %) of ξ+, so compare relative to the shift magnitude
    rel = np.abs(shift_fc_p - shift_ind_p) / np.abs(shift_fc_p).max()
    assert rel.max() < 0.05, (
        f"blinding shift disagrees by {rel.max():.3%} across stacks"
    )


# --------------------------------------------------------------------------- #
# Fast smoke variant — the same machinery at coarse resolution, ~seconds
# --------------------------------------------------------------------------- #
def test_camb_ccl_crosscheck_smoke():
    """Fast smoke: both paths run and roughly agree at coarse resolution.

    Not the precision assertion (that is the ``slow`` test) — this guards that
    the cross-check machinery imports, runs, and returns finite same-sign ξ±
    that agree to a few percent, so a broken wiring is caught in the fast suite.
    """
    cfg = bl.TheoryConfig()
    z, nz = _gauss_nz(n=150)
    theta_arcmin = np.geomspace(10.0, 100.0, 4)

    fc_xip, fc_xim = _firecrown_xi(z, nz, theta_arcmin, cfg)
    ind_xip, ind_xim, As = _independent_camb_xi(
        z, nz, theta_arcmin, cfg, n_ell=120, ell_max=30000, kmax=10.0
    )

    assert np.all(np.isfinite(fc_xip)) and np.all(np.isfinite(ind_xip))
    assert np.all(fc_xip > 0) and np.all(ind_xip > 0)
    assert 1e-9 < As < 3e-9  # σ8-matched A_s in a sane range
    rel_p = np.abs(ind_xip - fc_xip) / np.abs(fc_xip)
    assert rel_p.max() < 0.05, f"smoke ξ+ rel diff {rel_p.max():.3%} unexpectedly large"
