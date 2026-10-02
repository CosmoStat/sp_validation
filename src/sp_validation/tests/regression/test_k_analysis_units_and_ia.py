"""k-sensitivity calculations must run and implement the NLA shear spectrum.

CAMB is a toy backend with P_lin(k, z) = P_nl(k, z) = D(z)^2 P0(k),
D(z) = 1/(1+z), so the growth factor is known and no Boltzmann solve is needed.
The production cosmology keys, constants and units are left untouched: setup
failures can block the later IA assertions until the earlier defects are fixed.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import scipy.integrate as real_integrate

import sp_validation

REPO = Path(__file__).resolve().parents[4]
# Cross-checkout runs must exercise the package selected by PYTHONPATH.
ACTIVE_REPO = Path(sp_validation.__file__).resolve().parents[2]
if ACTIVE_REPO != REPO:
    REPO = ACTIVE_REPO

SCRIPT = REPO / "cosmo_inference" / "scripts" / "k_analysis.py"
A_IA = 0.83
# C1 = 5e-14 h^-2 Msun^-1 Mpc^3; rho_crit = 2.775e11 h^2 Msun Mpc^-3.
C1_RHOCRIT = 5e-14 * 2.77536627e11  # Dimensionless; h cancels.


class Reached(Exception):
    """Stop before an expensive calculation, without changing its inputs."""


def _growth(z):
    return 1.0 / (1.0 + np.asarray(z))


def _fake_camb():
    k = np.logspace(-4, np.log10(200.0), 300)
    p0 = 2e4 * (k / 0.02) / (1 + (k / 0.02) ** 2.5)  # Mpc^3
    state = {}

    class Results:
        def calc_power_spectra(self, pars):
            pass

        def _pk(self):
            z = state["z"]
            return k, z, _growth(z)[:, None] ** 2 * p0[None, :]

        def get_nonlinear_matter_power_spectrum(self, **kwargs):
            return self._pk()

        def get_linear_matter_power_spectrum(self, **kwargs):
            return self._pk()

    class Pars:
        def set_matter_power(self, redshifts, kmax):
            state["z"] = np.asarray(redshifts)

    mod = types.ModuleType("camb")
    mod.set_params = lambda **kwargs: Pars()
    mod.get_results = lambda pars: Results()
    return mod


def _load_script(monkeypatch):
    if not SCRIPT.exists():
        pytest.skip(f"k_analysis.py not found at {SCRIPT}")
    # Load shared dependencies before stubbing CAMB, so they cannot retain it.
    pytest.importorskip("cs_util.cosmo")
    monkeypatch.setitem(sys.modules, "camb", _fake_camb())
    spec = importlib.util.spec_from_file_location("k_analysis_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def nz_file(tmp_path):
    z = np.linspace(0.0, 4.5, 451)
    nz = np.exp(-0.5 * ((z - 0.7) / 0.25) ** 2)
    nz /= real_integrate.simpson(nz, x=z)
    path = tmp_path / "nz.txt"
    np.savetxt(path, np.c_[z, nz])
    return str(path)


@pytest.mark.xfail(
    strict=True,
    reason="#391: k_analysis reads nonexistent PLANCK18 As key",
)
def test_process_theta_reads_cosmology_keys_that_exist(monkeypatch, nz_file, tmp_path):
    """CAMB setup must use cs_util's PLANCK18 key A_s, not the nonexistent As.

    Correct behaviour gets as far as the comoving-distance quadrature; a
    sentinel stops there so this test needs no full xi calculation.
    """
    mod = _load_script(monkeypatch)

    def quad_sentinel(*args, **kwargs):
        raise Reached()

    monkeypatch.setattr(
        mod,
        "integrate",
        types.SimpleNamespace(quad=quad_sentinel, simpson=real_integrate.simpson),
    )
    with pytest.raises(Reached):
        mod.process_theta(5.0, nz_file, str(tmp_path / "out"))


@pytest.mark.xfail(
    strict=True,
    reason="#391: k_analysis setup and dimensional c block quad",
)
def test_comoving_distance_integrand_is_a_plain_float(monkeypatch, nz_file, tmp_path):
    """The c/H(z) comoving-distance quadrature must accept plain numeric inputs.

    Astropy's dimensional c Quantity cannot be handed to scipy.quad. A
    sentinel on simpson stops at the lensing kernel, proving quad completed.
    The earlier PLANCK18 key defect is deliberately not bypassed.
    """
    mod = _load_script(monkeypatch)

    def simpson_sentinel(*args, **kwargs):
        raise Reached()

    monkeypatch.setattr(
        mod,
        "integrate",
        types.SimpleNamespace(quad=real_integrate.quad, simpson=simpson_sentinel),
    )
    with pytest.raises(Reached):
        mod.process_theta(5.0, nz_file, str(tmp_path / "out"))


def _run_c_ell(monkeypatch, nz_file, tmp_path, ell=200.0, kmax=1e3):
    """Capture the unmodified nested C_ell before the expensive Hankel transform.

    Only CAMB is stubbed. No aliases or unit-stripping fixes are injected.
    Capture its locals to compare against independently calculated NLA values.
    """
    mod = _load_script(monkeypatch)
    grabbed = {}

    def tracer(frame, event, arg):
        if (
            event == "call"
            and frame.f_code.co_name == "xi"
            and frame.f_code.co_filename == str(SCRIPT)
        ):
            grabbed.update(frame.f_back.f_locals)
            raise Reached()
        return None

    previous_trace = sys.gettrace()
    sys.settrace(tracer)
    try:
        with pytest.raises(Reached):
            mod.process_theta(5.0, nz_file, str(tmp_path / "out"))
    finally:
        sys.settrace(previous_trace)
    c_ell = grabbed["C_ell"]
    inner = {}

    def tracer_locals(frame, event, arg):
        if frame.f_code is c_ell.__code__:

            def local(fr, ev, value):
                if ev == "return":
                    inner.update(fr.f_locals)
                return local

            return local
        return None

    c_no = c_ell(ell, kmax, False)
    sys.settrace(tracer_locals)
    try:
        c_ia = c_ell(ell, kmax, True)
    finally:
        sys.settrace(previous_trace)
    return c_ia, c_no, inner, grabbed


@pytest.mark.xfail(
    strict=True,
    reason="#391: k_analysis setup fails and NLA growth uses D^2",
)
def test_ia_growth_factor_is_d_not_d_squared(monkeypatch, nz_file, tmp_path):
    """The NLA growth factor is D(z), not the power ratio D(z)^2.

    The toy backend has P_lin(k,z) = D(z)^2 P0(k), D = 1/(1+z), so the
    reference is known by construction. CosmoSIS la_model takes the square
    root of the same power ratio. Setup defects must be fixed first.
    """
    _, _, loc, _ = _run_c_ell(monkeypatch, nz_file, tmp_path)
    np.testing.assert_allclose(
        loc["Dzs"],
        _growth(loc["z_valid"]),
        rtol=1e-3,
        err_msg="Dzs should be D(z)=1/(1+z), not D(z)^2",
    )


@pytest.mark.xfail(
    strict=True,
    reason="#391: k_analysis setup fails and NLA omits rho_crit",
)
def test_ia_amplitude_includes_rho_crit(monkeypatch, nz_file, tmp_path):
    """F(z) = -A_IA C1 rho_crit Omega_m / D(z), C1 rho_crit = 0.0139.

    C1 alone (5e-14 Mpc^3/Msun) has the wrong dimensions and is missing
    rho_crit. The reference is dimensionless and h-independent by
    construction. Setup defects must be fixed before reaching this check.
    """
    _, _, loc, captured = _run_c_ell(monkeypatch, nz_file, tmp_path)
    z = loc["z_valid"]
    f_ref = -A_IA * C1_RHOCRIT * captured["Omega_m"] / _growth(z)
    ratio = np.median(loc["P_ia"] / f_ref)
    assert ratio == pytest.approx(1.0, rel=0.05), (
        f"P_ia / (-A C1 rho_crit Om / D) = {ratio:.3e}"
    )


@pytest.mark.xfail(
    strict=True,
    reason="#391: k_analysis setup fails and omits the IG term",
)
def test_gi_term_counted_twice(monkeypatch, nz_file, tmp_path):
    """Shear-shear includes GI + IG, so the GI cross-term has a factor of two.

    Using the code's kernels and F, C(IA) - C(noIA) must equal the integral
    of c P/(H r^2) [2 W_g W_I F + W_I^2 F^2]. This algebraic reference is
    independent of the amplitude convention. Setup defects are not bypassed.
    """
    c_ia, c_no, loc, _ = _run_c_ell(monkeypatch, nz_file, tmp_path)
    pref = 299792.458 * loc["Pks"] / (loc["Hzs"] * loc["rzs"] ** 2)
    f, wg, wi, z = loc["P_ia"], loc["W_ggs"], loc["W_ias"], loc["z_valid"]
    gi1 = real_integrate.simpson(pref * wg * wi * f, x=z)
    ii = real_integrate.simpson(pref * wi**2 * f**2, x=z)
    expected = 2 * gi1 + ii
    got = c_ia - c_no
    assert got == pytest.approx(expected, rel=1e-6, abs=0), (
        f"C_IA - C_noIA = {got:.6e}, expected 2*GI+II = {expected:.6e}"
    )
