"""Fresh least-squares rho/tau leakage fits must be reproducible."""

from types import SimpleNamespace

import numpy as np
import pytest
from shear_psf_leakage.rho_tau_stat import PSFErrorFit

N_SAMPLES = 256


def _synthetic_fitter():
    """Build a two-parameter model with known rho/tau and diagonal covariance."""
    theta = np.logspace(0, 2, 8)
    rho0 = 1e-5 * theta**-0.8
    rho1 = 3e-7 * theta**-0.6
    rho2 = -2e-6 * theta**-0.7
    alpha, beta = 0.02, 1.5
    tau0 = alpha * rho0 + beta * rho2
    tau2 = alpha * rho2 + beta * rho1
    fit = PSFErrorFit.__new__(PSFErrorFit)
    fit.use_eta = False
    fit.use_fourth_moment = False
    fit.rho_stat_handler = SimpleNamespace(
        rho_stats={"theta": theta, "rho_0_p": rho0, "rho_1_p": rho1, "rho_2_p": rho2}
    )
    fit.tau_stat_handler = SimpleNamespace(tau_stats={"tau_0_p": tau0, "tau_2_p": tau2})
    rho_vec = np.concatenate([rho0, rho1, rho2])
    tau_vec = np.concatenate([tau0, tau2])
    fit.cov_rho = np.diag((0.1 * np.abs(rho_vec)) ** 2)
    fit.cov_tau = np.diag((0.1 * np.abs(tau_vec) + 1e-9) ** 2)
    return fit


@pytest.fixture
def restore_random_state():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="unseeded draws: least-squares draws use an unseeded random stream",
)
def test_lsq_samples_identical_for_identical_inputs(restore_random_state):
    """Two fresh fits must return identical samples and reported quantiles.

    The default cosmo_val rho_tau_method is lsq. Its alpha/beta/eta quantiles
    and xi_psf_sys mean/quantiles depend on these draws, so an unseeded global
    stream makes uncached reruns disagree. Exact equality is expected because
    the fit should be deterministic for identical inputs with a fixed internal
    Monte Carlo stream. Even 256 draws suffice to detect an unseeded stream;
    the test deliberately does not reset the stream between the two calls.
    """
    s1, r1, _ = _synthetic_fitter().get_least_squares_params_samples(
        npatch=None, n_samples=N_SAMPLES, verbose=False
    )
    s2, r2, _ = _synthetic_fitter().get_least_squares_params_samples(
        npatch=None, n_samples=N_SAMPLES, verbose=False
    )
    width = r1[2] - r1[0]
    dmed = np.abs(r1[1] - r2[1]) / width
    assert np.array_equal(s1, s2), (
        "lsq draws differ between identical fresh fits: "
        f"max|dsample|={np.max(np.abs(s1 - s2)):.3e}; "
        f"median shift (alpha, beta) = {dmed} of the 68% width"
    )
    np.testing.assert_array_equal(r1, r2)
