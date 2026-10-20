"""Least-squares PSF-leakage samples must follow the requested tau covariance."""

import inspect

import numpy as np
import pytest
from astropy.table import Table
from shear_psf_leakage.rho_tau_stat import PSFErrorFit, RhoStat, TauStat

from sp_validation.rho_tau import get_samples_lsq

CFG = dict(min_sep=1.0, max_sep=30.0, nbins=1, sep_units="arcmin")
SIGMA_TH = 1e-3
SIGMA_JK = 1e-1


def _write_fixture(out):
    # One theta bin: rho_0 = rho_1 = rho_3 = 1, other rho and all tau = 0.
    # Each of alpha/beta/eta has independent unit response to a tau component.
    rd = {"theta": [10.0]}
    for i in range(6):
        rd[f"rho_{i}_p"] = [1.0 if i in (0, 1, 3) else 0.0]
    td = {"theta": [10.0]}
    for i in (0, 2, 5):
        td[f"tau_{i}_p"] = [0.0]
    Table(rd).write(out / "rho_stats_fit.fits")
    Table(td).write(out / "tau_stats_fit.fits")
    np.save(out / "cov_tau_fit_th.npy", np.eye(3) * SIGMA_TH**2)
    np.save(out / "cov_tau_fit_jk.npy", np.eye(3) * SIGMA_JK**2)
    np.save(out / "cov_rho_fit_jk.npy", np.zeros((6, 6)))


def _run(fitter, cov_type):
    np.random.seed(23)
    if "base_tau" in inspect.signature(get_samples_lsq).parameters:
        # Tomography branch takes separate rho/tau basenames and a draw count.
        return get_samples_lsq(fitter, "fit", "fit", cov_type=cov_type, nsamples=256)
    return get_samples_lsq(fitter, "test", "fit", cov_type=cov_type)


@pytest.fixture
def restore_random_state():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def test_lsq_posterior_width_follows_cov_type_not_cached_th(
    tmp_path, monkeypatch, restore_random_state
):
    """A jk request after a th request must use jk covariance, not cached th draws.

    This one-bin unit-response model has tau = 0 and diagonal covariance
    sigma_tau^2 I, so each parameter's posterior std is sigma_tau by
    construction: 1e-3 for th and 1e-1 for jk. Using the same output directory
    must not let a basename-only cache silently retain the previous method's
    alpha/beta/eta errors and xi_psf_sys uncertainty bands.
    """
    _write_fixture(tmp_path)
    rho = RhoStat(output=str(tmp_path), treecorr_config=CFG)
    tau = TauStat(output=str(tmp_path), treecorr_config=CFG)
    fitter = PSFErrorFit(rho, tau, str(tmp_path))
    original = fitter.get_least_squares_params_samples

    def small_sample(**kwargs):
        # Only reduce Monte Carlo work; leave loading and cache logic untouched.
        kwargs.update(n_samples=256, verbose=False)
        return original(**kwargs)

    monkeypatch.setattr(fitter, "get_least_squares_params_samples", small_sample)
    s_th, _, _ = _run(fitter, "th")
    std_th = s_th.std(axis=0)
    assert np.allclose(std_th, SIGMA_TH, rtol=0.3), std_th

    s_jk, _, _ = _run(fitter, "jk")
    std_jk = s_jk.std(axis=0)
    assert np.allclose(std_jk, SIGMA_JK, rtol=0.3), (
        f"jk after th returned posterior std {std_jk}; expected ~{SIGMA_JK}, "
        f"not cached th width ~{SIGMA_TH}"
    )
