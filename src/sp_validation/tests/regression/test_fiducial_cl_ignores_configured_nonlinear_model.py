"""The pseudo-Cl covariance must use the configured CAMB nonlinear model.

The paper config requests mead2020_feedback at log T_AGN=7.8. Develop builds
its fiducial spectrum with CCL; the tomography branch uses get_fiducial_cl's
CAMB backend. Neither path may silently substitute a different nonlinear model.
Configuration routing is covered by test_pseudo_cl_producers_ignore_cosmo_val_config.
"""

from pathlib import Path

import numpy as np
import pyccl as ccl
import pytest
import yaml
from cs_util.cosmo import get_cosmo, get_theo_c_ell

import sp_validation
from sp_validation import pseudo_cl

ELL_CHECK = np.array([100.0, 300.0, 1000.0, 2000.0])
LMAX = 2000


def _config_cosmo_params():
    root = Path(__file__).resolve().parents[4]
    # When testing another checkout, read its config alongside its imported code.
    code_root = Path(sp_validation.__file__).resolve().parents[2]
    if code_root != root:
        root = code_root
    path = root / "papers/cosmo_val/config/config.yaml"
    return yaml.safe_load(path.read_text())["cosmo_val"]["cosmo_params"]


def _nz():
    z = np.linspace(0.01, 2.5, 250)
    return z, np.exp(-(((z - 0.7) / 0.3) ** 2))


def _pipeline_fiducial_cl(cosmo):
    """Use the fiducial-theory entry point selected by the active branch."""
    z, nz = _nz()
    ell = np.arange(1, LMAX + 1)
    if hasattr(pseudo_cl, "get_fiducial_cl"):
        cl = pseudo_cl.get_fiducial_cl(z, nz, LMAX, cosmo)["W1xW1"]
    else:
        cl = get_theo_c_ell(ell=ell, z=z, nz=nz, backend="ccl", cosmo=cosmo)["W1xW1"]
    return np.interp(ELL_CHECK, ell, cl)


def _reference_cl(params):
    """Independent CCL projection using CAMB's configured nonlinear P(k)."""
    z, nz = _nz()
    base = get_cosmo(**{k: v for k, v in params.items() if k != "extra_params"})
    cosmo = ccl.Cosmology(
        Omega_c=base["Omega_c"],
        Omega_b=base["Omega_b"],
        h=base["h"],
        sigma8=base["sigma8"],
        n_s=base["n_s"],
        m_nu=base["m_nu"],
        transfer_function="boltzmann_camb",
        matter_power_spectrum="camb",
        extra_parameters=params["extra_params"],
    )
    tracer = ccl.WeakLensingTracer(cosmo, dndz=(z, nz))
    return ccl.angular_cl(cosmo, tracer, tracer, ELL_CHECK)


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="#381: fiducial C_ell ignores AGN feedback",
)
def test_fiducial_cl_does_not_ignore_configured_agn_feedback():
    """The fiducial C_ell must change when the configured log T_AGN changes.

    Shifting log T_AGN from 7.3 to 8.3 moves HMCode-2020 baryonic suppression
    by tens of percent at k ~ 1-10 h/Mpc, hence several percent at ell ~ 1000.
    A bit-identical spectrum cannot be the configured mead2020_feedback model.
    Read the parameter's name from the config so correcting its spelling does
    not leave this test exercising an obsolete key.
    """
    params = _config_cosmo_params()
    camb = params["extra_params"]["camb"]
    feedback_key = next(k for k in camb if k.startswith("HMCode_log"))

    def with_feedback(log_temperature):
        settings = {**camb, feedback_key: log_temperature}
        varied = {**params, "extra_params": {"camb": settings}}
        return _pipeline_fiducial_cl(get_cosmo(**varied))

    ratio = with_feedback(8.3) / with_feedback(7.3)
    assert np.max(np.abs(ratio - 1)) > 0.01, (
        f"fiducial C_ell ignores {feedback_key}: "
        f"C(8.3)/C(7.3) at ell={ELL_CHECK.tolist()} = {ratio.tolist()}"
    )


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="#381: configured nonlinear model is ignored",
)
def test_fiducial_cl_does_not_substitute_halofit_for_configured_feedback():
    """The fiducial C_ell must match CAMB mead2020_feedback(log T_AGN=7.8).

    The independent CCL reference uses matter_power_spectrum='camb' and the
    config's own extra_parameters: the model requested for the covariance.
    A 2% tolerance exceeds CCL-versus-CAMB Limber scatter for this smooth n(z),
    but is well below the nonlinear-model discrepancy at ell ~ 1000-2000.
    """
    params = _config_cosmo_params()
    got = _pipeline_fiducial_cl(get_cosmo(**params))
    ref = _reference_cl(params)
    ratio = got / ref
    assert np.allclose(ratio, 1, rtol=0.02), (
        "fiducial C_ell is not the configured mead2020_feedback(7.8) model: "
        f"pipeline/reference at ell={ELL_CHECK.tolist()} = {ratio.tolist()}"
    )
