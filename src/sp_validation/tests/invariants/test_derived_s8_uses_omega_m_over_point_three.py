"""Derived S8 is Sigma8 times sqrt(Omega_m / 0.3), applied row-wise."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sp_validation

pytestmark = [pytest.mark.fast, pytest.mark.decision("inference.posterior_summary")]

REPO = Path(sp_validation.__file__).resolve().parents[2]
SCRIPT = REPO / "cosmo_inference" / "scripts" / "chain_postprocessing.py"


def _load_chain_postprocessing():
    if not SCRIPT.is_file():
        pytest.fail(f"chain postprocessing script missing: {SCRIPT}")
    spec = importlib.util.spec_from_file_location("chain_postprocessing", SCRIPT)
    if spec is None or spec.loader is None:
        pytest.fail(f"could not load chain postprocessing script: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_derived_s8_uses_omega_m_over_point_three():
    """The deterministic identity has false-alarm probability 0; rtol=1e-12 and
    atol=1e-14 allow only floating-point error on this fixture."""
    from getdist import MCSamples

    omega_m = np.array([0.3, 0.075, 1.2])
    sigma_8 = np.array([0.8, 0.6, 0.7])
    expected_s8 = np.array([0.8, 0.3, 1.4])

    class RecordingChain:
        def __init__(self):
            self.params = SimpleNamespace(OMEGA_M=omega_m, SIGMA_8=sigma_8)
            self.derived = None

        def getParams(self):
            return self.params

        def addDerived(self, values, name, label):
            self.derived = (np.asarray(values).copy(), name, label)

    module = _load_chain_postprocessing()
    lightweight_chain = RecordingChain()
    returned_chain = module.derive_parameter_S8(lightweight_chain)

    assert returned_chain is lightweight_chain
    got_s8, name, label = lightweight_chain.derived
    np.testing.assert_allclose(
        got_s8,
        expected_s8,
        rtol=1e-12,
        atol=1e-14,
        err_msg="derived S_8 must equal Sigma_8 * sqrt(Omega_m / 0.3) row-wise",
    )
    assert name == "S_8"
    assert label == "S_8"

    weights = np.random.default_rng(0).choice(
        np.array([0.5, 1.5, 3.0]), size=3, replace=False
    )
    getdist_chain = MCSamples(
        samples=np.column_stack((omega_m, sigma_8)),
        names=["OMEGA_M", "SIGMA_8"],
        weights=weights,
    )
    original_weights = getdist_chain.weights.copy()
    assert np.unique(original_weights).size == 3

    returned_chain = module.derive_parameter_S8(getdist_chain)

    assert returned_chain is getdist_chain
    np.testing.assert_allclose(
        getdist_chain.getParams().S_8,
        expected_s8,
        rtol=1e-12,
        atol=1e-14,
        err_msg="GetDist S_8 values must preserve the input row alignment",
    )
    np.testing.assert_array_equal(
        getdist_chain.weights,
        original_weights,
        err_msg="adding derived S_8 must leave GetDist sample weights unchanged",
    )
