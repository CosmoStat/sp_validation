"""Derived Omega_m must include baryons and use the chain's reduced Hubble h."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import sp_validation

REPO = Path(__file__).resolve().parents[4]
# Cross-checkout runs must exercise the package selected by PYTHONPATH.
ACTIVE_REPO = Path(sp_validation.__file__).resolve().parents[2]
if ACTIVE_REPO != REPO:
    REPO = ACTIVE_REPO

SCRIPT = REPO / "cosmo_inference" / "scripts" / "chain_postprocessing.py"


def _load():
    if not SCRIPT.exists():
        pytest.skip(f"{SCRIPT} missing")
    spec = importlib.util.spec_from_file_location("chain_postprocessing", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.xfail(
    strict=True,
    reason="#391: Omega_m uses H0 units for h and omits baryons",
)
def test_omega_m_uses_reduced_h_and_includes_baryons():
    """Protects the Omega_m derived from a CosmoSIS chain. The UNIONS values_*.ini
    files sample h0 as the reduced Hubble constant h (prior [0.64, 0.82]) and
    sample both omch2 and ombh2, so the physical matter density is
    Omega_m = (omch2 + ombh2) / h0**2 (neglecting neutrinos, as sample_S8.py
    does). For omch2=0.12, ombh2=0.023, h0=0.7 that is 0.143/0.49 = 0.29184;
    the reference is computed by hand, independent of the package."""
    from getdist import MCSamples

    mod = _load()
    omch2, ombh2, h0 = 0.12, 0.023, 0.7
    n = 5
    samples = np.column_stack(
        [np.full(n, omch2), np.full(n, ombh2), np.full(n, h0)]
    ) + 1e-9 * np.random.default_rng(0).standard_normal((n, 3))
    chain = MCSamples(samples=samples, names=["omch2", "ombh2", "h0"])

    chain = mod.derive_parameter_Om(chain)
    got = float(np.mean(chain.getParams().OMEGA_M))
    expected = (omch2 + ombh2) / h0**2
    assert got == pytest.approx(expected, rel=1e-4), (
        f"OMEGA_M = {got:.4f}, expected (omch2+ombh2)/h0^2 = {expected:.4f}"
    )
