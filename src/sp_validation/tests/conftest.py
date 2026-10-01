"""Fixtures shared across test modules."""

from pathlib import Path

import numpy as np
import pytest

PURE_EB_XI = Path(__file__).parent / "data" / "pure_eb_xi_fixture.npz"


@pytest.fixture
def pure_eb_xi():
    """Committed fine-grid ξ± of the synthetic coherent-shear catalogue.

    Exact-binning integration grid [1, 300]′ in 600 bins with its pair weights
    and edges, and the requested edges of a [15, 70]′ reporting grid in 6 bins,
    keyed by ``b_modes.calculate_pure_eb_correlation``'s parameters.
    """
    with np.load(PURE_EB_XI) as npz:
        return {k: (v.item() if v.ndim == 0 else v) for k, v in npz.items()}


@pytest.fixture
def toy_theory(monkeypatch):
    """Blinds shift by the analytic ``_synthetic.toy_shear``."""
    from _synthetic import toy_shear

    from sp_validation import theory

    monkeypatch.setattr(theory, "shear", toy_shear)
