"""Fixtures shared across test modules."""

from pathlib import Path

import numpy as np
import pytest

PURE_EB_XI = Path(__file__).parent / "data" / "pure_eb_xi_fixture.npz"


@pytest.fixture
def pure_eb_xi():
    """Committed ξ± of the synthetic coherent-shear catalogue.

    Exact-binning reporting [15, 70]′ in 6 bins and integration [1, 300]′ in
    600 bins, keyed by ``b_modes.pure_eb_from_xi``'s parameters.
    """
    with np.load(PURE_EB_XI) as npz:
        return {k: (v.item() if v.ndim == 0 else v) for k, v in npz.items()}
