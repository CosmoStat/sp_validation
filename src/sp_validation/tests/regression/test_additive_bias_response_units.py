"""Additive bias must be subtracted in the units in which it was estimated."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from sp_validation.cosmo_val import CosmologyValidation


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=("#382: additive bias is subtracted in the wrong units"),
)
@pytest.mark.parametrize("response", [0.5, 0.8, 0.92])
def test_calibrated_shear_removes_weighted_mean_for_nonunit_response(response):
    """Protect the shear feeding xi+/- and aperture-mass statistics.

    Ellipticities are a known offset (0.002, -0.0015) plus antisymmetric
    shape-noise pairs with identical weights, so their weighted mean is exactly
    the offset. Removing additive bias must give <g>_w = (0, 0), independently
    of R: g = (e - <e>_w)/R. Estimating c = <e/R>_w but subtracting it from raw
    e leaves <e>_w (R-1)/R^2, a constant shear whose square contaminates xi+.
    This exercises the real calculate_additive_bias and _calibrated_g methods;
    only catalogue I/O and the unrelated constructor are replaced by state.
    """
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.28, size=(2000, 2))
    noise = np.concatenate([noise, -noise])
    weights = rng.uniform(0.5, 2.0, size=2000)
    weights = np.concatenate([weights, weights])
    offset = np.array([0.002, -0.0015])
    ellipticity = noise + offset
    version = "SP_test"
    cv = CosmologyValidation.__new__(CosmologyValidation)
    cv.versions = [version]
    cv.cc = {
        version: {
            "shear": {"e1_col": "e1", "e2_col": "e2", "w_col": "w", "R": response}
        }
    }
    cv._results = {
        version: SimpleNamespace(
            dat_shear={"e1": ellipticity[:, 0], "e2": ellipticity[:, 1], "w": weights},
            temporarily_read_data=nullcontext,
        )
    }
    g1, g2 = cv._calibrated_g(version)
    mean_g = [np.average(g1, weights=weights), np.average(g2, weights=weights)]
    np.testing.assert_allclose(
        mean_g,
        [0.0, 0.0],
        atol=1e-12,
        rtol=0,
        err_msg=f"R={response}: additive correction leaves a constant shear",
    )
