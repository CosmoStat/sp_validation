"""The leakage fit must use the science catalogue's weights."""

import numpy as np

from sp_validation import catalog_builders


def test_leakage_uses_iv_weights(monkeypatch):
    cat = {"w_des": np.array([1.0, 0.0]), "w_iv": np.array([0.0, 1.0])}
    data = {"e1_PSF": np.array([1.0, 2.0]), "e2_PSF": np.array([2.0, 3.0])}

    def fit(gal, num_bins, weight_type):
        weights = gal[f"w_{weight_type}"]
        return np.average(gal["e1"], weights=weights), 0.0

    monkeypatch.setattr(
        catalog_builders.calibration, "get_alpha_leakage_per_object", fit
    )
    result = catalog_builders.compute_PSF_leakage(
        cat,
        (np.array([1.0, 3.0]), np.array([0.0, 0.0])),
        data,
        np.array([0, 1]),
        weight_type="iv",
    )
    assert result == (3.0, 0.0)
