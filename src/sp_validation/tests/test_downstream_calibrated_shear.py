"""Real- and harmonic-space estimators share calibrated shear."""

from types import SimpleNamespace

import numpy as np
import pytest

from sp_validation.cosmo_val import pseudo_cl
from sp_validation.cosmo_val.core import CosmologyValidation


@pytest.mark.parametrize("response,mean_subtracted", [(0.5, False), (1.0, True)])
def test_pseudo_cl_uses_xi_shear(monkeypatch, response, mean_subtracted):
    catalog = np.array(
        [(1.0, 2.0, 1.0), (3.0, -1.0, 3.0)],
        dtype=[("e1", "f8"), ("e2", "f8"), ("w", "f8")],
    )
    if mean_subtracted:
        for col in ("e1", "e2"):
            catalog[col] -= np.average(catalog[col], weights=catalog["w"])
    obj = CosmologyValidation.__new__(CosmologyValidation)
    obj.cc = {
        "test": {"shear": {"e1_col": "e1", "e2_col": "e2", "w_col": "w", "R": response}}
    }
    obj._results = {"test": SimpleNamespace(dat_shear=catalog)}
    obj._c1 = {"test": np.average(catalog["e1"] / response, weights=catalog["w"])}
    obj._c2 = {"test": np.average(catalog["e2"] / response, weights=catalog["w"])}
    monkeypatch.setattr(pseudo_cl, "open_entry", lambda entry: catalog)
    xi = obj._calibrated_g("test")
    harmonic = obj._pseudo_cl_catalogue("test")
    np.testing.assert_allclose(harmonic["e1"], xi[0])
    np.testing.assert_allclose(harmonic["e2"], xi[1])
    if mean_subtracted:
        np.testing.assert_allclose(harmonic["e1"], catalog["e1"])
        np.testing.assert_allclose(harmonic["e2"], catalog["e2"])
    else:
        np.testing.assert_allclose(xi[0], [-3.0, 1.0])
