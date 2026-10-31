"""Additive-bias producer and consumer must agree on component units."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from sp_validation.cosmo_val.core import CosmologyValidation


def test_des_additive_bias_producer_to_consumer():
    obj = object.__new__(CosmologyValidation)
    obj.versions = ["DES"]
    obj.cc = {
        "DES": {
            "shear": dict(
                e1_col="e1", e2_col="e2", w_col="w", R=1, R11="R11", R22="R22"
            )
        }
    }
    data = dict(
        e1=np.array([0.12, -0.04]),
        e2=np.array([0.08, 0.16]),
        w=np.ones(2),
        R11=np.full(2, 0.8),
        R22=np.full(2, 0.7),
    )
    obj._results = {
        "DES": SimpleNamespace(dat_shear=data, temporarily_read_data=nullcontext)
    }
    obj.print_start = obj.print_done = obj.print_magenta = lambda *args: None
    obj.calculate_additive_bias()
    np.testing.assert_allclose([obj.c1["DES"], obj.c2["DES"]], [0.04 / 0.8, 0.12 / 0.7])
    g = obj._calibrated_g("DES")
    np.testing.assert_allclose(np.average(g, axis=1, weights=data["w"]), 0, atol=1e-16)
