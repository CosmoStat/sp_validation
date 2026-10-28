"""Validate data histogram selection and response inputs."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def script(monkeypatch):
    root = Path(__file__).resolve().parents[3] / "scripts/validation/response"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("data_snr_m", root / "data_snr_m.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_histogram_validation(script, tmp_path):
    response = [[[1, 0], [0, 2]], [[3, 0], [0, 4]]]
    data = {
        "edges": [10, 15, 20],
        "ensemble_response": {"available": True},
        "weight_verification": {"w_des_is_exact_production_weight": True},
        "per_bin": {
            "N": [2, 3],
            "sum_w": [1, 2],
            "R_total": response,
            "R_shear": response,
            "R_sel": np.zeros((2, 2, 2)).tolist(),
        },
        "totals": {"selected": 5, "sum_w": 3, "R_total": [[2, 0], [0, 3]]},
        "convention": "[lower, upper)",
    }
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))
    result = script.load_histogram(path, [10, 15, 20])
    np.testing.assert_array_equal(result["sum_w"], [1, 2])
    np.testing.assert_array_equal(result["total"], response)
    with pytest.raises(ValueError, match="edges differ"):
        script.load_histogram(path, [10, 16, 20])
    data["per_bin"]["R_sel"][0][0][0] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        script.load_histogram(path, [10, 15, 20])
