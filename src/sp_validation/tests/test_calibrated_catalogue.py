"""cosmo_val reads a catalogue calibrated on disk as it is, with the weight its
response record names, and calibrates any other with its declared response."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import yaml

from sp_validation import response
from sp_validation.cosmo_val.core import CosmologyValidation
from sp_validation.tests.test_response import metacal_block, synthetic


def test_record_names_the_weight_and_the_shear_is_read_as_is(tmp_path):
    cal = response.calibrate_catalogue(
        synthetic(2000), metacal_block(weight_scheme="iv")
    )
    catalogue = tmp_path / "catalog"
    catalogue.mkdir()
    (catalogue / "calibrated.fits").touch()
    (catalogue / "response.json").write_text(
        json.dumps(response.record(cal["result"], "iv"))
    )
    (tmp_path / "dndz.txt").write_text("0.1 1.0\n")
    config = {
        "nz": {"subdir": str(tmp_path), "dndz": {"path": "dndz.txt"}},
        "paths": {"output": str(tmp_path / "out")},
        "release": {
            "blind": "none",
            "subdir": str(catalogue),
            "pipeline": "SP",
            "shear": {
                "path": "calibrated.fits",
                "response": "response.json",
                "e1_col": "e1",
                "e2_col": "e2",
            },
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    cv = CosmologyValidation(
        versions=["release"],
        catalog_config=str(tmp_path / "config.yaml"),
        npatch=1,
        theta_min=1.0,
        theta_max=250.0,
        nbins=20,
    )
    assert cv.cc["release"]["shear"]["w_col"] == "w_iv"

    data = dict(e1=cal["e"][0], e2=cal["e"][1], w_iv=cal["w"])
    cv._results = {
        "release": SimpleNamespace(dat_shear=data, temporarily_read_data=nullcontext)
    }
    g = cv._calibrated_g("release")
    np.testing.assert_array_equal(g[0], data["e1"])
    np.testing.assert_array_equal(g[1], data["e2"])


def test_scalar_response_entry_is_calibrated_as_one_sample():
    cv = object.__new__(CosmologyValidation)
    cv.versions = ["external"]
    cv.cc = {"external": {"shear": dict(e1_col="e1", e2_col="e2", w_col="w", R=0.8)}}
    e = np.array([[0.12, -0.04, 0.03], [0.08, 0.16, -0.02]])
    w = np.array([1.0, 2.0, 0.5])
    cv._results = {
        "external": SimpleNamespace(
            dat_shear=dict(e1=e[0], e2=e[1], w=w), temporarily_read_data=nullcontext
        )
    }
    cv.print_start = cv.print_done = cv.print_magenta = lambda *args: None
    g = np.array(cv._calibrated_g("external"))
    expected = (e - (e @ w / w.sum())[:, None]) / 0.8
    np.testing.assert_allclose(g, expected)
