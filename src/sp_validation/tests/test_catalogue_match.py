import ast
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from sp_validation import catalogue_match as cm


def manifest(tmp_path, tile, match, publication="one"):
    path = tmp_path / publication / tile / "tile_make_cat.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"unit": tile}
    if match is not None:
        data["catalogue_match"] = match
    path.write_text(json.dumps(data))
    return path


def match(n_external=100, n_paired=80, n_dropped=0):
    return {
        "n_external": n_external,
        "n_paired": n_paired,
        "n_sextractor": n_paired + n_dropped,
        "n_dropped": n_dropped,
        "n_external_only": n_external - n_paired,
        "external_paired_fraction": n_paired / n_external if n_external else 0,
        "external_warning_budget": max(20, 0.02 * n_external + 50),
    }


def test_missing_empty_and_zero_row_tiles(tmp_path):
    missing = manifest(tmp_path, "001.001", None)
    empty_match = match(0, 0)
    empty_match["external_warning_budget"] = 20
    empty = manifest(tmp_path, "002.002", empty_match)
    output = tmp_path / "audit.csv"
    ids = [1.001, 1.001, 3.003]
    summary = cm.write_catalogue_match_audit(ids, [missing, empty], output)
    rows = {row["tile_id"]: row for row in csv.DictReader(output.open())}
    assert ids == [1.001, 1.001, 3.003]
    assert summary == {
        "tiles": 3,
        "missing_manifest": 1,
        "missing_catalogue_match": 1,
        "warnings": 0,
    }
    assert rows["001.001"]["catalogue_rows"] == "2"
    assert rows["001.001"]["n_external"] == ""
    assert rows["001.001"]["manifest_path"] == str(missing)
    assert rows["002.002"]["catalogue_rows"] == "0"
    assert rows["002.002"]["external_paired_fraction"] == "0"
    assert rows["003.003"]["manifest_path"] == ""


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_dropped", 1),
        ("n_external_only", 2),
        ("external_paired_fraction", 0.2),
        ("external_warning_budget", -1),
        ("external_warning_budget", float("nan")),
        ("external_warning_budget", True),
        ("n_paired", -1),
    ],
)
def test_invalid_metadata(tmp_path, field, value):
    data = match()
    data[field] = value
    with pytest.raises(ValueError):
        cm.read_catalogue_match_manifests([manifest(tmp_path, "001.001", data)])


def test_custom_published_budget(tmp_path):
    data = match()
    data["external_warning_budget"] = 7.5
    path = manifest(tmp_path, "001.001", data)
    assert cm.read_catalogue_match_manifests([path])["001.001"]["match"] == data


def test_warning_boundary(tmp_path):
    paths = [
        manifest(tmp_path, "001.001", match(100, 48)),
        manifest(tmp_path, "002.002", match(100, 47)),
    ]
    output = tmp_path / "audit.csv"
    assert cm.write_catalogue_match_audit([], paths, output)["warnings"] == 1
    rows = list(csv.DictReader(output.open()))
    assert [r["external_warning"] for r in rows] == ["False", "True"]


def test_conflicting_duplicate_and_unit(tmp_path):
    first = manifest(tmp_path, "001.001", match())
    same = manifest(tmp_path, "001.001", match(), "two")
    different = manifest(tmp_path, "001.001", match(100, 79), "three")
    assert len(cm.read_catalogue_match_manifests([first, same])) == 1
    with pytest.raises(ValueError, match="Conflicting"):
        cm.read_catalogue_match_manifests([first, different])
    same.write_text(json.dumps({"unit": "002.002"}))
    with pytest.raises(ValueError, match="disagree"):
        cm.read_catalogue_match_manifests([same])


def test_tile_ids():
    assert cm.normalize_tile_id(51.232) == "051.232"
    assert cm.normalize_tile_id(b"051.232") == "051.232"
    for bad in ("051.232extra", "51.23", float("nan"), -1, 51.2325, True):
        with pytest.raises(ValueError):
            cm.normalize_tile_id(bad)


def test_float32_tile_counts(tmp_path):
    ids = np.array([51.232, 51.232, 52.233], dtype=np.float32)
    output = tmp_path / "audit.csv"
    cm.write_catalogue_match_audit(ids, [], output)
    rows = {
        row["tile_id"]: int(row["catalogue_rows"])
        for row in csv.DictReader(output.open())
    }
    assert rows == {"051.232": 2, "052.233": 1}


def test_distinct_numeric_representations_are_combined(tmp_path):
    ids = np.array([51.232, 51.23201])
    output = tmp_path / "audit.csv"
    cm.write_catalogue_match_audit(ids, [], output)
    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 1
    assert rows[0]["tile_id"] == "051.232"
    assert rows[0]["catalogue_rows"] == "2"


def test_missing_unit_and_missing_vs_present_metadata(tmp_path):
    missing = manifest(tmp_path, "001.001", None)
    present = manifest(tmp_path, "001.001", match(), "two")
    with pytest.raises(ValueError, match="Conflicting"):
        cm.read_catalogue_match_manifests([missing, present])
    missing.write_text("{}")
    with pytest.raises(ValueError, match="Missing unit"):
        cm.read_catalogue_match_manifests([missing])


DECISION = "dr6_catalogue.catalogue_match_tile_selection"


def record_pins(root):
    """Return the decision's ``Values:`` pins from the composed ASTRA record."""
    analysis_id, decision_id = DECISION.split(".")
    project_root = root / "scratch/cdaley/unions_astra"
    project = yaml.safe_load((project_root / "astra.yaml").read_text())
    analysis_root = project_root / project["analyses"][analysis_id]["path"]
    spec = yaml.safe_load((analysis_root / "astra.yaml").read_text())
    decision = spec["decisions"][decision_id]
    assert decision["default"] in decision["options"]
    rationale = decision["rationale"]
    assert rationale.count("Values:") == 1
    values = rationale.split("Values:", 1)[1].strip().removesuffix(".")
    pins = {}
    for entry in values.split(";"):
        name, value = [part.strip() for part in entry.split("=", 1)]
        pins[name] = ast.literal_eval(value)
    return pins


def assert_decision_pins(config=None):
    root = Path(__file__).resolve().parents[3]
    pins = record_pins(root)
    assert set(pins) == {"MIN_EXTERNAL_PAIRED_FRACTION", "mask_v2.0.yaml#N_EPOCH"}

    tag = f"@sc [decision:{DECISION}]"
    tagged_paths = (
        root / "src/sp_validation/catalogue_match.py",
        root / "config/calibration/mask_v2.0.yaml",
        root / "scripts/calibration/extract_info.py",
    )
    for path in tagged_paths:
        assert path.read_text().count(tag) == 1

    assert cm.MIN_EXTERNAL_PAIRED_FRACTION == pins["MIN_EXTERNAL_PAIRED_FRACTION"]
    if config is None:
        config = yaml.safe_load(
            (root / "config/calibration/mask_v2.0.yaml").read_text()
        )
    cuts = [item for item in config["dat"] if item.get("col_name") == "N_EPOCH"]
    assert len(cuts) == 1
    assert cuts[0]["kind"] == "greater_equal"
    assert cuts[0]["value"] == pins["mask_v2.0.yaml#N_EPOCH"]


def test_policy_pin():
    assert_decision_pins()


def test_policy_pin_detects_threshold_drift(monkeypatch):
    monkeypatch.setattr(cm, "MIN_EXTERNAL_PAIRED_FRACTION", 0.98)
    with pytest.raises(AssertionError):
        assert_decision_pins()


def test_policy_pin_detects_epoch_drift():
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load((root / "config/calibration/mask_v2.0.yaml").read_text())
    next(item for item in config["dat"] if item.get("col_name") == "N_EPOCH")[
        "value"
    ] = 1
    with pytest.raises(AssertionError):
        assert_decision_pins(config)
