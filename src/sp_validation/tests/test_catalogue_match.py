import ast
import csv
import json
from pathlib import Path

import pytest

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
    assert summary == {"tiles": 3, "missing_metadata": 2, "warnings": 0}
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


def policy_from_document(text):
    code = text.split("```python\n", 1)[1].split("```", 1)[0]
    return ast.literal_eval(ast.parse(code).body[0].value)


def test_policy_pin():
    root = Path(__file__).resolve().parents[3]
    document = (root / "docs/catalogue-match-selection.md").read_text()
    assert policy_from_document(document) == cm.MIN_EXTERNAL_PAIRED_FRACTION
    source = Path(cm.__file__).read_text()
    mutated = source.replace(
        "MIN_EXTERNAL_PAIRED_FRACTION = None", "MIN_EXTERNAL_PAIRED_FRACTION = 0.98"
    )
    assignment = next(
        node
        for node in ast.parse(mutated).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "MIN_EXTERNAL_PAIRED_FRACTION"
            for target in node.targets
        )
    )
    with pytest.raises(AssertionError):
        assert policy_from_document(document) == ast.literal_eval(assignment.value)
