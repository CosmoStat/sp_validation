"""Record integrity of the calibrated sample's decisions in ``dr6_catalogue``.

Every ``@sc`` decision citation of ``dr6_catalogue`` names a decision the
ASTRA record defines, every decision has a tagged site, and every ``Values:``
pin equals the committed value. A pin is ``mask_v2.0.yaml#path`` (the selection
config; ``dat.<column>.<key>`` reads that column's cut), ``v2.yaml#path`` (the
sample recipe) or ``module.CONSTANT`` (under ``sp_validation``). The
paired-fraction decision's pins are checked by ``test_catalogue_match``.
"""

import ast
import copy
import importlib
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
RECORD = ROOT / "scratch/cdaley/unions_astra"
FILES = {
    "mask_v2.0.yaml": "config/calibration/mask_v2.0.yaml",
    "v2.yaml": "config/calibration/sample/v2.yaml",
}
ELSEWHERE = {"catalogue_match_tile_selection"}
TAG = re.compile(r"@sc \[([^\]]*)\]")


def read_record():
    project = yaml.safe_load((RECORD / "astra.yaml").read_text())
    path = RECORD / project["analyses"]["dr6_catalogue"]["path"] / "astra.yaml"
    return yaml.safe_load(path.read_text())


def read_cited():
    """The ``dr6_catalogue`` decisions cited by ``@sc`` tags in the repository."""
    cited = set()
    for directory in ("src", "scripts", "config", "workflow"):
        for path in (ROOT / directory).rglob("*"):
            if path.suffix in (".py", ".yaml", ".yml", ".smk") and path != Path(
                __file__
            ):
                for meta in TAG.findall(path.read_text()):
                    for item in meta.split(","):
                        key, _, value = item.strip().partition(":")
                        if key == "decision" and value.startswith("dr6_catalogue."):
                            cited.add(value.removeprefix("dr6_catalogue."))
    return cited


def read_configs():
    return {
        name: yaml.safe_load((ROOT / path).read_text()) for name, path in FILES.items()
    }


def pins(decision):
    rationale = decision["rationale"]
    assert rationale.count("Values:") == 1
    values = rationale.split("Values:", 1)[1].strip().removesuffix(".")
    if values == "none":
        return {}
    entries = (entry.split("=", 1) for entry in values.split(";"))
    return {name.strip(): ast.literal_eval(value.strip()) for name, value in entries}


def committed(name, configs):
    if "#" not in name:
        module, constant = name.rsplit(".", 1)
        return getattr(importlib.import_module(f"sp_validation.{module}"), constant)
    file, path = name.split("#")
    value = configs[file]
    for key in path.split("."):
        if isinstance(value, list):
            (value,) = [cut for cut in value if cut["col_name"] == key]
        else:
            value = value[key]
    return value


def check(record, cited, configs):
    decisions = {k: v for k, v in record["decisions"].items() if k not in ELSEWHERE}
    cited = cited - ELSEWHERE
    assert not cited - set(decisions), f"undefined: {cited - set(decisions)}"
    assert not set(decisions) - cited, f"untagged: {set(decisions) - cited}"
    for name, decision in decisions.items():
        assert decision["default"] in decision["options"], name
        for pin, value in pins(decision).items():
            assert committed(pin, configs) == value, f"{pin} drifted"
    for output in record["outputs"]:
        assert set(output["decisions"]) <= set(record["decisions"]), output["id"]
        assert set(output["inputs"]) <= {i["id"] for i in record["inputs"]}


def test_record_matches_code():
    check(read_record(), read_cited(), read_configs())


def test_detects_selection_drift():
    configs = read_configs()
    configs["mask_v2.0.yaml"]["metacal"]["gal_rel_size_max"] = 3
    with pytest.raises(AssertionError, match="gal_rel_size_max drifted"):
        check(read_record(), read_cited(), configs)
    configs = read_configs()
    flags = configs["mask_v2.0.yaml"]["dat"][0]
    assert flags["col_name"] == "FLAGS"
    del flags["ignore_bits"]
    with pytest.raises(KeyError):
        check(read_record(), read_cited(), configs)


def test_detects_record_drift():
    record = copy.deepcopy(read_record())
    decision = record["decisions"]["footprint"]
    decision["rationale"] = decision["rationale"].replace("= 4096", "= 2048")
    with pytest.raises(AssertionError, match="footprint_nside drifted"):
        check(record, read_cited(), read_configs())


def test_detects_an_untagged_and_an_undefined_decision():
    with pytest.raises(AssertionError, match="untagged: {'overlap_cut'}"):
        check(read_record(), read_cited() - {"overlap_cut"}, read_configs())
    with pytest.raises(AssertionError, match="undefined: {'unrecorded'}"):
        check(read_record(), read_cited() | {"unrecorded"}, read_configs())
