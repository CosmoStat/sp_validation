"""Record integrity of the ``nz`` sub-analysis: its decisions, tags and pins agree.

Every ``@sc`` decision citation of ``nz`` names a decision the ASTRA record
defines, every decision has a tagged site, every ``Values:`` pin sits in a file
that cites its decision and equals the committed value there, and contract ids
are unique. A pin is ``v2.yaml#dotted.key`` (the n(z) recipe) or
``module.CONSTANT`` (under ``sp_validation``).
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
RECIPE = "config/calibration/nz/v2.yaml"
TAG = re.compile(r"@sc \[([^\]]*)\][ \t]*([\w.-]*)")


def read_record():
    project = yaml.safe_load((RECORD / "astra.yaml").read_text())
    path = RECORD / project["analyses"]["nz"]["path"] / "astra.yaml"
    return yaml.safe_load(path.read_text())["decisions"]


def read_sites():
    """Text of every repository file carrying an ``@sc`` tag, but this one."""
    sites = {}
    for directory in ("src", "scripts", "config", "workflow"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.suffix in (".py", ".yaml", ".yml") and path != Path(__file__):
                text = path.read_text()
                if "@sc [" in text:
                    sites[path.relative_to(ROOT).as_posix()] = text
    return sites


def read_recipe():
    return yaml.safe_load((ROOT / RECIPE).read_text())


def pins(decision):
    """The decision's ``Values:`` pins, ``{}`` for ``Values: none``."""
    rationale = decision["rationale"]
    assert rationale.count("Values:") == 1
    values = rationale.split("Values:", 1)[1].strip().removesuffix(".")
    if values == "none":
        return {}
    entries = (entry.split("=", 1) for entry in values.split(";"))
    return {name.strip(): ast.literal_eval(value.strip()) for name, value in entries}


def site(name):
    """The file a pin resolves to."""
    if "#" in name:
        assert name.split("#")[0] == Path(RECIPE).name
        return RECIPE
    return "src/sp_validation/" + name.rsplit(".", 1)[0].replace(".", "/") + ".py"


def committed(name, recipe):
    if "#" in name:
        value = recipe
        for key in name.split("#")[1].split("."):
            value = value[key]
        return value
    module, constant = name.rsplit(".", 1)
    return getattr(importlib.import_module(f"sp_validation.{module}"), constant)


def check(record, sites, recipe):
    cited, contracts = {}, []
    for path, text in sites.items():
        for meta, contract in TAG.findall(text):
            for item in meta.split(","):
                key, _, value = item.strip().partition(":")
                if key == "decision" and value.startswith("nz."):
                    cited.setdefault(value.removeprefix("nz."), set()).add(path)
            if contract:
                contracts.append(contract)
    assert not set(cited) - set(record), f"undefined: {set(cited) - set(record)}"
    assert not set(record) - set(cited), f"untagged: {set(record) - set(cited)}"
    repeated = {c for c in contracts if contracts.count(c) > 1}
    assert not repeated, f"repeated contract ids: {repeated}"
    for name, decision in record.items():
        assert decision["default"] in decision["options"]
        for pin, value in pins(decision).items():
            assert site(pin) in cited[name], f"{pin}: no {name} tag there"
            assert committed(pin, recipe) == value, f"{pin} drifted"


def test_record_matches_code():
    check(read_record(), read_sites(), read_recipe())


def test_detects_recipe_drift():
    recipe = read_recipe()
    recipe["gold_qc"]["nmad_factor"] = 4
    with pytest.raises(AssertionError, match="gold_qc.nmad_factor drifted"):
        check(read_record(), read_sites(), recipe)


def test_detects_constant_drift(monkeypatch):
    from sp_validation.nz import inputs

    monkeypatch.setattr(inputs, "OBJECT_ID", "RA_DEC")
    with pytest.raises(AssertionError, match="OBJECT_ID drifted"):
        check(read_record(), read_sites(), read_recipe())


def test_detects_record_drift():
    record = copy.deepcopy(read_record())
    decision = record["tomographic_bins"]
    decision["rationale"] = decision["rationale"].replace("count = 4", "count = 5")
    with pytest.raises(AssertionError, match="bins.count drifted"):
        check(record, read_sites(), read_recipe())


def test_detects_a_removed_pinned_site():
    sites = read_sites()
    tag = "@sc [decision:nz." + "gold_selection]"
    sites[RECIPE] = sites[RECIPE].replace(tag, "")
    with pytest.raises(AssertionError, match="no gold_selection tag there"):
        check(read_record(), sites, read_recipe())


def test_detects_an_untagged_decision():
    tag = "@sc [decision:nz." + "grid_step]"
    sites = {path: text.replace(tag, "") for path, text in read_sites().items()}
    with pytest.raises(AssertionError, match="untagged: {'grid_step'}"):
        check(read_record(), sites, read_recipe())


def test_detects_an_undefined_decision():
    sites = read_sites()
    sites["extra.py"] = "# @sc [decision:nz." + "unrecorded]\n"
    with pytest.raises(AssertionError, match="undefined: {'unrecorded'}"):
        check(read_record(), sites, read_recipe())


def test_detects_a_repeated_contract_id():
    sites = read_sites()
    sites["extra.py"] = "# @sc [label:convention] " + "deposit-conserves-moments\n"
    with pytest.raises(AssertionError, match="repeated contract ids"):
        check(read_record(), sites, read_recipe())
