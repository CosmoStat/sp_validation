"""Custody is declared on every catalogue entry and resolved in one place.

Blind records are hand-written here: the host never draws, so a record needs
no real seed.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from sp_validation import custody as cu

REPO = Path(__file__).resolve().parents[3]


def _catalogues(tmp_path, **blinds):
    """A parsed catalogue config: one entry per ``name=blind``, each reading
    its own file; ``None`` declares nothing."""
    entries = {
        "nz": {"subdir": "/nz"},
        "paths": {"output": "./output", "blinds": str(tmp_path / "blinds")},
    }
    for name, blind in blinds.items():
        shear = {
            "path": f"{name}.fits",
            "e1_col_corrected": "e1c",
            "e2_col_corrected": "e2c",
        }
        entries[name] = {"subdir": str(tmp_path), "shear": shear}
        if blind is not None:
            entries[name]["blind"] = blind
    return entries


def _record(tmp_path, name):
    """A blind's record in the registry; returns its commitment."""
    record = {"seed": f"seed-of-{name}", "envelope": {"S8": 0.075}, "draw_scheme": 2}
    (tmp_path / "blinds").mkdir(exist_ok=True)
    (tmp_path / "blinds" / f"{name}.blind.json").write_text(json.dumps(record))
    return cu.commitment(record)


def test_every_entry_declares_its_blind(tmp_path):
    cats = _catalogues(tmp_path, OPEN="none", MOCK="mock", Y3="y3", NEW=None)
    commitment = _record(tmp_path, "y3")
    assert cu.custody_of(cats, "OPEN").token == "none"
    assert cu.custody_of(cats, "MOCK").token == "mock"
    assert cu.custody_of(cats, "Y3").token == f"y3:{commitment}"
    with pytest.raises(cu.CustodyError, match="declares no `blind:`"):
        cu.custody_of(cats, "NEW")
    # _leak_corr and _seed<N> versions take their entry's custody
    for version in ("Y3_leak_corr", "Y3_seed00042_leak_corr"):
        assert cu.custody_of(cats, version) == cu.custody_of(cats, "Y3")


def test_a_missing_blind_fails_closed_naming_the_remedy(tmp_path):
    cats = _catalogues(tmp_path, SP_v9="y3")
    with pytest.raises(cu.CustodyError, match="get read access") as absent:
        cu.custody_of(cats, "SP_v9")
    _record(tmp_path, "other")
    with pytest.raises(cu.CustodyError, match="blinding init y3") as missing:
        cu.custody_of(cats, "SP_v9")
    for refusal in (absent, missing):
        assert "SP_v9 is blinded under y3" in str(refusal.value)
        assert "`blind: none` would unblind it" in str(refusal.value)


def test_entries_reading_one_file_declare_one_blind(tmp_path, monkeypatch):
    """Within the passed config, and against the repository config, which is
    authoritative for the files it names."""
    cats = _catalogues(tmp_path, TOY="y3", TWIN="none")
    cats["TWIN"]["shear"]["path"] = "TOY.fits"
    with pytest.raises(cu.CustodyError, match="declare one blind"):
        cu.declared(cats, "TWIN")
    cats["TWIN"]["blind"] = "y3"
    assert cu.declared(cats, "TWIN") == "y3"

    repo = tmp_path / "repo_cat_config.yaml"
    repo.write_text(yaml.safe_dump(_catalogues(tmp_path, TOY="y3")))
    monkeypatch.setattr(cu, "REPO_CAT_CONFIG", repo)
    personal = _catalogues(tmp_path, MINE="none")
    personal["MINE"]["shear"]["path"] = "TOY.fits"
    with pytest.raises(cu.CustodyError, match="authoritative"):
        cu.declared(personal, "MINE")
    assert cu.of_file(tmp_path / "TOY.fits") == {"y3"}
    assert cu.of_file(tmp_path / "elsewhere.fits") == set()


@pytest.mark.parametrize(
    "blinds, allowed",
    [
        ({"A": "y3", "B": "none"}, False),
        ({"A": "y3", "B": "y4"}, False),
        ({"A": "none", "B": "y3"}, False),
        ({"A": "y3", "B": "mock"}, True),
        ({"A": "y3", "B": "y3"}, True),
        ({"A": "none", "B": "mock"}, True),
    ],
)
def test_the_mixing_rule(blinds, allowed):
    """A blinded catalogue is shown only beside mocks and its own blind."""
    if allowed:
        cu.check_mix(blinds)
    else:
        with pytest.raises(cu.CustodyError, match="shows the blind's shift"):
            cu.check_mix(blinds)


def test_the_registry_stays_out_of_git_worktrees(tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    inside = {"paths": {"blinds": str(tmp_path / "repo" / "blinds")}}
    with pytest.raises(cu.CustodyError, match="inside the git worktree"):
        cu.registry(inside)
    root = {"paths": {"blinds": str(tmp_path / "repo")}}
    assert cu.registry(root) == (tmp_path / "repo").resolve()
    with pytest.raises(cu.CustodyError, match="absolute"):
        cu.registry({"paths": {"blinds": "blinds"}})


def test_the_commitment_is_the_forks_of_the_whole_record():
    smokescreen = pytest.importorskip("smokescreen")
    record = {"seed": "s", "envelope": {"S8": 0.075}, "draw_scheme": 2}
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    assert cu.commitment(record) == smokescreen.seed_commitment(canonical)


def test_every_catalogue_in_the_repository_resolves():
    cats = yaml.safe_load((REPO / "cosmo_val" / "cat_config.yaml").read_text())
    blinds = {
        name: cu.custody_of(cats, name).blind
        for name in cats
        if name not in cu.NOT_CATALOGUES
    }
    assert blinds and set(blinds.values()) <= {cu.NONE, cu.MOCK}, blinds


def test_host_and_job_resolve_the_same_custody(tmp_path):
    """`common.custody_token(v)` is what a job's CosmologyValidation seals under."""
    from sp_validation.cosmo_val import CosmologyValidation

    (tmp_path / "workflow").mkdir()
    (tmp_path / "workflow" / "common.py").write_text(
        (REPO / "workflow" / "common.py").read_text()
    )
    (tmp_path / "src").symlink_to(REPO / "src")
    (tmp_path / "cosmo_val").mkdir()
    cats = _catalogues(tmp_path, TOY="toy", TOY_OPEN="none", TOY_MOCK="mock")
    _record(tmp_path, "toy")
    (tmp_path / "cosmo_val" / "cat_config.yaml").write_text(yaml.safe_dump(cats))
    spec = importlib.util.spec_from_file_location(
        "toy_common", tmp_path / "workflow" / "common.py"
    )
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    common.CATALOG_CONFIG = yaml.safe_load(Path(common.CAT_CONFIG).read_text())

    for version in ("TOY", "TOY_leak_corr", "TOY_OPEN", "TOY_MOCK"):
        token = common.custody_token(version)
        job = CosmologyValidation(
            versions=[version],
            catalog_config=common.CAT_CONFIG,
            output_dir=str(tmp_path / "out"),
            custody={version: token},
        )
        interactive = CosmologyValidation(
            versions=[version],
            catalog_config=common.CAT_CONFIG,
            output_dir=str(tmp_path / "out"),
        )
        assert job.custody(version) == interactive.custody(version), version
        assert job.custody(version).token == token
    assert common.custody_token("TOY").startswith("toy:")
