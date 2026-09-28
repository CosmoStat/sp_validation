"""Custody is declared per base catalogue and resolved in one place (I12, I13).

``sp_validation.custody.custody_of`` reads a catalogue's declaration and the
blind registry beside the catalogue config. These tests pin every row of its
table on hand-written registries (the host never decrypts, so a record needs
no real seed), and that the host Snakemake and a container job resolve the same
custody for every version.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from sp_validation import custody as cu

REPO = Path(__file__).resolve().parents[3]


def _write_blind(registry, name, bases, *, revealed=None):
    """A registry record as ``blinding init`` writes it, with a stand-in seed."""
    record = registry / name
    record.mkdir(parents=True)
    seed = f"seed-of-{name}"
    (record / "commitment.json").write_text(
        json.dumps(
            {
                "blind": name,
                "seed_commitment": cu.seed_commitment(seed),
                "config_digest": "d" * 64,
                "draw_scheme": 2,
            }
        )
    )
    (record / "bases").write_text("".join(f"{b}\n" for b in bases))
    if revealed is not None:
        (record / "revealed.json").write_text(json.dumps({"seed": revealed}))
    return seed


def _catalogues(**declarations):
    """A parsed catalogue config: one entry per ``name=declaration``."""
    entries = {"nz": {"subdir": "/nz"}, "paths": {"output": "./output"}}
    for name, declaration in declarations.items():
        entries[name] = {
            "shear": {
                "path": f"{name}.fits",
                "e1_col_corrected": "e1_corrected",
                "e2_col_corrected": "e2_corrected",
            }
        }
        if isinstance(declaration, dict):
            entries[name].update(declaration)
        elif declaration is not None:
            entries[name]["blinding"] = declaration
    return entries


# --------------------------------------------------------------------------- #
# I12: the custody table
# --------------------------------------------------------------------------- #
def test_undeclared_catalogue_without_a_blind_fails_with_the_command(tmp_path):
    cats = _catalogues(SP_v9=None)
    with pytest.raises(cu.CustodyError) as err:
        cu.custody_of(cats, "SP_v9", registry=tmp_path / "blinds")
    message = str(err.value)
    assert "declares no custody for it, so it is blinded" in message
    assert "python -m sp_validation.blinding init" in message
    assert "share" in message


def test_the_printed_commands_name_the_config_as_it_was_given(tmp_path):
    """The operator pastes these commands, so they keep the launch's spelling
    of the checkout: on candide the plain /nXXdataN, never the /automnt path it
    resolves to."""
    (tmp_path / "real" / "cosmo_val").mkdir(parents=True)
    checkout = tmp_path / "checkout"
    checkout.symlink_to(tmp_path / "real", target_is_directory=True)
    registry = cu.registry_of(checkout / "cosmo_val" / "cat_config.yaml")
    with pytest.raises(cu.CustodyError) as err:
        cu.custody_of(_catalogues(SP_v9=None), "SP_v9", registry=registry)
    message = str(err.value)
    assert f"APPTAINERENV_PYTHONPATH={checkout}/src " in message
    assert f"--cat-config {checkout}/cosmo_val/cat_config.yaml" in message


def test_declared_blinded_without_a_blind_fails(tmp_path):
    cats = _catalogues(SP_v9="blinded")
    with pytest.raises(cu.CustodyError, match="no blind covers it"):
        cu.custody_of(cats, "SP_v9", registry=tmp_path / "blinds")


@pytest.mark.parametrize("declaration", [None, "blinded"])
def test_blinded_under_the_covering_blind(tmp_path, declaration):
    registry = tmp_path / "blinds"
    seed = _write_blind(registry, "y3", ["SP_v9"])
    c = cu.custody_of(_catalogues(SP_v9=declaration), "SP_v9", registry=registry)
    assert c.status == "blinded" and c.catalogue == "SP_v9" and c.blind == "y3"
    assert c.commitment == cu.seed_commitment(seed)
    assert c.token == f"blinded:SP_v9:y3:{c.commitment}"
    assert c.stamp["blinding"] == "blinded"
    assert c.stamp["blinding_commitment"] == c.commitment


def test_blinded_under_a_revealed_blind_fails(tmp_path):
    registry = tmp_path / "blinds"
    seed = _write_blind(registry, "y3", ["SP_v9"])
    (registry / "y3" / "revealed.json").write_text(json.dumps({"seed": seed}))
    with pytest.raises(cu.CustodyError, match="declare `blinding: unblinded`"):
        cu.custody_of(_catalogues(SP_v9=None), "SP_v9", registry=registry)


def test_unblinded_without_a_blind(tmp_path):
    c = cu.custody_of(
        _catalogues(SP_v9="unblinded"), "SP_v9", registry=tmp_path / "blinds"
    )
    assert (c.status, c.token) == ("unblinded", "unblinded:SP_v9")
    assert c.stamp == {"blinding": "unblinded", "blinding_catalogue": "SP_v9"}


def test_unblinding_a_concealed_catalogue_is_the_reveal(tmp_path):
    registry = tmp_path / "blinds"
    _write_blind(registry, "y3", ["SP_v9"])
    with pytest.raises(cu.CustodyError, match="blinding reveal y3"):
        cu.custody_of(_catalogues(SP_v9="unblinded"), "SP_v9", registry=registry)


def test_unblinded_after_a_reveal_whose_seed_matches(tmp_path):
    registry = tmp_path / "blinds"
    seed = _write_blind(registry, "y3", ["SP_v9"])
    (registry / "y3" / "revealed.json").write_text(json.dumps({"seed": seed}))
    c = cu.custody_of(_catalogues(SP_v9="unblinded"), "SP_v9", registry=registry)
    assert c.token == "unblinded:SP_v9"


def test_a_published_seed_that_misses_the_commitment_fails(tmp_path):
    registry = tmp_path / "blinds"
    _write_blind(registry, "y3", ["SP_v9"], revealed="not-the-seed")
    with pytest.raises(cu.CustodyError, match="commitment"):
        cu.custody_of(_catalogues(SP_v9="unblinded"), "SP_v9", registry=registry)


def test_mock(tmp_path):
    registry = tmp_path / "blinds"
    c = cu.custody_of(_catalogues(SP_v9="mock"), "SP_v9", registry=registry)
    assert (c.status, c.token) == ("mock", "mock:SP_v9")
    _write_blind(registry, "y3", ["SP_v9"])
    with pytest.raises(cu.CustodyError, match="mock"):
        cu.custody_of(_catalogues(SP_v9="mock"), "SP_v9", registry=registry)


def test_two_covering_blinds_fail(tmp_path):
    registry = tmp_path / "blinds"
    _write_blind(registry, "a", ["SP_v9"])
    _write_blind(registry, "b", ["SP_v9"])
    with pytest.raises(cu.CustodyError, match="a, b"):
        cu.custody_of(_catalogues(SP_v9=None), "SP_v9", registry=registry)


def test_a_variant_entry_may_not_declare_custody(tmp_path):
    cats = _catalogues(
        SP_v9="unblinded", SP_v9_ecut07={"base": "SP_v9", "blinding": "unblinded"}
    )
    with pytest.raises(cu.CustodyError, match="declare custody on SP_v9"):
        cu.custody_of(cats, "SP_v9_ecut07", registry=tmp_path / "blinds")


def test_an_unknown_declaration_fails(tmp_path):
    with pytest.raises(cu.CustodyError, match="blinded, unblinded or mock"):
        cu.custody_of(_catalogues(SP_v9="open"), "SP_v9", registry=tmp_path / "blinds")


def test_variants_share_their_base(tmp_path):
    """`_leak_corr`, `_seed<N>` and `base:` entries resolve to the base."""
    registry = tmp_path / "blinds"
    _write_blind(registry, "y3", ["SP_v9"])
    cats = _catalogues(SP_v9=None, SP_v9_ecut07={"base": "SP_v9"})
    base = cu.custody_of(cats, "SP_v9", registry=registry)
    for version in (
        "SP_v9_leak_corr",
        "SP_v9_seed00042",
        "SP_v9_seed00042_leak_corr",
        "SP_v9_leak_corr_seed00042",
        "SP_v9_ecut07",
        "SP_v9_ecut07_leak_corr",
    ):
        assert cu.custody_of(cats, version, registry=registry) == base, version


def test_an_entry_named_as_a_variant_shares_its_parents_custody(tmp_path):
    """An entry named `<entry>_leak_corr` or `<entry>_seed<N>` resolves through
    that entry, as the variant it is named for does. It may repeat the entry's
    `blinding` or `base`, as the copies `overwrite_config` writes do, but a
    declaration of its own is refused, never ignored."""
    registry = tmp_path / "blinds"
    _write_blind(registry, "y3", ["TOY"])
    cats = _catalogues(PARENT="unblinded", TOY=None, EC={"base": "PARENT"})
    copies = dict(
        cats, PARENT_leak_corr=dict(cats["PARENT"]), EC_seed7=dict(cats["EC"])
    )
    for version in ("PARENT_leak_corr", "EC_seed7"):
        custody = cu.custody_of(copies, version, registry=registry)
        assert custody.token == "unblinded:PARENT", version
    for alias, claim in {
        "PARENT_leak_corr": {"blinding": "blinded"},
        "PARENT_seed7": {"base": "TOY"},
    }.items():
        with pytest.raises(cu.CustodyError, match="declare custody on PARENT"):
            cu.custody_of({**cats, alias: claim}, alias, registry=registry)


def _reading(cats, path, *names):
    """Point ``names``' entries at one shear file, spelled two ways."""
    for i, name in enumerate(names):
        cats[name]["shear"]["path"] = path.name if i % 2 else str(path)
        cats[name]["subdir"] = str(path.parent)
    return cats


def test_catalogues_reading_one_file_share_one_blind(tmp_path):
    """A blind conceals a shear file, whatever entry reads it: a catalogue
    reading a concealed catalogue's file under any other custody is refused,
    on either side, until the blind covers it too."""
    registry = tmp_path / "blinds"
    seed = _write_blind(registry, "y3", ["TOY"])
    cats = _catalogues(TOY=None, TOY_A="unblinded", TOY_V={"base": "TOY"})
    _reading(cats, tmp_path / "toy.fits", "TOY", "TOY_A", "TOY_V")
    for version in ("TOY_A", "TOY", "TOY_leak_corr"):
        with pytest.raises(cu.CustodyError, match="not concealed alike") as err:
            cu.custody_of(cats, version, registry=registry)
        assert "TOY: blind y3" in str(err.value), version
        assert "TOY_A: public" in str(err.value), version

    two = _reading(
        _catalogues(TOY=None, TOY_B=None), tmp_path / "toy.fits", "TOY", "TOY_B"
    )
    _write_blind(tmp_path / "two", "y3", ["TOY"])
    _write_blind(tmp_path / "two", "y4", ["TOY_B"])
    with pytest.raises(cu.CustodyError, match="TOY: blind y3; TOY_B: blind y4"):
        cu.custody_of(two, "TOY", registry=tmp_path / "two")

    mock = _catalogues(OPEN="unblinded", OPEN_MOCK="mock")
    _reading(mock, tmp_path / "open.fits", "OPEN", "OPEN_MOCK")
    assert cu.custody_of(mock, "OPEN", registry=registry).status == "unblinded"

    del cats["TOY_A"]["blinding"]
    (registry / "y3" / "bases").write_text("TOY\nTOY_A\n")
    for version in ("TOY", "TOY_A", "TOY_V"):
        assert cu.custody_of(cats, version, registry=registry).blind == "y3"

    (registry / "y3" / "revealed.json").write_text(json.dumps({"seed": seed}))
    cats["TOY"]["blinding"] = "unblinded"
    cats["TOY_A"]["blinding"] = "unblinded"
    assert cu.custody_of(cats, "TOY_A", registry=registry).status == "unblinded"


def test_base_links_are_followed_and_checked():
    cats = _catalogues(
        A="unblinded", B={"base": "A"}, C={"base": "B"}, D={"base": "nowhere"}
    )
    assert cu.base_catalogue(cats, "C_leak_corr") == "A"
    with pytest.raises(cu.CustodyError, match="nowhere"):
        cu.base_catalogue(cats, "D")
    loop = _catalogues(E={"base": "F"}, F={"base": "E"})
    with pytest.raises(cu.CustodyError, match="cycle"):
        cu.base_catalogue(loop, "E")
    with pytest.raises(cu.CustodyError, match="not a catalogue"):
        cu.base_catalogue(cats, "paths")


def test_summary_is_one_line_per_base(tmp_path):
    registry = tmp_path / "blinds"
    _write_blind(registry, "y3", ["SP_v9"])
    cats = _catalogues(SP_v9=None, SP_v8="unblinded")
    lines = cu.summary(
        cats, ["SP_v9", "SP_v9_leak_corr", "SP_v8_leak_corr"], registry=registry
    )
    assert lines == [
        "[custody] SP_v9 (+ SP_v9_leak_corr): blinded under y3",
        "[custody] SP_v8 (+ SP_v8_leak_corr): unblinded",
    ]


def test_every_catalogue_in_the_repository_resolves():
    """Each entry of the committed cat_config has a custody, from the repo registry."""
    path = REPO / "cosmo_val" / "cat_config.yaml"
    cats = yaml.safe_load(path.read_text())
    registry = cu.registry_of(path)
    statuses = {
        name: cu.custody_of(cats, name, registry=registry).status
        for name in cats
        if name not in cu.NOT_CATALOGUES
    }
    assert statuses and set(statuses.values()) <= set(cu.STATUSES), statuses


# --------------------------------------------------------------------------- #
# I13: the host and a job resolve the same custody
# --------------------------------------------------------------------------- #
def _toy_checkout(tmp_path):
    """A checkout-shaped tree: workflow/common.py, src/, cosmo_val/{cat_config,blinds}."""
    (tmp_path / "workflow").mkdir()
    (tmp_path / "workflow" / "common.py").write_text(
        (REPO / "workflow" / "common.py").read_text()
    )
    (tmp_path / "src").symlink_to(REPO / "src")
    cats = _catalogues(
        TOY=None,
        TOY_OPEN="unblinded",
        TOY_MOCK="mock",
        TOY_ecut07={"base": "TOY"},
    )
    for entry in (cats["TOY"], cats["TOY_OPEN"], cats["TOY_MOCK"], cats["TOY_ecut07"]):
        entry["subdir"] = str(tmp_path)
    (tmp_path / "cosmo_val").mkdir()
    (tmp_path / "cosmo_val" / "cat_config.yaml").write_text(yaml.safe_dump(cats))
    _write_blind(tmp_path / "cosmo_val" / "blinds", "toy", ["TOY"])
    return tmp_path


def test_host_and_job_resolve_the_same_custody(tmp_path):
    """The host and a job agree on each version's custody and patch centres.

    `common.custody_token(v)` equals `CosmologyValidation.custody(v).token`, and
    the centres the xi rule declares are the file the job splits at, one per
    base catalogue.
    """
    from sp_validation.cosmo_val import CosmologyValidation

    root = _toy_checkout(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "toy_common", root / "workflow" / "common.py"
    )
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    common.CATALOG_CONFIG = yaml.safe_load(Path(common.CAT_CONFIG).read_text())

    versions = ["TOY", "TOY_leak_corr", "TOY_OPEN", "TOY_MOCK", "TOY_ecut07"]
    cv = CosmologyValidation(
        versions=versions,
        catalog_config=common.CAT_CONFIG,
        output_dir=str(common.COSMO_VAL),
    )
    for version in versions:
        assert common.custody_token(version) == cv.custody(version).token, version
        assert common.patches_path(version, 100) == cv.patch_centers_path(
            version, 100
        ), version
    assert common.custody_token("TOY").startswith("blinded:TOY:toy:")
    assert len({common.patches_path(v, 100) for v in versions}) == 3
