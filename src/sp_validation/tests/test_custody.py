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
from hypothesis import example, given, settings
from hypothesis import strategies as st

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
NO_BLIND = [
    "declares no custody for it, so it is blinded",
    "blinding init",
    "blinding share",
]
Y3 = {"y3": (["SP_v9"], None)}  # a blind over SP_v9, concealed
PUBLIC = {"y3": (["SP_v9"], "seed-of-y3")}  # the same, its seed published
V = "SP_v9_ecut07"
# case: (declarations, blinds {name: (bases, published seed)}, expected, version).
# expected is the custody's token, or the fragments of the refusal.
# fmt: off
TABLE = {
    "undeclared": ({"SP_v9": None}, {}, NO_BLIND),
    "blinded": ({"SP_v9": "blinded"}, {}, ["no blind covers it"]),
    "undeclared_covered": ({"SP_v9": None}, Y3, "blinded:SP_v9:y3:{y3}"),
    "blinded_covered": ({"SP_v9": "blinded"}, Y3, "blinded:SP_v9:y3:{y3}"),
    "blinded_revealed": ({"SP_v9": None}, PUBLIC, ["declare `blinding: unblinded`"]),
    "unblinded": ({"SP_v9": "unblinded"}, {}, "unblinded:SP_v9"),
    "unblinded_concealed": ({"SP_v9": "unblinded"}, Y3, ["blinding reveal y3"]),
    "unblinded_revealed": ({"SP_v9": "unblinded"}, PUBLIC, "unblinded:SP_v9"),
    "unblinded_wrong_seed": (
        {"SP_v9": "unblinded"}, {"y3": (["SP_v9"], "not-the-seed")}, ["its commitment"]
    ),
    "mock": ({"SP_v9": "mock"}, {}, "mock:SP_v9"),
    "mock_covered": ({"SP_v9": "mock"}, Y3, ["a mock is never blinded"]),
    "two_blinds": ({"SP_v9": None}, {**Y3, "b": (["SP_v9"], None)}, ["blinds: b, y3"]),
    "unknown": ({"SP_v9": "open"}, {}, ["blinded, unblinded or mock"]),
    "variant_declares": (
        {"SP_v9": "unblinded", V: {"base": "SP_v9", "blinding": "unblinded"}}, {},
        ["declare custody on SP_v9"], V,
    ),
    "alias_repeats": (
        {"SP_v9": "unblinded", "SP_v9_leak_corr": "unblinded"}, {}, "unblinded:SP_v9",
        "SP_v9_leak_corr",
    ),
    "alias_repeats_base": (
        {"SP_v9": "unblinded", V: {"base": "SP_v9"}, f"{V}_seed7": {"base": "SP_v9"}},
        {}, "unblinded:SP_v9", f"{V}_seed7",
    ),
    "alias_declares": (
        {"SP_v9": "unblinded", "SP_v9_leak_corr": "blinded"}, {},
        ["declare custody on SP_v9"], "SP_v9_leak_corr",
    ),
    "alias_names_another_base": (
        {"SP_v9": "unblinded", "TOY": None, "SP_v9_seed7": {"base": "TOY"}},
        {"y3": (["TOY"], None)}, ["declare custody on SP_v9"], "SP_v9_seed7",
    ),
}
# fmt: on


@pytest.mark.parametrize("case", TABLE)
def test_the_custody_table(tmp_path, case):
    """Every row of custody_of's table: the declaration on the base catalogue
    against the blinds covering it. An entry named as a variant of another
    (``<entry>_leak_corr``, ``<entry>_seed<N>``) may repeat that entry's
    custody, never declare another."""
    declarations, blinds, expected, *version = TABLE[case]
    version = version[0] if version else "SP_v9"
    registry = tmp_path / "blinds"
    commitments = {
        name: cu.seed_commitment(_write_blind(registry, name, bases, revealed=seed))
        for name, (bases, seed) in blinds.items()
    }
    cats = _catalogues(**declarations)
    if isinstance(expected, str):
        custody = cu.custody_of(cats, version, registry=registry)
        assert custody.token == expected.format(**commitments)
        return
    with pytest.raises(cu.CustodyError) as refused:
        cu.custody_of(cats, version, registry=registry)
    for fragment in expected:
        assert fragment in str(refused.value)


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


SUFFIXES = st.lists(
    st.one_of(st.just("_leak_corr"), st.integers(0, 99999).map("_seed{:05d}".format)),
    max_size=3,
)


@settings(max_examples=30, deadline=None)
@given(entry=st.sampled_from(["SP_v9", "SP_v9_ecut07"]), suffixes=SUFFIXES)
@example(entry="SP_v9", suffixes=["_leak_corr"])
@example(entry="SP_v9_ecut07", suffixes=["_seed00042", "_leak_corr"])
def test_every_variant_shares_its_base(tmp_path_factory, entry, suffixes):
    """``_leak_corr`` and ``_seed<N>`` versions, in any order, of a catalogue
    or of an entry naming its parent with ``base:``, resolve to the base's
    custody and blind."""
    registry = tmp_path_factory.mktemp("variants") / "blinds"
    _write_blind(registry, "y3", ["SP_v9"])
    cats = _catalogues(SP_v9=None, SP_v9_ecut07={"base": "SP_v9"})
    version = entry + "".join(suffixes)
    base = cu.custody_of(cats, "SP_v9", registry=registry)
    assert cu.custody_of(cats, version, registry=registry) == base


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
