"""Every catalogue entry declares its blind."""

from pathlib import Path

import pytest
import yaml

from sp_validation import custody as cu

REPO = Path(__file__).resolve().parents[3]


def _catalogues(tmp_path, **blinds):
    """A parsed catalogue config: one entry per ``name=blind``, each reading
    its own file; ``None`` declares nothing."""
    entries = {"nz": {"subdir": "/nz"}, "paths": {"output": "./output"}}
    for name, blind in blinds.items():
        entries[name] = {"subdir": str(tmp_path), "shear": {"path": f"{name}.fits"}}
        if blind is not None:
            entries[name]["blind"] = blind
    return entries


def test_every_entry_declares_its_blind(tmp_path):
    cats = _catalogues(tmp_path, OPEN="none", Y3="y3", NEW=None)
    assert cu.declared(cats, "OPEN") == "none"
    assert cu.declared(cats, "Y3") == "y3"
    with pytest.raises(ValueError, match="declares no `blind:`"):
        cu.declared(cats, "NEW")
    # _leak_corr and _seed<N> versions take their entry's blind
    for version in ("Y3_leak_corr", "Y3_seed00042_leak_corr"):
        assert cu.declared(cats, version) == "y3"


def test_entries_reading_one_file_declare_one_blind(tmp_path):
    cats = _catalogues(tmp_path, TOY="y3", TWIN="none")
    cats["TWIN"]["shear"]["path"] = "TOY.fits"
    with pytest.raises(ValueError, match="different blinds"):
        cu.declared(cats, "TWIN")
    cats["TWIN"]["blind"] = "y3"
    assert cu.declared(cats, "TWIN") == "y3"


def test_every_catalogue_in_the_repository_is_public():
    cats = yaml.safe_load((REPO / "cosmo_val" / "cat_config.yaml").read_text())
    blinds = {
        name: cu.declared(cats, name) for name in cats if name not in cu.NOT_CATALOGUES
    }
    assert blinds and set(blinds.values()) == {"none"}, blinds
