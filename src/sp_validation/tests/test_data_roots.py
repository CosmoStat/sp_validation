import sys
import types
from pathlib import Path

import yaml

from sp_validation import data_roots


def _roots_file(tmp_path, **roots):
    path = tmp_path / "roots.yaml"
    path.write_text(yaml.safe_dump(roots))
    return path


def test_first_existing_directory_wins(tmp_path):
    here = tmp_path / "here"
    here.mkdir()
    roots = _roots_file(tmp_path, data=[str(tmp_path / "absent"), str(here)])
    assert data_roots.expand("${data}/nz.txt", roots) == f"{here}/nz.txt"


def test_environment_overrides_file(tmp_path, monkeypatch):
    roots = _roots_file(tmp_path, data=[str(tmp_path)])
    monkeypatch.setenv("SPV_ROOT_DATA", "/elsewhere")
    data_roots.roots.cache_clear()
    assert data_roots.expand("${data}/nz.txt", roots) == "/elsewhere/nz.txt"
    data_roots.roots.cache_clear()


def test_unresolved_and_undeclared_tokens_stay(tmp_path):
    roots = _roots_file(tmp_path, data=[str(tmp_path / "absent")])
    config = {"a": ["${data}/x", "${other}/y", "{version}"], "n": 3}
    assert data_roots.expand(config, roots) == {
        "a": ["${data}/x", "${other}/y", "{version}"],
        "n": 3,
    }


def test_repo_configs_load():
    root = data_roots.ROOTS_FILE.parents[1]
    catalogues = data_roots.load_yaml(root / "cosmo_val" / "cat_config.yaml")
    redshift = catalogues["SP_v2.0.0"]["shear"]["redshift_path"]
    assert redshift.endswith("/v2.0.0/nz/nz_tomo.txt")


class FakeVault:
    """A vos.Client over a local directory standing in for the vault."""

    def __init__(self, root):
        self.root = root

    def _path(self, uri):
        return self.root / uri.removeprefix("vos:")

    def isdir(self, uri):
        return self._path(uri).is_dir()

    def get_node(self, uri, limit=None, force=False):
        children = [
            types.SimpleNamespace(name=p.name, isdir=p.is_dir)
            for p in sorted(self._path(uri).iterdir())
        ]
        return types.SimpleNamespace(node_list=children)

    def copy(self, uri, dest):
        Path(dest).write_bytes(self._path(uri).read_bytes())


def test_vos_root_resolves_and_fetches_once(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "rel/catalogs/calibrated/dataset").mkdir(parents=True)
    (vault / "rel/catalogs/calibrated/dataset/Npix=1.parquet").write_text("p")
    (vault / "rel/nz").mkdir()
    (vault / "rel/nz/nz_tomo.txt").write_text("nz")
    fake = types.ModuleType("vos")
    fake.Client = lambda **_: FakeVault(vault)
    monkeypatch.setitem(sys.modules, "vos", fake)
    monkeypatch.setattr(data_roots, "CACHE", tmp_path / "cache")
    roots = _roots_file(tmp_path, rel=[str(tmp_path / "absent"), "vos:rel"])
    config = data_roots.expand(
        {"shear": "${rel}/catalogs/calibrated", "nz": "${rel}/nz/nz_tomo.txt"}, roots
    )
    assert config["nz"] == "vos:rel/nz/nz_tomo.txt"
    local = data_roots.localise(config)
    assert Path(local["nz"]).read_text() == "nz"
    part = Path(local["shear"]) / "dataset/Npix=1.parquet"
    assert part.read_text() == "p"
    (vault / "rel/nz/nz_tomo.txt").write_text("changed")
    assert Path(data_roots.local(config["nz"])).read_text() == "nz"
    assert data_roots.local("/a/local/path") == "/a/local/path"
