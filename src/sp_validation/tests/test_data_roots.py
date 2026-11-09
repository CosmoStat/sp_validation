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
    redshift = catalogues["SP_v2.0"]["shear"]["redshift_path"]
    assert redshift.endswith("/nz_v2_bound9_full_tomo.txt")
    recipe = data_roots.load_yaml(root / "config/calibration/nz/v2_bound9.yaml")
    assert recipe["photometry"]["dr6_root"]
