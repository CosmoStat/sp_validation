"""The covariance depends on a shape-free cache by default in cosmo_val."""

from contextlib import contextmanager
from pathlib import Path

import yaml
from conftest import VERSIONS, parse_jobs


@contextmanager
def _implicit_noise_setting(toy):
    # Snakemake recursively merges CLI config; omitting a nested CLI key does
    # not remove the explicit setting in the paper template.
    path = toy.rundir / "config" / "config.yaml"
    original = path.read_text()
    config = yaml.safe_load(original)
    config["covariance"].pop("shape_noise", None)
    path.write_text(yaml.safe_dump(config))
    try:
        yield
    finally:
        path.write_text(original)


def _ini(toy, tomo_suffix=""):
    # Unmasked target: no site-specific footprint needed by this test.
    (toy.root / "data" / "nz_SP_v0.1_A.txt").touch()
    grid = toy.common.xi_grids(toy.config, toy.config["fiducial"])["integration"]
    return toy.common.covariance_path(
        VERSIONS[0],
        "g",
        grid["min_sep"],
        grid["max_sep"],
        grid["nbins"],
        "",
        tomo_suffix,
        suffix=".ini",
    )


def test_pair_count_dependency_and_ini(toy):
    target = _ini(toy)
    with _implicit_noise_setting(toy):
        result = toy.snakemake("-n", target)
    assert result.returncode == 0, result.stdout
    jobs = parse_jobs(result.stdout)
    (pair,) = [j for j in jobs if j.rule == "covariance_pair_counts"]
    (ini,) = [j for j in jobs if j.rule == "covariance_ini"]
    assert pair.input == [str(toy.root / "data" / f"{VERSIONS[0]}.fits")]
    assert any(p.endswith(".txt") and p in ini.input for p in pair.output)
    assert not any(".sacc" in p or "blind" in p for p in pair.input + pair.output)
    # Stand in for measured counts, then execute only the host-side ini writer.
    for p in pair.output:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).touch()
    result = toy.snakemake(target, "--allowed-rules", "covariance_ini")
    assert result.returncode == 0, result.stdout
    text = Path(target).read_text()
    assert "[tabulated inputs files]\nnpair_directory = " in text
    assert "npair_mm_file = " in text
    assert "\\n" not in text
    assert f"ellipticity_dispersion = {0.3 / 2**0.5!r}" in text
    cat_path = toy.root / "cosmo_val" / "cat_config.yaml"
    original = cat_path.read_text()
    entries = yaml.safe_load(original)
    entries[VERSIONS[0]]["cov_th"]["sigma_e_convention"] = "per_component"
    cat_path.write_text(yaml.safe_dump(entries))
    try:
        result = toy.snakemake("-F", target, "--allowed-rules", "covariance_ini")
        assert result.returncode == 0, result.stdout
        assert "ellipticity_dispersion = 0.3" in Path(target).read_text()
    finally:
        cat_path.write_text(original)


def test_analytic_flag_removes_pair_dependency(toy):
    result = toy.snakemake(
        "-n",
        "-F",
        _ini(toy),
        config=['covariance={"shape_noise":"analytic","default_masked":false}'],
    )
    assert result.returncode == 0, result.stdout
    jobs = parse_jobs(result.stdout)
    assert not any(j.rule == "covariance_pair_counts" for j in jobs)
    (ini,) = [j for j in jobs if j.rule == "covariance_ini"]
    assert not any("_pairs_" in p for p in ini.input)


def test_implicit_tomographic_default_stays_analytic(toy):
    with _implicit_noise_setting(toy):
        result = toy.snakemake("-n", _ini(toy, "_tomo"))
    assert result.returncode == 0, result.stdout
    jobs = parse_jobs(result.stdout)
    assert any(j.rule == "covariance_ini" for j in jobs)
    assert not any(j.rule == "covariance_pair_counts" for j in jobs)


def test_explicit_tomographic_pair_counts_refused(toy):
    result = toy.snakemake(
        "-n",
        _ini(toy, "_tomo"),
        config=['covariance={"shape_noise":"pair_counts","default_masked":false}'],
    )
    assert result.returncode != 0
    assert "non-tomographic xi only" in result.stdout
