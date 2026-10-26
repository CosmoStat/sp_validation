"""The covariance depends on a shape-free cache by default in cosmo_val."""

from pathlib import Path

import yaml
from conftest import VERSIONS, parse_jobs


def _ini(toy):
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
        "",
        suffix=".ini",
    )


def test_pair_count_dependency_and_ini(toy):
    target = _ini(toy)
    result = toy.snakemake("-n", target, config=['covariance={"default_masked":false}'])
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
