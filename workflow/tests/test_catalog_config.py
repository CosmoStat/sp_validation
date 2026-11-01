"""Run-specific catalogue registries reach producers, including mock names."""

import yaml
from conftest import parse_jobs


def test_external_mock_registry(toy, tmp_path):
    name = "GLASS_00001_blinded"
    catalogue = tmp_path / "mock.fits"
    catalogue.touch()
    registry = tmp_path / "catalogues.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                name: {
                    "blind": "mock-test",
                    "colour": "C0",
                    "marker": "o",
                    "subdir": str(tmp_path),
                    "shear": {
                        "path": str(catalogue),
                        "redshift_path": str(tmp_path / "nz.txt"),
                    },
                },
                "paths": {"blinds": str(tmp_path / "mock-blinds")},
            }
        )
    )
    target = str(
        toy.common._plain(toy.cosmo_val)
        / (f"{name}_xi_minsep=1.0_maxsep=250.0_nbins=20_npatch=100.sacc")
    )
    result = toy.snakemake(
        "-n",
        target,
        config=[
            f"catalog_config={registry}",
            f'versions=["{name}"]',
            f'fiducial={{"version":"{name}","mock_version":"{name}"}}',
        ],
    )
    assert result.returncode == 0, result.stdout
    (job,) = [job for job in parse_jobs(result.stdout) if job.rule == "xi"]
    assert [toy.common._plain(p) for p in job.input] == [toy.common._plain(catalogue)]
    assert job.wildcards["version"] == name

    # Record producer params without running a scientific measurement.
    from pathlib import Path

    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Path(target).touch()
    overrides = [
        f"catalog_config={registry}",
        f'versions=["{name}"]',
        f'fiducial={{"version":"{name}","mock_version":"{name}"}}',
    ]
    touched = toy.snakemake("--touch", "--force", target, config=overrides)
    assert touched.returncode == 0, touched.stdout
    unchanged = toy.snakemake("-n", target, config=overrides)
    assert unchanged.returncode == 0, unchanged.stdout
    assert not parse_jobs(unchanged.stdout)

    entries = yaml.safe_load(registry.read_text())
    entries[name]["colour"] = "C3"
    entries[name]["marker"] = "s"
    entries[name]["cov_th"] = {"A": 1234.0, "sigma_e": 0.27}
    registry.write_text(yaml.safe_dump(entries))
    restyled = toy.snakemake("-n", target, config=overrides)
    assert restyled.returncode == 0, restyled.stdout
    assert not parse_jobs(restyled.stdout)

    entries[name]["shear"]["redshift_path"] = str(tmp_path / "other_nz.txt")
    registry.write_text(yaml.safe_dump(entries))
    changed = toy.snakemake("-n", target, config=overrides)
    assert changed.returncode == 0, changed.stdout
    assert [job.rule for job in parse_jobs(changed.stdout)] == ["xi"]
    assert "params have changed" in changed.stdout.lower()
