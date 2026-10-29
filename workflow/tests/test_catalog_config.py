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
