"""GLASS suite configuration, checked without running any science jobs."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from conftest import REPO, parse_jobs


@pytest.mark.parametrize("paper", ["bmodes", "cosmo_val"])
def test_paper_glass_suite(paper):
    config = yaml.safe_load(
        (REPO / "papers" / paper / "config" / "config.yaml").read_text()
    )
    assert config["glass_mocks"]["data_dir"] == (
        "/n09data/guerrini/glass_mock_v1.4.6.3_v2/results"
    )
    assert "glass_mock_data_dir" not in config["inference"]
    assert config["glass_mocks"]["seed_range"] == [1, 350]


@pytest.mark.parametrize("suite", ["glass_mock_v1.4.6.3_v2", "another_suite"])
def test_glass_rules_share_suite(toy, tmp_path, suite):
    """Changing only data_dir moves mock inputs and suite-labelled outputs."""
    config = deepcopy(toy.config)
    data_dir = tmp_path / suite / "results"
    data_dir.mkdir(parents=True)
    config["glass_mocks"]["data_dir"] = str(data_dir)
    config["glass_mocks"]["seed_range"] = [1, 2]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    # Empty files are dry-run dependencies only; no science code reads them.
    for seed in range(1, 101):
        (data_dir / f"unions_glass_sim_{seed:05d}_4096.fits").touch()
    for seed in range(1, 3):
        (data_dir / f"xi_glass_mock_{seed:05d}_4096_nbins=20.fits").touch()
        (data_dir / f"cl_glass_mock_{seed:05d}_4096.npy").touch()

    version = config["fiducial"]["mock_version"]
    suffix = toy.common.fiducial_binning_suffix(config["fiducial"])
    tag = toy.common.pseudo_cl_tag(config)
    dependencies = [
        toy.root / "data" / "nz_SP_v0.1_A.txt",
        toy.cosmo_val / f"pseudo_cl_cov_{version}_{tag}.fits",
        *[
            toy.cosmo_val / "rho_tau_stats" / name
            for name in (
                f"rho_stats_{version}{suffix}.fits",
                f"tau_stats_{version}{suffix}.fits",
                f"cov_tau_{version}{suffix}_th.npy",
            )
        ],
    ]
    for path in dependencies:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    result = toy.snakemake(
        "-n",
        "-s",
        str(toy.root / "workflow" / "Snakefile"),
        "covariance_glass_mock",
        "glass_mock_all_xi",
        "glass_mock_all_pseudo_cl",
        "inference_glass_mocks",
        "mock_cosebis_bias_test",
        "--configfile",
        str(config_path),
    )
    assert result.returncode == 0, result.stdout
    jobs = parse_jobs(result.stdout)
    for rule, count, input_count in (
        ("covariance_glass_mock", 1, 4),
        ("glass_mock_xi_fine", 100, 1),
        ("glass_mock_pseudo_cl", 100, 1),
        ("inference_prep_glass_mock", 2, 2),
    ):
        selected = [job for job in jobs if job.rule == rule]
        assert len(selected) == count
        for job in selected:
            mock_inputs = [f for f in job.input if Path(f).parent == data_dir]
            assert len(mock_inputs) == input_count
            assert all(suite in Path(f).parts for f in job.output)

    bias = next(job for job in jobs if job.rule == "mock_cosebis_bias_test")
    assert str(toy.covariances[config["fiducial"]["version"], "g"]) in bias.input
    assert all(suite in Path(f).parts for f in bias.output)
