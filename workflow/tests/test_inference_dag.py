"""Inference starts from declared SACCs, not obsolete ξ-file producers."""

import yaml
from conftest import parse_jobs


def test_sacc_to_summary_dag(toy, tmp_path):
    inputs = {}
    for key in (
        "sacc",
        "covariance",
        "template",
        "values",
        "priors",
        "rho_stats",
        "tau_stats",
    ):
        path = tmp_path / key
        path.touch()
        inputs[key] = str(path)
    inputs.update(n_bins=1, threads=8)
    out = tmp_path / "out"
    config = tmp_path / "inference.yaml"
    config.write_text(
        yaml.safe_dump(
            {"inference": {"output_dir": str(out), "runs": {"mock": inputs}}}
        )
    )
    result = toy.snakemake(
        "-n",
        "-s",
        str(toy.root / "workflow/inference/Snakefile"),
        "inference_fiducial",
        "--configfile",
        str(config),
    )
    assert result.returncode == 0, result.stdout
    jobs = {j.rule: j for j in parse_jobs(result.stdout)}
    assert set(jobs) == {
        "inference_prep",
        "inference_config",
        "inference_chain",
        "inference_summary",
        "inference_fiducial",
    }
    assert set(jobs["inference_prep"].input) == {
        inputs[k] for k in ("sacc", "covariance", "rho_stats", "tau_stats")
    }
    assert set(jobs["inference_config"].input) == {
        str(out / "mock/data.fits"),
        *[inputs[k] for k in ("template", "values", "priors")],
    }
    assert str(out / "mock/chain.txt") in jobs["inference_summary"].input
    assert str(out / "mock/summary.json") in jobs["inference_fiducial"].input
    assert all("xi_plus_" not in p for job in jobs.values() for p in job.input)
