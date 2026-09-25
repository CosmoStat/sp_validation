"""DAG properties, checked through the host launcher (see conftest.py)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from conftest import HOST_PYTHON, REPO, VERSIONS, fake_image, on_candide, parse_jobs


def test_assemble_resolves(toy):
    """Each terminal file gathers every part and the analytic covariances.

    The ξ± block takes the CosmoCov covariance on the reporting grid, and the
    harmonic block is the part on the fiducial harmonic binning with the
    NaMaster covariance of that same binning.
    """
    result = toy.snakemake("-n", "assemble_sacc_all")
    assert result.returncode == 0, result.stdout
    jobs = [j for j in parse_jobs(result.stdout) if j.rule == "assemble_sacc"]
    grids = toy.common.xi_grids(toy.config, toy.config["fiducial"])
    reporting = toy.common.grid_binning(grids["reporting"])
    harmonic = toy.common.pseudo_cl_tag(toy.config)
    assert sorted(j.wildcards["version"] for j in jobs) == sorted(VERSIONS)
    for job in jobs:
        version = job.wildcards["version"]
        assert [Path(o).name for o in job.output] == [f"{version}.sacc"]
        assert {Path(f).name for f in job.input} == {
            f"{version}_xi_{reporting}.sacc",
            toy.covariances[version, "ng"].name,
            f"pseudo_cl_{version}_{harmonic}.sacc",
            f"pseudo_cl_cov_{version}_{harmonic}.fits",
            f"{version}_cosebis.sacc",
            f"{version}_pure_eb.sacc",
            f"rho_tau_{version}_{reporting}.sacc",
        }, job.input


def test_one_integration_grid(toy):
    """ξ± is measured on two grids, and both B-mode statistics share one.

    COSEBIs and pure-E/B read the same integration-grid part and the same
    CosmoCov covariance on that grid; no other binning is measured.
    """
    result = toy.snakemake("-n", "assemble_sacc_all")
    assert result.returncode == 0, result.stdout
    jobs = parse_jobs(result.stdout)
    grids = toy.common.xi_grids(toy.config, toy.config["fiducial"])

    measured = {
        (j.wildcards["version"], toy.common.grid_of(grids, j.wildcards))
        for j in jobs
        if j.rule == "xi"
    }
    assert len([j for j in jobs if j.rule == "xi"]) == len(measured), result.stdout
    assert measured == {(v, g) for v in VERSIONS for g in grids}, measured
    assert set(grids) == {"reporting", "integration"}

    tag = toy.common.grid_binning(grids["integration"])
    for version in VERSIONS:
        by_rule = {
            j.rule: set(j.input) for j in jobs if j.wildcards.get("version") == version
        }
        part = str(toy.cosmo_val / f"{version}_xi_{tag}.sacc")
        covariance = str(toy.covariances[version, "g"])
        assert by_rule["cv_cosebis"] == {part, covariance}, by_rule["cv_cosebis"]
        assert {part, covariance} <= by_rule["cv_pure_eb"], by_rule["cv_pure_eb"]


def test_outputs_stay_in_the_output_roots(toy):
    """Nothing the suite declares lands outside the configured output roots."""
    result = toy.snakemake("-n", "all")
    assert result.returncode == 0, result.stdout
    roots = [
        r.resolve()
        for r in (toy.cosmo_val, toy.cosmo_inference, toy.rundir / "results")
    ]
    outputs = [Path(o) for j in parse_jobs(result.stdout) for o in j.output]
    assert outputs, result.stdout
    strays = [
        o
        for o in outputs
        if not any((toy.rundir / o).resolve().is_relative_to(r) for r in roots)
    ]
    assert not strays, strays


@pytest.mark.parametrize(
    "python, snakemake_version",
    [("3.13.1", None), (None, "9.0.0")],
    ids=["python-minor", "snakemake"],
)
def test_image_parity_is_checked_at_launch(toy, tmp_path, python, snakemake_version):
    """An image whose Python minor or Snakemake differs stops the launch."""
    image = fake_image(
        tmp_path / "image",
        **({"python": python} if python else {}),
        **({"snakemake_version": snakemake_version} if snakemake_version else {}),
    )
    result = toy.snakemake("-n", "assemble_sacc_all", container=image)
    assert result.returncode != 0, result.stdout
    minor = ".".join((python or HOST_PYTHON).split(".")[:2])
    assert f"uv tool install --force --python {minor} snakemake==" in result.stdout
    assert "rule assemble_sacc" not in result.stdout


def test_unreadable_image_is_named_not_fatal(toy):
    """A registry tag cannot be inspected; the launch says so and goes on."""
    result = toy.snakemake(
        "-n", "assemble_sacc_all", container="docker://example.org/image:tag"
    )
    assert result.returncode == 0, result.stdout
    assert "parity unchecked" in result.stdout


def test_launch_reads_the_image_under_home(toy, tmp_path):
    """The launch finds your image under ~/.cache, whatever XDG_CACHE_HOME says.

    Jobs on other nodes run the image from the path the launching host
    resolved, and a cluster's XDG_CACHE_HOME is often node-local. The image here
    reports another Python, so reading it stops the launch.
    """
    home = tmp_path / "home"
    fake_image(home / ".cache" / "sp_validation" / "sandbox", python="3.13.1")
    env = {k: v for k, v in toy.env.items() if not k.startswith("SPV_")}
    env.update(HOME=str(home), XDG_CACHE_HOME=str(tmp_path / "node-local"))
    result = toy.snakemake("-n", "assemble_sacc_all", container=False, env=env)
    assert result.returncode != 0, result.stdout
    assert "uv tool install --force --python 3.13 snakemake==" in result.stdout


def test_image_sims_checks_parity_at_launch(toy, tmp_path):
    """The standalone image-sims workflow stops on a mismatched image too."""
    run = {
        "image_sims": {
            "sif": str(fake_image(tmp_path / "image", python="3.13.1")),
            "grids_base": str(tmp_path / "grids"),
            "mask_config": "mask.yaml",
            "match_radius_deg": 0.0002,
            "w_cols": ["none"],
            "pair_match": True,
            "n_bootstrap": 1,
            "bootstrap_seed": 0,
        }
    }
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(run))
    result = toy.snakemake(
        "-n",
        "-s",
        "workflow/image_sims/Snakefile",
        "--configfile",
        str(tmp_path / "run.yaml"),
        container=False,
        cwd=toy.root,
    )
    assert result.returncode != 0, result.stdout
    assert "uv tool install --force --python 3.13 snakemake==" in result.stdout


def _real_dry_run(paper, target):
    env = {k: v for k, v in os.environ.items() if k != "SNAKEMAKE_PROFILE"}
    env.update(PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "-n",
            "--profile",
            str(REPO / "workflow" / "profiles" / "candide"),
            target,
        ],
        cwd=REPO / "papers" / paper,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
        check=False,
    )


@pytest.mark.candide
@on_candide
@pytest.mark.parametrize(
    "paper, target", [("cosmo_val", "assemble_sacc_all"), ("bmodes", "all_tapestry")]
)
def test_papers_resolve_on_candide(paper, target):
    """The real paper DAGs resolve against the real catalogues and your image.

    Your image (the SIF or sandbox `spv-container` manages) is read, so parity
    was checked.
    """
    result = _real_dry_run(paper, target)
    assert result.returncode == 0, result.stdout
    assert "parity unchecked" not in result.stdout, (
        "no local image was read; run `spv-container pull`\n" + result.stdout
    )
