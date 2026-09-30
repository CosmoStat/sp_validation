"""DAG properties, checked through the host launcher (see conftest.py)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import REPO, VERSIONS, on_candide, parse_jobs


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
    """COSEBIs and pure-E/B share the integration-grid part and its covariance."""
    result = toy.snakemake("-n", "assemble_sacc_all")
    assert result.returncode == 0, result.stdout
    jobs = parse_jobs(result.stdout)
    grids = toy.common.xi_grids(toy.config, toy.config["fiducial"])

    tag = toy.common.grid_binning(grids["integration"])
    for version in VERSIONS:
        by_rule = {
            j.rule: set(j.input) for j in jobs if j.wildcards.get("version") == version
        }
        part = str(toy.cosmo_val / f"{version}_xi_{tag}.sacc")
        covariance = str(toy.covariances[version, "g"])
        assert by_rule["cv_cosebis"] == {part, covariance}, by_rule["cv_cosebis"]
        assert {part, covariance} <= by_rule["cv_pure_eb"], by_rule["cv_pure_eb"]


@pytest.mark.parametrize("named", [True, False], ids=["named", "unnamed"])
def test_outputs_stay_in_the_output_roots(toy, named):
    """Nothing the suite declares lands outside the configured output roots.

    A launch that names no COSMO_VAL writes into its own checkout's
    cosmo_val/output.
    """
    env = toy.env if named else {k: v for k, v in toy.env.items() if k != "COSMO_VAL"}
    cosmo_val = toy.cosmo_val if named else toy.root / "cosmo_val" / "output"
    result = toy.snakemake("-n", "all", env=env)
    assert result.returncode == 0, result.stdout
    roots = [
        r.resolve() for r in (cosmo_val, toy.cosmo_inference, toy.rundir / "results")
    ]
    outputs = [Path(o) for j in parse_jobs(result.stdout) for o in j.output]
    assert outputs, result.stdout
    strays = [
        o
        for o in outputs
        if not any((toy.rundir / o).resolve().is_relative_to(r) for r in roots)
    ]
    assert not strays, strays


def _apptainer_stub(tmp_path):
    """A PATH entry that answers where apptainer is not installed.

    The candide profile deploys with apptainer, whose version Snakemake reads
    even when it runs no job.
    """
    apptainer = tmp_path / "bin" / "apptainer"
    apptainer.parent.mkdir()
    apptainer.write_text("#!/bin/sh\necho apptainer version 1.3.4\n")
    apptainer.chmod(0o755)
    return apptainer.parent


def test_a_job_needs_no_launch_cache(toy, tmp_path):
    """A job starts on a node where the launch's XDG cache cannot exist.

    Jobs inherit the launching shell's environment, whose XDG_CACHE_HOME may be
    node-local. Under the candide profile, a Snakemake that cannot create that
    cache still starts, and a job's environment carries no XDG_CACHE_HOME for
    the Snakemake its job step starts.
    """
    blocker = tmp_path / "a-file"
    blocker.touch()
    candide = toy.root / "workflow" / "profiles" / "candide"
    result = toy.snakemake(
        "-n",
        "--profile",
        str(candide),
        "assemble_sacc_all",
        env=toy.env
        | {
            "XDG_CACHE_HOME": str(blocker / "cache"),
            "PATH": f"{_apptainer_stub(tmp_path)}{os.pathsep}{toy.env['PATH']}",
        },
    )
    assert result.returncode == 0, result.stdout

    snakefile = tmp_path / "Snakefile"
    snakefile.write_text(
        f"import sys\nsys.path.insert(0, {str(toy.root / 'workflow')!r})\n"
        "import common\n\n"
        'rule job:\n    output: "env.txt"\n'
        '    shell: "printenv XDG_CACHE_HOME > {output} || true"\n'
    )
    result = toy.snakemake(
        "-s",
        str(snakefile),
        "--directory",
        str(tmp_path),
        env=toy.env | {"XDG_CACHE_HOME": str(tmp_path / "launch-cache")},
    )
    assert result.returncode == 0, result.stdout
    assert (tmp_path / "env.txt").read_text() == ""


def _real_dry_run(paper, targets):
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
            *targets,
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
    "paper, targets",
    [
        ("cosmo_val", ["assemble_sacc_all"]),
        ("bmodes", ["paper"]),
    ],
    ids=["cosmo_val", "bmodes"],
)
def test_papers_resolve_on_candide(paper, targets):
    """The real paper DAGs resolve against the real catalogues and your image."""
    result = _real_dry_run(paper, targets)
    assert result.returncode == 0, result.stdout
