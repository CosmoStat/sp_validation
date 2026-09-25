"""DAG properties, checked through the host launcher (see conftest.py)."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import HOST_PYTHON, REPO, VERSIONS, fake_image, on_candide, parse_jobs


def test_assemble_resolves(toy):
    """P1: the terminal files resolve from a catalogue config and a checkout."""
    result = toy.snakemake("-n", "assemble_sacc_all")
    assert result.returncode == 0, result.stdout
    assembled = {Path(o).name for j in parse_jobs(result.stdout) for o in j.output}
    assert {f"{v}.sacc" for v in VERSIONS} <= assembled, result.stdout


def test_one_integration_grid(toy):
    """P1: ξ± is measured on two grids, and both B-mode statistics share one.

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
        (covariance,) = [
            f for f in by_rule["cv_pure_eb"] if Path(f).name.startswith("covariance_")
        ]
        assert "_g_" in Path(covariance).name
        assert by_rule["cv_cosebis"] == {part, covariance}, by_rule["cv_cosebis"]
        assert part in by_rule["cv_pure_eb"]


def test_outputs_stay_in_the_output_roots(toy):
    """P2: nothing the DAG declares lands outside the configured output roots."""
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
    """P3: an image whose Python minor or Snakemake differs stops the launch."""
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
    """P3: a registry tag cannot be inspected; the launch says so and goes on."""
    result = toy.snakemake(
        "-n", "assemble_sacc_all", container="docker://example.org/image:tag"
    )
    assert result.returncode == 0, result.stdout
    assert "parity unchecked" in result.stdout


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
    """P4: the real paper DAGs resolve against the real catalogues and image.

    A local image (your SIF or sandbox) is read, so parity was checked.
    """
    result = _real_dry_run(paper, target)
    assert result.returncode == 0, result.stdout
    if _resolve_image()[1] != "tag":
        assert "parity unchecked" not in result.stdout, result.stdout


def _resolve_image():
    spec = importlib.util.spec_from_file_location(
        "container", REPO / "src" / "sp_validation" / "container.py"
    )
    container = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(container)
    return container.resolve_image()
