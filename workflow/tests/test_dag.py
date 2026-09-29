"""DAG properties, checked through the host launcher (see conftest.py)."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from conftest import (
    PUBLIC,
    REPO,
    UNCOVERED,
    UNDECLARED,
    VERSIONS,
    on_candide,
    parse_jobs,
)


@pytest.fixture(scope="module")
def forced(toy):
    """``snakemake -F -n all`` on the toy: its output and every job it schedules."""
    result = toy.snakemake("-F", "-n", "all")
    assert result.returncode == 0, result.stdout
    return result.stdout, parse_jobs(result.stdout)


@pytest.fixture(scope="module")
def grids(toy):
    return toy.common.xi_grids(toy.config, toy.config["fiducial"])


def test_xi_leaves_a_measurement_only_as_a_part(toy, forced, grids):
    """No job reads or writes a ξ± text dump; the ξ± figures draw the parts."""
    jobs = forced[1]
    dumps = [
        f for j in jobs for f in j.input + j.output if re.search(r"_xi_.*\.txt$", f)
    ]
    assert not dumps, dumps
    reporting = toy.common.grid_binning(grids["reporting"])
    for rule in ("cv_plot_2pcf", "cv_ratio_xi_sys_xi"):
        (job,) = [j for j in jobs if j.rule == rule]
        assert {Path(f).name for f in job.input if "_xi_" in Path(f).name} == {
            f"{v}_xi_{reporting}.sacc" for v in VERSIONS
        }, job.input


def test_one_integration_grid(toy, forced, grids):
    """COSEBIs and pure-E/B share the integration-grid part and its covariance."""
    jobs = forced[1]
    tag = toy.common.grid_binning(grids["integration"])
    for version in VERSIONS:
        by_rule = {
            j.rule: set(j.input) for j in jobs if j.wildcards.get("version") == version
        }
        part = str(toy.cosmo_val / f"{version}_xi_{tag}.sacc")
        covariance = str(toy.covariances[version, "g"])
        assert by_rule["cv_cosebis"] == {part, covariance}, by_rule["cv_cosebis"]
        assert {part, covariance} <= by_rule["cv_pure_eb"], by_rule["cv_pure_eb"]


def test_custody_is_the_checkouts_and_no_rule_touches_a_blind(toy, forced, tmp_path):
    """A catalogue and its variant share one custody line; even a
    forced run schedules nothing that reads or writes the registry; and a
    config file declaring the catalogue public changes nothing."""
    output, jobs = forced
    line = f"[custody] {VERSIONS[0]} (+ {VERSIONS[1]}): blinded under toy"
    assert {x for x in output.splitlines() if x.startswith("[custody]")} == {line}
    registry = (toy.root / "blinds").resolve()
    assert not [j.rule for j in jobs if "blind" in j.rule]
    touched = [
        f
        for j in jobs
        for f in j.input + j.output
        if (toy.rundir / f).resolve().is_relative_to(registry)
    ]
    assert not touched, touched

    override = tmp_path / "override.yaml"
    override.write_text(yaml.safe_dump({VERSIONS[0]: {"blind": "none"}}))
    result = toy.snakemake("-n", "assemble_sacc_all", "--configfile", str(override))
    assert result.returncode == 0, result.stdout
    assert line in result.stdout


def test_a_custody_flip_reruns_the_catalogues_parts(toy, grids, tmp_path):
    """A part's params carry its catalogue's custody token, so declaring
    the catalogue public reruns the part and nothing else does."""
    env = toy.env | {"COSMO_VAL": str(tmp_path / "cosmo_val")}
    reporting = toy.common.grid_binning(grids["reporting"])
    part = tmp_path / "cosmo_val" / f"{VERSIONS[0]}_xi_{reporting}.sacc"
    # Records the job's params; the output itself must exist to be touched.
    touched = toy.snakemake("--touch", str(part), env=env)
    assert touched.returncode == 0, touched.stdout
    part.parent.mkdir(exist_ok=True)
    part.touch()

    def scheduled():
        result = toy.snakemake("-n", str(part), env=env)
        assert result.returncode == 0, result.stdout
        return [j.rule for j in parse_jobs(result.stdout)], result.stdout

    rules, output = scheduled()
    assert rules == [], output
    cat_config = toy.root / "cosmo_val" / "cat_config.yaml"
    declared = cat_config.read_text()
    flipped = yaml.safe_load(declared)
    flipped[VERSIONS[0]]["blind"] = "none"
    cat_config.write_text(yaml.safe_dump(flipped))
    try:
        rules, output = scheduled()
    finally:
        cat_config.write_text(declared)
    assert rules == ["xi"], output
    assert "params have changed" in output.lower(), output


LAUNCH_REFUSALS = {
    # A blind the registry does not hold, an entry declaring none, and a
    # blinded catalogue overlaid with a public one
    "no_blind": ([UNCOVERED], [f"{UNCOVERED} is blinded under gone", "init gone"]),
    "undeclared": ([UNDECLARED], ["declares no `blind:`"]),
    "mixed": ([VERSIONS[0], PUBLIC], ["shows the blind's shift"]),
}


@pytest.mark.parametrize("case", LAUNCH_REFUSALS)
def test_a_launch_the_dag_cannot_honour_stops_with_the_fix(toy, case):
    versions, message = LAUNCH_REFUSALS[case]
    listed = ", ".join(f'"{v}"' for v in versions)
    result = toy.snakemake("-n", "assemble_sacc_all", config=[f"versions=[{listed}]"])
    assert result.returncode != 0, result.stdout
    for fragment in message:
        assert fragment in result.stdout, result.stdout
    assert "rule assemble_sacc" not in result.stdout


def test_outputs_stay_in_the_output_roots(toy, forced):
    """Nothing the suite declares lands outside the configured output roots."""
    roots = [
        r.resolve()
        for r in (toy.cosmo_val, toy.cosmo_inference, toy.rundir / "results")
    ]
    outputs = [Path(o) for j in forced[1] for o in j.output]
    assert outputs
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
