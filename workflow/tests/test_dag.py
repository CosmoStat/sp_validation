"""DAG properties, checked through the host launcher (see conftest.py)."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from conftest import (
    REPO,
    STALE,
    UNCOVERED,
    VERSIONS,
    _load_module,
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
    """[P12] No job reads or writes a ξ± text dump; the ξ± figures draw the parts."""
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


def test_patch_centres_are_an_input_no_rule_draws(toy, forced, grids):
    """[P13] Every patched ξ± job, the variant's too, splits at its base
    catalogue's centres, and even a forced run writes none."""
    jobs = forced[1]
    npatch = grids["reporting"]["npatch"]
    assert npatch > 1
    centres = str(toy.cosmo_val / "patches" / f"{VERSIONS[0]}_npatch={npatch}.dat")
    for job in [j for j in jobs if j.rule == "xi"]:
        patched = int(job.wildcards["npatch"]) > 1
        assert [f for f in job.input if "/patches/" in f] == (
            [centres] if patched else []
        ), job
    drawn = [f for j in jobs for f in j.output if "/patches/" in f]
    assert not drawn, drawn


def test_custody_is_the_checkouts_and_no_rule_touches_a_blind(toy, forced, tmp_path):
    """[P9, P10, P8] A catalogue and its variant share one custody line; even a
    forced run schedules nothing that reads or writes the registry; and a
    config file declaring the catalogue unblinded changes nothing."""
    output, jobs = forced
    line = f"[custody] {VERSIONS[0]} (+ {VERSIONS[1]}): blinded under toy"
    assert [x for x in output.splitlines() if x.startswith("[custody]")] == [line]
    registry = (toy.root / "cosmo_val" / "blinds").resolve()
    assert not [j.rule for j in jobs if "blind" in j.rule]
    touched = [
        f
        for j in jobs
        for f in j.input + j.output
        if (toy.rundir / f).resolve().is_relative_to(registry)
    ]
    assert not touched, touched

    override = tmp_path / "override.yaml"
    override.write_text(yaml.safe_dump({VERSIONS[0]: {"blinding": "unblinded"}}))
    result = toy.snakemake("-n", "assemble_sacc_all", "--configfile", str(override))
    assert result.returncode == 0, result.stdout
    assert line in result.stdout


def _launch_refusals(toy, tmp_path):
    """Launches the DAG cannot honour: ``(args, Toy.snakemake kwargs, message)``."""
    campaign = tmp_path / "campaign.yaml"
    config = dict(toy.config, cosmo_val=dict(toy.config["cosmo_val"], type="mock"))
    campaign.write_text(yaml.safe_dump(config))
    return {
        # [P6] a blinded catalogue no blind covers: pull, then draw or share
        "no_blind": (
            [],
            dict(config=[f'versions=["{UNCOVERED}"]']),
            ["git pull first", "python -m sp_validation.blinding init", "share"],
        ),
        # [P11] unblinded under a concealed blind, and the removed campaign switch
        "unrevealed": (
            [],
            dict(config=[f'versions=["{STALE}"]']),
            ["blinding reveal stale"],
        ),
        "campaign_type": (
            ["--configfile", str(campaign)],
            {},
            ["custody is declared per catalogue in cosmo_val/cat_config.yaml"],
        ),
    }


@pytest.mark.parametrize("case", ["no_blind", "unrevealed", "campaign_type"])
def test_a_launch_the_dag_cannot_honour_stops_with_the_fix(toy, tmp_path, case):
    args, launch, message = _launch_refusals(toy, tmp_path)[case]
    result = toy.snakemake("-n", "assemble_sacc_all", *args, **launch)
    assert result.returncode != 0, result.stdout
    for fragment in message:
        assert fragment in result.stdout, result.stdout
    assert "rule assemble_sacc" not in result.stdout


def test_outputs_stay_in_the_output_roots(toy, forced):
    """[P2] Nothing the suite declares lands outside the configured output roots."""
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


def _stand_in_centres(paper, cosmo_val):
    """Touch the centres ``paper``'s patched ξ± grids read; a dry-run reads none."""
    common = _load_module(
        REPO / "workflow" / "common.py", f"{paper}_common", {"COSMO_VAL": cosmo_val}
    )
    common.CATALOG_CONFIG = yaml.safe_load(Path(common.CAT_CONFIG).read_text())
    config = yaml.safe_load(
        (REPO / "papers" / paper / "config" / "config.yaml").read_text()
    )
    for grid in common.xi_grids(config, config["fiducial"]).values():
        for version in config["versions"] if grid["npatch"] > 1 else ():
            centres = Path(common.patches_path(version, grid["npatch"]))
            centres.parent.mkdir(parents=True, exist_ok=True)
            centres.touch()


def _real_dry_run(paper, targets, cosmo_val):
    env = {k: v for k, v in os.environ.items() if k != "SNAKEMAKE_PROFILE"}
    env.update(PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1", COSMO_VAL=cosmo_val)
    _stand_in_centres(paper, cosmo_val)
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
def test_papers_resolve_on_candide(paper, targets, tmp_path):
    """The real paper DAGs resolve against the real catalogues and your image.

    The products land in a tree of their own, holding stand-in patch centres.
    """
    result = _real_dry_run(paper, targets, str(tmp_path))
    assert result.returncode == 0, result.stdout
