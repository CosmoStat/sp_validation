"""DAG properties, checked through the host launcher (see conftest.py)."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from conftest import (
    HOST_PYTHON,
    REPO,
    STALE,
    UNCOVERED,
    VERSIONS,
    _load_module,
    fake_image,
    on_candide,
    parse_jobs,
)


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


def test_xi_leaves_a_measurement_only_as_a_part(toy):
    """No job reads or writes a ξ± text dump; the ξ± figures draw the parts."""
    result = toy.snakemake("-n", "all")
    assert result.returncode == 0, result.stdout
    jobs = parse_jobs(result.stdout)
    dumps = [
        f for j in jobs for f in j.input + j.output if re.search(r"_xi_.*\.txt$", f)
    ]
    assert not dumps, dumps

    grids = toy.common.xi_grids(toy.config, toy.config["fiducial"])
    reporting = toy.common.grid_binning(grids["reporting"])
    for rule in ("cv_plot_2pcf", "cv_ratio_xi_sys_xi"):
        (job,) = [j for j in jobs if j.rule == rule]
        assert {Path(f).name for f in job.input if "_xi_" in Path(f).name} == {
            f"{v}_xi_{reporting}.sacc" for v in VERSIONS
        }, job.input


def test_patch_centres_are_an_input_no_rule_draws(toy):
    """Every patched ξ± job splits at its base catalogue's centres, and even a
    forced run writes none: a re-draw cannot be reproduced."""
    result = toy.snakemake("-F", "-n", "assemble_sacc_all")
    assert result.returncode == 0, result.stdout
    jobs = parse_jobs(result.stdout)
    npatch = toy.common.xi_grids(toy.config, toy.config["fiducial"])["reporting"][
        "npatch"
    ]
    assert npatch > 1
    centres = str(toy.cosmo_val / "patches" / f"{VERSIONS[0]}_npatch={npatch}.dat")

    xi = [j for j in jobs if j.rule == "xi"]
    assert {j.wildcards["version"] for j in xi} == set(VERSIONS)
    for job in xi:
        patched = int(job.wildcards["npatch"]) > 1
        assert [f for f in job.input if "/patches/" in f] == (
            [centres] if patched else []
        ), job
    drawn = [f for j in jobs for f in j.output if "/patches/" in f]
    assert not drawn, drawn


def test_a_tree_without_centres_stops_the_launch(toy, tmp_path):
    """A launch into a tree without the centres names the command that draws
    them, for the base catalogue."""
    npatch = toy.common.xi_grids(toy.config, toy.config["fiducial"])["reporting"][
        "npatch"
    ]
    result = toy.snakemake(
        "-n", "assemble_sacc_all", env={**toy.env, "COSMO_VAL": str(tmp_path)}
    )
    plain = toy.common._plain
    assert result.returncode != 0, result.stdout
    assert (
        f"python -m sp_validation.cosmo_val.patch_centers {VERSIONS[0]} {npatch} "
        f"--cat-config {plain(toy.root / 'cosmo_val' / 'cat_config.yaml')} "
        f"--output-dir {plain(tmp_path)}"
    ) in result.stdout, result.stdout


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


@pytest.mark.parametrize("root", ["COSMO_VAL", "COSMO_INFERENCE"])
def test_every_spelling_of_an_output_root_declares_the_same_paths(toy, tmp_path, root):
    """Snakemake keys its persistence records by path string, so a symlinked
    spelling of an output root declares the tree's resolved paths."""
    tree = Path(toy.env[root]).resolve()
    link = tmp_path / "link"
    link.symlink_to(tree, target_is_directory=True)
    result = toy.snakemake("-n", "all", env={**toy.env, root: str(link)})
    assert result.returncode == 0, result.stdout
    declared = [f for j in parse_jobs(result.stdout) for f in j.input + j.output]
    assert any(f.startswith(f"{tree}/") for f in declared), declared
    assert not [f for f in declared if f.startswith(f"{link}/")], declared


def test_output_roots_take_the_plain_spelling(toy):
    """A root given as /automnt/<disk>/... is declared as /<disk>/..., the one
    spelling every node has, on any host: a job step re-derives the launch's
    paths on its own node, and the node that owns a disk has neither
    /automnt/<disk> nor a /<disk> link to it, like the disk no host has here.
    A file target named in the plain spelling resolves, and no declared path
    lies under /automnt."""
    tree = Path("/n00data0/spv-dag-toy")  # a dry-run creates nothing
    env = {
        **toy.env,
        "COSMO_VAL": f"/automnt{tree}/val",
        "COSMO_INFERENCE": f"/automnt{tree}/inference",
    }
    common = _load_module(toy.root / "workflow" / "common.py", "plain_common", env)
    assert (common.COSMO_VAL, common.COSMO_INFERENCE) == (
        tree / "val",
        tree / "inference",
    )
    grids = toy.common.xi_grids(toy.config, toy.config["fiducial"])
    integration = toy.common.grid_binning(grids["integration"])
    target = tree / "val" / f"{VERSIONS[0]}_xi_{integration}.sacc"
    result = toy.snakemake("-n", str(target), env=env)
    assert result.returncode == 0, result.stdout
    declared = [f for j in parse_jobs(result.stdout) for f in j.input + j.output]
    assert str(target) in declared, declared
    assert not [f for f in declared if f.startswith("/automnt/")], declared


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
        container=False,
        env=toy.env | {"XDG_CACHE_HOME": str(tmp_path / "launch-cache")},
    )
    assert result.returncode == 0, result.stdout
    assert (tmp_path / "env.txt").read_text() == ""


def test_the_candide_profile_bounds_its_jobs(tmp_path):
    """A real launch through the candide profile needs no --jobs.

    Snakemake refuses a real run on a remote executor without a job bound. The
    target is up to date, so the launch submits nothing and runs on any host;
    the same launch through the profile stripped of its bound shows the refusal.
    """
    (tmp_path / "Snakefile").write_text(
        'rule done:\n    output: "done.txt"\n    shell: "touch {output}"\n'
    )
    (tmp_path / "done.txt").touch()
    env = {k: v for k, v in os.environ.items() if k != "SNAKEMAKE_PROFILE"}
    env["PATH"] = f"{_apptainer_stub(tmp_path)}{os.pathsep}{env['PATH']}"
    candide = REPO / "workflow" / "profiles" / "candide"
    unbounded = tmp_path / "unbounded"
    unbounded.mkdir()
    profile = yaml.safe_load((candide / "config.yaml").read_text())
    profile.pop("jobs", None)
    (unbounded / "config.yaml").write_text(yaml.safe_dump(profile))

    def launch(profile_dir):
        return subprocess.run(
            [sys.executable, "-m", "snakemake", "--profile", str(profile_dir)],
            cwd=tmp_path,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )

    result = launch(candide)
    assert result.returncode == 0, result.stdout
    assert "Nothing to be done" in result.stdout, result.stdout
    refused = launch(unbounded)
    assert refused.returncode != 0 and "--jobs" in refused.stdout, refused.stdout


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


def _custody_lines(output):
    return [line for line in output.splitlines() if line.startswith("[custody]")]


def test_a_catalogue_without_a_blind_stops_the_launch(toy):
    """A blinded catalogue with no blind fails at parse: pull, then draw or share."""
    result = toy.snakemake(
        "-n", "assemble_sacc_all", config=[f'versions=["{UNCOVERED}"]']
    )
    assert result.returncode != 0, result.stdout
    assert "git pull first" in result.stdout
    assert "python -m sp_validation.blinding init" in result.stdout
    assert "share" in result.stdout
    assert "rule assemble_sacc" not in result.stdout


def test_no_config_line_unblinds_a_catalogue(toy, tmp_path):
    """Custody is read from the checkout's cat_config, never from the merged config."""
    override = tmp_path / "override.yaml"
    override.write_text(yaml.safe_dump({VERSIONS[0]: {"blinding": "unblinded"}}))
    result = toy.snakemake("-n", "assemble_sacc_all", "--configfile", str(override))
    assert result.returncode == 0, result.stdout
    assert f"[custody] {VERSIONS[0]} (+ {VERSIONS[1]}): blinded under toy" in (
        _custody_lines(result.stdout)
    )


def test_a_catalogue_and_its_variant_share_one_custody(toy):
    result = toy.snakemake("-n", "assemble_sacc_all")
    assert result.returncode == 0, result.stdout
    assert _custody_lines(result.stdout) == [
        f"[custody] {VERSIONS[0]} (+ {VERSIONS[1]}): blinded under toy"
    ]


def test_no_rule_draws_or_touches_a_blind(toy):
    """Even a forced run schedules nothing that reads or writes the registry."""
    result = toy.snakemake("-F", "-n", "all")
    assert result.returncode == 0, result.stdout
    jobs = parse_jobs(result.stdout)
    registry = (toy.root / "cosmo_val" / "blinds").resolve()
    assert jobs
    assert not [j.rule for j in jobs if "blind" in j.rule]
    touched = [
        f
        for j in jobs
        for f in j.input + j.output
        if (toy.rundir / f).resolve().is_relative_to(registry)
    ]
    assert not touched, touched


def test_the_campaign_type_switch_is_refused(toy, tmp_path):
    """A config carrying cosmo_val.type stops the launch with the pointer."""
    config = dict(toy.config, cosmo_val=dict(toy.config["cosmo_val"], type="mock"))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    result = toy.snakemake("-n", "assemble_sacc_all", "--configfile", str(path))
    assert result.returncode != 0, result.stdout
    assert "custody is declared per catalogue in cosmo_val/cat_config.yaml" in (
        result.stdout
    )


def test_unblinding_a_concealed_catalogue_needs_the_reveal(toy):
    result = toy.snakemake("-n", "assemble_sacc_all", config=[f'versions=["{STALE}"]'])
    assert result.returncode != 0, result.stdout
    assert "blinding reveal stale" in result.stdout


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
        ("bmodes", ["all_tapestry", "results/ecut/SP_v1.4.6_ecut07.fits"]),
    ],
    ids=["cosmo_val", "bmodes"],
)
def test_papers_resolve_on_candide(paper, targets, tmp_path):
    """The real paper DAGs resolve against the real catalogues and your image.

    The e-cut catalogue reads its parent's catalogue entry. Your image (the SIF
    or sandbox `spv-container` manages) is read, so parity was checked. The
    products land in a tree of their own, holding stand-in patch centres.
    """
    result = _real_dry_run(paper, targets, str(tmp_path))
    assert result.returncode == 0, result.stdout
    assert "parity unchecked" not in result.stdout, (
        "no local image was read; run `spv-container pull`\n" + result.stdout
    )
