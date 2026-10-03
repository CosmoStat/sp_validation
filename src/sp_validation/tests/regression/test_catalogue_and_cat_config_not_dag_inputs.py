"""Snakemake must rerun readers when cat_config.yaml or a catalogue changes.

Repository paths are relative to this test file. The test
copies the checkout's workflow/, papers/cosmo_val/ (Snakefile, config,
cosmology JSON), cosmo_val/cat_config.yaml and src/sp_validation/container.py
into a temporary tree, points SP_v1.4.6.3's shear catalogue at a small dummy
file, plants up-to-date outputs for a set of catalogue-reading rules, and asks
Snakemake (dry run, the profiles' rerun triggers) what it would do after (a) an
edit to cat_config.yaml and (b) an in-place replacement of the catalogue.
Nothing scientific runs; only the DAG is evaluated.
"""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

# A Snakemake dry run in the container: minutes, so outside per-change CI.
pytestmark = pytest.mark.slow

WT = Path(__file__).resolve().parents[4]
VERSION = "SP_v1.4.6.3"
# The host-side Snakemake that launches the workflow (a uv tool on candide);
# the image's python does not carry it.
_SM = shutil.which("snakemake")
SNAKEMAKE = [_SM] if _SM else [sys.executable, "-m", "snakemake"]
TRIGGERS = ["mtime", "params", "input", "code"]  # workflow/profiles/*/config.yaml

# Rules whose scripts build a CosmologyValidation from cat_config (and so read
# the shear catalogue named there) for VERSION.
CONFIG_READERS = {
    "xi",
    "rho_tau_stats",
    "pseudo_cl",
    "pseudo_cl_cov",
    "cv_footprints",
    "cv_objectwise_leakage",
    "cv_weights",
    "cv_additive_bias",
}
# Of those, the ones that certainly read the shear catalogue file itself.
CATALOGUE_READERS = {"xi", "pseudo_cl", "cv_weights", "cv_additive_bias"}


def _snakemake(tree, *args):
    home = tree / "home"
    home.mkdir(exist_ok=True)
    env = dict(
        os.environ,
        COSMO_VAL=str(tree / "out"),
        COSMO_INFERENCE=str(tree / "inference"),
        PYTHONUNBUFFERED="1",
        HOME=str(home),
        XDG_CACHE_HOME=str(home / ".cache"),
        TMPDIR=str(home),
    )
    cmd = [
        *SNAKEMAKE,
        "--rerun-triggers",
        *TRIGGERS,
        "--config",
        "container=docker://example/none:latest",
        "-s",
        "Snakefile",
        "-n",
        "--nolock",
        "--cores",
        "1",
        *args,
    ]
    res = subprocess.run(
        cmd,
        cwd=tree / "papers" / "cosmo_val",
        env=env,
        capture_output=True,
        text=True,
        timeout=400,
    )
    out = res.stdout + res.stderr
    assert res.returncode == 0, out[-4000:]
    return out


def _scheduled(out):
    return set(re.findall(r"^\s*(?:local)?rule (\w+):\s*$", out, re.M))


def _outputs(out):
    files = []
    for line in out.splitlines():
        m = re.match(r"^\s*output:\s*(.+)$", line)
        if m:
            files += [f.strip() for f in m.group(1).split(",") if f.strip()]
    return files


@pytest.fixture
def tree(tmp_path):
    if _SM is None:
        pytest.importorskip(
            "snakemake", reason="Snakemake is required for DAG dry runs"
        )
    if not (WT / "papers" / "cosmo_val" / "Snakefile").exists():
        pytest.skip("not a checkout with papers/cosmo_val")
    t = tmp_path / "repo"
    shutil.copytree(
        WT / "workflow",
        t / "workflow",
        ignore=shutil.ignore_patterns("__pycache__", ".snakemake"),
    )
    pc = t / "papers" / "cosmo_val"
    pc.mkdir(parents=True)
    shutil.copy2(WT / "papers/cosmo_val/Snakefile", pc / "Snakefile")
    shutil.copytree(WT / "papers/cosmo_val/config", pc / "config")
    shutil.copytree(WT / "papers/cosmo_val/results/cosmology", pc / "results/cosmology")
    (t / "src/sp_validation").mkdir(parents=True)
    shutil.copy2(WT / "src/sp_validation/container.py", t / "src/sp_validation/")
    (t / "cosmo_val").mkdir()
    (t / "inference").mkdir()

    catalogue = t / "catalogue.fits"
    catalogue.write_text("dummy shear catalogue v1\n")
    text = (WT / "cosmo_val/cat_config.yaml").read_text()
    real = yaml.safe_load(text)[VERSION]["shear"]["path"]
    assert text.count(real) >= 1
    (t / "cosmo_val/cat_config.yaml").write_text(text.replace(real, str(catalogue)))

    cfg = yaml.safe_load((pc / "config/config.yaml").read_text())
    cv = cfg["cosmo_val"]
    tag = (
        f"minsep={float(cv['theta_min'])}_maxsep={float(cv['theta_max'])}"
        f"_nbins={cv['nbins']}_npatch={cv['npatch']}"
    )
    nb = cv["n_ell_bins"]
    out = t / "out"
    targets = [
        str(out / f"{VERSION}_xi_{tag}.txt"),
        str(out / f"rho_tau_stats/rho_stats_{VERSION}_{tag}.fits"),
        str(out / f"pseudo_cl_{VERSION}_{cv['binning']}_nbins={nb}.sacc"),
        str(out / f"pseudo_cl_cov_{VERSION}_{cv['binning']}_nbins={nb}.fits"),
        "cv_footprints",
        "cv_objectwise_leakage",
        "cv_weights",
        "cv_additive_bias",
    ]

    # Everything the workflow is made of, and its inputs, is old ...
    t0 = time.time() - 2000
    for p in t.rglob("*"):
        if p.is_file():
            os.utime(p, (t0, t0))
    first = _snakemake(t, *targets)
    assert CONFIG_READERS <= _scheduled(first), first[-3000:]
    # ... and the products are newer: a finished run.
    t1 = time.time() - 1000
    for f in _outputs(first):
        p = Path(f) if os.path.isabs(f) else pc / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        os.utime(p, (t1, t1))
    settled = _snakemake(t, *targets)
    assert not (_scheduled(settled) & CONFIG_READERS), settled[-3000:]
    return t, targets, catalogue


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#381: cat_config.yaml is not a DAG input to its readers",
)
def test_cat_config_edit_reruns_rules_that_read_it(tree):
    """An edit to cat_config.yaml must mark every rule that reads it stale.

    cat_config.yaml decides which catalogue, weight column, response R,
    leakage-corrected ellipticity columns and n(z) each version uses, so a change
    there changes xi±, rho/tau, pseudo-Cl and the cv diagnostics. After changing
    SP_v1.4.6.3's w_col on a settled tree, Snakemake (with the profiles'
    rerun-triggers) must schedule every rule in CONFIG_READERS; the right answer
    is by construction, since each of them passes cat_config to
    CosmologyValidation. On a workflow that passes cat_config only as a path
    string in params, the dry run reports nothing to do.
    """
    t, targets, _ = tree
    cc = t / "cosmo_val/cat_config.yaml"
    text = cc.read_text()
    block = text.index(f"\n{VERSION}:")
    i = text.index("w_col:", block)
    j = text.index("\n", i)
    cc.write_text(text[:i] + "w_col: w_iv" + text[j:])  # mtime: now
    out = _snakemake(t, *targets)
    missed = sorted(CONFIG_READERS - _scheduled(out))
    summary = "Nothing to be done" if "Nothing to be done" in out else out[-800:]
    assert not missed, (
        f"after editing {VERSION}.shear.w_col in cat_config.yaml, these rules were "
        f"not re-scheduled: {missed}; scheduled: {sorted(_scheduled(out))}; "
        f"snakemake: {summary}"
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#381: catalogue is not a DAG input to all its readers",
)
def test_catalogue_replaced_in_place_reruns_rules_that_read_it(tree):
    """Replacing the shear catalogue file must mark every rule that reads it stale.

    A catalogue overwritten in place (same path, new contents and mtime) changes
    every statistic measured from it. xi declares the catalogue as input and so
    re-runs; pseudo_cl, cv_weights and cv_additive_bias read the same file through
    cat_config and must re-run too.
    """
    t, targets, catalogue = tree
    catalogue.write_text("dummy shear catalogue v2\n")  # mtime: now
    out = _snakemake(t, *targets)
    missed = sorted(CATALOGUE_READERS - _scheduled(out))
    assert not missed, (
        f"after replacing {VERSION}'s shear catalogue in place, these rules were "
        f"not re-scheduled: {missed}; scheduled: {sorted(_scheduled(out))}"
    )
