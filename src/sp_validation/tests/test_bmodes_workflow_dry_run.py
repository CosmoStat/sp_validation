"""Back-pressure guard #2: the paper Snakemake workflows dry-run.

The reorg is allowed to change the rule graph; these guards only assert that
Snakemake can still parse each composed workflow and construct a dry run. One
guard covers papers/bmodes (config space, no cosmo_val block); a second covers
papers/cosmo_val, whose config DOES carry a cosmo_val block — so it is the only
one that includes cosmo_val.smk and hence the born-as-SACC + assemble rules.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# The workflow composes a catalog configfile and terminal inputs that live at
# candide-absolute paths (Snakefile line 5, workflow/common.py), so the dry
# run can only be constructed on the cluster. Same pattern as test_cosmo_val.
requires_candide_data = pytest.mark.skipif(
    not Path("/n17data/cdaley/unions").exists(),
    reason="candide-local workflow config/data (/n17data) absent — off-cluster",
)


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate repo root (no pyproject.toml above test)")


def _dry_run(workflow_dir, targets, *extra_snakemake_args):
    """Construct a dry run of the paper workflow at ``workflow_dir``.

    PYTHONUNBUFFERED satisfies the Snakefile's ``envvars:`` declaration. A dry
    run never dispatches jobs, so any inherited SNAKEMAKE_PROFILE is dropped
    rather than requiring its executor plugin. snakemake is invoked through
    sys.executable, since a bare python3.12 may resolve off PATH to an
    interpreter without it.
    """
    env = os.environ | {"PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1"}
    env.pop("SNAKEMAKE_PROFILE", None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            *targets,
            "--dry-run",
            "--cores",
            "1",
            "--configfile",
            "config/config.yaml",
            *extra_snakemake_args,
        ],
        cwd=workflow_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )


@requires_candide_data
def test_bmodes_workflow_dry_runs():
    """The paper B-mode workflow must still parse and dry-run cleanly."""
    result = _dry_run(_repo_root() / "papers/bmodes", ["all_tapestry"])
    assert result.returncode == 0, result.stdout


@requires_candide_data
def test_cosmo_val_workflow_assemble_dry_runs():
    """The cosmo_val workflow (the only one including cosmo_val.smk) resolves the
    born-as-SACC + assemble DAG, and assemble pulls the tagged pseudo-Cl + cov."""
    version = "SP_v1.4.6.3_leak_corr"
    result = _dry_run(_repo_root() / "papers/cosmo_val", ["assemble_sacc_all"])
    assert result.returncode == 0, result.stdout
    # assemble_sacc must pull the tagged pseudo-Cl part + its NaMaster
    # covariance (not the untagged cv_pseudo_cl diagnostic), plus every part.
    out = result.stdout
    assert "rule assemble_sacc:" in out, out
    assert f"pseudo_cl_analysis_{version}_blind=A_powspace_nbins=32.sacc" in out, out
    assert f"pseudo_cl_cov_{version}_blind=A_powspace_nbins=32.fits" in out, out
    for part in ("_xi_minsep=", "_cosebis.sacc", "_pure_eb.sacc", "rho_tau_"):
        assert part in out, f"missing {part} part in assemble DAG:\n{out}"
