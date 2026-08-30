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

    Returns the CompletedProcess. PYTHONUNBUFFERED satisfies the Snakefile's
    ``envvars:`` declaration without depending on the invoking shell. A dry run
    resolves the DAG only — it never dispatches jobs — so drop any inherited
    SNAKEMAKE_PROFILE (e.g. the login shell's "slurm" profile), which would
    otherwise force an executor plugin the test environment need not have. And
    invoke snakemake through sys.executable (the interpreter pytest, hence
    snakemake, lives in) — a bare python3.12 resolves off PATH to e.g. an
    intel-python without snakemake.
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
    born-as-SACC + assemble DAG, and assemble pulls the tagged pseudo-Cl + cov.

    Targets the assemble_sacc_all rule so every version's assemble_sacc job
    appears. The dry run resolves the DAG structure only — it never executes the
    assemble script — so the placeholder-cov opt-in (cosmo_val.allow_placeholder_cov)
    is irrelevant here; a real run would need it (or a wired --xi-cov) to proceed,
    which is the fail-loud-by-default behaviour asserted in test_assemble_sacc."""
    version = "SP_v1.4.6.3_leak_corr"
    result = _dry_run(_repo_root() / "papers/cosmo_val", ["assemble_sacc_all"])
    assert result.returncode == 0, result.stdout
    # assemble_sacc must be in the DAG and pull the tagged, blinded pseudo-Cl
    # part + its NaMaster covariance (not the untagged cv_pseudo_cl diagnostic),
    # plus all five per-statistic parts.
    out = result.stdout
    assert "rule assemble_sacc:" in out, out
    assert f"pseudo_cl_{version}_blind=A_powspace_nbins=32.sacc" in out, out
    assert f"pseudo_cl_cov_{version}_blind=A_powspace_nbins=32.fits" in out, out
    # The ξ± part is named by its reporting binning (one binning-agnostic `xi`
    # rule serves the reporting and integration grids alike).
    for part in ("_xi_minsep=", "_cosebis.sacc", "_pure_eb.sacc", "rho_tau_"):
        assert part in out, f"missing {part} part in assemble DAG:\n{out}"
