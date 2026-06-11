"""Back-pressure guard #2: the B-modes Snakemake workflow dry-runs.

The reorg is allowed to change the rule graph; this guard only asserts that
Snakemake can still parse the workflow and construct a dry run.
"""

import os
import subprocess
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


@requires_candide_data
def test_bmodes_workflow_dry_runs():
    """The paper B-mode workflow must still parse and dry-run cleanly."""
    workflow_dir = _repo_root() / "papers/bmodes"
    # PYTHONUNBUFFERED satisfies the Snakefile's `envvars:` declaration without
    # depending on the invoking shell's environment.
    env = os.environ | {"PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(
        [
            "python3.12",
            "-m",
            "snakemake",
            "all_tapestry",
            "--dry-run",
            "--cores",
            "1",
            "--configfile",
            "config/config.yaml",
        ],
        cwd=workflow_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout
