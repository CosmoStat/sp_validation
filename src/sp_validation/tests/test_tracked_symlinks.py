"""Back-pressure guard #5: every tracked symlink resolves."""

import subprocess
from pathlib import Path

import pytest

KNOWN_DANGLING_SYMLINKS = {
    Path("cosmo_inference/notebooks/2D_bmodes_paper_workflow/scripts/claims_server.py"),
    Path(
        "cosmo_inference/notebooks/2D_bmodes_paper_workflow/scripts/"
        "generate_claims_dashboard.py"
    ),
}


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate repo root (no pyproject.toml above test)")


def _tracked_symlinks() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-s"],
        cwd=_repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    return [
        _repo_root() / line.split(maxsplit=3)[3]
        for line in result.stdout.splitlines()
        if line.startswith("120000 ")
    ]


@pytest.mark.xfail(
    reason=(
        "pre-existing dangling tracked symlinks in the B-modes workflow scripts: "
        "claims_server.py and generate_claims_dashboard.py"
    ),
    strict=True,
)
def test_tracked_symlinks_resolve():
    """A move that breaks a tracked symlink must go red immediately."""
    symlinks = _tracked_symlinks()
    missing = [path.relative_to(_repo_root()) for path in symlinks if not path.exists()]
    assert symlinks, "no tracked symlinks discovered"
    stale_known = KNOWN_DANGLING_SYMLINKS - set(missing)
    assert not stale_known, (
        f"KNOWN_DANGLING_SYMLINKS entries are now fixed: {stale_known}"
    )
    assert not missing, f"tracked symlinks with missing targets: {missing}"
