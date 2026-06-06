"""Back-pressure guard #5: every tracked symlink resolves."""

import subprocess
from pathlib import Path


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


def test_tracked_symlinks_resolve():
    """A move that breaks a tracked symlink must go red immediately."""
    symlinks = _tracked_symlinks()
    missing = [path.relative_to(_repo_root()) for path in symlinks if not path.exists()]
    assert symlinks, "no tracked symlinks discovered"
    assert not missing, f"tracked symlinks with missing targets: {missing}"
