"""Back-pressure guard #5: every tracked symlink resolves."""

import subprocess
from pathlib import Path


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate repo root (no pyproject.toml above test)")


def _tracked_symlinks() -> list[Path]:
    """Tracked symlinks via git; full-tree walk where .git is absent.

    The CI image is built from the Git build context — tracked content only,
    no .git directory — so walking the tree there finds exactly the tracked
    symlinks and the guard keeps its teeth in-image.
    """
    root = _repo_root()
    result = subprocess.run(
        ["git", "ls-files", "-s"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode == 0:
        return [
            root / line.split(maxsplit=3)[3]
            for line in result.stdout.splitlines()
            if line.startswith("120000 ")
        ]
    return [
        path
        for path in root.rglob("*")
        if path.is_symlink() and ".git" not in path.parts
    ]


def test_tracked_symlinks_resolve():
    """A move that breaks a tracked symlink must go red immediately."""
    symlinks = _tracked_symlinks()
    missing = [path.relative_to(_repo_root()) for path in symlinks if not path.exists()]
    assert symlinks, "no tracked symlinks discovered"
    assert not missing, f"tracked symlinks with missing targets: {missing}"
