"""Back-pressure guard #6: moved internal directories leave no stale refs."""

import subprocess
from pathlib import Path

import pytest

# Add (old_internal_dir, new_location) pairs as the restructuring starts moving
# directories. Empty pre-reorg by design: the harness is the invariant.
MOVE_MAP: tuple[tuple[str, str], ...] = (
    ("cosmo_inference/notebooks/2D_bmodes_paper_workflow", "papers/bmodes"),
)
EXCLUDED_DIRS = {
    ".git",
    ".felt",
    ".snakemake",
    "__pycache__",
    "results",
    "output",
    "outputs",
}


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate repo root (no pyproject.toml above test)")


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=_repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    test_file = Path(__file__).resolve()
    files = []
    for name in result.stdout.splitlines():
        path = (_repo_root() / name).resolve()
        if path == test_file or any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


@pytest.mark.parametrize(("old_internal_dir", "new_location"), MOVE_MAP)
def test_no_dangling_internal_references(old_internal_dir, new_location):
    """Once a move is mapped, the old internal path must disappear from text."""
    hits = []
    for path in _tracked_files():
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if old_internal_dir in text:
            hits.append(path.relative_to(_repo_root()))

    assert not hits, (
        f"{old_internal_dir!r} moved to {new_location!r}, but stale references "
        f"remain in: {hits}"
    )
