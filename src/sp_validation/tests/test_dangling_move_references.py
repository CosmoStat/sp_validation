"""Back-pressure guard #6: moved internal directories leave no stale refs."""

import subprocess
from pathlib import Path

import pytest

# Add (old_internal_dir, new_location) pairs as the restructuring starts moving
# directories. Empty pre-reorg by design: the harness is the invariant.
#
# Entries here are *fully retired* paths: the old path string must no longer
# appear anywhere. Extractions-in-place do NOT belong here. The GLASS-mock
# generation core was lifted out of ``glass_mock/make_unions_glass_sim.py`` into
# the library at ``src/sp_validation/glass_mock.py``, but ``glass_mock/`` itself
# survives (CLI wrapper + postprocessing scripts), so the string ``glass_mock``
# is still live across cosmo_inference — registering it would be a false move.
MOVE_MAP: tuple[tuple[str, str], ...] = (
    ("cosmo_inference/notebooks/2D_bmodes_paper_workflow", "papers/bmodes"),
    ("rules/glass_mock_validation.smk", "workflow/rules/glass_mock.smk"),
    ("notebooks/cosmo_val/catalog_paper_plot", "papers/catalog"),
    ("cosmo_inference/notebooks/2D_harmonic_space_cosmic_shear_plots", "papers/harmonic"),
    ("notebooks/cosmo_val", "cosmo_val"),
)
EXCLUDED_DIRS = {
    ".git",
    ".felt",
    ".pytest_cache",
    ".snakemake",
    ".venv",
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
    """Tracked files via git; full-tree walk where .git is absent.

    The CI image is built from the Git build context — tracked content only,
    no .git directory — so walking the tree there scans exactly the tracked
    set and the guard keeps its teeth in-image.
    """
    root = _repo_root()
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    candidates = (
        [root / name for name in result.stdout.splitlines()]
        if result.returncode == 0
        else [path for path in root.rglob("*") if path.is_file()]
    )
    test_file = Path(__file__).resolve()
    files = []
    for candidate in candidates:
        path = candidate.resolve()
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
