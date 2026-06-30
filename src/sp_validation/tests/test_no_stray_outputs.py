"""STRUCTURAL GUARD: no generated outputs tracked under ``*/scripts/``.

Scripts are code, not artifacts. Plots, maps, FITS, and arrays they produce
belong in (gitignored) output directories, not committed beside the script.
This guard exists because a 761 KB PNG once snuck into a ``scripts/`` dir; it
fails loudly the next time a binary output is staged there.

Environment-independent: queries the git index via ``git ls-files`` and asserts
on the result, so it runs anywhere the repo is a git checkout (skips otherwise).

:Author: cdaley

"""

import subprocess
from pathlib import Path

import pytest

# Binary/output extensions that should never be tracked under a scripts/ dir.
_OUTPUT_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".pdf",
    ".fits",
    ".fits.gz",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
    ".pkl",
}

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _tracked_files():
    """Return the repo's tracked paths via git, or None if not a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return [line for line in out.stdout.splitlines() if line]


def test_no_tracked_outputs_under_scripts():
    """No tracked image/data outputs may live under any ``scripts/`` directory."""
    tracked = _tracked_files()
    if tracked is None:
        pytest.skip("not a git checkout — guard needs the git index")

    def is_output(path):
        suffixes = [s.lower() for s in Path(path).suffixes]
        return bool(suffixes) and (
            suffixes[-1] in _OUTPUT_SUFFIXES
            or "".join(suffixes[-2:]) in _OUTPUT_SUFFIXES  # e.g. .fits.gz
        )

    offenders = [f for f in tracked if "/scripts/" in f"/{f}" and is_output(f)]

    assert not offenders, (
        "Generated outputs are tracked under a scripts/ directory — move them to "
        "a gitignored output location and `git rm --cached` them:\n  "
        + "\n  ".join(sorted(offenders))
    )
