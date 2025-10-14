"""Lightweight checks that catalog paths referenced in ``cat_config.yaml`` exist.

These tests do not load the data; they only ensure that the configuration points
to readable files so that downstream validation runs fail fast if a path drifts.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, Tuple

import yaml


CATALOG_CONFIG = (
    Path(__file__).resolve().parents[3] / "notebooks" / "cosmo_val" / "cat_config.yaml"
)


def _resolve(base: Path, candidate: str) -> Path:
    """Return an absolute path given a base directory and a candidate string."""
    candidate_path = Path(candidate)
    return candidate_path if candidate_path.is_absolute() else base / candidate_path


def _iter_catalog_entries(config: Dict[str, Dict]) -> Iterator[Tuple[str, Dict]]:
    """Yield (name, entry) pairs for catalog-like entries in the config."""
    for name, entry in config.items():
        if not isinstance(entry, dict):
            continue
        if "subdir" not in entry:
            continue
        yield name, entry


def test_catalog_files_exist():
    """Ensure every shear/star/psf entry references an on-disk file."""
    config = yaml.safe_load(CATALOG_CONFIG.read_text())

    missing = defaultdict(set)

    for name, entry in _iter_catalog_entries(config):
        base = Path(entry["subdir"])
        assert base.is_absolute(), f"{name}: subdir must be absolute ({base})"

        if name == "nz":
            dndz_path = _resolve(base, entry["dndz"]["path"])
            assert (
                dndz_path.parent.is_dir()
            ), f"{name}: dndz parent directory missing ({dndz_path.parent})"
            continue

        for block_name in ("shear", "star", "psf"):
            block = entry.get(block_name)
            if not block:
                continue
            resolved_path = _resolve(base, block["path"])
            if not resolved_path.is_file():
                missing[name].add(block_name)

    assert not missing, (
        "Catalog configuration references missing files: "
        f"{ {name: sorted(blocks) for name, blocks in missing.items()} }"
    )
