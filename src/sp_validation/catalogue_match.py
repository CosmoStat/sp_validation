"""Audit published ShapePipe catalogue-match metadata without selecting objects."""

import csv
import json
import math
import numbers
import re
from collections import Counter
from pathlib import Path

import numpy as np

# @sc [decision:dr6_catalogue.catalogue_match_tile_selection]
MIN_EXTERNAL_PAIRED_FRACTION = None

_COUNT_FIELDS = (
    "n_sextractor",
    "n_external",
    "n_paired",
    "n_dropped",
    "n_external_only",
)
_MATCH_FIELDS = (*_COUNT_FIELDS, "external_paired_fraction", "external_warning_budget")


def normalize_tile_id(value):
    """Return a canonical tile ID, accepting catalogue numeric representations."""
    if isinstance(value, bytes):
        value = value.decode("ascii")
    if isinstance(value, str):
        if not re.fullmatch(r"\d{1,3}\.\d{3}", value):
            raise ValueError(f"Invalid tile ID: {value!r}")
        return f"{int(value.split('.')[0]):03d}.{value.split('.')[1]}"
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number) and 0 <= number < 1000:
            rounded = round(number, 3)
            # Numeric catalogue columns may be stored as float32.
            if rounded < 1000 and abs(number - rounded) <= 0.00004:
                return f"{rounded:07.3f}"
    raise ValueError(f"Invalid tile ID: {value!r}")


def read_catalogue_match_manifests(manifest_paths):
    """Read explicit manifests.

    Raises:
        ValueError: Invalid metadata or conflicting metadata for one tile,
            including manifests with and without catalogue_match.
    """
    records = {}
    for path in map(Path, manifest_paths):
        with path.open() as stream:
            manifest = json.load(stream)
        if "unit" not in manifest:
            raise ValueError(f"Missing unit in {path}")
        tile = normalize_tile_id(manifest["unit"])
        if (
            path.name != "tile_make_cat.json"
            or normalize_tile_id(path.parent.name) != tile
        ):
            raise ValueError(f"Manifest path and unit disagree: {path}")
        match = manifest.get("catalogue_match")
        if match is not None:
            if not isinstance(match, dict):
                raise ValueError(f"Invalid catalogue_match in {path}")
            try:
                match = {key: match[key] for key in _MATCH_FIELDS}
            except KeyError as exc:
                raise ValueError(f"Missing {exc} in {path}") from exc
            for key in _COUNT_FIELDS:
                if type(match[key]) is not int or match[key] < 0:
                    raise ValueError(f"Invalid {key} in {path}")
            if (
                match["n_paired"] + match["n_dropped"] != match["n_sextractor"]
                or match["n_paired"] + match["n_external_only"] != match["n_external"]
            ):
                raise ValueError(f"Inconsistent catalogue-match counts in {path}")
            fraction = match["external_paired_fraction"]
            budget = match["external_warning_budget"]
            expected_fraction = (
                match["n_paired"] / match["n_external"] if match["n_external"] else 0.0
            )
            if (
                isinstance(fraction, bool)
                or not isinstance(fraction, (int, float))
                or not math.isclose(
                    fraction, expected_fraction, rel_tol=1e-9, abs_tol=1e-9
                )
            ):
                raise ValueError(f"Invalid fraction in {path}")
            if (
                isinstance(budget, bool)
                or not isinstance(budget, (int, float))
                or not math.isfinite(budget)
                or budget < 0
            ):
                raise ValueError(f"Invalid budget in {path}")
        if tile in records and records[tile]["match"] != match:
            raise ValueError(
                f"Conflicting catalogue-match metadata for {tile}: "
                f"{records[tile]['manifest_path']} and {path}"
            )
        if tile not in records:
            records[tile] = {"manifest_path": str(path), "match": match}
    return records


def write_catalogue_match_audit(tile_ids, manifest_paths, output_path):
    """Write one row per catalogue or manifest tile and return summary counts."""
    counts = Counter()
    unique_ids, row_counts = np.unique(tile_ids, return_counts=True)
    for tile, count in zip(unique_ids, row_counts):
        counts[normalize_tile_id(tile)] += int(count)
    records = read_catalogue_match_manifests(manifest_paths)
    fields = (
        "tile_id",
        "catalogue_rows",
        "manifest_path",
        *_MATCH_FIELDS,
        "external_warning",
        "min_external_paired_fraction",
    )
    summary = {
        "tiles": 0,
        "missing_manifest": 0,
        "missing_catalogue_match": 0,
        "warnings": 0,
    }
    with Path(output_path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for tile in sorted(counts.keys() | records.keys()):
            record = records.get(tile, {})
            match = record.get("match")
            row = {
                "tile_id": tile,
                "catalogue_rows": counts[tile],
                "manifest_path": record.get("manifest_path"),
                "min_external_paired_fraction": MIN_EXTERNAL_PAIRED_FRACTION,
            }
            if match is None:
                key = (
                    "missing_catalogue_match" if tile in records else "missing_manifest"
                )
                summary[key] += 1
            else:
                warning = match["n_external_only"] > match["external_warning_budget"]
                row.update(match, external_warning=warning)
                summary["warnings"] += warning
            summary["tiles"] += 1
            writer.writerow(row)
    return summary
