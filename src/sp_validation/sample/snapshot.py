"""The ShapePipe snapshot a calibrated sample is built from.

A snapshot is a directory holding ``MANIFEST.json``, ShapePipe's list of the
published tiles and their object counts. A sample built from it records the
snapshot's name, tile count and the manifest's SHA-256 in its HATS properties
(:func:`record`); a ``cat_config`` entry names the snapshot it expects under
``shear.snapshot``, and :func:`check` fails when the catalogue it points at
was built from another.
"""

import hashlib
import json
from pathlib import Path

from .. import io

#: The catalogue properties holding the record, with the record's own keys.
PROPERTIES = {
    "name": "snapshot_name",
    "tiles": "snapshot_tiles",
    "manifest_sha256": "snapshot_manifest_sha256",
}


def read(directory):
    """Return the snapshot at ``directory``.

    Returns
    -------
    dict
        ``name`` (the directory's), ``manifest_sha256``, and ``objects``, the
        object count of each tile ID (``"194.248"``)
    """
    directory = Path(directory)
    content = (directory / "MANIFEST.json").read_bytes()
    manifest = json.loads(content)
    if manifest["status"] != "complete":
        raise ValueError(f"snapshot {directory} is {manifest['status']}, not complete")
    return dict(
        name=directory.name,
        manifest_sha256=hashlib.sha256(content).hexdigest(),
        objects={tile: entry["objects"] for tile, entry in manifest["tiles"].items()},
    )


def record(snapshot):
    """Return the record of ``snapshot`` a sample carries: its name, tile count
    and manifest SHA-256."""
    return dict(
        name=snapshot["name"],
        tiles=len(snapshot["objects"]),
        manifest_sha256=snapshot["manifest_sha256"],
    )


def recorded(catalogue):
    """Return the snapshot record of the HATS catalogue ``catalogue``, ``None``
    for a catalogue without one."""
    directory = io.hats_catalogue_dir(catalogue)
    if directory is None:
        return None
    properties = io._properties(Path(directory) / "hats.properties")
    if not all(key in properties for key in PROPERTIES.values()):
        return None
    found = {name: properties[key] for name, key in PROPERTIES.items()}
    return dict(found, tiles=int(found["tiles"]))


def check(entry):
    """Fail when a shear entry's catalogue was built from another snapshot.

    @sc [decision:dr6_catalogue.snapshot_record] entry-snapshot-is-catalogue-snapshot
    An entry with a ``snapshot`` block (``name``, ``tiles``,
    ``manifest_sha256``) is measured only on a catalogue recording exactly
    that snapshot; an entry without one is not checked.
    """
    expected = entry.get("snapshot")
    if expected is None:
        return
    found = recorded(entry["path"])
    if found != expected:
        raise ValueError(
            f"{entry['path']} was built from snapshot {found}, "
            f"its cat_config entry expects {expected}"
        )
