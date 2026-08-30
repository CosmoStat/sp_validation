"""Blinding file-name conventions, with no dependencies.

Separate from :mod:`sp_validation.blinding` so the Snakemake DAG build can
import the path layout without pulling in numpy, CCL and smokescreen.
"""

import os


def init_paths(blind_dir):
    """The fixed custody state ``blinding.blind_init`` writes in ``blind_dir``."""
    return {
        "commitment": os.path.join(blind_dir, "commitment.json"),
        "bundle": os.path.join(blind_dir, "blind_seed.encrpt"),
        "key": os.path.join(blind_dir, "blind_seed.key"),
    }


def part_paths(part_path):
    """Blinded-output and escrow-bundle paths beside a part file."""
    stem, ext = os.path.splitext(str(part_path))
    return {
        "blinded": f"{stem}_blinded{ext or '.fits'}",
        "escrow": f"{stem}_escrow.encrpt",
        "escrow_key": f"{stem}_escrow.key",
    }
