#!/usr/bin/env python3
"""Print the local SIF that Snakemake pulls the workflow image into.

For host-side callers that are *outside* Snakemake and therefore need a file
rather than a ``docker://`` tag -- the papers/bmodes/scripts drivers, and
interactive ``apptainer exec``. Resolving it here keeps the image in exactly
one place (``CONTAINER_URI`` in workflow/common.py, ``apptainer-prefix`` in the
driving profile) instead of a path anyone has to maintain by hand.

    CONTAINER=$(workflow/scripts/container_path.py)

Stdlib only, so it runs under bare ``python3`` on the login node with no
environment to activate. Exits non-zero if the image has not been pulled yet;
the message says how to pull it.
"""

import hashlib
import re
import sys
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parent.parent
PROFILE = WORKFLOW / "profiles/candide/config.yaml"


def _grep(path, pattern, what):
    match = re.search(pattern, path.read_text(), re.MULTILINE)
    if match is None:
        sys.exit(f"{path}: could not find {what}")
    return match.group(1).strip().strip("\"'")


uri = _grep(WORKFLOW / "common.py", r'^CONTAINER_URI = "(.+)"$', "CONTAINER_URI")
prefix = _grep(PROFILE, r"^apptainer-prefix:\s*(\S+)", "apptainer-prefix")

sif = Path(prefix) / f"{hashlib.md5(uri.encode()).hexdigest()}.simg"
if not sif.exists():
    sys.exit(
        f"{sif} not pulled yet -- run any snakemake target with "
        f"--profile {PROFILE.parent} once, or pull it directly (workflow/README.md)."
    )
print(sif)
