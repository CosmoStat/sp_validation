"""Multiplicative and additive shear bias of the image simulations.

For every shear component the manifest pairs a ``+g`` and a ``-g`` simulation;
their per-tile response sums (left by the calibration step) give ``m`` and ``c``
(``response.shear_bias``), and deleting each tile from every simulation at once
gives their covariance (``response.shear_bias_jackknife``). ``weight_table``
names the table every simulation is weighted with: a DES weight-table product,
``self`` (each simulation's own table, refitted per deletion) or ``unit``.

The results yaml carries its provenance: the manifest's hash, both repositories'
branch and commit, and the image's revision.

``snakemake`` is the global Snakemake's ``script:`` directive injects.
"""

import hashlib
import os
import re
import subprocess

import numpy as np
import yaml

from sp_validation import response

params = snakemake.params  # noqa: F821


def git(repo, *args):
    """A git fact of ``repo``, or ``None`` if it is not a checkout."""
    try:
        return subprocess.run(
            ["git", "-C", repo, *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def image_revision():
    """``org.opencontainers.image.revision`` of the mounted image.

    Apptainer names the mounted image in ``APPTAINER_CONTAINER``; its OCI
    labels are scanned as plain bytes, chunk by chunk.
    """
    path = os.environ.get("APPTAINER_CONTAINER")
    if not path or not os.path.isfile(path):
        return None
    pattern = re.compile(rb'org\.opencontainers\.image\.revision"?[:=]"?([0-9a-f]{7,40})')
    tail = b""
    with open(path, "rb") as fh:
        while chunk := fh.read(8 << 20):
            if match := pattern.search(tail + chunk):
                return match.group(1).decode()
            tail = chunk[-128:]
    return None


def repository(repo):
    return {
        "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git(repo, "rev-parse", "HEAD"),
    }


with open(snakemake.input.manifest, "rb") as fh:  # noqa: F821
    text = fh.read()
manifest = yaml.safe_load(text)

pairs = {
    p["component"]: (
        response.Sums.load(params.sums[p["plus"]]),
        response.Sums.load(params.sums[p["minus"]]),
    )
    for p in manifest["pairs"]
}
choice = params.weight_table
if choice == "self":
    table = response.refit(params.des_min_count)
elif choice == "unit":
    table = response.unit_table(next(iter(pairs.values()))[0].edges)
else:
    table = response.read_table(choice)

g_in = manifest["shear_amplitude"]
values = response.shear_bias(pairs, g_in, table)
tiles, _, covariance = response.shear_bias_jackknife(pairs, g_in, table)
errors = np.sqrt(np.diag(covariance))

results = {key: float(value) for key, value in values.items()}
results.update({f"{key}_err": float(e) for key, e in zip(values, errors, strict=True)})
results.update(
    mc_covariance=covariance.tolist(),
    covariance_order=list(values),
    jackknife_tiles=[str(t) for t in tiles],
    weight_table=choice,
    shear_amplitude=g_in,
    provenance={
        "manifest_sha256": hashlib.sha256(text).hexdigest(),
        "sp_validation": repository(params.sp_validation_repo),
        "shapepipe": repository(params.shapepipe_repo),
        "container": {"sif": params.sif, "ghcr_revision": image_revision()},
    },
)

with open(snakemake.output.results, "w") as fh:  # noqa: F821
    yaml.safe_dump(results, fh, sort_keys=False)

print(f"weights: {choice}; {len(tiles)} tiles; g_in = {g_in}")
for key in values:
    print(f"  {key} = {results[key]:+.5f} +- {results[key + '_err']:.5f}")
