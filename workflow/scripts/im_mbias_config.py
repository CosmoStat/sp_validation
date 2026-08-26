"""Assemble ``m_bias_config.yaml`` for the image-sims m-bias estimator.

Combines the manifest's injected-shear facts, this run's science knobs and
git/container provenance into the single config ``compute_m_bias_image_sims.py``
reads.  ``provenance`` rides along as a top-level block: the estimator copies it
verbatim into its results yaml, so a result file records which manifest, repo
commits and container produced the number.

`snakemake` is injected as a module global by Snakemake's `script:` preamble
before this file runs (`from snakemake.script import snakemake` is
IDE-hint-only and raises ImportError if actually executed).
"""

import hashlib
import re
import subprocess

import yaml

params = snakemake.params  # noqa: F821
manifest_path = snakemake.input["manifest"]  # noqa: F821


def _git(repo, *args):
    """Read a git fact from ``repo``; ``None`` if it is not a checkout."""
    try:
        return subprocess.run(
            ["git", "-C", repo, *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _sif_revision(sif_path):
    """``org.opencontainers.image.revision`` from the SIF's OCI labels.

    Plain-text scan of the image file -- no exec, no container start.
    """
    try:
        with open(sif_path, "rb") as fh:
            blob = fh.read()
    except OSError:
        return None
    m = re.search(
        rb'org\.opencontainers\.image\.revision"?[:=]"?([0-9a-f]{7,40})', blob
    )
    return m.group(1).decode() if m else None


with open(manifest_path) as fh:
    manifest = yaml.safe_load(fh)
with open(manifest_path, "rb") as fh:
    manifest_sha256 = hashlib.sha256(fh.read()).hexdigest()

mbias_cfg = {
    "grids_dir": params.grids_base,
    "num": params.num,
    "catalog_name": params.cat_name,
    # Injected shear: from the manifest, the single source of truth.
    "shear_amplitude": manifest["shear_amplitude"],
    "branches": list(manifest["branches"]),
    "pairs": manifest["pairs"],
    "match_radius_deg": params.match_radius_deg,
    "w_cols": list(params.w_cols),
    "pair_match": params.pair_match,
    "n_bootstrap": params.n_bootstrap,
    "bootstrap_seed": params.bootstrap_seed,
    "results_dir": params.results_dir,
    "output_path": params.results,
    "provenance": {
        "manifest_sha256": manifest_sha256,
        "sp_validation": {
            "branch": _git(
                params.sp_validation_repo, "rev-parse", "--abbrev-ref", "HEAD"
            ),
            "commit": _git(params.sp_validation_repo, "rev-parse", "HEAD"),
        },
        "shapepipe": {
            "branch": _git(
                params.shapepipe_repo, "rev-parse", "--abbrev-ref", "HEAD"
            ),
            "commit": _git(params.shapepipe_repo, "rev-parse", "HEAD"),
        },
        "container": {
            "sif": params.sif,
            "ghcr_revision": _sif_revision(params.sif),
        },
    },
}

with open(snakemake.output["cfg"], "w") as fh:  # noqa: F821
    yaml.safe_dump(mbias_cfg, fh)
