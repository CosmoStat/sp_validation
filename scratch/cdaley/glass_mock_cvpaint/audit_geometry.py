"""Verify every coordinate and weight before sharing geometry-only pair counts."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path("/n17data/cdaley/scratch/glass-mock-cv/cvpaint")
cc = yaml.safe_load((ROOT / "cat_config.yaml").read_text())
results = {}
for version in ("GLASS1_clean", "GLASS1_blinded", "GLASS1_psfleak", "GLASS1_addB"):
    entry = cc[version]["shear"]
    hashes = {key: hashlib.sha256() for key in ("ra_col", "dec_col", "w_col")}
    sums = np.zeros(2)
    with h5py.File(entry["path"], "r") as handle:
        dataset = handle["data"]
        n = len(dataset)
        names = [
            entry.get(key, {"ra_col": "RA", "dec_col": "Dec"}.get(key))
            for key in hashes
        ]
        for lo in range(0, n, 250000):
            data = dataset.fields(names)[lo : lo + 250000]
            for key in hashes:
                hashes[key].update(np.ascontiguousarray(data[entry[key]]).tobytes())
            w = np.asarray(data[entry["w_col"]], float)
            sums += np.array([np.sum(w), np.sum(w * w)])
    results[version] = dict(
        rows=n,
        sha256={key: value.hexdigest() for key, value in hashes.items()},
        sum_w=float(sums[0]),
        sum_w2=float(sums[1]),
    )
    print(version, "geometry hash complete", flush=True)
reference = results["GLASS1_clean"]
assert all(value == reference for value in results.values()), (
    "Geometry/weights differ; refuse shared pair counts"
)
(ROOT / "geometry_audit.json").write_text(json.dumps(results, indent=2))
config = yaml.safe_load((ROOT / "config.yaml").read_text())
config["covariance"]["pair_counts_reference"] = {
    v: "GLASS1_clean" for v in results if v != "GLASS1_clean"
}
(ROOT / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
(ROOT / "geometry_verified").touch()
print(
    "All rows have identical geometry and weights; enabled explicit geometry-only references.",
    flush=True,
)
