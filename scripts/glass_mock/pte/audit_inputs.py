#!/usr/bin/env python3
"""Check mock-only geometry, weights, and iNKA noise moments across skies."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

root = Path("/n17data/cdaley/scratch/glass-mock-cv/pte")
p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=10)
a = p.parse_args()
reg = yaml.safe_load((root / "cat_config.yaml").read_text())
records = {}
for seed in range(1, a.seeds + 1):
    name = f"GLASSpte_s{seed:02d}"
    ent = reg[name]
    h = hashlib.sha256()
    energy = 0.0
    weights = 0.0
    with h5py.File(ent["shear"]["path"], "r") as f:
        d = f["data"]
        for start in range(0, len(d), 250_000):
            x = d.fields(["RA", "Dec", "w_des", "e1", "e2"])[start : start + 250_000]
            for col in ("RA", "Dec", "w_des"):
                h.update(np.ascontiguousarray(x[col]).tobytes())
            w = x["w_des"].astype("f8")
            e1 = x["e1"].astype("f8")
            e2 = x["e2"].astype("f8")
            energy += np.sum(0.5 * (e1 * e1 + e2 * e2) * w * w)
            weights += np.sum(w * w)
        n = len(d)
    meta = json.loads(
        (root.parent / f"mocks/v1.4.6.3/seed_{seed:05d}/maps/maps.json").read_text()
    )
    map_config = meta["config"].copy()
    map_config.pop("seed")
    records[name] = {
        "rows": n,
        "geometry_weight_sha256": h.hexdigest(),
        "weighted_shape_moment": energy / weights,
        "cov_th": ent["cov_th"],
        "mask": ent["mask"],
        "nz": ent["shear"]["redshift_path"],
        "map_config_except_seed": map_config,
        "source_weights": meta["source_weights"],
        "spectra_cache": meta["spectra_cache"],
    }
    print(name, json.dumps(records[name]), flush=True)
ref = records["GLASSpte_s01"]
for name, r in records.items():
    if any(
        r[k] != ref[k]
        for k in (
            "rows",
            "geometry_weight_sha256",
            "cov_th",
            "mask",
            "nz",
            "map_config_except_seed",
            "source_weights",
            "spectra_cache",
        )
    ):
        raise ValueError(f"{name}: seed-dependent geometry or covariance metadata")
    r["noise_moment_fractional_difference_from_seed1"] = (
        r["weighted_shape_moment"] / ref["weighted_shape_moment"] - 1
    )
(root / "input_audit.json").write_text(json.dumps(records, indent=2))
print(
    "Geometry/weight bytes and covariance metadata identical; painted-shape moments recorded separately."
)
