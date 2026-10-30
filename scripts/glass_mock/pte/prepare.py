#!/usr/bin/env python3
"""Prepare an isolated registry of known-truth painted mock catalogues."""

import argparse
import copy
import json
from pathlib import Path

import yaml

ROOT = Path("/n17data/cdaley/scratch/glass-mock-cv/pte")
REPO = Path("/n17data/cdaley/scratch/glass-mock-cv/pte-wt")
MOCKS = ROOT.parent / "mocks/v1.4.6.3"
parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=10)
a = parser.parse_args()
base = yaml.safe_load((REPO / "cosmo_val/cat_config.yaml").read_text())["SP_v1.4.6.3"]
config = yaml.safe_load((ROOT.parent / "cvmock/config.yaml").read_text())
registry = {}
for seed in range(1, a.seeds + 1):
    name = f"GLASSpte_s{seed:02d}"
    d = MOCKS / f"seed_{seed:05d}"
    if not (d / "painted.h5").is_file():
        raise FileNotFoundError(d / "painted.h5")
    meta = json.loads((d / "maps/maps.json").read_text())
    entry = copy.deepcopy(base)
    entry.update(blind="none", subdir=str(d), label=name, patch_number=1)
    entry["shear"] = dict(
        path=str(d / "painted.h5"),
        R=1.0,
        ra_col="RA",
        dec_col="Dec",
        w_col="w_des",
        e1_col="e1",
        e2_col="e2",
        redshift_path=meta["config"]["nz_path"],
    )
    registry[name] = entry
registry["nz"] = {
    "subdir": str(ROOT),
    "dndz": {"path": registry["GLASSpte_s01"]["shear"]["redshift_path"]},
}
registry["paths"] = {
    "output": str(ROOT / "output"),
    "blinds": str(ROOT / "mock_blinds"),
}
(ROOT / "cat_config.yaml").write_text(yaml.safe_dump(registry, sort_keys=False))
config.update(
    versions=[f"GLASSpte_s{i:02d}" for i in range(1, a.seeds + 1)],
    catalog_config=str(ROOT / "cat_config.yaml"),
)
config["fiducial"].update(version="GLASSpte_s01", mock_version="GLASSpte_s01", npatch=1)
config["cosmo_val"]["npatch"] = 1
config["tools"]["python_executable"] = str(ROOT / "onecov-python")
(ROOT / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
print(
    f"Prepared {a.seeds} entries; covariance parameters from SP_v1.4.6.3 registry, independent of seed."
)
