"""Reproduce the full-area delivery, optionally rebuilding the assignment model."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from verify import verify

from sp_validation.nz_pipeline import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe")
    parser.add_argument("output")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rebuild-model", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.recipe).read_text())
    run(config, args.output, workers=args.workers, rebuild=args.rebuild_model)
    reference = yaml.safe_load(Path(__file__).with_name("reference.yaml").read_text())
    verify(reference, args.output)
    if args.rebuild_model:
        new = np.load(Path(args.output) / "som_assignment_model.npz")
        old = np.load(config["inputs"]["model"])
        keys = ["code", "center", "scale", "group_map", "edges", "gold"]
        result = {key: bool(np.array_equal(new[key], old[key])) for key in keys}
        result["max_code_difference"] = float(abs(new["code"] - old["code"]).max())
        result["gold_mismatches"] = int(np.count_nonzero(new["gold"] != old["gold"]))
        (Path(args.output) / "rebuild.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        assert all(result[key] for key in keys), result
