"""Combine production residual m/c across independent simulation grids."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(root):
    rows = {}
    for grid in range(1, 5):
        product = json.loads((root / f"production_{grid}.json").read_text())
        p = product["weights"]["w_des"]
        rows[str(grid)] = dict(
            value=[p[k] for k in ("m1", "c1", "m2", "c2")],
            error=[p[k + "_err"] for k in ("m1", "c1", "m2", "c2")],
            covariance=p["mc_covariance"],
            deleted=p["mc_jackknife"],
            labels=product["jackknife_tiles"],
            clipping=json.loads((root / f"clipping_{grid}.json").read_text()),
        )
    values = np.array([r["value"] for r in rows.values()])
    errors = np.array([r["error"] for r in rows.values()])
    weights = errors**-2
    mean = dict(
        value=(np.sum(weights * values, axis=0) / weights.sum(axis=0)).tolist(),
        error=np.sqrt(1 / weights.sum(axis=0)).tolist(),
    )
    result = dict(order=["m1", "c1", "m2", "c2"], grids=rows, combined=mean)
    (root / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
