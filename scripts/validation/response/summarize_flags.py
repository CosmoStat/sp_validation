"""Combine FLAGS changes with paired shared-tile jackknife errors."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(root):
    """Compute a paired difference per grid before inverse-variance averaging."""
    values, errors = [], []
    for grid in range(1, 5):
        a = json.loads((root / f"base_{grid}.json").read_text())
        b = json.loads((root / f"noflags_{grid}.json").read_text())
        x, y = a["result"], b["result"]
        if x["jackknife_tiles"] != y["jackknife_tiles"]:
            raise ValueError("Mismatched tile deletions")
        diff = np.asarray(y["mc_jackknife"]) - np.asarray(x["mc_jackknife"])
        centered = diff - diff.mean(axis=0)
        cov = (len(diff) - 1) / len(diff) * centered.T @ centered
        value = np.array([y["m1"] - x["m1"], y["m2"] - x["m2"]])
        error = np.sqrt(np.diag(cov))[[0, 2]]
        values.append(value)
        errors.append(error)
        print(
            grid,
            "delta",
            value,
            "paired error",
            error,
            "counts",
            a["counts"],
            b["counts"],
        )
    values, errors = np.asarray(values), np.asarray(errors)
    weights = 1 / errors**2
    print(
        "combined delta",
        np.sum(values * weights, axis=0) / weights.sum(axis=0),
        "paired error",
        1 / np.sqrt(weights.sum(axis=0)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summarize(args.directory)
