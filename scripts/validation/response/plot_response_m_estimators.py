"""Compare paired, matched branch-weighted and full-population estimators."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(root, output, production=None):
    results = [
        json.loads((root / f"grid_{grid}/validation.json").read_text())
        for grid in range(1, 5)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    x = np.arange(1, 5)
    for component, ax in enumerate(axes, 1):
        rows = [r["m"][f"m{component}"] for r in results]
        for offset, key, label in (
            (-0.12, "new_paired", "Matched, symmetric pair weights (Martin)"),
            (0, "intersection_branch_weighted", "Matched, separate branch weights"),
            (0.12, "new_unpaired", "Full populations, separate branch weights"),
        ):
            error = (
                [r["unpaired_refit_W_tile_jk_error"] * 1000 for r in rows]
                if key == "new_unpaired"
                else None
            )
            ax.errorbar(
                x + offset,
                [r[key] * 1000 for r in rows],
                yerr=error,
                fmt="o",
                capsize=3,
                label=label,
            )
        if production is not None:
            data = json.loads((production / "summary.json").read_text())["grids"]
            idx = 2 * (component - 1)
            ax.errorbar(
                x + 0.24,
                [data[str(i)]["value"][idx] * 1000 for i in x],
                yerr=[data[str(i)]["error"][idx] * 1000 for i in x],
                fmt="s",
                capsize=3,
                label="Full populations, independent v2 table",
            )
        ax.axhline(0, color="black", lw=0.8)
        ax.set(xticks=x, xlabel="Grid", ylabel=rf"$m_{component}$ ($10^{{-3}}$)")
        ax.grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "m_estimators.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--production", type=Path)
    args = parser.parse_args()
    plot(args.root, args.output, args.production)
