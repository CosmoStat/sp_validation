"""Figures and a table of the n(z) twin ladder (``nz_twin.py``'s records).

``ladder.png``: Δ⟨z⟩ per bin on each rung, under each redshift model, with
the realisation scatter, against the ±σ⟨z⟩ the data's ensemble quotes.
``zero_points.png``: what the zero-point maps do: Δ⟨z⟩ on ``zp_map`` minus
``success`` of the same seed, and the rms over sky regions of each bin's true
mean. Prints the table the figures draw.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNGS = ["representative", "egs", "success", "zp_patch", "zp_map"]
MODELS = ["colour", "bpz"]


def ensemble_sigma(path):
    r = np.load(path)
    z, draws = r["z"], r["realisations"]
    mean = np.trapezoid(draws * z, z, axis=-1) / np.trapezoid(draws, z, axis=-1)
    return mean.std(0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path)
    p.add_argument(
        "--realisations", type=Path, required=True, help="the data's realisations.npz"
    )
    p.add_argument("--figures", type=Path, required=True)
    a = p.parse_args(argv)
    a.figures.mkdir(parents=True, exist_ok=True)
    sigma = ensemble_sigma(a.realisations)
    delta = defaultdict(dict)
    extra = defaultdict(dict)
    for rec in map(json.loads, a.records.read_text().splitlines()):
        key = (rec["rung"], rec["model"])
        delta[key][rec["seed"]] = [b["delta"] for b in rec["bins"]]
        extra[key][rec["seed"]] = rec
    n_bin = len(sigma)
    bins = np.arange(1, n_bin + 1)

    print("ensemble σ⟨z⟩", np.round(sigma, 4))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, m in zip(axes, MODELS):
        ax.fill_between(
            bins, -sigma, sigma, step="mid", color="0.88", label="data's ensemble ±σ⟨z⟩"
        )
        rungs = [r for r in RUNGS if delta[(r, m)]]
        for j, r in enumerate(rungs):
            d = np.array(list(delta[(r, m)].values()))
            x = bins + (j - (len(rungs) - 1) / 2) * 0.12
            ax.errorbar(
                x,
                d.mean(0),
                d.std(0),
                fmt="o",
                ms=4,
                capsize=2,
                label=f"{r} ({len(d)})",
            )
            print(
                f"{m:7s} {r:15s} mean",
                np.round(d.mean(0), 4),
                "scatter",
                np.round(d.std(0), 4),
            )
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(bins)
        ax.set_xlabel("tomographic bin")
        ax.set_title(f"redshift model: {m}")
    axes[0].set_ylabel("Δ⟨z⟩ = calibrated − true")
    axes[1].legend(fontsize=7, loc="best")
    fig.suptitle("SOM n(z) on the catalogue twin: Δ⟨z⟩ per rung (mean ± scatter)")
    fig.tight_layout()
    fig.savefig(a.figures / "ladder.png", dpi=150)

    if not delta[("zp_map", MODELS[0])]:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for j, m in enumerate(MODELS):
        seeds = sorted(set(delta[("zp_map", m)]) & set(delta[("success", m)]))
        shift = np.array(
            [
                np.subtract(delta[("zp_map", m)][s], delta[("success", m)][s])
                for s in seeds
            ]
        )
        region = np.array([extra[("zp_map", m)][s]["region_rms"] for s in seeds])
        x = bins + (j - 0.5) * 0.2
        axes[0].errorbar(
            x,
            shift.mean(0),
            shift.std(0),
            fmt="o",
            capsize=2,
            label=f"{m} ({len(seeds)} maps)",
        )
        axes[1].plot(x, region.mean(0), "o", label=m)
        print(
            f"{m:7s} zp_map − success mean",
            np.round(shift.mean(0), 4),
            "rms",
            np.round(np.sqrt((shift**2).mean(0)), 4),
            "region rms",
            np.round(region.mean(0), 4),
        )
    axes[0].fill_between(bins, -sigma, sigma, step="mid", color="0.88")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("Δ⟨z⟩ change from the zero-point map")
    axes[1].set_ylabel("rms over 54 deg² regions of true ⟨z⟩ change")
    for ax in axes:
        ax.set_xticks(bins)
        ax.set_xlabel("tomographic bin")
        ax.legend(fontsize=8)
    fig.suptitle("Zero-point maps forward-modelled through the calibration")
    fig.tight_layout()
    fig.savefig(a.figures / "zero_points.png", dpi=150)


if __name__ == "__main__":
    main()
