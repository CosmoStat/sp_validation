"""Figures and a table of the n(z) twin ladder (``nz_twin.py``'s records).

``--run`` is the recipe's calibration run on the data (its
``realisations.npz`` gives the quoted σ⟨z⟩, and its ensemble inputs give the
zero-point term alone).

- ``ladder.png``: Δ⟨z⟩ per bin on each rung, under each redshift model, with
  the realisation scatter, against the data's ±σ⟨z⟩.
- ``zero_points.png``: each zero-point rung's change of Δ⟨z⟩ from its base
  rung of the same seed (``egs``, or ``success`` where a model has one): mean
  and rms over realisations, the ``zp_ensemble`` rms against the ensemble's
  zero-point term, and the rms over sky regions of the change in a bin's true
  mean.

Prints the numbers the figures draw.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from sp_validation.nz import ensemble

RUNGS = ["representative", "egs", "success", "zp_patch", "zp_map", "zp_ensemble"]
ZP = ["zp_patch", "zp_map", "zp_ensemble"]
MODELS = ["colour", "bpz", "bpz_desi"]


def mean_sigma(z, draws):
    mean = np.trapezoid(draws * z, z, axis=-1) / np.trapezoid(draws, z, axis=-1)
    return mean.std(0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--figures", type=Path, required=True)
    a = p.parse_args(argv)
    a.figures.mkdir(parents=True, exist_ok=True)
    r = np.load(a.run / "realisations.npz")
    sigma = mean_sigma(r["z"], r["realisations"])
    settings = yaml.safe_load((a.run / "recipe.yaml").read_text())["ensemble"]
    zp_only = dict(draws=1000, seed=1, zero_points=settings["zero_points"])
    cal = ensemble.Calibration.load(a.run)
    sigma_zp = mean_sigma(cal.grid, ensemble.realisations(cal, zp_only)[0])

    delta, extra = defaultdict(dict), defaultdict(dict)
    for rec in map(json.loads, a.records.read_text().splitlines()):
        key = (rec["rung"], rec["model"])
        delta[key][rec["seed"]] = np.array([b["delta"] for b in rec["bins"]])
        extra[key][rec["seed"]] = rec
    bins = np.arange(1, len(sigma) + 1)
    print("data σ⟨z⟩", sigma.round(4), " zero-point term alone", sigma_zp.round(4))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, m in zip(axes, MODELS):
        ax.fill_between(
            bins, -sigma, sigma, step="mid", color="0.88", label="data's ±σ⟨z⟩"
        )
        rungs = [g for g in RUNGS if delta[(g, m)]]
        for j, g in enumerate(rungs):
            d = np.array(list(delta[(g, m)].values()))
            x = bins + (j - (len(rungs) - 1) / 2) * 0.12
            ax.errorbar(
                x,
                d.mean(0),
                d.std(0),
                fmt="o",
                ms=4,
                capsize=2,
                label=f"{g} ({len(d)})",
            )
            print(
                f"{m:8s} {g:14s} mean",
                d.mean(0).round(4),
                "± se",
                (d.std(0) / np.sqrt(len(d))).round(4),
                "scatter",
                d.std(0).round(4),
            )
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(bins)
        ax.set_xlabel("tomographic bin")
        ax.set_title(f"redshift model: {m}")
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("Δ⟨z⟩ = calibrated − true")
    fig.suptitle("SOM n(z) on the catalogue twin: Δ⟨z⟩ per rung (mean ± scatter)")
    fig.tight_layout()
    fig.savefig(a.figures / "ladder.png", dpi=150)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, g in enumerate(ZP):
        for j, m in enumerate(MODELS):
            base = "success" if delta[("success", m)] else "egs"
            seeds = sorted(set(delta[(g, m)]) & set(delta[(base, m)]))
            if not seeds:
                continue
            shift = np.array([delta[(g, m)][s] - delta[(base, m)][s] for s in seeds])
            rms = np.sqrt((shift**2).mean(0))
            print(
                f"{m:8s} {g} − {base}: mean",
                shift.mean(0).round(4),
                "rms",
                rms.round(4),
                f"({len(seeds)})",
            )
            x = bins + (j - 1) * 0.15
            if g == "zp_ensemble":
                axes[1].plot(x, rms, "o", color=f"C{j}", label=m)
            elif g == "zp_map":
                axes[0].errorbar(
                    x, shift.mean(0), shift.std(0), fmt="o", capsize=2, color=f"C{j}",
                    label=f"{m}: map ({len(seeds)})",
                )  # fmt: skip
                region = np.array([extra[(g, m)][s]["region_rms"] for s in seeds])
                axes[2].plot(x, region.mean(0), "o", color=f"C{j}", label=m)
                print(f"{m:8s} zp_map region rms", region.mean(0).round(4))
            else:
                axes[0].plot(x, shift.mean(0), "x", color=f"C{j}", label=f"{m}: patch")
    axes[1].step(bins, sigma_zp, where="mid", color="k", label="ensemble's term")
    axes[0].fill_between(bins, -sigma, sigma, step="mid", color="0.88")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("change of Δ⟨z⟩ (map: mean ± rms)")
    axes[1].set_ylabel("rms change of Δ⟨z⟩, ensemble's EGS offsets")
    axes[2].set_ylabel("rms over 54 deg² regions, true ⟨z⟩ change")
    for ax in axes:
        ax.set_xticks(bins)
        ax.set_xlabel("tomographic bin")
        ax.legend(fontsize=7)
    fig.suptitle("Zero points forward-modelled through the calibration")
    fig.tight_layout()
    fig.savefig(a.figures / "zero_points.png", dpi=150)


if __name__ == "__main__":
    main()
