"""Draw the clustering-redshift figures from the products they compare.

Reads ``clustering.json`` from the recipe's ``clustering.output`` (written by
``sp_validation.nz.clustering``) and the SOM n(z) of the calibration run in the
recipe's ``output``, and writes PNGs to the figure directory:

- ``clustering_gate.png``: the known-answer gate, the clustering n(z) of a
  held-out tracer half against its true n(z);
- ``clustering_nz.png``: each gold bin's n(z) b(z), divided by its fitted
  amplitude and bias tilt, against the SOM n(z) (and the alternative
  calibrators' when the run has them).

Run it inside the container:

    python scripts/calibration/plot_clustering.py RECIPE FIGURE_DIR
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_nz import COLORS, save

from sp_validation.data_roots import load_yaml
from sp_validation.nz.clustering import candidates


def gate(path, result):
    g = result["gate"]
    z, dz = np.array(g["z"]), np.diff(g["z"]).mean()
    fig, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained")
    ax.step(
        np.r_[z - dz / 2, z[-1] + dz / 2],
        np.r_[g["truth"], g["truth"][-1]],
        where="post",
        color="0.3",
        lw=1.2,
        label="true n(z) of the held-out half",
    )
    ax.errorbar(
        z,
        g["n"],
        g["err"],
        fmt="o",
        ms=4,
        color=COLORS[0],
        label=f"clustering, χ² = {g['chi2']:.1f} / {g['dof']}",
    )
    ax.axhline(0, color="k", lw=0.5)
    s = g["shift"]
    ax.set(
        xlabel="z",
        ylabel="n(z)",
        title=f"Known answer: ⟨z⟩ {g['mean_z']:.3f} ± {g['mean_z_err']:.3f} "
        f"(true {g['mean_z_truth']:.3f}), shift {s['best']:+.3f} "
        f"[{s['lo']:+.3f}, {s['hi']:+.3f}]",
    )
    ax.title.set_fontsize(9)
    ax.legend(fontsize=8)
    save(fig, path)


def bins(path, result, grid, cands):
    lo, hi = np.array(result["z_lo"]), np.array(result["z_hi"])
    z = (lo + hi) / 2
    styles = {"calibration": ("k", "-"), "alternative calibrators": (COLORS[1], "--")}
    labels = {
        "calibration": "SOM n(z)",
        "alternative calibrators": "SOM n(z), alternative calibrators",
    }
    fig, axes = plt.subplots(
        len(result["bins"]), 1, figsize=(7, 8.5), sharex=True, layout="constrained"
    )
    for k, (b, ax) in enumerate(zip(result["bins"], axes)):
        fit = b["fits"]["calibration"]
        tilt = (1 + z) ** fit["alpha"]
        A = fit["amplitude"]
        ax.errorbar(
            z,
            np.array(b["nb"]) / tilt / A,
            np.array(b["nb_err"]) / tilt / A,
            fmt="o",
            ms=3,
            lw=1,
            color=COLORS[0],
            label="clustering × DESI DR1",
        )
        for name, hists in cands.items():
            ax.plot(
                grid,
                hists[k],
                color=styles[name][0],
                ls=styles[name][1],
                lw=1.2,
                label=labels[name],
            )
        ax.axvspan(*b["core"], color=COLORS[0], alpha=0.07, lw=0)
        ax.axhline(0, color="k", lw=0.5)
        shift = fit["profiled_shift"]
        text = (
            f"bin {k + 1}   χ² = "
            + ", ".join(f"{f['chi2']:.1f}" for f in b["fits"].values())
            + f" / {fit['dof']}   profile Δz = {shift['best']:+.3f} "
            f"[{shift['lo']:+.3f}, {shift['hi']:+.3f}]"
            f"  α(profile) = {shift['alpha']:.2f}"
            + (" (interval at prior edge)" if shift["alpha_boundary"] else "")
        )
        ax.text(
            0.99, 0.93, text, transform=ax.transAxes, ha="right", va="top", fontsize=8
        )
        ax.set_ylabel("n(z)")
    axes[0].legend(fontsize=7, loc="center right")
    axes[-1].set(xlabel="z", xlim=(0, 2.6))
    r_lo, r_hi = result["annulus"]
    fig.suptitle(
        f"Gold bins × spectroscopic slices, {r_lo:g}–{r_hi:g} Mpc/h; fit window shaded",
        fontsize=10,
    )
    save(fig, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", help="the n(z) recipe with a clustering section")
    parser.add_argument("figures", help="directory to write the PNGs to")
    args = parser.parse_args(argv)

    config = load_yaml(args.recipe)
    out = Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)
    path = Path(config["clustering"]["output"]) / "clustering.json"
    result = json.loads(path.read_text())
    gate(out / "clustering_gate.png", result)
    bins(out / "clustering_nz.png", result, *candidates(config["output"]))


if __name__ == "__main__":
    main()
