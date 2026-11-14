"""Run the SOM n(z) calibration on mock galaxies with known redshifts.

Each realisation draws calibrators and sources (:mod:`sp_validation.nz.mock`),
calibrates them with the committed recipe (:func:`sp_validation.nz.pipeline.calibrate`)
and an ensemble of the bootstrap term only, and scores each bin k by

    Δ⟨z⟩_k = ⟨z⟩ of the calibrated n(z) − true shear-weighted ⟨z⟩ of the
             sources with TOM_BIN_ID = k + 1,

the same statistic on every rung. The naive estimate (unweighted mean
spec-z of the calibrators with BPZ in the bin) is scored against the same
truth.

False alarms, fixed before the first run. Per rung and bin:

- bias: |mean Δ| > 3 standard errors (≈1% family-wise over 4 bins);
- calibrated uncertainty: σ_boot / scatter outside [0.75, 1.33], about
  2.5σ of the scatter's own error (1/√(2(N−1)) ≈ 11% at N = 40);
- the naive estimator fails when its |mean Δ| > 3 standard errors and
  exceeds the realisation scatter of the calibrated Δ.

Writes ``ladder.json`` and ``rung<r>.png`` to the output directory, and each
realisation's calibration products to ``rung<r>/<seed>/``.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from sp_validation.data_roots import load_yaml
from sp_validation.nz import calibration as nz
from sp_validation.nz import mock, pipeline

RECIPE = Path(__file__).parents[2] / "config/calibration/nz/v2.yaml"
BIAS_SE, RATIO = 3.0, (0.75, 1.33)


def recipe(side=None, epochs=None, groups=None, draws=500):
    """The committed recipe with the ensemble reduced to its bootstrap term."""
    config = load_yaml(RECIPE)
    config["som"].update(
        {k: v for k, v in dict(side=side, epochs=epochs).items() if v is not None}
    )
    config["groups"] = groups or config["groups"]
    config["ensemble"] = dict(draws=draws, seed=0, bootstrap=True)
    return config


def realisation(seed, *, rung, config, n_cal, n_src, output, keep_products=True):
    """Calibrate one mock realisation and score it."""
    cal, src, z = mock.draw(rung, seed, n_cal, n_src)
    out = Path(output) / f"rung{rung}" / f"{seed:03d}"
    summary = pipeline.calibrate(config, cal, src, out)
    tom = pq.read_table(out / "tom_bin.parquet", columns=["TOM_BIN_ID"])
    tom = tom["TOM_BIN_ID"].to_numpy()
    arrays = np.load(out / "calibration_arrays.npz")
    grid, edges = arrays["grid"], arrays["edges"]
    draws = np.load(out / "realisations.npz")["realisations"]
    boot = np.trapezoid(draws * grid, grid, axis=-1)
    w = src["weight"]
    m, _ = nz.z_magnitudes(src["mag"], config["z2_offset"])
    complete = nz.detected(m).all(1)
    record = dict(seed=seed, retention=summary["retention"], bins=[])
    truth_hists = []
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = tom == k + 1
        prelim = complete & (src["photoz"] >= lo) & (src["photoz"] < hi)
        in_bin = (cal["photoz"] >= lo) & (cal["photoz"] < hi)
        b = summary["bins"][k]
        truth = float(np.average(z[sel], weights=w[sel]))
        record["bins"].append(
            dict(
                calibrated=b["mean"],
                truth=truth,
                truth_pre_gold=float(np.average(z[prelim], weights=w[prelim])),
                naive=float(cal["z"][in_bin].mean()),
                sigma_boot=float(boot[:, k].std()),
                boot_mean_shift=float(boot[:, k].mean() - b["mean"]),
                gold_fraction=b["gold_fraction"],
                weight_fraction=b["weight_fraction"],
                calibrators_in_bin=b["calibrators_in_bpZ_bin"],
            )
        )
        truth_hists.append(nz.deposit(z[sel], w[sel] / w[sel].sum(), grid))
        if seed == 0:
            naive = nz.deposit(
                cal["z"][in_bin], np.full(in_bin.sum(), 1 / in_bin.sum()), grid
            )
            record.setdefault("naive_hist", []).append(naive.tolist())
    if seed == 0:
        record["grid"] = grid.tolist()
        record["edges"] = edges.tolist()
        record["calibrated_hist"] = arrays["bin_hists"].tolist()
        record["truth_hist"] = np.array(truth_hists).tolist()
    if not keep_products:
        for f in out.iterdir():
            f.unlink()
        out.rmdir()
    return record


def score(records):
    """Per-bin ensemble statistics of a rung's realisations."""
    out = []
    n = len(records)
    for k in range(len(records[0]["bins"])):
        b = [r["bins"][k] for r in records]
        delta = np.array([x["calibrated"] - x["truth"] for x in b])
        naive = np.array([x["naive"] - x["truth"] for x in b])
        sigma = np.array([x["sigma_boot"] for x in b])
        scatter = delta.std(ddof=1)
        pull = delta / sigma
        se = scatter / np.sqrt(n)
        naive_se = naive.std(ddof=1) / np.sqrt(n)
        ratio = sigma.mean() / scatter
        out.append(
            dict(
                bin=k + 1,
                mean_delta=delta.mean(),
                se=se,
                scatter=scatter,
                sigma_boot=sigma.mean(),
                ratio=ratio,
                pull_mean=pull.mean(),
                pull_std=pull.std(ddof=1),
                naive_delta=naive.mean(),
                naive_se=naive_se,
                naive_vs_pre_gold=np.mean(
                    [x["naive"] - x["truth_pre_gold"] for x in b]
                ),
                gold_shift=np.mean([x["truth"] - x["truth_pre_gold"] for x in b]),
                gold_fraction=np.mean([x["gold_fraction"] for x in b]),
                weight_fraction=np.mean([x["weight_fraction"] for x in b]),
                boot_mean_shift=np.mean([x["boot_mean_shift"] for x in b]),
                biased=bool(abs(delta.mean()) > BIAS_SE * se),
                miscalibrated=bool(not RATIO[0] <= ratio <= RATIO[1]),
                naive_fails=bool(
                    abs(naive.mean()) > BIAS_SE * naive_se
                    and abs(naive.mean()) > scatter
                ),
            )
        )
    return [
        {k: (float(v) if isinstance(v, np.floating) else v) for k, v in b.items()}
        for b in out
    ]


def figure(records, stats, rung, path):
    """n(z) of one realisation and Δ⟨z⟩ across realisations, per bin."""
    import matplotlib.pyplot as plt

    first = next(r for r in records if "grid" in r)
    grid = np.array(first["grid"])
    n_bin = len(stats)
    colours = dict(truth="#4d4d4d", calibrated="#0072b2", naive="#d55e00")
    fig, axes = plt.subplots(
        2, n_bin, figsize=(3.2 * n_bin, 6), constrained_layout=True
    )
    for k in range(n_bin):
        ax = axes[0, k]
        ax.fill_between(
            grid,
            first["truth_hist"][k],
            step=None,
            color=colours["truth"],
            alpha=0.25,
            lw=0,
            label="true (TOM_BIN sources)",
        )
        ax.plot(
            grid,
            first["calibrated_hist"][k],
            color=colours["calibrated"],
            lw=1.5,
            label="SOM direct calibration",
        )
        ax.plot(
            grid,
            first["naive_hist"][k],
            color=colours["naive"],
            lw=1.2,
            ls="--",
            label="naive (in-bin spectra)",
        )
        lo, hi = first["edges"][k], first["edges"][k + 1]
        ax.axvspan(lo, hi, color="0.9", zorder=-1)
        ax.set_xlim(0, 2)
        ax.set_xlabel("z")
        ax.text(
            0.97,
            0.95,
            f"bin {k + 1}\nBPZ [{lo:.2f}, {hi:.2f})",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        if k == 0:
            ax.set_ylabel("n(z), realisation 0")
            ax.legend(fontsize=7, loc="center right", frameon=False)

        ax = axes[1, k]
        s = stats[k]
        delta = np.array(
            [r["bins"][k]["calibrated"] - r["bins"][k]["truth"] for r in records]
        )
        naive = np.array(
            [r["bins"][k]["naive"] - r["bins"][k]["truth"] for r in records]
        )
        lim = max(4 * s["scatter"], 1.2 * np.abs(naive).max(), 0.005)
        bins = np.linspace(-lim, lim, 41)
        ax.hist(delta, bins, color=colours["calibrated"], alpha=0.7, label="calibrated")
        ax.hist(naive, bins, color=colours["naive"], alpha=0.5, label="naive")
        x = np.linspace(-lim, lim, 400)
        scale = len(delta) * (bins[1] - bins[0])
        ax.plot(
            x,
            scale
            * np.exp(-0.5 * ((x - s["mean_delta"]) / s["sigma_boot"]) ** 2)
            / (np.sqrt(2 * np.pi) * s["sigma_boot"]),
            color="k",
            lw=1,
            label="N(mean, σ_boot)",
        )
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_xlabel(r"$\Delta\langle z\rangle$")
        ax.text(
            0.97,
            0.95,
            f"⟨Δ⟩ = {s['mean_delta']:+.4f} ± {s['se']:.4f}\n"
            f"scatter {s['scatter']:.4f}\nσ_boot/scatter {s['ratio']:.2f}\n"
            f"naive ⟨Δ⟩ = {s['naive_delta']:+.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85),
        )
        ax.set_ylim(0, 1.6 * ax.get_ylim()[1])
        if k == 0:
            ax.set_ylabel(f"realisations (N = {len(records)})")
            fig.legend(
                *ax.get_legend_handles_labels(),
                loc="lower center",
                ncol=3,
                fontsize=8,
                frameon=False,
                bbox_to_anchor=(0.5, -0.04),
            )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rungs", type=int, nargs="+", default=[0, 1])
    p.add_argument("--realisations", type=int, default=40)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=1, help="realisations at once")
    p.add_argument("--output", required=True)
    p.add_argument("--n-cal", type=int, default=9920)
    p.add_argument("--n-src", type=int, default=2_000_000)
    p.add_argument("--side", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--groups", type=int)
    p.add_argument("--draws", type=int, default=500, help="bootstrap draws")
    p.add_argument("--drop-products", action="store_true")
    p.add_argument(
        "--figures-only",
        action="store_true",
        help="rescore and redraw from the output's saved records",
    )
    a = p.parse_args(argv)
    output = Path(a.output)
    if a.figures_only:
        ladder = json.loads((output / "ladder.json").read_text())
        for rung in a.rungs:
            records = json.loads((output / f"rung{rung}.records.json").read_text())
            stats = score(records)
            ladder["rungs"][str(rung)] = dict(realisations=len(records), bins=stats)
            figure(records, stats, rung, output / f"rung{rung}.png")
        (output / "ladder.json").write_text(json.dumps(ladder, indent=2) + "\n")
        return
    output.mkdir(parents=True, exist_ok=True)
    config = recipe(a.side, a.epochs, a.groups, a.draws)
    seeds = range(a.first_seed, a.first_seed + a.realisations)
    ladder = dict(
        n_cal=a.n_cal,
        n_src=a.n_src,
        som=config["som"],
        groups=config["groups"],
        bootstrap_draws=a.draws,
        thresholds=dict(bias_se=BIAS_SE, ratio=list(RATIO)),
        rungs={},
    )
    for rung in a.rungs:
        run = partial(
            realisation,
            rung=rung,
            config=config,
            n_cal=a.n_cal,
            n_src=a.n_src,
            output=output,
            keep_products=not a.drop_products,
        )
        with ProcessPoolExecutor(a.jobs) as pool:
            records = list(pool.map(run, seeds))
        (output / f"rung{rung}.records.json").write_text(json.dumps(records))
        stats = score(records)
        ladder["rungs"][str(rung)] = dict(realisations=len(records), bins=stats)
        figure(records, stats, rung, output / f"rung{rung}.png")
    (output / "ladder.json").write_text(json.dumps(ladder, indent=2) + "\n")


if __name__ == "__main__":
    main()
