"""Draw the figures of an n(z) calibration from the products it wrote.

Reads ``calibration_arrays.npz``, ``som_model.npz``, ``summary.json`` and
``recipe.yaml`` from the run directory (see ``sp_validation.nz.pipeline``)
and writes PNGs to the figure directory:

- ``som_plane.png``: calibrators, source weight, mean source BPZ and the
  groups' spec-z mean, scatter and count across the SOM;
- ``gold_masks.png``: each bin's gold groups and the weight gold removes;
- ``bpz_vs_specz.png``: the calibrators' BPZ against their spec-z, with the
  bin edges;
- ``nz_vs_reference.png``: each bin's n(z), against ``--reference`` if given;
- ``summed_bins_vs_v1.png`` (with ``--published``): the summed n(z) and its
  cumulative distribution against a non-tomographic n(z);
- ``som_calibration_mechanics.png``: how source occupancy weights the spectra.

Run it inside the container:

    python scripts/calibration/plot_nz.py RUN_DIR FIGURE_DIR
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LogNorm

COLORS = ("#146e7c", "#d2791c", "#322d61", "#898989")
plt.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans"})


def save(fig, path):
    fig.savefig(path, dpi=100, bbox_inches="tight", pil_kwargs={"optimize": True})
    plt.close(fig)


def curve(path):
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1:]


def normalise(y, z):
    return y / np.trapezoid(y, z)


def cumulative(y, z):
    return np.r_[0, np.cumsum((y[1:] + y[:-1]) * np.diff(z) / 2)]


def som_plane(path, arrays, gm, side):
    z, group, cell = arrays["z"], arrays["cal_group"], arrays["cal_cell"]
    ncell, ngroup = len(gm), int(gm.max()) + 1
    src_weight_cell = arrays["source_weight_cell"]
    mean_p = np.divide(
        arrays["source_photoz_cell"],
        src_weight_cell,
        out=np.full(ncell, np.nan),
        where=src_weight_cell > 0,
    )
    count = np.bincount(group, minlength=ngroup)
    mean = np.divide(
        np.bincount(group, weights=z, minlength=ngroup),
        count,
        out=np.full(ngroup, np.nan),
        where=count > 0,
    )
    square = np.divide(
        np.bincount(group, weights=z * z, minlength=ngroup),
        count,
        out=np.full(ngroup, np.nan),
        where=count > 0,
    )
    std = np.sqrt(np.maximum(0, square - mean**2))
    std[count < 2] = np.nan

    panels = [
        ("Calibrators per cell", np.bincount(cell, minlength=ncell), "log"),
        ("Source shear weight per cell", src_weight_cell, "log"),
        ("Mean spec-z per group", mean[gm], "z"),
        ("Spec-z scatter per group", std[gm], None),
        ("Mean source BPZ per cell", mean_p, None),
        ("Calibrators per group", count[gm], "log"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for ax, (title, values, scale) in zip(axes.flat, panels):
        plane = np.ma.masked_invalid(np.asarray(values, float).reshape(side, side))
        if scale == "log":
            plane = np.ma.masked_where(plane <= 0, plane)
            im = ax.imshow(plane, origin="lower", cmap="viridis", norm=LogNorm())
        elif scale == "z":
            im = ax.imshow(plane, origin="lower", cmap="viridis", vmin=0, vmax=2)
        else:
            im = ax.imshow(plane, origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    save(fig, path)


def gold_masks(path, arrays, gm, side, summary):
    gold = arrays["gold"]
    fig, axes = plt.subplots(1, len(gold), figsize=(15, 4), layout="constrained")
    for k, ax in enumerate(np.atleast_1d(axes)):
        ax.imshow(
            gold[k, gm].reshape(side, side),
            origin="lower",
            cmap="YlGn",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        lost = 100 * (1 - summary["bins"][k]["gold_fraction"])
        ax.set_title(f"Bin {k + 1}: {lost:.1f}% weight lost")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Per-bin gold groups (green) and rejected groups (pale)", y=1.08)
    save(fig, path)


def bpz_vs_specz(path, arrays):
    z, p, edges = arrays["z"], arrays["cal_photoz"], arrays["edges"]
    valid = np.isfinite(p) & (p >= edges[0]) & (p < edges[-1])
    dz = (p[valid] - z[valid]) / (1 + z[valid])
    nmad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
    outlier = np.mean(np.abs(dz) > 0.15)
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    h = ax.hist2d(
        z[valid],
        p[valid],
        bins=100,
        range=[[0, 2], [0, 1.2]],
        norm=LogNorm(),
        cmap="viridis",
    )
    fig.colorbar(h[3], ax=ax, label="calibrators")
    ax.plot([0, 2], [0, 2], color="white", lw=1)
    for edge in edges:
        ax.axhline(edge, color="#d1495b", lw=1)
    ax.set(
        xlim=(0, 2),
        ylim=(0, 1.2),
        xlabel="Spectroscopic redshift",
        ylabel="DR6 BPZ",
        title=f"BPZ-selected calibrators: NMAD {nmad:.3f}, "
        f"outliers {100 * outlier:.1f}%",
    )
    save(fig, path)


def nz_vs_reference(path, arrays, label, reference):
    grid, edges = arrays["grid"], arrays["edges"]
    nbin = len(edges) - 1
    fig, axes = plt.subplots(
        (nbin + 1) // 2,
        2,
        figsize=(11, 3.75 * ((nbin + 1) // 2)),
        sharex=True,
        layout="constrained",
    )
    for k, ax in enumerate(axes.flat[:nbin]):
        ax.plot(
            grid,
            normalise(arrays["bin_hists"][k], grid),
            color=COLORS[0],
            lw=1.7,
            label=label,
        )
        if reference:
            tz, ref = reference[1]
            ax.plot(
                tz,
                normalise(ref[:, k], tz),
                color=COLORS[1],
                lw=1.4,
                label=reference[0],
            )
        ax.axvspan(edges[k], edges[k + 1], color="0.92", zorder=0)
        ax.set(
            title=f"Bin {k + 1}: BPZ [{edges[k]:.2f}, {edges[k + 1]:.2f})",
            xlabel="Redshift z",
            ylabel="n(z)",
            xlim=(0, 2.2),
        )
        ax.legend(fontsize=8)
    versus = f" vs {reference[0]}" if reference else ""
    fig.suptitle(f"Per-bin n(z): {label}{versus}")
    save(fig, path)


def summed_bins(path, arrays, summary, label, reference, published):
    """The bins summed by their weight fractions, against a published n(z).

    The published n(z) is normalised on its own: it describes a different
    selection.
    """
    grid = arrays["grid"]
    fractions = np.array([b["weight_fraction"] for b in summary["bins"]])
    curves = [(label, grid, normalise(fractions @ arrays["bin_hists"], grid))]
    colors = [COLORS[0]]
    if reference:
        tz, ref = reference[1]
        curves.append((reference[0], tz, normalise(ref.sum(axis=1), tz)))
        colors.append(COLORS[1])
    pz, pub = published[1]
    curves.append((published[0], pz, normalise(pub[:, 0], pz)))
    colors.append(COLORS[2])

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7.5),
        gridspec_kw={"height_ratios": [2, 1]},
        layout="constrained",
    )
    for (name, z, y), color in zip(curves, colors):
        mean = np.trapezoid(z * y, z)
        axes[0].plot(z, y, color=color, lw=2, label=f"{name}, mean {mean:.4f}")
        axes[1].plot(z, cumulative(y, z), color=color, lw=2, label=name)
    axes[0].set(xlim=(0, 2.2), ylabel="Normalized n(z)")
    axes[0].legend()
    axes[1].set(
        xlim=(0, 2.2),
        ylim=(0, 1.02),
        xlabel="Redshift z",
        ylabel="Cumulative probability",
    )
    save(fig, path)


def mechanics(path, arrays, gm, side):
    z, group = arrays["z"], arrays["cal_group"]
    weights, grid = arrays["cal_weights"], arrays["grid"]
    ngroup = int(gm.max()) + 1
    ncal = np.bincount(group, minlength=ngroup)
    mean_z = np.divide(
        np.bincount(group, weights=z, minlength=ngroup),
        ncal,
        out=np.full(ngroup, np.nan),
        where=ncal > 0,
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    image = axes[0].imshow(
        mean_z[gm].reshape(side, side), origin="lower", cmap="viridis", vmin=0, vmax=2
    )
    axes[0].set(
        title="SOM groups colored by mean spec-z", xlabel="SOM cell", ylabel="SOM cell"
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(image, ax=axes[0], label="Mean spectroscopic z")
    for k in range(weights.shape[0]):
        count = np.bincount(group, weights=(weights[k] > 0), minlength=ngroup)
        group_weight = np.bincount(group, weights=weights[k], minlength=ngroup)
        ratio = np.divide(
            group_weight,
            count,
            out=np.full(ngroup, np.nan),
            where=(count > 0) & (group_weight > 0),
        )
        ratio = ratio[np.isfinite(ratio)]
        axes[1].hist(
            ratio,
            bins=np.geomspace(ratio.min(), ratio.max(), 32),
            histtype="step",
            lw=1.6,
            color=COLORS[k % len(COLORS)],
            label=f"Bin {k + 1}",
        )
    axes[1].set(
        xscale="log",
        xlabel="Source shear weight / calibrator",
        ylabel="Groups",
        title="Group occupancy sets each spectrum's weight",
    )
    axes[1].legend(fontsize=8)
    bins = np.r_[grid, grid[-1] + grid[1] - grid[0]]
    raw = np.histogram(z, bins=bins)[0].astype(float)
    calibrated = np.histogram(z, bins=bins, weights=weights.sum(axis=0))[0]
    axes[2].plot(
        grid, normalise(raw, grid), color=COLORS[3], lw=1.3, label="Unweighted spectra"
    )
    axes[2].plot(
        grid,
        normalise(calibrated, grid),
        color=COLORS[0],
        lw=1.8,
        label="Source-weighted spectra",
    )
    axes[2].set(
        xlim=(0, 2),
        xlabel="Spectroscopic redshift",
        ylabel="Normalized n(z)",
        title="Sum the per-group weighted spectra",
    )
    axes[2].legend(fontsize=8)
    fig.suptitle(
        "SOM direct calibration: map similar photometry, "
        "then transfer source occupancy to spectra"
    )
    save(fig, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="the calibration's output directory")
    parser.add_argument("figures", help="directory to write the PNGs to")
    parser.add_argument("--label", default="This calibration")
    parser.add_argument(
        "--reference",
        nargs=2,
        metavar=("NZ_TOMO", "LABEL"),
        help="a tomographic n(z) file to overlay on each bin",
    )
    parser.add_argument(
        "--published",
        nargs=2,
        metavar=("NZ", "LABEL"),
        help="a non-tomographic n(z) file to compare the summed bins with",
    )
    args = parser.parse_args(argv)

    run, out = Path(args.run), Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)
    arrays = np.load(run / "calibration_arrays.npz")
    gm = np.load(run / "som_model.npz")["group_map"]
    summary = json.loads((run / "summary.json").read_text())
    side = yaml.safe_load((run / "recipe.yaml").read_text())["som"]["side"]
    reference = args.reference and (args.reference[1], curve(args.reference[0]))
    published = args.published and (args.published[1], curve(args.published[0]))

    som_plane(out / "som_plane.png", arrays, gm, side)
    gold_masks(out / "gold_masks.png", arrays, gm, side, summary)
    bpz_vs_specz(out / "bpz_vs_specz.png", arrays)
    nz_vs_reference(out / "nz_vs_reference.png", arrays, args.label, reference)
    if published:
        summed_bins(
            out / "summed_bins_vs_v1.png",
            arrays,
            summary,
            args.label,
            reference,
            published,
        )
    mechanics(out / "som_calibration_mechanics.png", arrays, gm, side)


if __name__ == "__main__":
    main()
