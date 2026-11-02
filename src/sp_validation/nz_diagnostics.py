"""Figures for the direct SOM redshift calibration."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

COLORS = ("#146e7c", "#d2791c", "#322d61", "#898989")


def _save(fig, path):
    fig.savefig(path, dpi=100, bbox_inches="tight", pil_kwargs={"optimize": True})
    plt.close(fig)


def _curve(path):
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1:]


def _normalise(y, z):
    return y / np.trapezoid(y, z)


def _mechanics(path, gm, group, z, weights, grid):
    ngroup = int(gm.max()) + 1
    ncal = np.bincount(group, minlength=ngroup)
    zsum = np.bincount(group, weights=z, minlength=ngroup)
    mean_z = np.divide(zsum, ncal, out=np.full(ngroup, np.nan), where=ncal > 0)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    image = axes[0].imshow(
        mean_z[gm].reshape(101, 101), origin="lower", cmap="viridis", vmin=0, vmax=2
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
    raw = np.histogram(z, bins=np.r_[grid, grid[-1] + grid[1] - grid[0]])[0].astype(
        float
    )
    calibrated = np.zeros(len(grid))
    for k in range(weights.shape[0]):
        calibrated += np.histogram(
            z, bins=np.r_[grid, grid[-1] + grid[1] - grid[0]], weights=weights[k]
        )[0]
    raw = raw / np.trapezoid(raw, grid)
    calibrated = calibrated / np.trapezoid(calibrated, grid)
    axes[2].plot(grid, raw, color=COLORS[3], lw=1.3, label="Unweighted spectra")
    axes[2].plot(
        grid, calibrated, color=COLORS[0], lw=1.8, label="Source-weighted spectra"
    )
    axes[2].set(
        xlim=(0, 2),
        xlabel="Spectroscopic redshift",
        ylabel="Normalized n(z)",
        title="Sum the per-group weighted spectra",
    )
    axes[2].legend(fontsize=8)
    fig.suptitle(
        "SOM direct calibration: map similar photometry, then transfer source occupancy to spectra"
    )
    _save(fig, path)


def render(config, output_dir):
    """Render calibration figures from saved products and explicit input paths.

    Required config keys are ``calibration_arrays``, ``model``, ``summary``
    and ``source_catalogue``. The source catalogue NPZ contains ``weight``,
    ``photoz`` and ``mag``. An optional ``source_assignment`` NPZ can provide
    precomputed ``cell`` values; otherwise cells are assigned in chunks.
    Optional keys ``published_nz`` and ``tile_nz`` point to text n(z) files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = np.load(config["calibration_arrays"])
    model = np.load(config["model"])
    summary = __import__("json").loads(Path(config["summary"]).read_text())
    source = np.load(config["source_catalogue"])

    z = arrays["z"]
    p = arrays["cal_photoz"]
    group = arrays["cal_group"]
    cell = arrays["cal_cell"]
    edges = arrays["edges"]
    gold = arrays["gold"]
    weights = arrays["cal_weights"]
    grid = arrays["grid"]
    gm = model["group_map"]
    src_w = source["weight"]
    src_p = source["photoz"]
    src_mag = source["mag"]
    usable = (
        np.isfinite(src_mag).all(axis=1)
        & (src_mag > 0).all(axis=1)
        & (src_mag < 50).all(axis=1)
        & (src_w > 0)
        & np.isfinite(src_p)
        & (src_p >= edges[0])
        & (src_p < edges[-1])
    )
    if config.get("source_assignment"):
        cells = np.load(config["source_assignment"])["cell"][usable]
    else:
        from . import nz_calibration

        valid_mag = src_mag[usable]
        cells = np.empty(len(valid_mag), dtype=np.int32)
        chunk_size = config.get("assignment_chunk_size", 200_000)
        for start in range(0, len(valid_mag), chunk_size):
            stop = min(start + chunk_size, len(valid_mag))
            _, cells[start:stop] = nz_calibration.assign(
                valid_mag[start:stop], model, workers=config.get("workers", 1)
            )
    cells, src_w, src_p = cells, src_w[usable], src_p[usable]
    ncell = len(gm)
    src_weight_cell = np.bincount(cells, weights=src_w, minlength=ncell)
    src_p_cell = np.bincount(cells, weights=src_w * src_p, minlength=ncell)
    mean_p = np.divide(
        src_p_cell,
        src_weight_cell,
        out=np.full(ncell, np.nan),
        where=src_weight_cell > 0,
    )
    cal_count = np.bincount(cell, minlength=ncell)
    n_group = int(gm.max()) + 1
    spec_count = np.bincount(group, minlength=n_group)
    spec_sum = np.bincount(group, weights=z, minlength=n_group)
    spec_sq = np.bincount(group, weights=z * z, minlength=n_group)
    spec_mean = np.divide(
        spec_sum, spec_count, out=np.full(n_group, np.nan), where=spec_count > 0
    )
    spec_std = np.sqrt(
        np.maximum(
            0,
            np.divide(
                spec_sq, spec_count, out=np.full(n_group, np.nan), where=spec_count > 0
            )
            - spec_mean**2,
        )
    )
    spec_std[spec_count < 2] = np.nan

    # SOM plane: sparse calibrators and source occupancy share the trained map.
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    maps = [
        cal_count,
        src_weight_cell,
        spec_mean[gm],
        spec_std[gm],
        mean_p,
        np.zeros(ncell),
    ]
    titles = [
        "Calibrators per cell",
        "Source shear weight per cell",
        "Mean spec-z per group",
        "Spec-z scatter per group",
        "Mean source BPZ per cell",
        "Ward-group boundaries",
    ]
    for ax, values, title in zip(axes.flat, maps, titles):
        plane = np.asarray(values).reshape(101, 101)
        if title.startswith("Ward"):
            groups = gm.reshape(101, 101)
            plane = np.zeros_like(groups, dtype=float)
            plane[1:, :] = groups[1:, :] != groups[:-1, :]
            plane[:, 1:] += groups[:, 1:] != groups[:, :-1]
        m = np.ma.masked_invalid(plane)
        if title.startswith("Calibrators") or title.startswith("Source"):
            m = np.ma.masked_where(m <= 0, m)
            im = ax.imshow(m, origin="lower", cmap="viridis", norm=LogNorm())
        else:
            im = ax.imshow(m, origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    _save(fig, out / "som_plane.png")

    # Per-bin gold support and weight-per-calibrator distributions.
    fig, axes = plt.subplots(1, len(edges) - 1, figsize=(15, 4), layout="constrained")
    for k, ax in enumerate(np.atleast_1d(axes)):
        passed = gold[k, gm].reshape(101, 101)
        ax.imshow(
            passed, origin="lower", cmap="YlGn", vmin=0, vmax=1, interpolation="nearest"
        )
        ax.set_title(
            f"Bin {k + 1}: {100 * (1 - summary['bins'][k]['gold_fraction']):.1f}% weight lost"
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Per-bin gold groups (green) and rejected groups (pale)", y=1.08)
    _save(fig, out / "gold_masks.png")

    fig, axes = plt.subplots(1, len(edges) - 1, figsize=(15, 4), layout="constrained")
    for k, ax in enumerate(np.atleast_1d(axes)):
        in_bin = np.isfinite(p) & (p >= edges[k]) & (p < edges[k + 1])
        ng = np.bincount(group[in_bin], minlength=n_group)
        wg = np.bincount(group, weights=weights[k], minlength=n_group)
        ratio = np.divide(
            wg, ng, out=np.full(n_group, np.nan), where=(ng > 0) & (wg > 0)
        )
        vals = ratio[np.isfinite(ratio)]
        ax.hist(vals, bins=np.geomspace(vals.min(), vals.max(), 36), color=COLORS[0])
        ax.set_xscale("log")
        ax.set(
            title=f"Bin {k + 1}",
            xlabel="source shear weight / calibrator",
            ylabel="groups",
        )
    _save(fig, out / "reweighting.png")

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
        title=f"BPZ-selected calibrators: NMAD {nmad:.3f}, outliers {100 * outlier:.1f}%",
    )
    _save(fig, out / "bpz_vs_specz.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, layout="constrained")
    tile_path = config.get("tile_nz")
    if tile_path:
        tz, tile = _curve(tile_path)
    for k, ax in enumerate(axes.flat):
        full_curve = _normalise(arrays["bin_hists"][k], grid)
        ax.plot(grid, full_curve, color=COLORS[0], lw=1.7, label="Full area, z + z2")
        if tile_path:
            ax.plot(
                tz,
                _normalise(tile[:, k], tz),
                color=COLORS[1],
                lw=1.4,
                label="600-tile reference",
            )
        ax.axvspan(edges[k], edges[k + 1], color="0.92", zorder=0)
        ax.set(
            title=f"Bin {k + 1}: BPZ [{edges[k]:.2f}, {edges[k + 1]:.2f})",
            xlabel="Redshift z",
            ylabel="n(z)",
        )
        ax.set_xlim(0, 2.2)
        ax.legend(fontsize=8)
    fig.suptitle("Per-bin n(z): full area compared with 600-tile calibration")
    _save(fig, out / "nz_full_vs_600tiles.png")

    # Coverage at each selection stage, comparing canonical-z-only and z+z2.
    ret = summary["retention"]
    stage_names = ["all", "complete_photometry", "bpz_bounds", "gold"]
    labels = ["All", "Complete\nphotometry", "BPZ\nbounds", "Gold"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    width = 0.34
    for j, (key, label, color) in enumerate(
        (("zonly", "Canonical z only", COLORS[2]), ("both", "z + z2", COLORS[0]))
    ):
        values = [100 * ret[key][name] for name in stage_names]
        bars = ax.bar(x + (j - 0.5) * width, values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylabel="Retained bound-9 shear weight (%)",
        ylim=(0, 110),
        title="Coverage with and without z2",
    )
    ax.legend()
    _save(fig, out / "coverage_z2.png")

    # The sum uses the file columns as delivered; published v1 is normalized
    # independently because it represents a different selected population.
    if config.get("published_nz"):
        pz, published = _curve(config["published_nz"])
        fractions = np.array([b["weight_fraction"] for b in summary["bins"]])
        total = fractions @ arrays["bin_hists"]
        total = _normalise(total, grid)
        published = _normalise(published[:, 0], pz)
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(10, 7.5),
            gridspec_kw={"height_ratios": [2, 1]},
            layout="constrained",
        )
        axes[0].plot(
            grid,
            total,
            color=COLORS[0],
            lw=2,
            label=f"Four-bin sum, mean {np.trapezoid(grid * total, grid):.4f}",
        )
        if tile_path:
            tile_total = _normalise(tile.sum(axis=1), tz)
            axes[0].plot(
                tz,
                tile_total,
                color=COLORS[1],
                lw=1.7,
                label=f"600-tile sum, mean {np.trapezoid(tz * tile_total, tz):.4f}",
            )
        axes[0].plot(
            pz,
            published,
            color=COLORS[2],
            lw=2,
            label=f"Published v1.4.6.3, mean {np.trapezoid(pz * published, pz):.4f}",
        )
        axes[0].set(xlim=(0, 2.2), ylabel="Normalized n(z)")
        axes[0].legend()
        axes[1].plot(
            grid,
            np.r_[0, np.cumsum((total[1:] + total[:-1]) * np.diff(grid) / 2)],
            color=COLORS[0],
            lw=2,
            label="Four-bin sum",
        )
        if tile_path:
            axes[1].plot(
                tz,
                np.r_[
                    0, np.cumsum((tile_total[1:] + tile_total[:-1]) * np.diff(tz) / 2)
                ],
                color=COLORS[1],
                lw=1.7,
                label="600-tile sum",
            )
        axes[1].plot(
            pz,
            np.r_[0, np.cumsum((published[1:] + published[:-1]) * np.diff(pz) / 2)],
            color=COLORS[2],
            lw=2,
            label="Published v1.4.6.3",
        )
        axes[1].set(
            xlim=(0, 2.2),
            ylim=(0, 1.02),
            xlabel="Redshift z",
            ylabel="Cumulative probability",
        )
        _save(fig, out / "summed_bins_vs_v1.png")

    _mechanics(out / "som_calibration_mechanics.png", gm, group, z, weights, grid)
    return sorted(path.name for path in out.glob("*.png"))
