"""Summarize the minimum-count ladder; run in the allocated container."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def summarize(root, dest, chosen):
    dest.mkdir(parents=True, exist_ok=True)
    grids = [
        json.loads((root / f"sparse_grid_{g}.json").read_text()) for g in range(1, 5)
    ]
    public = json.loads((root / "sparse_public.json").read_text())
    # Optional follow-up thresholds avoid rerunning the expensive initial ladder.
    extra = root / "low-count"
    if extra.exists():
        for g, result in enumerate(grids, 1):
            low = json.loads((extra / f"sparse_grid_{g}.json").read_text())
            result["ladder"].update(low["ladder"])
            result.setdefault("grid4_deletion", {}).update(low["grid4_deletion"])
            for branch in result["occupancy"]:
                result["occupancy"][branch]["thresholds"].update(
                    low["occupancy"][branch]["thresholds"]
                )
        low = json.loads((extra / "sparse_public.json").read_text())
        for policy in ("reconstructed", "released"):
            public[policy]["thresholds"].update(low[policy]["thresholds"])
    ns = sorted(map(int, grids[0]["ladder"]))
    for result in grids:
        result["ladder"] = {str(n): result["ladder"][str(n)] for n in ns}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for g, (ax, result) in enumerate(zip(axes.flat, grids, strict=True), 1):
        for i, color in zip((0, 2), ("C0", "C1"), strict=True):
            for policy, style in (("refit", "-o"), ("fixed", "--")):
                values = [result["ladder"][str(n)][policy]["error"][i] for n in ns]
                ax.plot(
                    ns, values, style, color=color, label=f"m{i // 2 + 1}, {policy}-W"
                )
        ax.axvline(chosen, color=".5", lw=1, label=f"chosen N_min={chosen}")
        ax.set(
            title=f"Grid {g}",
            ylabel="Jackknife standard error of m",
            xscale="symlog",
            xlabel="Minimum training count",
        )
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xticks(
            [0, 3, 10, 30, 100, 300, 1000],
            labels=["0", "3", "10", "30", "100", "300", "1000"],
        )
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(dest / "refit_w_errors.png", dpi=170)
    plt.close(fig)

    samples = [x["occupancy"]["1z2z"] for x in grids] + [public["reconstructed"]]
    names = [f"Grid {g}" for g in range(1, 5)] + ["Public v1.4.6.3"]
    fig = plt.figure(figsize=(15, 7))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.3])
    top = outer[0].subgridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.06])
    bottom = outer[1].subgridspec(1, 3)
    for j, (sample, name) in enumerate(zip(samples, names, strict=True)):
        ax = fig.add_subplot(top[0, j])
        count = np.array(sample["count"])
        im = ax.imshow(
            np.ma.masked_equal(count.T, 0),
            origin="lower",
            norm=LogNorm(1, 1e6),
            aspect="auto",
            cmap="viridis",
        )
        ax.set(title=name, xlabel="SNR cell")
        if j == 0:
            ax.set_ylabel("Size cell")
        else:
            ax.tick_params(labelleft=False)
    fig.colorbar(im, cax=fig.add_subplot(top[0, 5]), label="Training objects per cell")
    for panel, key, ylabel in (
        (0, "object_fraction", "Objects in sparse cells (%)"),
        (1, "weight_fraction", "Weight in sparse cells (%)"),
    ):
        ax = fig.add_subplot(bottom[0, panel])
        for sample, name in zip(samples, names, strict=True):
            ax.plot(
                ns[1:],
                [100 * sample["thresholds"][str(n)][key] for n in ns[1:]],
                "-o",
                label=name,
            )
        ax.axvline(chosen, color=".5", lw=1)
        ax.set(
            xscale="log",
            xlabel="Count threshold N (cells with count < N)",
            ylabel=ylabel,
        )
        ax.set_yscale("symlog", linthresh=0.001)
        ax.set_ylim(0, 110)
        ax.grid(alpha=0.2)
        if panel == 0:
            ax.legend(fontsize=8)
    ax = fig.add_subplot(bottom[0, 2])
    for result, name in zip(grids, names[:4], strict=True):
        noise = result["sampling_noise"]
        counts = list(map(int, noise))
        ax.loglog(counts, [noise[str(n)]["median"] for n in counts], "-o", label=name)
    ax.set(xlabel="Resampled training count", ylabel="Median SD(W) / mean(W)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(dest / "occupancy.png", dpi=170)
    plt.close(fig)

    lines = [
        "# Minimum-count ladder",
        "",
        "Each cell lists m; refit-W error; fixed-W error.",
        "",
        "| Grid | N_min | m1; sigma_refit; sigma_fixed | m2; sigma_refit; sigma_fixed |",
        "| --- | ---: | --- | --- |",
    ]
    shifts = {}
    diagnostics = {}
    for g, result in enumerate(grids, 1):
        for n in ns:
            item = result["ladder"][str(n)]
            cells = [
                f"{item['value'][i]:+.6f}; {item['refit']['error'][i]:.6f}; "
                f"{item['fixed']['error'][i]:.6f}"
                for i in (0, 2)
            ]
            lines.append(f"| {g} | {n} | {cells[0]} | {cells[1]} |")
        a, b = result["ladder"][str(chosen)], result["ladder"]["0"]
        delta = np.array(a["value"]) - np.array(b["value"])
        deleted = np.array(a["refit"]["deleted"]) - np.array(b["refit"]["deleted"])
        d = deleted - deleted.mean(axis=0)
        cov = (len(d) - 1) / len(d) * d.T @ d
        diagnostics[str(g)] = {}
        for n in ns:
            item = result["ladder"][str(n)]
            jk = np.array(item["refit"]["deleted"])[:, (0, 2)]
            terms = (jk - jk.mean(axis=0)) ** 2
            diagnostics[str(g)][str(n)] = dict(
                max_variance_fraction=(terms.max(axis=0) / terms.sum(axis=0)).tolist(),
                worst_labels=[item["refit"]["labels"][k] for k in terms.argmax(axis=0)],
            )
        shifts[str(g)] = dict(
            delta=delta.tolist(),
            error=np.sqrt(np.diag(cov)).tolist(),
            sigma_fraction=(delta / a["refit"]["error"]).tolist(),
        )
    lines += [
        "",
        "## Occupancy of no-external-shear training samples",
        "",
        "| Sample | Objects | Occupied / 400 | Min / median / max occupied count | Objects below chosen N (%) | Weight below chosen N (%) |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for sample, name in zip(samples, names, strict=True):
        q = sample["quantiles"]
        t = sample["thresholds"][str(chosen)]
        lines.append(
            f"| {name} | {sample['total']:,} | {sample['occupied']} | "
            f"{q[0]:g} / {q[2]:g} / {q[4]:g} | "
            f"{100 * t['object_fraction']:.6g} | {100 * t['weight_fraction']:.6g} |"
        )
    lines += [
        "",
        "## Occupancy shares across the ladder",
        "",
        "Entries are object percentage / unregularized DES weight percentage in cells below N.",
        "",
        "| N | Grid 1 | Grid 2 | Grid 3 | Grid 4 | Public release |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for n in ns:
        parts = []
        for sample in samples:
            t = sample["thresholds"][str(n)]
            parts.append(
                f"{100 * t['object_fraction']:.6g} / {100 * t['weight_fraction']:.6g}"
            )
        lines.append(f"| {n} | " + " | ".join(parts) + " |")
    lines += [
        "",
        "## Sampling noise of W",
        "",
        "Median SD(W)/mean(W), in percent, across parent cells with at least 100 objects; 300 within-cell bootstrap draws per count.",
        "",
        "| Resampled count | Grid 1 | Grid 2 | Grid 3 | Grid 4 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for n in (2, 3, 10, 30, 100, 300, 1000):
        parts = [f"{100 * d['sampling_noise'][str(n)]['median']:.2f}" for d in grids]
        lines.append(f"| {n} | " + " | ".join(parts) + " |")
    lines += [
        "",
        "## Public-release affected fractions",
        "",
        "| N_min | Cells below N | Objects | Object fraction (%) | Reconstructed DES weight (%) | Released weight (%) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for n in ns:
        t = public["reconstructed"]["thresholds"][str(n)]
        old = public["released"]["thresholds"][str(n)]
        lines.append(
            f"| {n} | {t['cells']} | {t['objects']} | "
            f"{100 * t['object_fraction']:.6f} | {100 * t['weight_fraction']:.6f} | "
            f"{100 * old['weight_fraction']:.6f} |"
        )
    (dest / "sparse_ladder.md").write_text("\n".join(lines) + "\n")
    (dest / "sparse_shifts.json").write_text(json.dumps(shifts, indent=2))
    (dest / "sparse_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    for g, result in enumerate(grids, 1):
        (dest / f"sparse_grid_{g}.json").write_text(json.dumps(result, indent=2))
    (dest / "sparse_public.json").write_text(json.dumps(public, indent=2))
    print("\n".join(lines))
    print("SHIFTS", json.dumps(shifts))
    print("VARIANCE_FRACTIONS", json.dumps(diagnostics))
    values = np.array([x["ladder"][str(chosen)]["value"] for x in grids])
    errors = np.array([x["ladder"][str(chosen)]["refit"]["error"] for x in grids])
    iv = errors**-2
    print(
        "GRID_AVERAGE",
        np.sum(iv * values, axis=0) / iv.sum(axis=0),
        iv.sum(axis=0) ** -0.5,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("--chosen", required=True, type=int)
    a = p.parse_args()
    summarize(a.root, a.dest, a.chosen)
