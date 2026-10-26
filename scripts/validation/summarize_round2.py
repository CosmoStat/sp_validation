"""Summarize round-2 measurements; run inside the allocated container."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run(source, dest):
    grids = [
        json.loads((source / f"weights_grid_{i}.json").read_text()) for i in range(1, 5)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for c, ax in enumerate(axes):
        for offset, policy, label in (
            (-0.16, "des_refit", "DES (refit W)"),
            (0, "iv_exact", "IV (branch errors)"),
            (0.16, "iv_frozen", "IV (frozen; diagnostic)"),
        ):
            ax.errorbar(
                np.arange(1, 5) + offset,
                [100 * g[policy]["value"][2 * c] for g in grids],
                yerr=[100 * g[policy]["error"][2 * c] for g in grids],
                fmt="o",
                capsize=2,
                label=label,
            )
        ax.axhline(0, color="k", lw=0.5)
        ax.set(
            xlabel="Simulation grid",
            ylabel=f"Residual m{c + 1} [%]",
            xticks=range(1, 5),
        )
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(dest / "iv_des_m.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for c, ax in enumerate(axes):
        for policy, label in (("des_fixed", "Fixed W"), ("des_refit", "Refit W")):
            ax.plot(
                range(1, 5),
                [100 * g[policy]["error"][2 * c] for g in grids],
                "o-",
                label=label,
            )
        ax.set(
            xlabel="Simulation grid",
            ylabel=f"Jackknife error on m{c + 1} [percentage points]",
            xticks=range(1, 5),
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "refit_w_errors.png", dpi=180)
    plt.close(fig)
    iv = {
        str(i): {k: v for k, v in g.items() if k.startswith("iv_")}
        for i, g in enumerate(grids, 1)
    }
    des = {
        str(i): {k: v for k, v in g.items() if k.startswith("des_")}
        for i, g in enumerate(grids, 1)
    }
    (dest / "iv_results.json").write_text(json.dumps(iv, indent=2))
    (dest / "refit_w_results.json").write_text(json.dumps(des, indent=2))
    if not (source / "pairs_public.json").exists():
        return
    samples = ["public", "1", "2", "3", "4"]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.4, 1]},
    )
    reports = {}
    for name in samples:
        d = json.loads((source / f"pairs_{name}.json").read_text())
        reports[name] = d
        axes[0 if name == "public" else 1].errorbar(
            d["theta"],
            np.array(d["fraction"]) * 100,
            yerr=np.array(d["error"]) * 100,
            fmt="o-",
            ms=3,
            capsize=2,
            label="Public v1.4.6.3" if name == "public" else f"Grid {name}",
        )
        # Compact NPZ preserves all covariance and delete-one samples.
        data = np.load(source / f"pairs_{name}.npz")
        np.savez(dest / f"stage1_{name}.npz", **data)
    d = reports["public"]
    xi = np.loadtxt(d["xi_source"], skiprows=3, max_rows=20)
    for key, col, errcol, marker, label in (
        ("xip_shift_over_sigma", 3, 7, "o", r"$\Delta\xi_+/\sigma_+$"),
        ("xim_shift_over_sigma", 4, 8, "s", r"$\Delta\xi_-/\sigma_-$"),
    ):
        error = (
            np.array(d["error"])
            / (1 + np.array(d["fraction"])) ** 2
            * abs(xi[:, col] / xi[:, errcol])
        )
        d[key + "_error"] = error.tolist()
        axes[2].errorbar(
            d["theta"],
            d[key],
            yerr=error,
            fmt=marker + "-",
            capsize=2,
            ms=3,
            label=label,
        )
    for ax in axes[:2]:
        ax.set(ylabel=r"$\delta R_{2pt}/R_{global}^2$ [%]")
        ax.legend(fontsize=8, ncol=2)
    axes[2].set(
        xlabel="Separation [arcmin]",
        ylabel=r"Implied shift / $\sigma_\xi$",
        xscale="log",
    )
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    fig.savefig(dest / "stage1_response_jackknife.png", dpi=180)
    plt.close(fig)
    (dest / "stage1_results.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    run(a.source, a.dest)
