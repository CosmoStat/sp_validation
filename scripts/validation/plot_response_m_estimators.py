#!/usr/bin/env python3
"""Plot residual m by grid, including the production v2 data table."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "docs/validation/response"


def load(name):
    return json.loads((ROOT / name).read_text())


paired = load("production_comparison.json")
iv = load("iv_results.json")
v2 = load("v2_data_table.json")["grids"]
vals = {
    k: [[], []]
    for k in (
        "Paired (Martin)",
        "Self-fitted, refit-W",
        "Other three grids",
        "IV exact",
        "(d) v2 data table",
    )
}
errs = {k: [[], []] for k in vals}
for i in range(1, 5):
    sparse_row = load(f"sparse_grid_{i}.json")["ladder"]["3"]
    sparse = sparse_row["refit"]
    indep = load(f"independent_grid_{i}.json")["b"]
    prod = paired[i - 1]["previously_validated_m"]
    ivdata = iv[str(i)]["iv_exact"]
    data_table = v2[str(i)]
    sources = [
        (
            "Paired (Martin)",
            [prod["m1"]["martin_paired"], prod["m2"]["martin_paired"]],
            [0, 0],
        ),
        (
            "Self-fitted, refit-W",
            [sparse_row["value"][0], sparse_row["value"][2]],
            [sparse["error"][0], sparse["error"][2]],
        ),
        (
            "Other three grids",
            [indep["value"][0], indep["value"][2]],
            [indep["error"][0], indep["error"][2]],
        ),
        (
            "IV exact",
            [ivdata["value"][0], ivdata["value"][2]],
            [ivdata["error"][0], ivdata["error"][2]],
        ),
    ]
    sources.append(
        (
            "(d) v2 data table",
            [data_table["value"][0], data_table["value"][2]],
            [data_table["error"][0], data_table["error"][2]],
        )
    )
    for name, v, e in sources:
        for c in range(2):
            vals[name][c].append(v[c])
            errs[name][c].append(e[c])

colors = ["#555555", "#d55e00", "#0072b2", "#009e73", "#cc79a7"]
x = np.arange(1, 5)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
for c, ax in enumerate(axes):
    for j, name in enumerate(vals):
        y = np.array(vals[name][c]) * 1e3
        e = np.array(errs[name][c]) * 1e3
        ax.errorbar(
            x + (j - 2) * 0.13,
            y,
            yerr=e,
            fmt="o",
            capsize=3,
            color=colors[j],
            label=name,
        )
        w = np.where(e > 0, 1 / e**2, 0)
        mean = np.sum(w * y) / np.sum(w) if np.sum(w) else np.mean(y)
        sem = np.sqrt(1 / np.sum(w)) if np.sum(w) else 0
        ax.axhspan(mean - sem, mean + sem, color=colors[j], alpha=0.10, lw=0)
        ax.plot([0.65, 4.35], [mean, mean], color=colors[j], lw=1, alpha=0.75)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xticks=x, xlabel="Grid", ylabel=rf"$m_{c + 1}$ ($10^{{-3}}$)")
    ax.grid(axis="y", alpha=0.2)
axes[0].set_title("m1")
axes[1].set_title("m2")
axes[1].legend(frameon=False, fontsize=8, loc="best")
fig.suptitle("Residual multiplicative bias by grid")
fig.tight_layout()
fig.savefig(ROOT / "m_estimators.png", dpi=180, bbox_inches="tight")
