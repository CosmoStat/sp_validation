"""Check regression jackknife sensitivity to large sky cells using cached moments."""

import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/n17data/cdaley/scratch/glass-mock-cv/cvpaint")
a = np.load(ROOT / "objectwise_moments.npz")
xx, xy, count = a["xx"], a["xy"], a["count"]
results = {}


def slopes(a, b):
    fits = np.stack([np.linalg.solve(a, k) for k in b])
    return np.stack(
        [fits[:, 0, 0], fits[:, 1, 1], 0.5 * (fits[:, 0, 0] + fits[:, 1, 1])], axis=1
    )


point = slopes(xx.sum(axis=0), xy.sum(axis=0))
for nside in (2, 4, 8):
    factor = (8 // nside) ** 2
    gx = xx.reshape(-1, factor, 3, 3).sum(axis=1)
    gy = xy.reshape(-1, factor, 3, 3, 2).sum(axis=1)
    n = count.reshape(-1, factor).sum(axis=1)
    keep = n > 0
    gx, gy, n = gx[keep], gy[keep], n[keep]
    deleted = np.stack(
        [slopes(gx.sum(axis=0) - x, gy.sum(axis=0) - y) for x, y in zip(gx, gy)]
    )
    centered = deleted - deleted.mean(axis=0)
    covariance = (len(n) - 1) / len(n) * np.einsum("gki,gkj->kij", centered, centered)
    error = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
    results[str(nside)] = dict(
        npatch=len(n),
        errors=error.tolist(),
        pulls=(point / error).tolist(),
        two_sided_gaussian_p=[
            [math.erfc(abs(v) / math.sqrt(2)) for v in row] for row in point / error
        ],
    )
(ROOT / "objectwise_patch_scales.json").write_text(json.dumps(results, indent=2))
reported = json.loads((ROOT / "objectwise_early.json").read_text())["GLASS1_clean"][
    "tomo_bin_all"
]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for j, (ax, key) in enumerate(zip(axes, ("a11", "a22", "aii_mean"))):
    values = [point[0, j]] * 4 + [point[2, j], point[1, j]]
    errors = (
        [reported[key][1]]
        + [results[str(n)]["errors"][0][j] for n in (8, 4, 2)]
        + [results["8"]["errors"][2][j], results["8"]["errors"][1][j]]
    )
    for k, (v, e) in enumerate(zip(values, errors)):
        ax.errorbar(k, v, yerr=e, fmt="o", capsize=3)
    ax.axhline(0, color="k", lw=0.7)
    ax.set(
        title=key,
        ylabel="PSF leakage slope (1σ bars)",
        xticks=np.arange(6),
        xticklabels=[
            "Full\nobject",
            "Full\nJK8",
            "Full\nJK4",
            "Full\nJK2",
            "Signal\nJK8",
            "Noise\nJK8",
        ],
    )
fig.suptitle(
    "Object-wise leakage: error sensitivity to spatial cell scale (JK2 has few, unequal groups)"
)
fig.tight_layout()
fig.savefig(ROOT / "figures/objectwise_patch_scales.png", dpi=170)
plt.close(fig)
print("Saved objectwise_patch_scales.json and corresponding figure.", flush=True)
