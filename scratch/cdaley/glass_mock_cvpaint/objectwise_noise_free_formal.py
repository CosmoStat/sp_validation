"""Compare independent-object and spatial errors for the noise-free shear fit."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/n17data/cdaley/scratch/glass-mock-cv/cvpaint")
a = np.load(ROOT / "psf_pixel_operator.npz")
pix, design, xx = a["pixels"], a["design"], a["xx"]
MAPS = ROOT.parent / "mocks/v1.4.6.3/seed_00001/maps"
g1 = np.load(MAPS / "gamma1.npy", mmap_mode="r")
g2 = np.load(MAPS / "gamma2.npy", mmap_mode="r")
y = np.column_stack((g1[pix], -g2[pix])).astype(float)
rhs = design.T @ y
fit = np.linalg.solve(xx, rhs)
n = json.loads((ROOT / "geometry_audit.json").read_text())["GLASS1_clean"]["rows"]
rss = float(np.sum(design[:, 2, None] * y * y) - np.sum(fit * rhs))
scale = rss / (2 * n - 6)
vcov = scale * np.linalg.inv(xx)
errors = [
    float(np.sqrt(vcov[0, 0])),
    float(np.sqrt(vcov[1, 1])),
    float(0.5 * np.sqrt(vcov[0, 0] + vcov[1, 1])),
]
point = [float(fit[0, 0]), float(fit[1, 1]), float(0.5 * (fit[0, 0] + fit[1, 1]))]
audit = json.loads((ROOT / "objectwise_audit.json").read_text())
formal = dict(
    model="Same six-parameter weighted mixed-affine model and pooled residual variance as lmfit independent-object Hessian.",
    point=point,
    errors=errors,
    weighted_rss=rss,
    dof=2 * n - 6,
    comparison={
        key: dict(
            independent_object_error=error,
            jk8_error=audit["jackknife"]["8"]["errors"]["noise_free_shear"][key],
            jk4_error=audit["jackknife"]["4"]["errors"]["noise_free_shear"][key],
        )
        for key, error in zip(("a11", "a22", "aii_mean"), errors)
    },
)
(ROOT / "objectwise_noise_free_formal.json").write_text(json.dumps(formal, indent=2))
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for j, (ax, key) in enumerate(zip(axes, ("a11", "a22", "aii_mean"))):
    es = [
        errors[j],
        formal["comparison"][key]["jk8_error"],
        formal["comparison"][key]["jk4_error"],
    ]
    for k, error in enumerate(es):
        ax.errorbar(k, point[j], yerr=error, fmt="o", capsize=3)
    ax.axhline(0, color="k", lw=0.7)
    ax.set(
        title=key,
        xticks=[0, 1, 2],
        xticklabels=["Per object", "JK8", "JK4"],
        ylabel="Noise-free shear slope (1σ bars)",
    )
fig.suptitle(
    "Noise-free shear fit: independent galaxies are not independent cosmic-shear modes"
)
fig.tight_layout()
fig.savefig(ROOT / "figures/objectwise_noise_free_formal.png", dpi=170)
plt.close(fig)
print("Saved noise-free formal/spatial error comparison and figure.", flush=True)
