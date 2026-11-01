"""Separate shot noise from correlated shear in the object-wise PSF fit.

The weighted normal equations reproduce the pipeline's affine, mixed-component
regression. Fixed HEALPix cells define a spatial delete-one-group jackknife;
two cell scales expose sensitivity to patch size and incomplete footprint cells.
"""

import json
from pathlib import Path

import h5py
import healpy as hp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/n17data/cdaley/scratch/glass-mock-cv/cvpaint")
MOCK = ROOT.parent / "mocks/v1.4.6.3/seed_00001/painted.h5"
columns = [
    "RA",
    "Dec",
    "w_des",
    "e1_PSF",
    "e2_PSF",
    "e1",
    "e2",
    "e1_noise",
    "e2_noise",
    "g1_true",
    "g2_true",
]
# x=(PSF1,PSF2,1), y=(e1,e2); datasets full/noise/signal.
xx = np.zeros((768, 3, 3))
xy = np.zeros((768, 3, 3, 2))
count = np.zeros(768, np.int64)
with h5py.File(MOCK, "r") as handle:
    dataset = handle["data"]
    for start in range(0, len(dataset), 250000):
        d = dataset.fields(columns)[start : start + 250000]
        pix = hp.ang2pix(8, d["RA"], d["Dec"], lonlat=True, nest=True)
        w = np.asarray(d["w_des"], float)
        x = np.array([d["e1_PSF"], d["e2_PSF"], np.ones(len(d))], float).T
        y = np.stack(
            [
                np.array([d[a], d[b]], float).T
                for a, b in [
                    ("e1", "e2"),
                    ("e1_noise", "e2_noise"),
                    ("g1_true", "g2_true"),
                ]
            ],
            axis=1,
        )
        count += np.bincount(pix, minlength=768)
        for i in range(3):
            for j in range(i, 3):
                contribution = np.bincount(
                    pix, weights=w * x[:, i] * x[:, j], minlength=768
                )
                xx[:, i, j] += contribution
                if i != j:
                    xx[:, j, i] += contribution
            for kind in range(3):
                for component in range(2):
                    xy[:, kind, i, component] += np.bincount(
                        pix, weights=w * x[:, i] * y[:, kind, component], minlength=768
                    )
        if start % 5000000 == 0:
            print("Rows accumulated:", start + len(d), flush=True)
np.savez(ROOT / "objectwise_moments.npz", xx=xx, xy=xy, count=count)


def slopes(xx, xy):
    fits = np.stack([np.linalg.solve(xx, kind) for kind in xy])
    return np.stack(
        [fits[:, 0, 0], fits[:, 1, 1], 0.5 * (fits[:, 0, 0] + fits[:, 1, 1])], axis=1
    )


def jackknife(nside):
    factor = (8 // nside) ** 2
    group_xx = xx.reshape(-1, factor, 3, 3).sum(axis=1)
    group_xy = xy.reshape(-1, factor, 3, 3, 2).sum(axis=1)
    group_count = count.reshape(-1, factor).sum(axis=1)
    keep = group_count > 0
    group_xx, group_xy, group_count = group_xx[keep], group_xy[keep], group_count[keep]
    total_xx, total_xy = group_xx.sum(axis=0), group_xy.sum(axis=0)
    best = slopes(total_xx, total_xy)
    deleted = np.stack(
        [slopes(total_xx - a, total_xy - b) for a, b in zip(group_xx, group_xy)]
    )
    centered = deleted - deleted.mean(axis=0)
    cov = (
        (len(deleted) - 1)
        / len(deleted)
        * np.einsum("gki,gkj->kij", centered, centered)
    )
    return best, cov, group_count, deleted


best, cov, group_count, deleted = jackknife(8)
result = {
    "model": "e1=a11*PSF1+a12*PSF2+c1; e2=a21*PSF1+a22*PSF2+c2; inverse-variance weights w_des",
    "components": ["a11", "a22", "aii_mean"],
    "datasets": ["full", "rotated_noise", "noise_free_shear"],
    "point_estimates": {
        name: dict(zip(("a11", "a22", "aii_mean"), map(float, values)))
        for name, values in zip(("full", "rotated_noise", "noise_free_shear"), best)
    },
    "jackknife": {},
    "caveat": "Fixed HEALPix groups are spatial but have unequal occupied area and weights. Delete-one-group jackknife is a sensitivity estimate, not an ensemble-calibrated covariance. Compare nside4 and8.",
}
for nside in (4, 8):
    point, cc, nn, dd = jackknife(nside)
    result["jackknife"][str(nside)] = {
        "npatch": len(nn),
        "pixel_area_deg2": float(hp.nside2pixarea(nside, degrees=True)),
        "min_rows": int(nn.min()),
        "max_rows": int(nn.max()),
        "covariances": cc.tolist(),
        "errors": {
            name: dict(zip(("a11", "a22", "aii_mean"), map(float, np.sqrt(np.diag(c)))))
            for name, c in zip(("full", "rotated_noise", "noise_free_shear"), cc)
        },
    }
    np.savez(
        ROOT / f"objectwise_jk_nside{nside}.npz",
        point=point,
        covariance=cc,
        delete_one=dd,
        rows=nn,
    )
pipeline = json.loads((ROOT / "objectwise_early.json").read_text())
reported = pipeline["GLASS1_clean"]["tomo_bin_all"]
result["pipeline_point_max_difference"] = float(
    max(
        abs(best[0, j] - reported[key][0])
        for j, key in enumerate(("a11", "a22", "aii_mean"))
    )
)
result["pipeline_per_object_errors"] = {
    key: reported[key][1] for key in ("a11", "a22", "aii_mean")
}
result["composition_residual_slopes"] = dict(
    zip(("a11", "a22", "aii_mean"), map(float, best[0] - best[1] - best[2]))
)
(ROOT / "objectwise_audit.json").write_text(json.dumps(result, indent=2))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for j, (ax, key) in enumerate(zip(axes, ("a11", "a22", "aii_mean"))):
    labels = [
        "Full mock: per-object error",
        "Full mock: spatial JK nside8",
        "Full mock: spatial JK nside4",
        "Noise-free shear: JK nside8",
        "Rotated noise: JK nside8",
    ]
    values = [best[0, j], best[0, j], best[0, j], best[2, j], best[1, j]]
    errors = [
        reported[key][1],
        result["jackknife"]["8"]["errors"]["full"][key],
        result["jackknife"]["4"]["errors"]["full"][key],
        result["jackknife"]["8"]["errors"]["noise_free_shear"][key],
        result["jackknife"]["8"]["errors"]["rotated_noise"][key],
    ]
    for k, (value, error) in enumerate(zip(values, errors)):
        ax.errorbar(k, value, yerr=error, fmt="o", capsize=3, label=labels[k])
    ax.axhline(0, color="k", lw=0.7)
    ax.set(
        title=key,
        ylabel="PSF leakage slope",
        xticks=np.arange(5),
        xticklabels=[
            "Full\nobject",
            "Full\nJK8",
            "Full\nJK4",
            "Signal\nJK8",
            "Noise\nJK8",
        ],
    )
axes[0].legend(fontsize=7)
fig.suptitle(
    "Correlated shear versus independent-object regression uncertainty; identical PSF/weights"
)
fig.tight_layout()
fig.savefig(ROOT / "figures/objectwise_spatial_audit.png", dpi=170)
plt.close(fig)
print(
    "Saved objectwise_audit.json and figures/objectwise_spatial_audit.png; inspect before interpreting.",
    flush=True,
)
