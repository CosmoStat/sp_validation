"""Project ten independent mock skies onto the fixed object-wise PSF design.

Aggregating the weighted PSF design per generation pixel is exactly the same
linear regression as sampling each gamma map at every galaxy. Seed 1 is checked
against the independently accumulated stored g_true regression.
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
BASE = ROOT.parent / "mocks/v1.4.6.3"
operator = ROOT / "psf_pixel_operator.npz"
if operator.exists():
    a = np.load(operator)
    pixels, design, xx = a["pixels"], a["design"], a["xx"]
else:
    design = np.zeros((hp.nside2npix(4096), 3))
    xx = np.zeros((3, 3))
    with h5py.File(BASE / "seed_00001/painted.h5", "r") as handle:
        dataset = handle["data"]
        for start in range(0, len(dataset), 250000):
            d = dataset.fields(["RA", "Dec", "w_des", "e1_PSF", "e2_PSF"])[
                start : start + 250000
            ]
            pix = hp.ang2pix(4096, np.mod(d["RA"], 360), d["Dec"], lonlat=True)
            w = np.asarray(d["w_des"], float)
            x = np.array([d["e1_PSF"], d["e2_PSF"], np.ones(len(d))], float).T
            xx += x.T @ (w[:, None] * x)
            np.add.at(design, pix, w[:, None] * x)
            if start % 10000000 == 0:
                print("Operator rows:", start + len(d), flush=True)
    pixels = np.flatnonzero(design[:, 2])
    design = design[pixels]
    np.savez(operator, pixels=pixels, design=design, xx=xx)

estimates = []
reference = None
for seed in range(1, 11):
    maps = BASE / f"seed_{seed:05d}/maps"
    metadata = json.loads((maps / "maps.json").read_text())
    config = dict(metadata["config"])
    config.pop("seed")
    if reference is None:
        reference = config
    assert config == reference, "Ensemble sky configurations differ"
    g1 = np.load(maps / "gamma1.npy", mmap_mode="r")
    g2 = np.load(maps / "gamma2.npy", mmap_mode="r")
    rhs = np.column_stack(
        (
            design.T @ np.asarray(g1[pixels], float),
            -design.T @ np.asarray(g2[pixels], float),
        )
    )
    fit = np.linalg.solve(xx, rhs)
    estimates.append([fit[0, 0], fit[1, 1], 0.5 * (fit[0, 0] + fit[1, 1])])
    print("Sky projected:", seed, flush=True)
    del g1, g2
estimates = np.asarray(estimates)
audit = json.loads((ROOT / "objectwise_audit.json").read_text())
point = np.array(
    [
        audit["point_estimates"]["noise_free_shear"][k]
        for k in ("a11", "a22", "aii_mean")
    ]
)
assert np.max(np.abs(estimates[0] - point)) < 1e-8, (
    "Pixel operator does not reproduce stored g_true regression"
)
result = dict(
    columns=["a11", "a22", "aii_mean"],
    noise_free_estimates=estimates.tolist(),
    mean=estimates.mean(axis=0).tolist(),
    std=estimates.std(axis=0, ddof=1).tolist(),
    seed1_absolute_pull_using_ensemble_scatter=(
        estimates[0] / estimates.std(axis=0, ddof=1)
    ).tolist(),
    seed1_operator_max_difference=float(np.max(np.abs(estimates[0] - point))),
    caveat="Ten skies give an imprecise variance estimate; noise-free cosmic-shear scatter only, not an ensemble of full noisy object-wise fits.",
)
(ROOT / "objectwise_sky_ensemble.json").write_text(json.dumps(result, indent=2))
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for j, (ax, key) in enumerate(zip(axes, result["columns"])):
    ax.plot(np.arange(1, 11), estimates[:, j], "o")
    ax.axhline(0, color="k", lw=0.7)
    ax.axhspan(
        -result["std"][j],
        result["std"][j],
        color="0.8",
        alpha=0.5,
        label="Ten-sky standard deviation",
    )
    ax.set(
        xlabel="Independent GLASS sky seed",
        ylabel="Noise-free shear PSF-regression slope",
        title=key,
    )
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures/objectwise_sky_ensemble.png", dpi=170)
plt.close(fig)
print("Saved ten-sky object-wise signal regression and figure.", flush=True)
