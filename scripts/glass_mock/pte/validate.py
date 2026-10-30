#!/usr/bin/env python3
"""Validate painted-mock SACC vectors against their pixelized GLASS skies."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from astropy.io import fits
from scipy import linalg
from scipy.special import jv
from scipy.stats import chi2, kstest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sp_validation import sacc_io

root = Path("/n17data/cdaley/scratch/glass-mock-cv/pte")
parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=10)
a = parser.parse_args()
figdir = root / "figures"
figdir.mkdir(exist_ok=True)
base = "covariance_GLASSpte_s01_ng_minsep=1.0_maxsep=250.0_nbins=20_masked"
cxi = np.loadtxt(root / "inference/data/covariance" / base / (base + ".dat"))
if cxi.shape != (40, 40):
    raise ValueError(f"Unexpected OneCovariance matrix shape: {cxi.shape}")
with fits.open(root / "output/pseudo_cl_cov_GLASSpte_s01_powspace_nbins=32.fits") as hd:
    cov = {
        x: np.asarray(hd["COVAR_" + x + "_" + x].data, float)
        for x in ["EE", "EB", "BB"]
    }


def pte(x, c):
    x = np.asarray(x, float)
    vals = linalg.eigvalsh(c)
    if vals[0] <= vals[-1] * 1e-12:
        raise ValueError("Covariance is not positive definite at the declared rank")
    q = float(x @ linalg.solve(c, x, assume_a="pos"))
    return {"chi2": q, "dof": len(x), "pte": float(chi2.sf(q, len(x)))}


# Finite-band flat-sky spin-2 transform: no uncontrolled extrapolation of
# the noisy realization spectrum beyond the GLASS band limit.
edges = np.geomspace(1.0, 250.0, 21)
# Uniform pair density within each annulus, rather than unknown fine-grid counts.
q = np.linspace(0, 1, 33)
theta = np.sqrt(edges[:-1, None] ** 2 * (1 - q) + edges[1:, None] ** 2 * q)
fine_path = root / "output/GLASSpte_s01_pairs_minsep=0.08_maxsep=300.0_nbins=1000.npz"
with np.load(fine_path) as fine:
    fine_theta = fine["theta"].copy()
    fine_weight = fine["weight"].copy()
    fine_edges = fine["edges"].copy()
lo = np.maximum(edges[:-1, None], fine_edges[None, :-1])
hi = np.minimum(edges[1:, None], fine_edges[None, 1:])
fraction = np.maximum(hi**2 - lo**2, 0) / np.diff(fine_edges**2)[None, :]
weights = fraction * fine_weight[None, :]
theta_all = np.concatenate((theta.ravel(), fine_theta))
ls_transform = np.arange(2, 8193)
arg = np.outer(theta_all * np.pi / (180 * 60), ls_transform + 0.5)
kernels = [jv(order, arg) * ((ls_transform + 0.5) / (2 * np.pi)) for order in (0, 4)]
del arg
results = {}
residuals = []
raw_residuals = []
xi_standardized = []
xi_uniform = []
ell0 = None
for seed in range(1, a.seeds + 1):
    name = f"GLASSpte_s{seed:02d}"
    part = sacc_io.load(
        root / f"output/pseudo_cl_{name}_powspace_nbins=32.sacc", blind="none"
    )
    ell, ee, bb, eb, window = sacc_io.get_pseudo_cl(part, (0, 0))
    xi = sacc_io.load(
        root / f"output/{name}_xi_minsep=1.0_maxsep=250.0_nbins=20_npatch=1.sacc",
        blind="none",
    )
    _, xp, xm = sacc_io.get_xi(xi, (0, 0), grid="reporting")
    path = root.parent / f"mocks/v1.4.6.3/seed_{seed:05d}/maps/spectra.npz"
    with np.load(path) as sky:
        ls = np.asarray(sky["ell"])
        real = np.asarray(sky["realization_ee"])
        ensemble = np.asarray(sky["cl_ee_pixel"])

    def convolve(c):
        return np.asarray(window.weight).T @ np.interp(
            window.values, ls, c, left=0, right=0
        )

    own = convolve(real)
    target = convolve(ensemble)
    if ell0 is None:
        ell0 = ell
    elif not np.array_equal(ell0, ell):
        raise ValueError("Inconsistent ell grid")
    residuals.append((ee - own) / np.sqrt(np.diag(cov["EE"])))
    raw_residuals.append(ee - own)
    if not np.array_equal(ls[2:], ls_transform):
        raise ValueError("Unexpected GLASS ell grid")
    theory = []
    theory_uniform = []
    for kernel in kernels:
        vals = kernel @ real[2:]
        uniform = vals[: theta.size].reshape(theta.shape)
        fv = vals[theta.size :]
        theory_uniform.append(np.trapezoid(uniform, q, axis=1))
        theory.append((weights @ fv) / np.sum(weights, axis=1))
    diff = np.concatenate([xp - theory[0], xm - theory[1]])
    udiff = np.concatenate([xp - theory_uniform[0], xm - theory_uniform[1]])
    xi_standardized.append(diff / np.sqrt(np.diag(cxi)))
    xi_uniform.append(udiff / np.sqrt(np.diag(cxi)))
    row = {
        "BB": pte(bb, cov["BB"]),
        "EB": pte(eb, cov["EB"]),
        "EE_realization": pte(ee - own, cov["EE"]),
        "EE_ensemble": pte(ee - target, cov["EE"]),
        "xi_plus": pte(diff[:20], cxi[:20, :20]),
        "xi_minus": pte(diff[20:], cxi[20:, 20:]),
        "xi_combined": pte(diff, cxi),
        "xi_plus_uniform": pte(udiff[:20], cxi[:20, :20]),
        "xi_minus_uniform": pte(udiff[20:], cxi[20:, 20:]),
    }
    results[name] = row
    print(name, json.dumps(row), flush=True)
keys = list(next(iter(results.values())))
summary = {}
for k in keys:
    ks = kstest([r[k]["pte"] for r in results.values()], "uniform")
    summary[k] = {"ks_p": float(ks.pvalue), "ks_d": float(ks.statistic)}
mean_residual = np.mean(raw_residuals, axis=0)
mean_test = pte(mean_residual, cov["EE"] / a.seeds)
noise = []
for seed in (1, 2):
    with h5py.File(
        root.parent / f"mocks/v1.4.6.3/seed_{seed:05d}/painted.h5", "r"
    ) as f:
        d = f["data"].fields(["e1_noise", "e2_noise"])[:1_000_000]
        noise.append(np.column_stack([d["e1_noise"], d["e2_noise"]]))
noise_corr = np.corrcoef(noise[0].T, noise[1].T)[:2, 2:].tolist()
(root / "metrics.json").write_text(
    json.dumps(
        {
            "per_seed": results,
            "uniform_ks": summary,
            "xi_theory": "Finite ell 2..8192 flat-sky spin-2 Bessel sum of already-pixelized realization_ee; measured 1000-bin pair-weight average at bin centers with fractional annular overlap",
            "ee_mean_residual_chi2": mean_test,
            "noise_component_correlation_seed1_seed2_1M": noise_corr,
            "theta_bin_mean_standardized_residual_squared": np.mean(
                np.square(xi_standardized), axis=0
            ).tolist(),
            "theta_bin_mean_standardized_residual_squared_uniform": np.mean(
                np.square(xi_uniform), axis=0
            ).tolist(),
            "theta_bin_mean_standardized_residual": np.mean(
                xi_standardized, axis=0
            ).tolist(),
            "covariance": "One shared seed-1 iNKA covariance and one shared OneCovariance matrix; includes cosmic variance; not a conditional covariance on the observed full-sky realization",
        },
        indent=2,
    )
)
fig, axes = plt.subplots(len(keys), 2, figsize=(10, 2.5 * len(keys)))
for row, k in zip(axes, keys):
    p = np.sort([v[k]["pte"] for v in results.values()])
    n = len(p)
    row[0].hist(p, bins=np.linspace(0, 1, 6), edgecolor="k")
    row[0].axhline(n / 5, color="k", ls="--")
    row[0].set(
        title=f"{k}: KS p={summary[k]['ks_p']:.3g}", xlabel="PTE", ylabel="Count"
    )
    row[1].step(np.r_[0, p, 1], np.r_[0, np.arange(1, n + 1) / n, 1], where="post")
    row[1].plot([0, 1], [0, 1], "k--")
    row[1].set(xlabel="PTE", ylabel="Empirical CDF", xlim=(0, 1), ylim=(0, 1))
fig.tight_layout()
fig.savefig(figdir / "pte_uniformity.png", dpi=150)
plt.close(fig)
r = np.stack(residuals)
fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(
    ell0, r.mean(axis=0), yerr=r.std(axis=0, ddof=1) / np.sqrt(len(r)), fmt="o", ms=3
)
ax.axhline(0, color="k")
ax.set(
    xscale="log",
    xlabel="Multipole ell",
    ylabel="Mean (EE − realized truth) / marginal sigma",
    title=f"{a.seeds} GLASS skies",
)
fig.tight_layout()
fig.savefig(figdir / "ee_residual_mean.png", dpi=150)
plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for i, ax in enumerate(axes):
    vals = np.asarray(xi_standardized)[:, i * 20 : (i + 1) * 20]
    ax.plot(
        np.sqrt(edges[:-1] * edges[1:]),
        np.mean(vals**2, axis=0),
        "o-",
        label="Measured fine pair weights",
    )
    uv = np.asarray(xi_uniform)[:, i * 20 : (i + 1) * 20]
    ax.plot(
        np.sqrt(edges[:-1] * edges[1:]),
        np.mean(uv**2, axis=0),
        "--",
        label="Uniform annular weighting",
    )
    ax.axhline(1, color="k", lw=1)
    ax.set(
        xscale="log",
        xlabel="theta [arcmin]",
        title=r"$\xi_+$" if i == 0 else r"$\xi_-$",
    )
axes[0].set_ylabel("Mean squared marginal standardized residual")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(figdir / "xi_bin_residuals.png", dpi=150)
plt.close(fig)
print("Inspect all figures and investigate extreme PTEs before reporting.")
