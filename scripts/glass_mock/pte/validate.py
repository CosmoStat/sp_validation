#!/usr/bin/env python3
"""Validate painted-mock SACC vectors against their pixelized GLASS skies."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pyccl as ccl
from astropy.io import fits
from scipy import linalg
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


# CCL's correlation routine is a linear transform of the supplied pixelized
# spectrum; cosmology is not involved in this transform.
cosmo = ccl.Cosmology(
    Omega_c=0.30966 - 0.04897, Omega_b=0.04897, h=0.6766, sigma8=0.8102, n_s=0.9665
)
edges = np.geomspace(1.0, 250.0, 21)
# Uniform pair density within each annulus, rather than unknown fine-grid counts.
q = np.linspace(0, 1, 33)
theta = np.sqrt(edges[:-1, None] ** 2 * (1 - q) + edges[1:, None] ** 2 * q)
results = {}
residuals = []
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
    theory = []
    for typ in ["GG+", "GG-"]:
        vals = ccl.correlation(
            cosmo, ell=ls[2:], C_ell=real[2:], theta=theta.ravel() / 60, type=typ
        ).reshape(theta.shape)
        theory.append(np.trapezoid(vals, q, axis=1))
    diff = np.concatenate([xp - theory[0], xm - theory[1]])
    row = {
        "BB": pte(bb, cov["BB"]),
        "EB": pte(eb, cov["EB"]),
        "EE_realization": pte(ee - own, cov["EE"]),
        "EE_ensemble": pte(ee - target, cov["EE"]),
        "xi_plus": pte(diff[:20], cxi[:20, :20]),
        "xi_minus": pte(diff[20:], cxi[20:, 20:]),
        "xi_combined": pte(diff, cxi),
    }
    results[name] = row
    print(name, json.dumps(row), flush=True)
keys = list(next(iter(results.values())))
summary = {
    k: {
        "ks_p": float(kstest([r[k]["pte"] for r in results.values()], "uniform").pvalue)
    }
    for k in keys
}
(root / "metrics.json").write_text(
    json.dumps(
        {
            "per_seed": results,
            "uniform_ks": summary,
            "xi_theory": "CCL spin-2 transform of pixelized realization_ee, uniform pair density within 20 annuli; ell <= 8192; not measured pair-weight averaging",
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
print("Inspect both figures and investigate extreme PTEs before reporting.")
