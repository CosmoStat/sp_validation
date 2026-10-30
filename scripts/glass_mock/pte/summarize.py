import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from sp_validation import sacc_io

r = Path("/n17data/cdaley/scratch/glass-mock-cv/pte")
m = json.loads((r / "metrics.json").read_text())
a = json.loads((r / "input_audit.json").read_text())
rows = list(m["per_seed"].values())
for key in (
    "BB",
    "EB",
    "EE_realization",
    "EE_ensemble",
    "xi_plus",
    "xi_minus",
    "xi_combined",
):
    q = np.array([x[key]["chi2"] for x in rows])
    p = np.array([x[key]["pte"] for x in rows])
    print(
        key,
        "mean chi2",
        q.mean(),
        "pte range",
        p.min(),
        p.max(),
        "KS",
        m["uniform_ks"][key],
    )
for key in ("xi_plus", "xi_minus"):
    print(
        key,
        "max dchi2 uniform",
        max(abs(x[key]["chi2"] - x[key + "_uniform"]["chi2"]) for x in rows),
    )
v = [x["noise_moment_fractional_difference_from_seed1"] for x in a.values()]
print("weighted moment fractional range", min(v), max(v))
with fits.open(r / "output/pseudo_cl_cov_GLASSpte_s01_powspace_nbins=32.fits") as h:
    sig = np.sqrt(np.diag(h["COVAR_EE_EE"].data))
res = []
for i in range(1, 11):
    s = sacc_io.load(
        r / f"output/pseudo_cl_GLASSpte_s{i:02d}_powspace_nbins=32.sacc", blind="none"
    )
    ell, ee, bb, eb, w = sacc_io.get_pseudo_cl(s, (0, 0))
    with np.load(r.parent / f"mocks/v1.4.6.3/seed_{i:05d}/maps/spectra.npz") as sky:
        pred = w.weight.T @ np.interp(
            w.values, sky["ell"], sky["realization_ee"], left=0, right=0
        )
    res.append((ee - pred) / sig)
print("EE ell mean std-err")
for multipole, x in zip(ell, np.array(res).T):
    print(f"{multipole:.1f} {x.mean():.4f} {x.std(ddof=1) / np.sqrt(10):.4f}")
edges = np.geomspace(1, 250, 21)
center = np.sqrt(edges[:-1] * edges[1:])
x = np.array(m["theta_bin_mean_standardized_residual_squared"])
print("theta and mean bin chi2 plus minus")
for theta, p, n in zip(center, x[:20], x[20:]):
    print(f"{theta:.3f} {p:.4f} {n:.4f}")
