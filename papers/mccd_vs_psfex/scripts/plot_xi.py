"""theta xi_pm of MCCD (v1.3.6.3) against PSFEx (v1.4.6.3), leakage-corrected.

Error bars are the TreeCorr jackknife sigmas (npatch 100); the shaded band is
the fiducial scale cut 12-83 arcmin used in the likelihood.

Usage: python plot_xi.py <cosmo_val_output dir> <out.png>
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TAG = "_xi_minsep=1.0_maxsep=250.0_nbins=20_npatch=100.txt"
RUNS = [
    ("SP_v1.4.6.3_leak_corr", "PSFEx (v1.4.6.3)", "#1f77b4", -0.02),
    ("SP_v1.3.6.3_leak_corr", "MCCD (v1.3.6.3)", "#d62728", 0.02),
]

indir, out = sys.argv[1], sys.argv[2]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
for ver, label, col, dx in RUNS:
    path = f"{indir}/{ver}{TAG}"
    with open(path) as f:
        names = [line for line in f if line.startswith("#   r_nom")][0].lstrip("#").split()
    t = dict(zip(names, np.loadtxt(path, comments="#", max_rows=20).T))
    th = t["meanr"] * (1 + dx)
    for ax, k in zip(axes, ("xip", "xim")):
        ax.errorbar(th, 1e4 * th * t[k], 1e4 * th * t[f"sigma_{k}"], fmt="o", ms=3,
                    color=col, label=label, capsize=2, lw=1)
for ax, k in zip(axes, (r"\xi_+", r"\xi_-")):
    ax.axvspan(12, 83, color="0.9", zorder=0)
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\theta$ [arcmin]")
    ax.set_ylabel(rf"$10^4\,\theta\,{k}$")
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(out, dpi=130)
print(out)
