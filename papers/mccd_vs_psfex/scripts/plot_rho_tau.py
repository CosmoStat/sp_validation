"""rho_0,1,2 and tau_0,2 (+ components) of MCCD against PSFEx.

rho_0: PSF ellipticity autocorrelation; rho_1: PSF-residual autocorrelation;
rho_2: PSF x residual; tau_0/tau_2: galaxy shear x PSF / x residual. Error
bars: sqrt(var) columns (jackknife for rho, as stored) and the diagonal of the
analytic tau covariance. Shaded: the 12-83 arcmin scale cut.

Usage: python plot_rho_tau.py <rho_tau_stats dir> <out.png>
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

SUFFIX = "_minsep=1.0_maxsep=250.0_nbins=20_npatch=100"
RUNS = [
    ("SP_v1.4.6.3_leak_corr", "PSFEx (v1.4.6.3)", "#1f77b4", 0.97),
    ("SP_v1.3.6.3_leak_corr", "MCCD (v1.3.6.3)", "#d62728", 1.03),
]
PANELS = [
    ("rho", "rho_0_p", r"$\rho_0^+$"),
    ("rho", "rho_1_p", r"$\rho_1^+$"),
    ("rho", "rho_2_p", r"$\rho_2^+$"),
    ("tau", "tau_0_p", r"$\tau_0^+$"),
    ("tau", "tau_2_p", r"$\tau_2^+$"),
]

d, out = sys.argv[1], sys.argv[2]
fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
for ver, label, col, fx in RUNS:
    stats = {k: fits.getdata(f"{d}/{k}_stats_{ver}{SUFFIX}.fits") for k in ("rho", "tau")}
    n = len(stats["tau"])
    ctau = np.sqrt(np.diag(np.load(f"{d}/cov_tau_{ver}{SUFFIX}_th.npy")))
    for ax, (k, c, lab) in zip(axes, PANELS):
        t = stats[k]
        th = t["theta"] * fx
        if k == "tau":
            err = ctau[:n] if c == "tau_0_p" else ctau[n : 2 * n]
        else:
            err = np.sqrt(t["var" + c])
        ax.errorbar(th, th * t[c], th * err, fmt="o", ms=3, color=col, label=label, capsize=2, lw=1)
        ax.set_title(lab)
for ax in axes:
    ax.axvspan(12, 83, color="0.9", zorder=0)
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\theta$ [arcmin]")
axes[0].set_ylabel(r"$\theta\times$ statistic")
axes[0].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(out, dpi=110)
print(out)
