"""GLS fit of the PSF-leakage parameters (alpha, beta) from rho/tau statistics.

tau_0+ = alpha rho_0+ + beta rho_2+ and tau_2+ = alpha rho_2+ + beta rho_1+,
fitted over all 20 bins (1-250 arcmin) with the analytic tau covariance
(first 2 x nbins block of cov_tau_*_th.npy, as cosmosis_fitting.py uses it).
Reproduces the priors_psf.ini centres from PSFEx v1.4.6.3; the MCCD fit
centres priors_psf_SP_v1.3.6.3_leak_corr.ini.

Usage: python fit_leakage.py <rho_tau_stats dir> VERSION [VERSION ...]
"""

import sys

import numpy as np
from astropy.io import fits

SUFFIX = "_minsep=1.0_maxsep=250.0_nbins=20_npatch=100"

d = sys.argv[1]
for v in sys.argv[2:]:
    r = fits.getdata(f"{d}/rho_stats_{v}{SUFFIX}.fits")
    t = fits.getdata(f"{d}/tau_stats_{v}{SUFFIX}.fits")
    n = len(t)
    C = np.load(f"{d}/cov_tau_{v}{SUFFIX}_th.npy")[: 2 * n, : 2 * n]
    y = np.concatenate([t["tau_0_p"], t["tau_2_p"]])
    X = np.column_stack(
        [
            np.concatenate([r["rho_0_p"], r["rho_2_p"]]),
            np.concatenate([r["rho_2_p"], r["rho_1_p"]]),
        ]
    )
    Ci = np.linalg.inv(C)
    F = X.T @ Ci @ X
    p = np.linalg.solve(F, X.T @ Ci @ y)
    e = np.sqrt(np.diag(np.linalg.inv(F)))
    res = y - X @ p
    print(
        f"{v}: alpha = {p[0]:.4f} +/- {e[0]:.4f}, beta = {p[1]:.3f} +/- {e[1]:.3f},"
        f" chi2 = {res @ Ci @ res:.1f} / {2 * n - 2}"
    )
