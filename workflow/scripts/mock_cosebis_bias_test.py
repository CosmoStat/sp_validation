"""Gather: mock COSEBIS bias test from pre-computed per-mock results.

Loads per-mock COSEBIS B_n from scatter jobs, propagates CosmoCov ξ±
covariance to COSEBIS space, tests mean B_n = 0 at σ/√N precision.

Two-panel figure:
  Top: B_n / σ_analytic for each mock (faded) + mean with σ/√N error bars
  Bottom: σ_empirical / σ_analytic per mode
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import treecorr
from scipy.stats import chi2 as chi2_dist

from cosmo_numba.B_modes.cosebis import COSEBIS
from sp_validation.b_modes import scale_cut_to_bins

from plotting_utils import FIG_WIDTH_SINGLE, PAPER_MPLSTYLE

plt.style.use(PAPER_MPLSTYLE)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Load scatter results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cosebis_paths = snakemake.input.cosebis
xi_ref_path = snakemake.input.xi_ref
cov_path = snakemake.input.cov
nmodes = snakemake.params.nmodes
theta_min = snakemake.params.theta_min
theta_max = snakemake.params.theta_max
figure_path = snakemake.output.figure
evidence_path = snakemake.output.evidence

n_mocks = len(cosebis_paths)
all_Bn = np.zeros((n_mocks, nmodes))
all_En = np.zeros((n_mocks, nmodes))

for i, path in enumerate(cosebis_paths):
    data = np.load(path)
    all_Bn[i] = data["Bn"]
    all_En[i] = data["En"]

print(f"Loaded {n_mocks} mock COSEBIS results, {nmodes} modes each")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analytic COSEBIS covariance from ξ± covariance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

gg_ref = treecorr.GGCorrelation(
    min_sep=0.5, max_sep=500, nbins=1000, sep_units="arcmin"
)
gg_ref.read(xi_ref_path)

start, stop = scale_cut_to_bins(gg_ref, theta_min, theta_max)
theta_cut = gg_ref.meanr[start:stop].astype(float)
nbins_xi = len(gg_ref.meanr)

cov_xipm = np.loadtxt(cov_path)
inds = np.arange(start, stop)
cov_inds = np.concatenate([inds, inds + nbins_xi])

cosebis_obj = COSEBIS(np.min(theta_cut), np.max(theta_cut), nmodes, precision=120)
cov_cosebis = cosebis_obj.cosebis_covariance_from_xipm_covariance(
    theta_cut, cov_xipm[cov_inds[:, None], cov_inds]
)
cov_B = cov_cosebis[nmodes:, nmodes:]
sigma_analytic = np.sqrt(np.diag(cov_B))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Statistics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

mean_Bn = all_Bn.mean(axis=0)
sigma_empirical = all_Bn.std(axis=0, ddof=1)
sigma_ratio = sigma_empirical / sigma_analytic

bias_significance = mean_Bn / (sigma_analytic / np.sqrt(n_mocks))
max_bias = np.max(np.abs(bias_significance))

cov_mean = cov_B / n_mocks
chi2_val = mean_Bn @ np.linalg.solve(cov_mean, mean_Bn)
pte = chi2_dist.sf(chi2_val, nmodes)

for n_sub in [5, 8]:
    cov_sub = cov_B[:n_sub, :n_sub] / n_mocks
    chi2_sub = mean_Bn[:n_sub] @ np.linalg.solve(cov_sub, mean_Bn[:n_sub])
    pte_sub = chi2_dist.sf(chi2_sub, n_sub)
    print(f"  Modes 1-{n_sub}: chi2={chi2_sub:.2f}/{n_sub}, PTE={pte_sub:.3f}")

print(f"\nAll {nmodes} modes: chi2={chi2_val:.2f}/{nmodes}, PTE={pte:.3f}")
print(f"Max |bias significance|: {max_bias:.2f}σ")
print(f"σ_emp/σ_ana range: [{sigma_ratio.min():.3f}, {sigma_ratio.max():.3f}]")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

modes = np.arange(1, nmodes + 1)
color = sns.color_palette("husl", 4)[0]

fig, (ax_bn, ax_ratio) = plt.subplots(
    2,
    1,
    figsize=(FIG_WIDTH_SINGLE, 3.5),
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    sharex=True,
)

ax_bn.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
ax_bn.axhspan(-2, 2, color="0.92", alpha=0.5, zorder=0, label=r"$\pm 2\sigma$")

for i in range(n_mocks):
    ax_bn.plot(
        modes, all_Bn[i] / sigma_analytic, "o", color=color, ms=2, alpha=0.15, zorder=1
    )

ax_bn.errorbar(
    modes,
    mean_Bn / sigma_analytic,
    yerr=np.full(nmodes, 1.0 / np.sqrt(n_mocks)),
    fmt="o",
    color=color,
    ms=4,
    lw=0.8,
    capsize=2,
    zorder=5,
    label=rf"Mean ($N={n_mocks}$)",
)

ax_bn.text(
    0.98,
    0.95,
    rf"$\chi^2/{nmodes} = {chi2_val:.1f}$, PTE $= {pte:.2f}$",
    transform=ax_bn.transAxes,
    fontsize=7,
    ha="right",
    va="top",
)

ax_bn.set_ylabel(r"$B_n / \sigma_n$")
ax_bn.set_ylim(-4, 4)
ax_bn.legend(fontsize=6, loc="upper left")

ax_ratio.axhline(1, color="black", lw=0.8, ls="--", alpha=0.5)
ax_ratio.bar(modes, sigma_ratio, color=color, alpha=0.6, width=0.7)
ax_ratio.set_ylabel(r"$\sigma_\mathrm{emp} / \sigma_\mathrm{ana}$")
ax_ratio.set_xlabel("COSEBIS mode $n$")
ax_ratio.set_ylim(0, 2)
ax_ratio.set_xticks(modes)

fig.suptitle(
    rf"COSEBIS $B_n$ bias test: {n_mocks} zero-B GLASS mocks",
    fontsize=8,
    y=0.98,
)

Path(figure_path).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(figure_path, dpi=300, bbox_inches="tight")
print(f"Saved {figure_path}")
plt.close(fig)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Evidence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

evidence = {
    "spec": "mock_cosebis_bias_test",
    "timestamp": datetime.now().isoformat(),
    "n_mocks": n_mocks,
    "nmodes": nmodes,
    "scale_cut": [theta_min, theta_max],
    "chi2": float(chi2_val),
    "dof": nmodes,
    "pte": float(pte),
    "max_bias_significance": float(max_bias),
    "per_mode_bias_significance": bias_significance.tolist(),
    "sigma_ratio_mean": float(sigma_ratio.mean()),
    "sigma_ratio_std": float(sigma_ratio.std()),
    "sigma_ratio_per_mode": sigma_ratio.tolist(),
    "mean_Bn": mean_Bn.tolist(),
    "sigma_analytic": sigma_analytic.tolist(),
    "sigma_empirical": sigma_empirical.tolist(),
}

Path(evidence_path).parent.mkdir(parents=True, exist_ok=True)
with open(evidence_path, "w") as f:
    json.dump(evidence, f, indent=2)
print(f"Saved {evidence_path}")
