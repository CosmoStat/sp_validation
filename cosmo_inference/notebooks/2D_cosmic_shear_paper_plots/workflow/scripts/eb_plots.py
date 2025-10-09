"""
Plot Data 2PCF as well as best-fit model, B-modes, and leakage systematics.
"""

# %%
# if interactive
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython import get_ipython
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

ipython = get_ipython()

# enable autoreload for interactive sessions
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
else:
    # Force unbuffered stdout and stderr
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)  # line-buffered
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

from sp_validation.cosmo_val import CosmologyValidation

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")
    os.chdir(
        "/n17data/cdaley/unions/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/workflow/scripts"
    )
    from snakemake_helpers import get_pipe

    snakemake = get_pipe("eb_plots", workdir="../")
else:
    from snakemake.script import snakemake

params = snakemake.params
config = snakemake.config
plt.style.use(config["plot_style"])

# %%
cwd = os.getcwd()
os.chdir("/n17data/cdaley/unions/sp_validation/notebooks/cosmo_val")

cv = CosmologyValidation(
    versions=[config["version"]],
    data_base_dir="/n17data/mkilbing/astro/data",
)
# %%
eb_results = cv.calculate_pure_eb(version=config["version"], **config["pure_eb"])
# %%
cosebis_results = cv.calculate_cosebis(version=config["version"], **config["cosebis"])
cosebis_results_fiducial = cosebis_results[tuple(config["fiducial_scale_cut"])]

# %%
os.chdir(cwd)

# %%
theta_model, xi_plus_theory, xi_minus_theory = np.loadtxt(
    snakemake.input.bestfit_xi_shear, unpack=True
)

theta_model, xi_plus_sys, xi_minus_sys = np.loadtxt(
    snakemake.input.bestfit_xi_sys, unpack=True
)


# %%
def populate_eb_results(results, start_p, start_m, stop=None):
    results["start_p"] = start_p
    results["start_m"] = start_m
    stop = stop if stop is not None else results["gg"].nbins - 1
    results["stop"] = stop

    nbins, npatch = results["gg"].nbins, results["gg"].npatch1
    hartlap_factor = (npatch - nbins - 2) / (npatch - 1)

    starts = {
        key: start_p if "xip" in key else start_m
        for key in ["xip_E", "xim_E", "xip_B", "xim_B", "xip_amb", "xim_amb"]
    }
    nbins_eff = {key: stop - starts[key] for key in starts}

    # Compute and store covariance blocks and stds in results with explicit names
    for i, key in enumerate(["xip_E", "xim_E", "xip_B", "xim_B", "xip_amb", "xim_amb"]):
        cov_block = results["cov"][
            nbins * i : nbins * (i + 1), nbins * i : nbins * (i + 1)
        ]
        std_block = np.sqrt(np.diag(cov_block))
        results[f"cov_{key}"] = cov_block
        results[f"std_{key}"] = std_block

    chi2s = {
        key: results[key][starts[key] : stop]
        @ (
            hartlap_factor
            * np.linalg.inv(
                results[f"cov_{key}"][starts[key] : stop, starts[key] : stop]
            )
        )
        @ results[key][starts[key] : stop]
        for key in ["xip_E", "xim_E", "xip_B", "xim_B"]
    }
    # Store chi2s in results
    for key in chi2s:
        results[f"{key}_chi2"] = chi2s[key]

    xip_B_pte, xim_B_pte = [
        stats.chi2.sf(chi2s[key], nbins_eff[key]) for key in ["xip_B", "xim_B"]
    ]
    # Store PTEs in results
    results["xip_B_pte"] = xip_B_pte
    results["xim_B_pte"] = xim_B_pte

    stop = results["stop"]
    for pm in "pm":
        results[f"xi{pm}_B_ptes"] = []
        for start in range(stop):
            nbins_eff = stop - start

            chi2s = {
                key: results[key][start:stop]
                @ (
                    hartlap_factor
                    * np.linalg.inv(results[f"cov_{key}"][start:stop, start:stop])
                )
                @ results[key][start:stop]
                for key in [f"xi{pm}_E", f"xi{pm}_B"]
            }

            results[f"xi{pm}_B_ptes"].append(
                stats.chi2.sf(chi2s[f"xi{pm}_B"], nbins_eff)
            )
    return results


populate_eb_results(eb_results, 10, 15, eb_results["gg"].nbins - 2)

# %%
# Plot Pure 2PCFs
gg, gg_int = eb_results["gg"], eb_results["gg_int"]
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

axs[0].plot(
    theta_model,
    theta_model * (xi_plus_theory + xi_plus_sys) / 1e-4,
    color="black",
    # ms=12,
    label=r"Best-fit $\xi_{+}^{\rm th} + \xi_{+}^{\rm sys}$",
)

axs[0].plot(
    theta_model,
    theta_model * (xi_plus_theory) / 1e-4,
    color="darkmagenta",
    ls="--",
    # ms=12,
    label=r"Best-fit $\xi_{+}^{\rm th}$",
)

axs[0].plot(
    theta_model,
    theta_model * xi_plus_sys / 1e-4,
    color="dodgerblue",
    # ms=12,
    label=r"Best-fit $\xi_{+}^{\rm sys}$",
)

axs[0].errorbar(
    gg.meanr,
    gg.meanr * gg.xip / 1e-4,
    yerr=gg.meanr * np.sqrt(gg.varxip) / 1e-4,
    fmt="k.",
    # ms=12,
    # label=r"$\xi_{+}=\xi_{+}^{E}+\xi_{+}^{B}+\xi_{+}^{\mathrm{amb}}$",
    label=r"$\xi_{+}$",
)

axs[0].errorbar(
    gg.meanr,
    gg.meanr * eb_results["xip_B"] / 1e-4,
    yerr=gg.meanr * eb_results["std_xip_B"] / 1e-4,
    color="crimson",
    ls="",
    marker=".",
    # ms=12,
    label=rf"$\xi_{{+}}^{{B}}, {{\rm PTE}}={np.round(eb_results['xip_B_pte'],4)}$",
)


axs[0].axvspan(0, gg.left_edges[eb_results["start_p"]], color="gray", alpha=0.1)
axs[0].axvspan(gg.right_edges[eb_results["stop"]], 1000, color="gray", alpha=0.1)
axs[0].axhline(0, ls="--", color="k", alpha=0.5)
axs[0].set_xscale("log")
axs[0].set_xticks([0.1, 1, 10, 100], [r"$0.1$", r"$1$", r"$10$", r"$100$"])
axs[0].set_xlabel(r"$\theta$ (arcmin)")
axs[0].set_ylabel(r"$\theta\xi\times10^{4}$")
axs[0].legend(loc="upper left")
axs[0].set_ylim(-0.45, 1.9)
axs[0].set_xlim(1, 170)
axs[0].set_title(r"$\xi_{+}$")

axs[1].plot(
    theta_model,
    theta_model * (xi_minus_theory + xi_minus_sys) / 1e-4,
    color="black",
    # ms=12,
    label=r"Best-fit $\xi_{-}^{\rm th} + \xi_{-}^{\rm sys}$",
)

axs[1].plot(
    theta_model,
    theta_model * (xi_minus_theory) / 1e-4,
    color="darkmagenta",
    ls="--",
    # ms=12,
    label=r"Best-fit $\xi_{-}^{\rm th}$",
)

axs[1].plot(
    theta_model,
    theta_model * xi_minus_sys / 1e-4,
    color="dodgerblue",
    # ms=12,
    label=r"Best-fit $\xi_{-}^{\rm sys}$",
)

axs[1].errorbar(
    gg.meanr,
    gg.meanr * gg.xim / 1e-4,
    yerr=gg.meanr * np.sqrt(gg.varxim) / 1e-4,
    fmt="k.",
    # ms=12,
    # label=r"$\xi_{-}=\xi_{-}^{E}-\xi_{-}^{B}+\xi_{-}^{\mathrm{amb}}$",
    label=r"$\xi_{-}$",
)

axs[1].errorbar(
    gg.meanr,
    gg.meanr * eb_results["xim_B"] / 1e-4,
    yerr=gg.meanr * eb_results["std_xim_B"] / 1e-4,
    color="crimson",
    ls="",
    marker=".",
    # ms=12,
    # label=r"$\xi_{-}^{B}$",
    label=rf"$\xi_{{-}}^{{B}}, {{\rm PTE}}={np.round(eb_results['xim_B_pte'],4)}$",
)

axs[1].axvspan(0, gg.left_edges[eb_results["start_m"]], color="gray", alpha=0.1)
axs[1].axvspan(gg.right_edges[eb_results["stop"]], 1000, color="gray", alpha=0.1)
axs[1].axhline(0, ls="--", color="k", alpha=0.5)
axs[1].set_xscale("log")
axs[1].set_xlabel(r"$\theta$ (arcmin)")
axs[1].legend(loc="upper left")
axs[1].set_ylim(-0.45, 1.9)
axs[1].set_title(r"$\xi_{-}$")
axs[1].set_xticks([0.1, 1, 10, 100], [r"$0.1$", r"$1$", r"$10$", r"$100$"])
axs[1].set_xlim(1, 170)

plt.savefig(snakemake.output["xis"], bbox_inches="tight", dpi=300)


# %%
# Plot 2PCF covariance
def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


fig, ax = plt.subplots(figsize=(4, 4))
im = ax.matshow(correlation_from_covariance(eb_results["cov"]), cmap="vlag")

for ticks in (plt.xticks, plt.yticks):
    ticks(
        range(10, 111, 20),
        [
            r"$\xi_+^{E}$",
            r"$\xi_-^{E}$",
            r"$\xi_+^{B}$",
            r"$\xi_-^{B}$",
            r"$\xi_+^{\rm amb}$",
            r"$\xi_-^{\rm amb}$",
        ],
    )
    ticks(range(0, 121, 20), minor=True)
ax.tick_params(axis="both", which="major", length=0)

im.set_clim(-1, 1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size/pad as needed
plt.colorbar(im, cax=cax)
# ax.set_title("Pure E/B Sampled Correlation Matrix")
plt.savefig(snakemake.output["xis_corr"], dpi=300, bbox_inches="tight")

# %%
# Plot COSEBIS
plt.figure(figsize=(4, 4))
plt.errorbar(
    cosebis_results_fiducial["modes"],
    cosebis_results_fiducial["E"] * 1e11,
    yerr=np.sqrt(np.diag(cosebis_results_fiducial["cov_E"])) * 1e11,
    label=rf"COSEBIs E-modes; $\sqrt{{\chi_0^2}}$ = {cosebis_results_fiducial['E_snr']:.2f}",
)
plt.errorbar(
    cosebis_results_fiducial["modes"],
    cosebis_results_fiducial["B"] * 1e11,
    yerr=np.sqrt(np.diag(cosebis_results_fiducial["cov_B"])) * 1e11,
    c="crimson",
    label=rf"COSEBIs B-modes; PTE $B_0$ = {cosebis_results_fiducial['B0_pte']:.2f}, $B_{{\rm all}}$ = {cosebis_results_fiducial['B_pte']:.2f}",
)

plt.axhline(0, ls="--", color="k")
plt.legend()
plt.xlabel(r"$n$ (mode)")
plt.ylabel(r"$E_n,B_n \times 10^{11}$")
plt.savefig(snakemake.output["cosebis"], dpi=300, bbox_inches="tight")

# %%
# Plot COSEBIS covariance

fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(cosebis_results_fiducial["cov_EB"], cmap="coolwarm")

nmodes = len(cosebis_results_fiducial["E"])
for ticks in (plt.xticks, plt.yticks):
    ticks(
        np.arange(nmodes / 2, nmodes * 2, nmodes),
        [r"$E_n$", r"$B_n$"],
    )
    ticks([nmodes], minor=True)

clim = np.max(np.abs(cosebis_results_fiducial["cov_EB"]))
im.set_clim(-clim, clim)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax)
plt.savefig(snakemake.output["cosebis_cov"], dpi=300, bbox_inches="tight")

# %%
plt.figure(figsize=(4, 4))
plt.plot(
    gg.meanr[0 : eb_results["stop"]],
    eb_results["xip_B_ptes"],
    label=r"$\xi_{B+}$",
    marker="$+$",
    c="dodgerblue",
    ms=8,
    ls="",
)
plt.plot(
    gg.meanr[0 : eb_results["stop"]],
    eb_results["xim_B_ptes"],
    c="crimson",
    label=r"$\xi_{B-}$",
    marker="$-$",
    ms=8,
    ls="",
)

B_ptes = [cosebis_results[scale_cut]["B_pte"] for scale_cut in cosebis_results]

plt.plot(
    [scale_cut[0] for scale_cut in cosebis_results],
    B_ptes,
    c="k",
    label=r"$B_{n}$",
    marker=".",
    ms=8,
    ls="",
)

plt.axhspan(0, 0.05, ls="--", color="k", alpha=0.5)
plt.axhspan(0.95, 1, ls="--", color="k", alpha=0.5)

plt.xscale("log")
# plt.yscale("log")
plt.ylim(0, 1)
plt.xlabel(r"$\theta$ (arcmin)")
plt.ylabel(r"PTE")
plt.legend()
# plt.title("B-mode PTEs as a function of lower scale cut")
plt.savefig(snakemake.output["ptes"], dpi=300, bbox_inches="tight")
