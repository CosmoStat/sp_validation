# %%
# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import treecorr
from cosmo_val import CosmologyValidation

# %%
cv = CosmologyValidation(
    versions=["SP_v1.4_noalpha"],
    data_base_dir="/n17data/mkilbing/astro/data/",
    npatch=100,
)
# %%
# %%memray_flamegraph
# cv.plot_rho_stats()
# %%
# cv.plot_tau_stats()
# # %%
# if cv.rho_tau_method != "none":
#     cv.plot_rho_tau_fits()
# # %%
# cv.plot_footprints()
# # %%
# cv.plot_scale_dependent_leakage()
# # %%
# cv.plot_objectwise_leakage()
# # %%
# cv.plot_ellipticity()
# # %%
# cv.plot_separation()
# # %%
# cv.plot_2pcf()
# # %%
# cv.plot_aperture_mass_dispersion()
# %%
# import treecorr
# self = cv
# ver = self.versions[0]

# treecorr_config_int = {
#     **self.treecorr_config,
#     "nbins": 100,
# }
# gg_int = treecorr.GGCorrelation(treecorr_config_int)

# with self.results[ver].temporarily_load_data():
#     R = self.cc[ver]["shear"]["R"]
#     g1 = (
#         self.results[ver].dat_shear[self.cc[ver]["shear"]["e1_col"]]
#         - self.c1[ver]
#     ) / R
#     g2 = (
#         self.results[ver].dat_shear[self.cc[ver]["shear"]["e2_col"]]
#         - self.c2[ver]
#     ) / R

#     cat = treecorr.Catalog(
#         ra=self.results[ver].dat_shear["RA"],
#         dec=self.results[ver].dat_shear["Dec"],
#         g1=g1,
#         g2=g2,
#         w=self.results[ver].dat_shear["w"],
#         ra_units=self.treecorr_config["ra_units"],
#         dec_units=self.treecorr_config["dec_units"],
#         npatch=25,
#     )
#     del g1, g2, R

# self.print_cyan("Integration binning")
# gg_int.process(cat)

# %%
# gg_int = cv.calculate_cosebis(theta_min=0.04, theta_max=500, nbins=2000)
# gg_int = cv.calculate_cosebis()
# %%
# gg = cv.cat_ggs[cv.versions[0]]
# %%
from cosmo_numba.B_modes.schneider2022 import get_pure_EB_modes

# theta = np.geomspace(
#     cv.treecorr_config["min_sep"],
#     cv.treecorr_config["max_sep"],
#     cv.treecorr_config["nbins"],
# )

theta_min_int = 0.04
theta_max_int = 500
nbins_int = 1000

# theta_min_int = cv.treecorr_config["min_sep"]
# theta_max_int = cv.treecorr_config["max_sep"]
# nbins_int = 100

# %%
# pureEB_vector, cov = cv.calculate_pure_eb()
theta, theta_int, gg, gg_int = cv.calculate_pure_eb(
    theta_min=cv.treecorr_config["min_sep"],
    theta_max=cv.treecorr_config["max_sep"],
    nbins=cv.treecorr_config["nbins"],
    theta_min_int=theta_min_int,
    theta_max_int=theta_max_int,
    nbins_int=nbins_int,
)
# theta = np.geomspace(theta_min_int, theta_max_int, cv.treecorr_config["nbins"])
# theta_int = np.geomspace(theta_min_int, theta_max_int, nbins_int)


# %%
def pure_EB(corrs):
    gg, gg_int = corrs
    return get_pure_EB_modes(
        theta=gg.meanr,
        xip=gg.xip,
        xim=gg.xim,
        theta_int=gg_int.meanr,
        xip_int=gg_int.xip,
        xim_int=gg_int.xim,
        tmin=theta[0],
        tmax=theta[-1],
    )


# %%
xip_E, xim_E, xip_B, xim_B, xip_amb, xim_amb = pure_EB([gg, gg_int])
cov_eb = treecorr.estimate_multi_cov(
    [gg, gg_int], "jackknife", func=lambda x: np.hstack(pure_EB(x))
)
# %%
import suisai
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm, LogNorm, SymLogNorm


cmap = suisai.cm.forestdawn

# %%

def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


fig, ax = plt.subplots(figsize=(9, 9))
im = ax.matshow(correlation_from_covariance(cov_eb), cmap=cmap)

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
ax.set_title("Pure E/B Correlation Matrix")
plt.savefig(
    cv.cc["paths"]["output"] + "/pure_EB_corr_matrix.png", dpi=300, bbox_inches="tight"
)

# %%

nbins = gg.nbins
npatch = cv.npatch


hartlap_factor = (npatch - nbins - 2) / (npatch - 1)
start, stop = 1, nbins
nbins_eff = nbins - start + stop

cov_xip_E, cov_xim_E, cov_xip_B, cov_xim_B, cov_xip_amb, cov_xim_amb = (
    cov_eb[nbins * i : nbins * (i + 1), nbins * i : nbins * (i + 1)] for i in range(6)
)

std_xip_E, std_xim_E, std_xip_B, std_xim_B, std_xip_amb, std_xim_amb = (
    np.sqrt(np.diag(cov))
    for cov in (cov_xip_E, cov_xim_E, cov_xip_B, cov_xim_B, cov_xip_amb, cov_xim_amb)
)

assume_uncorrelated = False
if assume_uncorrelated:
    xip_E_sig, xim_E_sig, xip_B_sig, xim_B_sig = (
        np.sqrt(
            xi[start:stop] @ np.linalg.inv(np.diag(np.diag(cov_xi[start:stop, start:stop]))) @ xi[start:stop]
        )
        for (xi, cov_xi) in zip(
            (xip_E, xim_E, xip_B, xim_B), (cov_xip_E, cov_xim_E, cov_xip_B, cov_xim_B)
        )
    )
    print(xip_E_sig, xim_E_sig, xip_B_sig, xim_B_sig)
else:
    xip_E_chi2, xim_E_chi2, xip_B_chi2, xim_B_chi2 = (
        xi[start:stop] @ (hartlap_factor * np.linalg.inv(cov_xi[start:stop, start:stop])) @ xi[start:stop]
        for (xi, cov_xi) in zip(
            (xip_E, xim_E, xip_B, xim_B), (cov_xip_E, cov_xim_E, cov_xip_B, cov_xim_B)
        )
    )
    print(xip_E_chi2, xim_E_chi2, xip_B_chi2, xim_B_chi2)

# %%
from scipy import stats

xip_B_pte, xim_B_pte = [stats.chi2.sf(chi2, nbins_eff) for chi2 in (xip_B_chi2, xim_B_chi2)]
print(xip_B_pte, xim_B_pte)

# %%
for i in range(2):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    i == 1 and axs[0].errorbar(
        gg.meanr,
        gg.meanr * gg.xip / 1e-4,
        yerr=gg.meanr * np.sqrt(gg.varxip) / 1e-4,
        fmt="k.",
        # ms=12,
        label=r"$\xi_{+}=\xi_{+}^{E}+\xi_{+}^{B}+\xi_{+}^{\mathrm{amb}}$",
    )
    axs[0].errorbar(
        gg.meanr,
        gg.meanr * xip_E / 1e-4,
        yerr=gg.meanr * std_xip_E / 1e-4,
        color="g",
        ls="",
        marker=".",
        # ms=12,
        label=rf"$\xi_{{+}}^{{E}}, \sqrt{{\chi_0^2}}={np.round(np.sqrt(xip_E_chi2),1)}$",
    )
    axs[0].errorbar(
        gg.meanr,
        gg.meanr * xip_B / 1e-4,
        yerr=gg.meanr * std_xip_B / 1e-4,
        color="r",
        ls="",
        marker=".",
        # ms=12,
        label=rf"$\xi_{{+}}^{{B}}, {{\rm PTE}}={np.round(xip_B_pte,4)}$",
    )
    axs[0].errorbar(
        gg.meanr,
        gg.meanr * xip_amb / 1e-4,
        yerr=gg.meanr * std_xip_amb / 1e-4,
        marker=".",
        # ms=12,
        ls="",
        color="purple",
        label=r"$\xi_{+}^{\mathrm{amb}}$",
    )
    axs[0].axhline(0, ls="--", color="k", alpha=0.5)
    axs[0].set_xscale("log")
    axs[0].set_xticks([0.1, 1, 10, 100], [r"$0.1$", r"$1$", r"$10$", r"$100$"])
    axs[0].set_xlabel(r"$\theta$ (arcmin)")
    axs[0].set_ylabel(r"$\theta\xi\times10^{4}$")
    axs[0].legend(loc="upper left")
    axs[0].set_ylim(-0.55, 1.9)
    axs[0].set_title(r"$\xi_{+}$")

    i == 1 and axs[1].errorbar(
        gg.meanr,
        gg.meanr * gg.xim / 1e-4,
        yerr=gg.meanr * np.sqrt(gg.varxim) / 1e-4,
        fmt="k.",
        # ms=12,
        label=r"$\xi_{-}=\xi_{-}^{E}-\xi_{-}^{B}+\xi_{-}^{\mathrm{amb}}$",
    )
    axs[1].errorbar(
        gg.meanr,
        gg.meanr * xim_E / 1e-4,
        yerr=gg.meanr * std_xim_E / 1e-4,
        color="g",
        ls="",
        marker=".",
        # ms=12,
        label=rf"$\xi_{{-}}^{{E}}, \sqrt{{\chi_0^2}}={np.round(np.sqrt(xim_E_chi2),1)}$",
    )
    axs[1].errorbar(
        gg.meanr,
        gg.meanr * xim_B / 1e-4,
        yerr=gg.meanr * std_xim_B / 1e-4,
        color="r",
        ls="",
        marker=".",
        # ms=12,
        # label=r"$\xi_{-}^{B}$",
        label=rf"$\xi_{{-}}^{{B}}, {{\rm PTE}}={np.round(xim_B_pte,4)}$",
    )
    axs[1].errorbar(
        gg.meanr,
        gg.meanr * xim_amb / 1e-4,
        yerr=gg.meanr * std_xim_amb / 1e-4,
        marker=".",
        ls="",
        # ms=12,
        color="purple",
        label=r"$\xi_{-}^{\mathrm{amb}}$",
    )
    axs[1].axhline(0, ls="--", color="k", alpha=0.5)
    axs[1].set_xscale("log")
    axs[1].set_xlabel(r"$\theta$ (arcmin)")
    axs[1].legend(loc="upper left")
    axs[1].set_ylim(-0.55, 1.9)
    axs[1].set_title(r"$\xi_{-}$")
    axs[1].set_xticks([0.1, 1, 10, 100], [r"$0.1$", r"$1$", r"$10$", r"$100$"])

    fig.suptitle("Pure E/B Correlation Functions")
    plt.savefig(f"output/pure_EB_modes{'' if i == 0 else '_vs_total'}.png", dpi=300)
# sns.despine()


# %%
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
axs[0].errorbar(
    gg_int.meanr,
    gg_int.meanr * gg_int.xip / 1e-4,
    # yerr=gg_int.meanr * np.sqrt(gg_int.varxip) / 1e-4,
    fmt="k.",
    ms=3,
    alpha=0.5,
    label=rf"$\xi_{{+}}$, Integration: ${theta_min_int} < \theta < {theta_max_int}$, {nbins_int} bins",
)
axs[0].errorbar(
    gg.meanr,
    gg.meanr * gg.xip / 1e-4,
    yerr=gg.meanr * np.sqrt(gg.varxip) / 1e-4,
    ls="",
    marker=".",
    c="crimson",
    ms=12,
    label=rf"$\xi_{{+}}$, Reporting: \ \ \ ${cv.treecorr_config['min_sep']} < \theta < {cv.treecorr_config['max_sep']}$, {cv.treecorr_config['nbins']} bins",
)
axs[0].axhline(0, ls="--", color="k", alpha=0.5)
axs[0].set_xscale("log")
axs[0].set_xticks([0.1, 1, 10, 100], [r"$0.1$", r"$1$", r"$10$", r"$100$"])
axs[0].set_xlabel(r"$\theta$ (arcmin)")
axs[0].set_ylabel(r"$\theta\xi\times10^{4}$")
axs[0].legend(loc="upper left")
axs[0].set_ylim(-0.55, 1.9)
axs[0].set_title(r"$\xi_{+}$")

axs[1].errorbar(
    gg_int.meanr,
    gg_int.meanr * gg_int.xim / 1e-4,
    # yerr=gg_int.meanr * np.sqrt(gg_int.varxim) / 1e-4,
    fmt="k.",
    ms=3,
    alpha=0.5,
    label=rf"$\xi_{{-}}$, Integration: ${theta_min_int} < \theta < {theta_max_int}$, {nbins_int} bins",
)
axs[1].errorbar(
    gg.meanr,
    gg.meanr * gg.xim / 1e-4,
    yerr=gg.meanr * np.sqrt(gg.varxim) / 1e-4,
    ls="",
    marker=".",
    c="crimson",
    ms=12,
    label=rf"$\xi_{{-}}$, Reporting: \ \ \ ${cv.treecorr_config['min_sep']} < \theta < {cv.treecorr_config['max_sep']}$, {cv.treecorr_config['nbins']} bins",
)
axs[1].axhline(0, ls="--", color="k", alpha=0.5)
axs[1].set_xscale("log")
axs[1].set_xlabel(r"$\theta$ (arcmin)")
axs[1].legend(loc="upper left")
axs[1].set_ylim(-0.55, 1.9)
axs[1].set_title(r"$\xi_{-}$")
axs[1].set_xticks([0.1, 1, 10, 100], [r"$0.1$", r"$1$", r"$10$", r"$100$"])

fig.suptitle("Correlation Functions Integration vs. Reporting (100 patches for cov)")
plt.savefig(f"output/pure_xi_integration_vs_reporting.png", dpi=300)
# %%
plt.plot(gg_int.meanr, np.sqrt(gg_int.varxip), label=r"$\xi_{+}$ uncertainty")
plt.plot(gg_int.meanr, np.sqrt(gg_int.varxim), c="crimson", label=r"$\xi_{-}$ uncertainty")
plt.plot(gg_int.meanr, 1 / np.sqrt(gg_int.npairs) / 5, label="npairs / 5")

# plt.plot(gg.meanr, gg.npairs / 50, ls="", marker=".", c='crimson')
plt.xscale("log")
plt.yscale("log")
plt.legend()
# %%
