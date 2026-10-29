"""PSF leakage: ρ statistics, the τ fit and its posterior, the propagated
systematic, and every estimate of the leakage coefficient α side by side.

The τ model is the one the ρ/τ fit uses (shear_psf_leakage, with η):

    τ0 = α ρ0 + β ρ2 + η ρ5,  τ2 = α ρ2 + β ρ1 + η ρ4,  τ5 = α ρ5 + β ρ4 + η ρ3

evaluated here on the cached posterior samples; nothing is refitted.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

from . import style as S

# PSF figures carry no E/B mode, so they use the page accents rather than the
# mode hues: one accent for the fitted model, three validated categorical
# slots for the ρ statistics within a panel.
MODEL = "#3D5BA0"  # page cobalt
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

TAU_MODEL = {
    0: ((0, 0), (1, 2), (2, 5)),
    2: ((0, 2), (1, 1), (2, 4)),
    5: ((0, 5), (1, 4), (2, 3)),
}
PARAMS = (r"\alpha", r"\beta", r"\eta")


def tau_model_samples(rt, k, samples):
    """τ_{k,+}(θ) for each posterior sample, shape ``(n_samples, n_theta)``."""
    model = 0
    for param, rho_k in TAU_MODEL[k]:
        model = model + samples[:, [param]] * rt["rho"][rho_k]["p"][None, :]
    return model


def _power_of_ten(values):
    top = np.nanmax(np.abs(values))
    return int(np.floor(np.log10(top))) if top > 0 else 0


def tau_fit(rt, out):
    """τ_{0,2,5}+ with the posterior band of the fitted model."""
    theta = rt["theta_tau"]
    samples = rt["fit"]["flat_samples"]
    with S.figure_style():
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
        for ax, k in zip(axes, (0, 2, 5)):
            tau = rt["tau"][k]["p"]
            sigma = np.sqrt(np.clip(rt["tau"][k]["var_p"], 0, None))
            model = tau_model_samples(rt, k, samples)
            lo, med, hi = np.percentile(model, [16, 50, 84], axis=0)
            p = _power_of_ten(np.concatenate([tau + sigma, tau - sigma]))
            scale = 10.0**p
            S.zero_line(ax)
            ax.fill_between(theta, lo / scale, hi / scale, color=MODEL, alpha=0.3, lw=0)
            ax.plot(theta, med / scale, color=MODEL, lw=1.1)
            ax.errorbar(
                theta,
                tau / scale,
                yerr=sigma / scale,
                marker="o",
                color=S.INK,
                **S.DATA_VECTOR_MARK,
            )
            S.theta_axis(ax)
            ax.set_xticks([1, 10, 100])
            ax.set_xticklabels(["$1$", "$10$", "$100$"])
            ax.set_xlabel(r"$\theta$ [arcmin]")
            ax.set_ylabel(rf"$\tau_{{{k},+}} \times 10^{{{-p}}}$")
            S.label_box(ax, rf"$\tau_{{{k}}}$", loc="upper right")
        handles = [
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                color=S.INK,
                markersize=3.5,
                label=r"measured $\tau_{k,+}$",
            ),
            Line2D(
                [],
                [],
                color=MODEL,
                lw=4,
                alpha=0.4,
                label=r"$\alpha, \beta, \eta$ posterior (68%)",
            ),
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=2)
        return S.save(fig, out, "full")


def _contour_levels(h, fractions=(0.68, 0.95)):
    flat = np.sort(h.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    return sorted(flat[np.searchsorted(cum, f)] for f in fractions)


def posterior(rt, out, markers=(0, 1, 1)):
    """Corner plot of the (α, β, η) posterior from the cached samples."""
    samples = rt["fit"]["flat_samples"]
    n = samples.shape[1]
    with S.figure_style():
        fig, axes = plt.subplots(n, n, figsize=(3.6, 3.6))
        lims = [np.percentile(samples[:, i], [0.3, 99.7]) for i in range(n)]
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                ax.tick_params(labelsize=7, top=False, right=False)
                if j > i:
                    ax.set_visible(False)
                    continue
                if i == j:
                    x = samples[:, i]
                    ax.hist(
                        x,
                        bins=40,
                        range=lims[i],
                        histtype="stepfilled",
                        color=MODEL,
                        alpha=0.25,
                    )
                    ax.hist(
                        x, bins=40, range=lims[i], histtype="step", color=MODEL, lw=0.9
                    )
                    lo, med, hi = np.percentile(x, [16, 50, 84])
                    ax.set_title(
                        rf"${PARAMS[i]} = {med:.3g}^{{+{hi - med:.2g}}}"
                        rf"_{{-{med - lo:.2g}}}$",
                        fontsize=7.5,
                    )
                    ax.set_yticks([])
                    if markers[i] is not None:
                        ax.axvline(markers[i], color=S.INK_MUTED, lw=0.7, ls=":")
                else:
                    h, xe, ye = np.histogram2d(
                        samples[:, j], samples[:, i], bins=40, range=[lims[j], lims[i]]
                    )
                    h = gaussian_filter(h, 1.0)
                    levels = _contour_levels(h)
                    xc, yc = (xe[1:] + xe[:-1]) / 2, (ye[1:] + ye[:-1]) / 2
                    ax.contourf(
                        xc,
                        yc,
                        h.T,
                        levels=[levels[0], h.max()],
                        colors=[MODEL],
                        alpha=0.18,
                    )
                    ax.contourf(
                        xc,
                        yc,
                        h.T,
                        levels=[levels[1], h.max()],
                        colors=[MODEL],
                        alpha=0.38,
                    )
                    ax.contour(
                        xc, yc, h.T, levels=levels, colors=[MODEL], linewidths=0.7
                    )
                    for v, axis in ((markers[j], ax.axvline), (markers[i], ax.axhline)):
                        if v is not None:
                            axis(v, color=S.INK_MUTED, lw=0.7, ls=":")
                    ax.set_ylim(lims[i])
                ax.set_xlim(lims[j])
                ax.locator_params(nbins=3)
                if i == n - 1:
                    ax.set_xlabel(rf"${PARAMS[j]}$")
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(rf"${PARAMS[i]}$")
                elif j > 0 and i != j:
                    ax.set_yticklabels([])
        fig.get_layout_engine().set(hspace=0.02, wspace=0.02)
        return S.save(fig, out, "half")


def rho_stats(rt, out):
    """|ρ_{k,+}| on a log axis: filled markers positive, open negative."""
    theta = rt["theta"]
    groups = [
        ((0, 1, 2), r"$\rho_{0,1,2}$: ellipticity"),
        ((3, 4, 5), r"$\rho_{3,4,5}$: size-weighted"),
    ]
    with S.figure_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
        for ax, (ks, label) in zip(axes, groups):
            for c, k in zip(SERIES, ks):
                rho = rt["rho"][k]["p"]
                sig = rt["rho"][k]["sigma_p"]
                pos = rho > 0
                ax.errorbar(
                    theta,
                    np.abs(rho),
                    yerr=sig,
                    color=c,
                    ls="-",
                    lw=0.7,
                    marker="none",
                    elinewidth=0.6,
                    capsize=0,
                    alpha=0.9,
                )
                ax.plot(
                    theta[pos],
                    rho[pos],
                    "o",
                    color=c,
                    ms=3.4,
                    mec=c,
                    label=rf"$\rho_{k}$",
                )
                ax.plot(
                    theta[~pos],
                    -rho[~pos],
                    "o",
                    color=c,
                    ms=3.4,
                    mfc=S.RAG,
                    mec=c,
                    mew=0.9,
                )
            ax.set_yscale("log")
            S.theta_axis(ax)
            ax.set_xticks([1, 10, 100])
            ax.set_xticklabels(["$1$", "$10$", "$100$"])
            ax.set_xlabel(r"$\theta$ [arcmin]")
            ax.legend(loc="upper right", ncol=3, handletextpad=0.2, columnspacing=0.8)
            S.header(ax, label)
        axes[0].set_ylabel(r"$|\rho_{k,+}(\theta)|$")
        axes[0].set_ylim(top=axes[0].get_ylim()[1] * 8)
        S.header(axes[1], r"open markers: $\rho < 0$", side="right")
        return S.save(fig, out, "full")


def xi_sys_ratio(rt, xi, out, reference=0.1):
    """ξ+ PSF/data ratio, and signed θξ− in native units near its zero crossing.

    The ξ+ ratio band propagates only PSF-coefficient uncertainty, holding the
    blinded denominator fixed. The ξ− panel does not divide by the noisy data:
    show separate analytic data errors and the posterior PSF model band.
    """
    fit = rt["fit"]
    theta = np.asarray(fit["theta"])
    with S.figure_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
        ax = axes[0]
        ratio = fit["samples_plus"] / xi["xip"][None, :]
        lo, med, hi = np.percentile(ratio, [16, 50, 84], axis=0)
        for r in (-reference, reference):
            ax.axhline(r, color=S.INK_MUTED, lw=0.7, ls=(0, (2, 2)))
        S.zero_line(ax)
        ax.fill_between(theta, lo, hi, color=MODEL, alpha=0.3, lw=0)
        ax.plot(theta, med, color=MODEL, lw=1.1)
        ax.set_ylabel(r"$\xi_+^{\rm sys} / \xi_+$")
        top = max(np.nanmax(hi), reference) * 1.25
        bot = min(np.nanmin(lo), -reference) * 1.25
        ax.set_ylim(max(bot, -1.0), min(top, 1.0))
        S.label_box(ax, r"$\xi_+$", loc="upper left")

        ax = axes[1]
        lo, med, hi = np.percentile(
            fit["samples_minus"] * theta[None, :], [16, 50, 84], axis=0
        )
        S.zero_line(ax)
        ax.fill_between(
            theta, lo, hi, color=MODEL, alpha=0.3, lw=0, label=r"PSF model (68%)"
        )
        ax.plot(theta, med, color=MODEL, lw=1.1)
        data_theta = np.asarray(xi["theta"])
        ax.errorbar(
            data_theta,
            data_theta * xi["xim"],
            yerr=data_theta * xi["sigma_xim"],
            marker="o",
            color=S.INK,
            label=r"$\xi_-$ (blinded)",
            **S.DATA_VECTOR_MARK,
        )
        ax.set_ylabel(r"$\theta\,\xi_-(\theta)$ [arcmin]")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.legend(loc="upper left")
        for ax in axes:
            S.theta_axis(ax)
            ax.set_xticks([1, 10, 100])
            ax.set_xticklabels(["$1$", "$10$", "$100$"])
            ax.set_xlabel(r"$\theta$ [arcmin]")
        return S.save(fig, out, "full")


def theta_tau(rt, out, corrected=None):
    """θτ0,2,5 with the raw fit; a corrected input needs only θ, τ and covariance.

    Nonpositive theory variances have no real standard deviation: retain those
    points but omit their error bars rather than presenting a zero uncertainty.
    Each catalogue uses its own pair-weighted mean separation.
    """
    with S.figure_style():
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
        for ax, k in zip(axes, (0, 2, 5)):
            theta = np.asarray(rt["theta_tau"])
            model = tau_model_samples(rt, k, rt["fit"]["flat_samples"]) * theta[None, :]
            lo, med, hi = np.percentile(model, [16, 50, 84], axis=0)
            S.zero_line(ax)
            ax.fill_between(theta, lo, hi, color=MODEL, alpha=0.3, lw=0)
            ax.plot(theta, med, color=MODEL, lw=1.1)
            inputs = [(rt, "raw", S.INK, "o")]
            if corrected is not None:
                inputs.append((corrected, "leakage-corrected", SERIES[1], "s"))
            for data, label, color, marker in inputs:
                th = np.asarray(data["theta_tau"])
                val = th * np.asarray(data["tau"][k]["p"])
                var = np.asarray(data["tau"][k]["var_p"])
                ax.plot(th, val, color=color, marker=marker, ms=3, lw=0.7, label=label)
                valid = np.isfinite(var) & (var > 0)
                ax.errorbar(
                    th[valid],
                    val[valid],
                    yerr=th[valid] * np.sqrt(var[valid]),
                    color=color,
                    fmt="none",
                    **S.DATA_VECTOR_MARK,
                )
            S.theta_axis(ax)
            ax.set_xticks([1, 10, 100])
            ax.set_xticklabels(["$1$", "$10$", "$100$"])
            ax.set_xlabel(r"$\theta$ [arcmin]")
            ax.set_ylabel(rf"$\theta\,\tau_{{{k},+}}$ [arcmin]")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        handles, labels = axes[0].get_legend_handles_labels()
        handles.append(Line2D([], [], color=MODEL, lw=4, alpha=0.4))
        labels.append(r"raw $\alpha,\beta,\eta$ posterior (68%)")
        fig.legend(handles, labels, loc="outside upper center", ncol=len(labels))
        return S.save(fig, out, "full")


def xi_sys_comparison(rt, out, xi=None):
    """Compare fitted PSF systematic, τ0²/ρ0 and optional blinded ξ+.

    ``xi`` comes from products.load_xi, including the run covariance errors.
    Its own pair-weighted angular grid is retained.
    """
    theta = np.asarray(rt["theta"])
    # Pair-weighted mean separations differ slightly within the same bins.
    if not (
        np.allclose(theta, rt["theta_tau"], rtol=0.002)
        and np.allclose(theta, rt["fit"]["theta"])
    ):
        raise ValueError("rho, tau and PSF fit angular grids differ")
    rho = np.asarray(rt["rho"][0]["p"])
    tau = np.asarray(rt["tau"][0]["p"])
    with np.errstate(divide="ignore", invalid="ignore"):
        simple = np.where(rho != 0, tau**2 / rho, np.nan)
    lo, med, hi = np.percentile(rt["fit"]["samples_plus"], [16, 50, 84], axis=0)
    with S.figure_style():
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        ax.fill_between(theta, lo, hi, color=MODEL, alpha=0.3, lw=0)
        ax.plot(theta, med, color=MODEL, label=r"$\alpha,\beta,\eta$ model")
        ax.plot(
            theta,
            simple,
            color=SERIES[1],
            marker="o",
            ms=3,
            label=r"$\tau_{0,+}^2/\rho_{0,+}$",
        )
        values = [lo, med, hi, simple]
        if xi is not None:
            xip = np.asarray(xi["xip"])
            ax.errorbar(
                xi["theta"],
                xip,
                yerr=xi["sigma_xip"],
                color=S.INK,
                marker="o",
                label=r"$\xi_+$ (blinded)",
                **S.DATA_VECTOR_MARK,
            )
            values.append(xip)
        S.theta_axis(ax)
        if np.all(np.concatenate(values) > 0):
            ax.set_yscale("log")
        else:
            S.zero_line(ax)
            ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_xlabel(r"$\theta$ [arcmin]")
        ax.set_ylabel(r"$\xi_+(\theta)$")
        ax.legend()
        return S.save(fig, out, "half")


ALPHA_ROWS = [
    ("aii_mean", r"$\langle a_{ii} \rangle$", "object-wise"),
    ("a11", r"$a_{11}$", "object-wise"),
    ("a22", r"$a_{22}$", "object-wise"),
    ("alpha_mean", r"$\bar\alpha$", "scale-dependent"),
    ("alpha_0", r"$\alpha(0)$", "scale-dependent"),
    ("alpha_1", r"$\alpha(\theta_{\min})$", "scale-dependent"),
]


def alpha_estimates(leakage, rt, out):
    """Forest plot of every PSF-leakage coefficient estimate (± 1σ)."""
    rows = []
    for key, label, group in ALPHA_ROWS:
        if key in leakage:
            value, err = leakage[key]
            rows.append((label, group, value, err, err))
    a = rt["fit"]["flat_samples"][:, 0]
    lo, med, hi = np.percentile(a, [16, 50, 84])
    rows.append((r"$\alpha$", r"$\rho/\tau$ fit", med, med - lo, hi - med))
    with S.figure_style():
        fig, ax = plt.subplots(figsize=(3.6, 2.9))
        y = np.arange(len(rows))[::-1].astype(float)
        groups = [r[1] for r in rows]
        # a small gap between estimator groups
        for i in range(1, len(rows)):
            if groups[i] != groups[i - 1]:
                y[i:] -= 0.6
        ax.axvline(0, color=S.INK, lw=0.7, ls=(0, (4, 3)), alpha=0.6)
        for yi, (label, group, v, em, ep) in zip(y, rows):
            ax.errorbar(
                [v],
                [yi],
                xerr=[[em], [ep]],
                fmt="o",
                color=S.INK,
                ms=3.6,
                elinewidth=0.9,
                capsize=1.8,
            )
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows])
        ax.tick_params(axis="y", length=0, right=False)
        ax.tick_params(axis="x", top=False)
        prev = None
        for yi, g in zip(y, groups):
            if g != prev:
                ax.text(
                    0.98,
                    yi + 0.42,
                    g,
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="bottom",
                    fontsize=7.5,
                    color=S.INK_MUTED,
                )
                prev = g
        ax.set_ylim(y.min() - 0.7, y.max() + 0.9)
        ax.set_xlabel("PSF leakage coefficient")
        return S.save(fig, out, "half")
