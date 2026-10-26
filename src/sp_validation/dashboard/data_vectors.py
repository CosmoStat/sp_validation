"""The correlation-function hero figure and the three B-mode data vectors.

Each function draws one figure from loaded products (see
:mod:`sp_validation.dashboard.products`) and writes it to ``out``. The B-mode
figures follow Paper II's single-data-vector figures (pure E/B Fig. 2,
COSEBIs Fig. 3, pseudo-C_ell Fig. 4): same estimator-family hues, marker
fills, normalisation and grey shading of excluded scales. For the dashboard's
row of three they share one layout -- two stacked panels on a common x axis
-- and carry their null-test PTEs in the panel headers.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

from sp_validation.b_modes import bins_from_scale_cut

from . import style as S

XI_SCALE = 1e-4
THIRD_HEIGHT = 3.3


def _theta_ticks(ax):
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(["$1$", "$10$", "$100$"])


def _pte_header(ax, text):
    ax.set_title(text, loc="right", fontsize=8.5, color=S.INK_MUTED, pad=3)


def xi_pm(xi, out, layout="wide", blind=None):
    """Blinded reporting-grid ξ± (θξ × 10⁴) with analytic error bars.

    ``layout="wide"`` puts ξ+ and ξ− side by side for the page column;
    ``"narrow"`` stacks them for phones.
    """
    with S.figure_style():
        if layout == "wide":
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
            size = "full"
        else:
            fig, axes = plt.subplots(2, 1, figsize=(3.6, 4.6), sharex=True)
            size = "half"
        theta = xi["theta"]
        for ax, key, label in zip(axes, ("xip", "xim"), (r"$\xi_+$", r"$\xi_-$")):
            S.zero_line(ax)
            ax.errorbar(
                theta,
                theta * xi[key] / XI_SCALE,
                yerr=theta * xi[f"sigma_{key}"] / XI_SCALE,
                marker="o",
                color=S.COLOR_TOTAL,
                markerfacecolor=S.COLOR_TOTAL,
                **{**S.DATA_VECTOR_MARK, "markersize": 3.8},
            )
            S.theta_axis(ax)
            _theta_ticks(ax)
            S.label_box(ax, label, loc="upper left", fontsize=11)
        for ax in axes if layout == "wide" else axes[-1:]:
            ax.set_xlabel(r"$\theta$ [arcmin]")
        for ax in axes[:1] if layout == "wide" else axes:
            ax.set_ylabel(r"$\theta\,\xi \times 10^{4}$")
        lo = min(np.min(theta * (xi[k] - xi[f"sigma_{k}"])) for k in ("xip", "xim"))
        hi = max(np.max(theta * (xi[k] + xi[f"sigma_{k}"])) for k in ("xip", "xim"))
        pad = 0.08 * (hi - lo)
        axes[0].set_ylim((lo - pad) / XI_SCALE, (hi + 2.2 * pad) / XI_SCALE)
        if blind:
            fig.suptitle(" ", fontsize=7)
            S.blind_stamp(fig, blind)
        return S.save(fig, out, size)


def _jitter(theta, i, n, spread=0.07):
    return theta * np.exp(spread * (i - (n - 1) / 2))


def pure_eb(pe, out, scale_cut, ptes, blind=None):
    """Pure E / B / ambiguous decomposition of ξ± (Paper II Fig. 2), stacked."""
    series = [
        ("", "o", S.COLOR_TOTAL, 1.0, True, "total"),
        ("_E", "s", S.COLOR_E, 0.5, True, "$E$"),
        ("_B", "X", S.COLOR_B, 1.0, True, "$B$"),
        ("_amb", "v", S.COLOR_AMB, 0.55, True, "amb."),
    ]
    with S.figure_style():
        fig, axes = plt.subplots(2, 1, figsize=(3.0, THIRD_HEIGHT), sharex=True)
        theta = pe["theta"]
        start, stop = bins_from_scale_cut(pe["left_edges"], pe["right_edges"],
                                          scale_cut)
        shade = (pe["left_edges"][start], pe["right_edges"][stop - 1])
        values = []
        for ax, xi, sign in zip(axes, ("xip", "xim"), "+-"):
            S.zero_line(ax)
            for i, (suffix, marker, color, alpha, filled, _) in enumerate(series):
                y = theta * pe[f"{xi}{suffix}"] / XI_SCALE
                dy = theta * pe[f"sigma_{xi}{suffix}"] / XI_SCALE
                values.append((y - dy, y + dy))
                ax.errorbar(
                    _jitter(theta, i, len(series)),
                    y,
                    yerr=dy,
                    marker=marker,
                    color=color,
                    alpha=alpha,
                    markerfacecolor=color if filled else "none",
                    **S.DATA_VECTOR_MARK,
                )
            S.theta_axis(ax)
            _theta_ticks(ax)
            S.shade_outside(ax, *shade)
            ax.set_ylabel(r"$\theta\,\xi \times 10^{4}$")
            S.label_box(ax, rf"$\xi_{sign}$", loc="upper left")
            _pte_header(
                ax, rf"$\xi_{sign}^B$ PTE ${S.fmt_pte(ptes.get(xi + '_B'))}$"
            )
        axes[-1].set_xlabel(r"$\theta$ [arcmin]")
        # Shared y range from the in-cut points, so the excluded large-scale
        # ambiguous modes don't set the scale.
        inside = np.zeros(len(theta), bool)
        inside[start:stop] = True
        lo = min(v[0][inside].min() for v in values)
        hi = max(v[1][inside].max() for v in values)
        pad = 0.12 * (hi - lo)
        for ax in axes:
            ax.set_ylim(lo - pad, hi + 2.5 * pad)
        handles = [
            Line2D([], [], ls="none", marker=m, color=c, alpha=a,
                   markersize=4, label=lab)
            for _, m, c, a, _, lab in series
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=4,
                   handletextpad=0.1, columnspacing=0.9, borderaxespad=0.1)
        if "combined" in ptes:
            fig.text(0.99, 0.003, rf"joint $\xi_\pm^B$ PTE "
                     rf"${S.fmt_pte(ptes['combined'])}$", ha="right", va="bottom",
                     fontsize=8, color=S.INK_MUTED)
        if blind:
            S.blind_stamp(fig, blind, where="bottom")
        return S.save(fig, out, "third")


def cosebis(co, out, full_cut=(1.0, 250.0)):
    """B-mode COSEBIs in units of their uncertainty (Paper II Fig. 3).

    Fiducial cuts (filled, COSEBIs purple) over all scales (open, light tint),
    each in its own panel.
    """
    fid = co["fiducial"]
    rows = [
        (fid, "o", True, S.COLOR_COSEBIS),
        (full_cut, "D", False, S.COLOR_COSEBIS_ALL),
    ]
    n = np.arange(1, co["nmodes"] + 1)
    with S.figure_style():
        fig, axes = plt.subplots(2, 1, figsize=(3.0, THIRD_HEIGHT), sharex=True,
                                 sharey=True)
        extent = 0
        for ax, (cut, marker, filled, color) in zip(axes, rows):
            r = co["cuts"][cut]
            y = r["Bn"] / r["sigma_B"]
            extent = max(extent, np.max(np.abs(y)) + 1)
            S.zero_line(ax)
            ax.errorbar(
                n, y, yerr=np.ones_like(y), marker=marker, color=color,
                markerfacecolor=color if filled else "none",
                markeredgecolor=color, **S.DATA_VECTOR_MARK,
            )
            ax.set_ylabel(r"$B_n / \sigma_n$")
            S.label_box(ax, rf"${cut[0]:g}' < \theta < {cut[1]:g}'$",
                        loc="upper left", fontsize=8.5)
            _pte_header(ax, rf"$B_n$ PTE ${S.fmt_pte(r['pte_B'])}$")
        lim = np.ceil(extent + 0.6)
        axes[0].set_ylim(-lim, lim + 1.2)
        axes[0].yaxis.set_major_locator(plt.MultipleLocator(2))
        axes[-1].set_xlim(0.4, n[-1] + 0.6)
        axes[-1].set_xticks(n[::2] if len(n) > 12 else n)
        axes[-1].set_xlabel(r"COSEBI mode $n$")
        _spacer_legend(fig)
        return S.save(fig, out, "third")


def _spacer_legend(fig):
    """Reserve the legend band the pure E/B figure uses, keeping rows aligned."""
    fig.legend(handles=[Line2D([], [], ls="none", label=" ")],
               loc="outside upper center", handlelength=0)


def _cl_cut_mask(cl, ell_cut):
    return (cl["ell_lo"] >= ell_cut[0]) & (cl["ell_hi"] <= ell_cut[1])


def cl_pte(cl, key, ell_cut=None):
    """χ² PTE of one spectrum against zero, over bins inside ``ell_cut``."""
    mask = np.ones(len(cl["ell"]), bool) if ell_cut is None else _cl_cut_mask(
        cl, ell_cut)
    d = cl[key][mask]
    cov = cl["cov"][f"{key}_{key}"][np.ix_(mask, mask)]
    return float(stats.chi2.sf(d @ np.linalg.solve(cov, d), mask.sum()))


def pseudo_cl(cl, out, ell_cut):
    """C_ℓ^BB and C_ℓ^EB in units of their uncertainty (Paper II Fig. 4)."""
    rows = [
        ("BB", "o", True, S.COLOR_CL_BB, r"$C_\ell^{BB}$"),
        ("EB", "D", False, S.COLOR_CL_EB, r"$C_\ell^{EB}$"),
    ]
    ell = cl["ell"]
    with S.figure_style():
        fig, axes = plt.subplots(2, 1, figsize=(3.0, THIRD_HEIGHT), sharex=True,
                                 sharey=True)
        extent = 0
        for ax, (key, marker, filled, color, label) in zip(axes, rows):
            y = cl[key] / cl[f"sigma_{key}"]
            extent = max(extent, np.max(np.abs(y)) + 1)
            S.zero_line(ax)
            ax.errorbar(
                ell, y, yerr=np.ones_like(y), marker=marker, color=color,
                markerfacecolor=color if filled else "none",
                markeredgecolor=color, **S.DATA_VECTOR_MARK,
            )
            ax.set_xscale("function", functions=(np.sqrt, np.square))
            ax.set_ylabel(r"$C_\ell / \sigma$")
            S.label_box(ax, label, loc="upper left")
            _pte_header(
                ax,
                rf"PTE ${S.fmt_pte(cl_pte(cl, key, ell_cut))}$"
                rf" (all $\ell$: ${S.fmt_pte(cl_pte(cl, key))}$)",
            )
        mask = _cl_cut_mask(cl, ell_cut)
        lo = cl["ell_lo"][mask][0]
        hi = cl["ell_hi"][mask][-1]
        for ax in axes:
            ax.set_xlim(ell.min() * 0.85, ell.max() * 1.04)
            S.shade_outside(ax, lo, hi)
            ax.set_xticks([100, 400, 900, 1600])
            ax.set_xticklabels(["$100$", "$400$", "$900$", "$1600$"])
            ax.set_xticks([i * 100 for i in range(1, 21)], minor=True)
        lim = np.ceil(extent + 0.6)
        axes[0].set_ylim(-lim, lim + 1.2)
        axes[0].yaxis.set_major_locator(plt.MultipleLocator(2))
        axes[-1].set_xlabel(r"$\ell$")
        _spacer_legend(fig)
        return S.save(fig, out, "third")
