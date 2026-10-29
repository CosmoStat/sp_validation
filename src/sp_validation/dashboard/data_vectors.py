"""Data-vector figures: the ξ± and C_ℓ two-point functions, and the three
B-mode data vectors as Paper II draws them.

Each function draws one figure from loaded products (see
:mod:`sp_validation.dashboard.products`) and writes it to ``out``. The B-mode
figures reproduce Paper II's single-data-vector figures -- pure E/B (Fig. 2),
COSEBIs (Fig. 3), C_ℓ BB/EB (Fig. 4) -- with the same estimator-family hues,
marker fills, normalisation, and grey shading of what lies outside the
fiducial selection; their PTEs live in the PTE table, not in the figures.
``layout="narrow"`` stacks multi-panel figures for phones.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from sp_validation.b_modes import bins_from_scale_cut

from . import style as S

XI_SCALE = 1e-4


def _legend(fig, handles, ncol):
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=ncol,
        handletextpad=0.2,
        columnspacing=1.0,
        borderaxespad=0.2,
    )


def _marker(marker, color, label, filled=True, alpha=1.0):
    return Line2D(
        [],
        [],
        ls="none",
        marker=marker,
        color=color,
        alpha=alpha,
        markerfacecolor=color if filled else "none",
        markeredgecolor=color,
        markersize=4,
        markeredgewidth=0.8,
        label=label,
    )


def xi_pm(xi, out, layout="wide", blind=None):
    """Blinded reporting-grid θξ± with analytic error bars."""
    with S.figure_style():
        if layout == "wide":
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
        else:
            fig, axes = plt.subplots(2, 1, figsize=(3.6, 4.6), sharex=True)
        theta = xi["theta"]
        for ax, key, sign in zip(axes, ("xip", "xim"), "+-"):
            S.zero_line(ax)
            ax.errorbar(
                theta,
                theta * xi[key] / XI_SCALE,
                yerr=theta * xi[f"sigma_{key}"] / XI_SCALE,
                marker="o",
                color=S.COLOR_TOTAL,
                **{**S.DATA_VECTOR_MARK, "markersize": 3.8},
            )
            S.theta_axis(ax)
            S.theta_ticks(ax)
            S.label_box(ax, rf"$\xi_{sign}$")
        for ax in axes if layout == "wide" else axes[-1:]:
            ax.set_xlabel(r"$\theta$ [arcmin]")
        for ax in axes[:1] if layout == "wide" else axes:
            ax.set_ylabel(r"$\theta\,\xi_\pm \times 10^{4}$")
        lo = min(np.min(theta * (xi[k] - xi[f"sigma_{k}"])) for k in ("xip", "xim"))
        hi = max(np.max(theta * (xi[k] + xi[f"sigma_{k}"])) for k in ("xip", "xim"))
        pad = 0.08 * (hi - lo)
        axes[0].set_ylim((lo - pad) / XI_SCALE, (hi + 2.2 * pad) / XI_SCALE)
        if blind:
            fig.suptitle(" ", fontsize=7)
            S.blind_stamp(fig, blind)
        return S.save(fig, out, "full" if layout == "wide" else "half")


def _cl_cut_mask(cl, ell_cut):
    return (cl["ell_lo"] >= ell_cut[0]) & (cl["ell_hi"] <= ell_cut[1])


def _cl_shade(cl, ell_cut):
    mask = _cl_cut_mask(cl, ell_cut)
    return cl["ell_lo"][mask][0], cl["ell_hi"][mask][-1]


def ell_cl(cl, out, ell_cut, layout="wide", blind=None):
    """ℓ C_ℓ for EE, BB and EB on a linear ℓ axis (EE blinded)."""
    ell = cl["ell"]
    series = [
        ("EE", "o", True, S.COLOR_TOTAL, r"$EE$"),
        ("BB", "o", True, S.COLOR_CL_BB, r"$BB$"),
        ("EB", "D", False, S.COLOR_CL_EB, r"$EB$"),
    ]
    top = np.max(ell * (cl["EE"] + cl["sigma_EE"]))
    p = int(np.floor(np.log10(top)))
    scale = 10.0**p
    size = (7.2, 2.7) if layout == "wide" else (3.6, 2.7)
    with S.figure_style():
        fig, ax = plt.subplots(figsize=size)
        S.zero_line(ax)
        bandwidth = cl["ell_hi"] - cl["ell_lo"] + 1
        for (key, marker, filled, color, _), off in zip(series, (-0.18, 0.0, 0.18)):
            ax.errorbar(
                ell + off * bandwidth,
                ell * cl[key] / scale,
                yerr=ell * cl[f"sigma_{key}"] / scale,
                marker=marker,
                color=color,
                markerfacecolor=color if filled else "none",
                markeredgecolor=color,
                **S.DATA_VECTOR_MARK,
            )
        ax.set_xlim(0, cl["ell_hi"][-1] + 30)
        S.shade_outside(ax, *_cl_shade(cl, ell_cut))
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(rf"$\ell\, C_\ell \times 10^{{{-p}}}$")
        S.label_box(ax, r"$C_\ell$")
        _legend(fig, [_marker(m, c, lab, f) for _, m, f, c, lab in series], 3)
        if blind:
            S.blind_stamp(fig, blind, where="bottom")
        return S.save(fig, out, "full" if layout == "wide" else "half")


def _jitter(theta, i, n, spread=0.07):
    return theta * np.exp(spread * (i - (n - 1) / 2))


PURE_EB_SERIES = [
    ("", "o", S.COLOR_TOTAL, 1.0, r"$\xi_\pm$ (total)"),
    ("_E", "s", S.COLOR_E, 0.5, r"$\xi_\pm^E$"),
    ("_B", "X", S.COLOR_B, 1.0, r"$\xi_\pm^B$"),
    ("_amb", "v", S.COLOR_AMB, 0.55, r"$\xi_\pm^{\rm amb}$"),
]


def pure_eb(pe, out, scale_cut, layout="wide", blind=None):
    """Pure E / B / ambiguous decomposition of θξ± (Paper II Fig. 2)."""
    theta = pe["theta"]
    start, stop = bins_from_scale_cut(pe["left_edges"], pe["right_edges"], scale_cut)
    shade = (pe["left_edges"][start], pe["right_edges"][stop - 1])
    with S.figure_style():
        if layout == "wide":
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
        else:
            fig, axes = plt.subplots(2, 1, figsize=(3.6, 4.6), sharex=True, sharey=True)
        bounds = []
        for ax, xi, sign in zip(axes, ("xip", "xim"), "+-"):
            S.zero_line(ax)
            for i, (suffix, marker, color, alpha, _) in enumerate(PURE_EB_SERIES):
                y = theta * pe[f"{xi}{suffix}"] / XI_SCALE
                dy = theta * pe[f"sigma_{xi}{suffix}"] / XI_SCALE
                bounds.append((y - dy)[start:stop].min())
                bounds.append((y + dy)[start:stop].max())
                ax.errorbar(
                    _jitter(theta, i, len(PURE_EB_SERIES)),
                    y,
                    yerr=dy,
                    marker=marker,
                    color=color,
                    alpha=alpha,
                    **S.DATA_VECTOR_MARK,
                )
            S.theta_axis(ax)
            S.theta_ticks(ax)
            S.shade_outside(ax, *shade)
            S.label_box(ax, rf"$\xi_{sign}$")
        for ax in axes if layout == "wide" else axes[-1:]:
            ax.set_xlabel(r"$\theta$ [arcmin]")
        for ax in axes[:1] if layout == "wide" else axes:
            ax.set_ylabel(r"$\theta\,\xi \times 10^{4}$")
        # The in-cut points set the range, as in Paper II.
        lo, hi = min(bounds), max(bounds)
        pad = 0.12 * (hi - lo)
        axes[0].set_ylim(lo - pad, hi + 2.2 * pad)
        _legend(
            fig, [_marker(m, c, lab, alpha=a) for _, m, c, a, lab in PURE_EB_SERIES], 4
        )
        if blind:
            S.blind_stamp(fig, blind, where="bottom")
        return S.save(fig, out, "full" if layout == "wide" else "half")


def cosebis(co, out, full_cut=(1.0, 250.0), n_subset=6):
    """B-mode COSEBIs in units of their uncertainty (Paper II Fig. 3).

    Fiducial cuts filled in the COSEBIs purple, the full range open in its
    tint; modes beyond the ``n <= n_subset`` subset are shaded, the way scales
    outside the fiducial cuts are shaded elsewhere.
    """
    fid = co["fiducial"]
    rows = [
        (fid, -0.15, "o", True, S.COLOR_COSEBIS, "fiducial"),
        (full_cut, 0.15, "D", False, S.COLOR_COSEBIS_ALL, "full range"),
    ]
    n = np.arange(1, co["nmodes"] + 1)
    with S.figure_style():
        fig, ax = plt.subplots(figsize=(3.6, 2.45))
        extent = 0
        handles = []
        for cut, off, marker, filled, color, name in rows:
            r = co["cuts"][cut]
            y = r["Bn"] / r["sigma_B"]
            extent = max(extent, np.max(np.abs(y)) + 1)
            ax.errorbar(
                n + off,
                y,
                yerr=np.ones_like(y),
                marker=marker,
                color=color,
                markerfacecolor=color if filled else "none",
                markeredgecolor=color,
                **S.DATA_VECTOR_MARK,
            )
            handles.append(
                _marker(marker, color, rf"{name} ${cut[0]:g}'\!-\!{cut[1]:g}'$", filled)
            )
        S.zero_line(ax)
        ax.set_xlim(0.4, n[-1] + 0.6)
        ax.axvspan(n_subset + 0.5, n[-1] + 0.6, **S.EXCLUDED_SHADE)
        ax.set_xticks(n)
        ax.set_xticklabels([f"${i}$" if i % 2 == 0 else "" for i in n])
        ax.set_xlabel(r"COSEBI mode $n$")
        ax.set_ylabel(r"$B_n / \sigma_n$")
        lim = np.ceil(extent + 0.4)
        ax.set_ylim(-lim, lim)
        ax.yaxis.set_major_locator(plt.MultipleLocator(2))
        S.label_box(ax, "COSEBIs")
        _legend(fig, handles, 2)
        return S.save(fig, out, "half")


def cl_bmodes(cl, out, ell_cut):
    """C_ℓ^BB and C_ℓ^EB in units of their uncertainty (Paper II Fig. 4)."""
    rows = [
        ("BB", "o", True, S.COLOR_CL_BB, r"$C_\ell^{BB}$"),
        ("EB", "D", False, S.COLOR_CL_EB, r"$C_\ell^{EB}$"),
    ]
    ell = cl["ell"]
    with S.figure_style():
        fig, axes = plt.subplots(2, 1, figsize=(3.6, 2.9), sharex=True, sharey=True)
        extent = 0
        for ax, (key, marker, filled, color, label) in zip(axes, rows):
            y = cl[key] / cl[f"sigma_{key}"]
            extent = max(extent, np.max(np.abs(y)) + 1)
            S.zero_line(ax)
            ax.errorbar(
                ell,
                y,
                yerr=np.ones_like(y),
                marker=marker,
                color=color,
                markerfacecolor=color if filled else "none",
                markeredgecolor=color,
                **S.DATA_VECTOR_MARK,
            )
            ax.set_xscale("function", functions=(np.sqrt, np.square))
            ax.set_ylabel(rf"{label}$/\sigma$")
            S.label_box(ax, label)
        for ax in axes:
            ax.set_xlim(ell.min() * 0.85, cl["ell_hi"][-1] * 1.02)
            S.shade_outside(ax, *_cl_shade(cl, ell_cut))
            ax.set_xticks([100, 400, 900, 1600])
            ax.set_xticklabels(["$100$", "$400$", "$900$", "$1600$"])
            ax.set_xticks([i * 100 for i in range(1, 21)], minor=True)
        lim = np.ceil(extent + 0.4)
        axes[0].set_ylim(-lim, lim)
        axes[0].yaxis.set_major_locator(plt.MultipleLocator(2))
        axes[-1].set_xlabel(r"$\ell$")
        return S.save(fig, out, "half")
