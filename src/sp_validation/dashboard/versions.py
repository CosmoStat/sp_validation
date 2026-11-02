"""Catalogue-version comparisons (Paper II Figs. 7-9), drawn only when a run
carries two or more versions.

Each figure overlays the versions with jittered markers in the version
palette and, per bin, a box spanning the versions' values with a bar at the
reference (first) version. ``datasets`` is a list of ``(label, products)``
with the reference first; products are the dicts of
:mod:`sp_validation.dashboard.products`.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from sp_validation.b_modes import bins_from_scale_cut

from . import style as S
from .data_vectors import XI_SCALE, _cl_cut_mask


def version_colors(n):
    """Paper II version palette: ``seaborn.husl_palette(n, l=0.4)``."""
    import seaborn as sns

    return sns.husl_palette(n, l=0.4)


def _boxes(ax, x_left, x_right, values, ref):
    """Box spanning ``values`` (versions x bins) per bin, bar at ``ref``."""
    lo, hi = values.min(0), values.max(0)
    for xl, xr, a, b, r in zip(x_left, x_right, lo, hi, ref):
        ax.add_patch(
            Rectangle(
                (xl, a),
                xr - xl,
                b - a,
                fill=False,
                edgecolor=S.INK,
                lw=0.8,
                alpha=0.75,
                zorder=1,
            )
        )
        ax.hlines(r, xl, xr, color=S.INK, lw=1.3, zorder=1)


def _series(ax, x, y, yerr, i, color, label=None):
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker=S.VERSION_MARKERS[i % 4],
        color=color,
        markerfacecolor="none",
        markeredgecolor=color,
        label=label,
        zorder=2,
        **S.DATA_VECTOR_MARK,
    )


def _log_offsets(n, spread=0.05):
    return np.exp(spread * (np.arange(n) - (n - 1) / 2))


def pure_eb(datasets, out, scale_cut):
    """Total θξ± (top) and ξ±^B/σ (bottom) across versions."""
    n = len(datasets)
    colors = version_colors(n)
    off = _log_offsets(n)
    pad = np.exp(0.05 * (n - 1) / 2 + 0.04)
    ref = datasets[0][1]
    theta = ref["theta"]
    start, stop = bins_from_scale_cut(ref["left_edges"], ref["right_edges"], scale_cut)
    shade = (ref["left_edges"][start], ref["right_edges"][stop - 1])
    with S.figure_style():
        fig, axes = plt.subplots(
            2, 2, figsize=(7.2, 4.0), sharex=True, height_ratios=(3, 2)
        )
        for col, xi in enumerate(("xip", "xim")):
            top = np.array([theta * d[xi] / XI_SCALE for _, d in datasets])
            bot = np.array([d[f"{xi}_B"] / d[f"sigma_{xi}_B"] for _, d in datasets])
            _boxes(axes[0, col], theta / pad, theta * pad, top, top[0])
            _boxes(axes[1, col], theta / pad, theta * pad, bot, bot[0])
            for i, (label, d) in enumerate(datasets):
                _series(
                    axes[0, col],
                    theta * off[i],
                    top[i],
                    theta * d[f"sigma_{xi}"] / XI_SCALE,
                    i,
                    colors[i],
                    label if col == 0 else None,
                )
                _series(
                    axes[1, col],
                    theta * off[i],
                    bot[i],
                    np.ones_like(theta),
                    i,
                    colors[i],
                )
            sign = "+" if xi == "xip" else "-"
            S.label_box(axes[0, col], rf"$\xi_{sign}$")
            S.zero_line(axes[1, col])
            for ax in axes[:, col]:
                S.theta_axis(ax)
                S.shade_outside(ax, *shade)
            axes[1, col].set_xticks([1, 10, 100])
            axes[1, col].set_xticklabels(["$1$", "$10$", "$100$"])
            axes[1, col].set_xlabel(r"$\theta$ [arcmin]")
        axes[0, 0].set_ylabel(r"$\theta\,\xi \times 10^{4}$")
        axes[1, 0].set_ylabel(r"$\xi^B / \sigma$")
        for row in range(2):
            axes[row, 1].sharey(axes[row, 0])
            axes[row, 1].tick_params(labelleft=False)
        fig.legend(loc="outside upper center", ncol=min(n, 4))
        return S.save(fig, out, "full")


def cosebis(datasets, out, full_cut=(1.0, 250.0)):
    """B_n/σ_n across versions, all scales (top) and fiducial cut (bottom)."""
    n = len(datasets)
    colors = version_colors(n)
    offsets = 0.13 * (np.arange(n) - (n - 1) / 2)
    half = 0.13 * (n - 1) / 2 + 0.12
    fid = datasets[0][1]["fiducial"]
    nmodes = datasets[0][1]["nmodes"]
    modes = np.arange(1, nmodes + 1)
    with S.figure_style():
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=True, sharey=True)
        for ax, cut in zip(axes, (full_cut, fid)):
            vals = np.array(
                [d["cuts"][cut]["Bn"] / d["cuts"][cut]["sigma_B"] for _, d in datasets]
            )
            _boxes(ax, modes - half, modes + half, vals, vals[0])
            for i, (label, _) in enumerate(datasets):
                _series(
                    ax,
                    modes + offsets[i],
                    vals[i],
                    np.ones(nmodes),
                    i,
                    colors[i],
                    label if cut == full_cut else None,
                )
            S.zero_line(ax)
            ax.set_ylabel(r"$B_n / \sigma_n$")
            S.header(ax, rf"${cut[0]:g}' < \theta < {cut[1]:g}'$", side="right")
        axes[-1].set_xlim(0.4, nmodes + 0.6)
        axes[-1].set_xticks(modes)
        axes[-1].set_xlabel(r"COSEBI mode $n$")
        fig.legend(loc="outside upper center", ncol=min(n, 4))
        return S.save(fig, out, "full")


def cl(datasets, out, ell_cut):
    """C_ℓ^BB/σ and C_ℓ^EB/σ across versions."""
    n = len(datasets)
    colors = version_colors(n)
    ref = datasets[0][1]
    ell = ref["ell"]
    width = ref["ell_hi"] - ref["ell_lo"] + 1
    offsets = 0.16 * (np.arange(n) - (n - 1) / 2)
    half = (0.16 * (n - 1) / 2 + 0.1) * width
    mask = _cl_cut_mask(ref, ell_cut)
    with S.figure_style():
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=True, sharey=True)
        for ax, key in zip(axes, ("BB", "EB")):
            vals = np.array([d[key] / d[f"sigma_{key}"] for _, d in datasets])
            _boxes(ax, ell - half, ell + half, vals, vals[0])
            for i, (label, _) in enumerate(datasets):
                _series(
                    ax,
                    ell + offsets[i] * width,
                    vals[i],
                    np.ones_like(ell),
                    i,
                    colors[i],
                    label if key == "BB" else None,
                )
            S.zero_line(ax)
            ax.set_xscale("function", functions=(np.sqrt, np.square))
            ax.set_xlim(ell.min() * 0.85, ell.max() * 1.04)
            S.shade_outside(ax, ref["ell_lo"][mask][0], ref["ell_hi"][mask][-1])
            ax.set_ylabel(rf"$C_\ell^{{{key}}} / \sigma$")
            S.label_box(ax, rf"$C_\ell^{{{key}}}$")
        axes[-1].set_xticks([100, 400, 900, 1600])
        axes[-1].set_xticklabels(["$100$", "$400$", "$900$", "$1600$"])
        axes[-1].set_xlabel(r"$\ell$")
        fig.legend(loc="outside upper center", ncol=min(n, 4))
        return S.save(fig, out, "full")
