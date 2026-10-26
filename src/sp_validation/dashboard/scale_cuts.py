"""B-mode PTEs as a function of the scale cut (Paper II Figs. 8-9 style).

Each heatmap cell is the PTE of the B-mode null test over the bins between a
lower cut (x) and an upper cut (y); the fiducial cut is outlined. The colour
map is anchored at the PTE threshold: solid blue below it, solid red above
one minus it, a smooth gradient between.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import stats

from sp_validation.b_modes import bins_from_scale_cut

from . import style as S


def cl_pte_matrix(cl, key="BB"):
    """PTE of ``key`` against zero for every contiguous band range [i, j]."""
    data = cl[key]
    cov = cl["cov"][f"{key}_{key}"]
    n = len(data)
    pte = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            d = data[i : j + 1]
            c = cov[i : j + 1, i : j + 1]
            pte[i, j] = stats.chi2.sf(d @ np.linalg.solve(c, d), j - i + 1)
    return pte


def _tick_subset(n, k=6):
    return np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))


def _heatmap(ax, pte, lo_labels, hi_labels, fid, label, cmap, norm):
    """Triangular PTE heatmap on bin indices; ``fid`` is ``(i, j)`` inclusive."""
    n = pte.shape[0]
    grid = np.arange(n + 1) - 0.5
    mesh = ax.pcolormesh(grid, grid, pte.T, cmap=cmap, norm=norm,
                         edgecolors="face", linewidth=0, rasterized=True)
    ax.set_facecolor("#e4dccb")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ticks = _tick_subset(n)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"${lo_labels[t]}$" for t in ticks], rotation=45,
                       ha="right", rotation_mode="anchor")
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"${hi_labels[t]}$" for t in ticks])
    ax.tick_params(which="both", top=False, right=False, length=2.5)
    if fid is not None:
        i, j = fid
        ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False,
                               edgecolor=S.INK, lw=1.6, zorder=5))
    S.label_box(ax, label, loc="lower right", fontsize=9)
    return mesh


def _fmt_angle(v):
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}".rstrip("0").rstrip(".")


def config_space(pe, out, scale_cut, layout="wide"):
    """ξ+^B, ξ−^B and joint ξ±^B PTEs over all reporting-bin scale cuts."""
    start, stop = bins_from_scale_cut(pe["left_edges"], pe["right_edges"],
                                      scale_cut)
    fid = (start, stop - 1)
    lo = [_fmt_angle(v) for v in pe["left_edges"]]
    hi = [_fmt_angle(v) for v in pe["right_edges"]]
    panels = [
        ("xip_B", r"$\xi_+^B$"),
        ("xim_B", r"$\xi_-^B$"),
        ("combined", r"$\xi_\pm^B$ joint"),
    ]
    cmap, norm = S.pte_cmap(), S.pte_norm()
    with S.figure_style():
        if layout == "wide":
            fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True)
        else:
            fig, axes = plt.subplots(3, 1, figsize=(3.6, 9.6), sharex=True)
        for ax, (key, label) in zip(axes, panels):
            mesh = _heatmap(ax, pe["pte_matrices"][key], lo, hi, fid, label,
                            cmap, norm)
        for ax in axes if layout == "wide" else axes[-1:]:
            ax.set_xlabel(r"$\theta_{\min}$ [arcmin]")
        for ax in axes[:1] if layout == "wide" else axes:
            ax.set_ylabel(r"$\theta_{\max}$ [arcmin]")
        S.pte_colorbar(fig, mesh, axes, shrink=0.9 if layout == "wide" else 0.5,
                       aspect=25, pad=0.02,
                       location="right" if layout == "wide" else "bottom")
        return S.save(fig, out, "full" if layout == "wide" else "half")


def harmonic_and_cosebis(cl, co, out, ell_cut, layout="wide"):
    """C_ℓ^BB PTEs over band ranges, beside COSEBIs B_n PTEs per cached cut."""
    pte = cl_pte_matrix(cl, "BB")
    inside = np.flatnonzero((cl["ell_lo"] >= ell_cut[0]) & (cl["ell_hi"] <= ell_cut[1]))
    fid = (inside[0], inside[-1])
    lo = [f"{v:.0f}" for v in cl["ell_lo"]]
    hi = [f"{v:.0f}" for v in cl["ell_hi"]]
    cmap, norm = S.pte_cmap(), S.pte_norm()
    with S.figure_style():
        if layout == "wide":
            fig, (ax_cl, ax_co) = plt.subplots(
                1, 2, figsize=(7.2, 3.0), width_ratios=(1, 1.15))
        else:
            fig, (ax_cl, ax_co) = plt.subplots(2, 1, figsize=(3.6, 6.6),
                                               height_ratios=(1.25, 1))
        mesh = _heatmap(ax_cl, pte, lo, hi, fid, r"$C_\ell^{BB}$", cmap, norm)
        ax_cl.set_xlabel(r"$\ell_{\min}$")
        ax_cl.set_ylabel(r"$\ell_{\max}$")
        _cosebis_cuts(ax_co, co, cmap, norm)
        S.pte_colorbar(fig, mesh, [ax_cl, ax_co], aspect=25, pad=0.02,
                       shrink=0.9 if layout == "wide" else 0.6,
                       location="right" if layout == "wide" else "bottom")
        return S.save(fig, out, "full" if layout == "wide" else "half")


def _cosebis_cuts(ax, co, cmap, norm, threshold=S.PTE_THRESHOLD):
    """One row per cached COSEBIs cut: the B_n PTE on a [0, 1] axis."""
    cuts = sorted(co["cuts"], key=lambda c: (c[1] - c[0], c[0]), reverse=True)
    y = np.arange(len(cuts))[::-1]
    ax.axvspan(0, threshold, color=cmap(0), alpha=0.18, lw=0)
    ax.axvspan(1 - threshold, 1, color=cmap(cmap.N - 1), alpha=0.18, lw=0)
    for yi, cut in zip(y, cuts):
        p = co["cuts"][cut]["pte_B"]
        is_fid = cut == co["fiducial"]
        ax.hlines(yi, 0, 1, color=S.RULE, lw=0.6, zorder=1)
        ax.scatter([p], [yi], s=46 if is_fid else 30, color=cmap(norm(p)),
                   edgecolor=S.INK, linewidth=1.4 if is_fid else 0.6, zorder=3,
                   marker="s" if is_fid else "o")
        ax.text(min(p + 0.035, 0.97), yi + 0.18, f"${S.fmt_pte(p)}$",
                fontsize=7.5, color=S.INK_MUTED, va="bottom",
                ha="left" if p < 0.85 else "right")
    ax.set_yticks(y)
    ax.set_yticklabels([rf"${c[0]:g}'\!-\!{c[1]:g}'$" for c in cuts])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, len(cuts) - 0.3)
    ticks = (threshold, 0.5, 1 - threshold)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"${t:g}$" for t in ticks])
    ax.tick_params(which="both", top=False, right=False, left=False)
    ax.set_xlabel(r"$B_n$ PTE ($n \leq %d$)" % co["nmodes"])
    S.label_box(ax, "COSEBIs", loc="lower right", fontsize=9)
