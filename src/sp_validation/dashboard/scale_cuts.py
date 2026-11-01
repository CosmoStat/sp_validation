"""B-mode PTEs as a function of the scale cut (Paper II Figs. 8-9).

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
    mesh = ax.pcolormesh(
        grid,
        grid,
        pte.T,
        cmap=cmap,
        norm=norm,
        edgecolors="face",
        linewidth=0,
        rasterized=True,
    )
    ax.set_facecolor("#e4dccb")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ticks = _tick_subset(n)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [f"${lo_labels[t]}$" for t in ticks],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"${hi_labels[t]}$" for t in ticks])
    ax.tick_params(which="both", top=False, right=False, length=2.5)
    if fid is not None:
        i, j = fid
        ax.add_patch(
            Rectangle(
                (i - 0.5, j - 0.5), 1, 1, fill=False, edgecolor=S.INK, lw=1.6, zorder=5
            )
        )
    S.label_box(ax, label, loc="lower right")
    return mesh


def _fmt_angle(v):
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}".rstrip("0").rstrip(".")


def config_space(pe, cg, out, scale_cut, layout="wide"):
    """ξ+^B, ξ−^B and COSEBIs B_n (n ≤ 6) PTEs over all reporting-bin scale cuts.

    ``cg`` is the COSEBIs grid product (:mod:`.cosebis_grid`); without it the
    COSEBIs panel is left out.
    """
    lo = [_fmt_angle(v) for v in pe["left_edges"]]
    hi = [_fmt_angle(v) for v in pe["right_edges"]]
    start, stop = bins_from_scale_cut(pe["left_edges"], pe["right_edges"], scale_cut)
    panels = [
        (pe["pte_matrices"]["xip_B"], (start, stop - 1), r"$\xi_+^B$"),
        (pe["pte_matrices"]["xim_B"], (start, stop - 1), r"$\xi_-^B$"),
    ]
    if cg is not None:
        c0, c1 = bins_from_scale_cut(cg["left_edges"], cg["right_edges"], scale_cut)
        panels.append((cg["pte_B_n6"], (c0, c1 - 1), r"COSEBIs $B_n$"))
    cmap, norm = S.pte_cmap(), S.pte_norm()
    n = len(panels)
    with S.figure_style():
        if layout == "wide":
            fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.75), sharey=True)
        else:
            fig, axes = plt.subplots(n, 1, figsize=(3.6, 3.2 * n), sharex=True)
        for ax, (pte, fid, label) in zip(axes, panels):
            mesh = _heatmap(ax, pte, lo, hi, fid, label, cmap, norm)
        for ax in axes if layout == "wide" else axes[-1:]:
            ax.set_xlabel(r"$\theta_{\min}$ [arcmin]")
        for ax in axes[:1] if layout == "wide" else axes:
            ax.set_ylabel(r"$\theta_{\max}$ [arcmin]")
        S.pte_colorbar(
            fig,
            mesh,
            axes,
            shrink=0.9 if layout == "wide" else 0.5,
            aspect=25,
            pad=0.02,
            location="right" if layout == "wide" else "bottom",
        )
        return S.save(fig, out, "full" if layout == "wide" else "half")


def harmonic(cl, out, ell_cut):
    """C_ℓ^BB PTEs over every contiguous band range (Paper II Fig. 9)."""
    pte = cl_pte_matrix(cl, "BB")
    inside = np.flatnonzero((cl["ell_lo"] >= ell_cut[0]) & (cl["ell_hi"] <= ell_cut[1]))
    fid = (inside[0], inside[-1])
    lo = [f"{v:.0f}" for v in cl["ell_lo"]]
    hi = [f"{v:.0f}" for v in cl["ell_hi"]]
    cmap, norm = S.pte_cmap(), S.pte_norm()
    with S.figure_style():
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        mesh = _heatmap(ax, pte, lo, hi, fid, r"$C_\ell^{BB}$", cmap, norm)
        ax.set_xlabel(r"$\ell_{\min}$")
        ax.set_ylabel(r"$\ell_{\max}$")
        S.pte_colorbar(fig, mesh, [ax], aspect=25, pad=0.02, shrink=0.9)
        return S.save(fig, out, "half")
