"""Tomographic figures: one panel per pair of source bins.

Every two-point figure is a lower triangle: pair (i, j), i ≤ j, sits in row j,
column i, named above its frame with any PTEs on the right of that header, so
no annotation sits on the data; the empty upper triangle holds the legend.
B-mode null tests that have no natural physical scale (C_ℓ, COSEBIs) are drawn
in units of their own standard deviation over ±1σ and ±2σ bands, one y range
for every pair; power spectra use the square-root ℓ axis (:func:`style.ell_axis`).

The C_ℓ grids and the τ rows are drawn by Sacha Guerrini's harmonic-space and
PSF functions (:func:`sp_validation.cosmo_val.pseudo_cl.plot_pseudo_cl_grid`,
:func:`sp_validation.cosmo_val.psf_systematics.plot_tau_stats_tomo`) through
their style hooks; this module adds bands, headers and legends on the axes they
return.

Two-point inputs are SACC objects as :func:`sp_validation.sacc_io.load` returns
them. E-mode vectors are drawn as stored, concealed under their blind, with no
theory beside them.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from sp_validation import sacc_io

from . import style as S

XI_SCALE = 1e-4
# Page accents for the source bins, ordered by redshift.
BIN_COLORS = ["#3D5BA0", "#3F8278", "#C49333", "#BC4538"]
PSF_MODEL = "#3D5BA0"  # page cobalt, as in dashboard.psf


def pairs(nbin):
    return [(i, j) for i in range(nbin) for j in range(i, nbin)]


def pair_label(p):
    return rf"${p[0] + 1}\times{p[1] + 1}$"


def sigma(s):
    return np.sqrt(np.diag(s.covariance.dense))


def series(s, dtype, pair, tag, err):
    """``(x, value, σ)`` of one data type for one bin pair, in SACC order."""
    tr = (sacc_io.source_name(pair[0]), sacc_io.source_name(pair[1]))
    r = np.asarray(s.indices(dtype, tr), int)
    x = np.array([s.data[k].tags[tag] for k in r], float)
    return x, s.mean[r], err[r]


def triangle(nbin, height=6.2, sharey=False):
    """Lower-triangle grid: ``(fig, {pair: ax}, key_ax)``.

    ``key_ax`` is the empty top-right cell, for the legend.
    """
    fig, axes = plt.subplots(
        nbin, nbin, figsize=(7.2, height), sharex=True, sharey=sharey
    )
    for i in range(nbin):
        for j in range(i + 1, nbin):
            axes[i, j].set_axis_off()
    fig.get_layout_engine().set(hspace=0.035, wspace=0.03)
    return fig, {p: axes[p[1], p[0]] for p in pairs(nbin)}, axes[0, -1]


def pair_header(ax, p, ptes=()):
    """Pair name above the frame (left); PTEs (``[(name, value)]``) on the right,
    each in cinnabar when it falls below the threshold."""
    ax.set_title(pair_label(p), loc="left", fontsize=9, pad=3)
    anchor, xy, off = ax, (1, 1), (0, 3.5)
    for name, v in reversed([x for x in ptes if x[1] is not None]):
        anchor = ax.annotate(
            f"{name} {S.fmt_pte(v)}",
            xy=xy,
            xycoords=anchor,
            xytext=off,
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=S.CINNABAR if v < S.PTE_THRESHOLD else S.INK_MUTED,
            annotation_clip=False,
        )
        xy, off = (0, 0), (-7, 0)
    return anchor


def _bands():
    from matplotlib.patches import Patch

    return [
        Patch(color=S.INK, alpha=0.11, lw=0, label=r"$\pm1\sigma$"),
        Patch(color=S.INK, alpha=0.045, lw=0, label=r"$\pm2\sigma$"),
    ]


def _key(ax, handles, title=None):
    ax.legend(
        handles=handles,
        loc="upper right",
        title=title,
        title_fontsize=8.5,
        handletextpad=0.3,
        alignment="right",
    )


def _marker(marker, color, label, filled=True, ms=4.5):
    return Line2D(
        [],
        [],
        ls="none",
        marker=marker,
        color=color,
        markersize=ms,
        markerfacecolor=color if filled else "none",
        markeredgecolor=color,
        markeredgewidth=0.9,
        label=label,
    )


def _label_edges(fig_axes, nbin, xlabel, ylabel):
    for p, ax in fig_axes.items():
        if p[1] == nbin - 1:
            ax.set_xlabel(xlabel)
        if p[0] == 0:
            ax.set_ylabel(ylabel)


def _sigma_units(ax, x, y, marker, color, filled):
    ax.plot(
        x,
        y,
        ls="none",
        marker=marker,
        color=color,
        markersize=3.2,
        markerfacecolor=color if filled else S.RAG,
        markeredgecolor=color,
        markeredgewidth=0.8,
        zorder=3,
    )


def _sigma_ylim(ax, extent):
    lim = max(4.0, np.ceil(extent + 0.3))
    ax.set_ylim(-lim, lim)
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))


CL_B_ROWS = [  # data type, PTE key, marker, filled, colour, name
    (sacc_io.CL_BB, "BB", "o", True, S.COLOR_CL_BB, "BB"),
    (sacc_io.CL_EB, "EB", "D", False, S.COLOR_CL_EB, "EB"),
]
ELL_MINOR = [t for t in range(100, 2001, 100) if t not in S.ELL_TICKS]


def _grid(nbin, sharey):
    """The triangle as the harmonic-space grid functions index it."""
    fig, ax_of, key = triangle(nbin, sharey=sharey)
    return fig, ax_of, key, np.array(fig.axes).reshape(nbin, nbin)


def _spectra(s, dtypes, nbin):
    """``(spectra, tomo_bins)`` of one SACC in the layout of
    :func:`sp_validation.cosmo_val.pseudo_cl.plot_pseudo_cl_grid`."""
    err = sigma(s)
    spectra = {}
    for p in pairs(nbin):
        d = {}
        for pol, dtype in dtypes.items():
            d["ell"], value, e = series(s, dtype, p, "ell", err)
            d[pol] = (value, e)
        spectra[(p[0] + 1, p[1] + 1)] = d
    bins = {"ids": list(range(1, nbin + 1)), "pairs": list(spectra)}
    return {"data": spectra}, {"data": bins}


def _cl_grid(s, axs, dtypes, pol_style, ell_range, **kw):
    """Draw with Sacha Guerrini's tomographic C_ℓ grid in the house style."""
    from sp_validation.cosmo_val.pseudo_cl import plot_pseudo_cl_grid

    spectra, bins = _spectra(s, dtypes, len(axs))
    plot_pseudo_cl_grid(
        spectra,
        bins,
        ["data"],
        {"data": ("", None)},
        list(dtypes),
        savefig=None,
        show=False,
        axs=axs,
        pol_style=pol_style,
        pair_label=False,
        ell_range=ell_range,
        minor_ticks=ELL_MINOR,
        zero_line={"color": S.INK, "ls": (0, (4, 3)), "lw": 0.7, "alpha": 0.6},
        **kw,
    )
    return spectra["data"]


def cl_bmodes(s, out, ptes=None, nbin=4, ell_range=(6, 2100)):
    """Pseudo-C_ℓ BB and EB per bin pair in units of their σ, on the √ℓ axis.

    Drawn by :func:`~sp_validation.cosmo_val.pseudo_cl.plot_pseudo_cl_grid`
    (``sigma_units``, BB and EB offset by half a bandpower); this adds the
    ±1σ/±2σ bands, the pair headers with ``ptes``
    (``{pair: {"BB": pte, "EB": pte}}``) and the legend.
    """
    ptes = ptes or {}
    with S.figure_style():
        fig, ax_of, key, axs = _grid(nbin, sharey=True)
        style = {
            n: {
                "fmt": m,
                "color": c,
                "markersize": 3.2,
                "markeredgewidth": 0.8,
                "markerfacecolor": c if f else S.RAG,
                "zorder": 3,
            }
            for _, n, m, f, c, _ in CL_B_ROWS
        }
        data = _cl_grid(
            s,
            axs,
            {k: dt for dt, k, *_ in CL_B_ROWS},
            style,
            ell_range,
            sigma_units=True,
            pol_offset=0.5,
        )
        extent = max(
            np.max(np.abs(d[k][0] / d[k][1]))
            for d in data.values()
            for _, k, *_ in CL_B_ROWS
        )
        for p, ax in ax_of.items():
            S.sigma_bands(ax)
            pair_header(
                ax, p, [(n, ptes.get(p, {}).get(k)) for _, k, _, _, _, n in CL_B_ROWS]
            )
        _sigma_ylim(ax, extent)
        _label_edges(ax_of, nbin, r"$\ell$", r"$C_\ell / \sigma_\ell$")
        key.set_visible(True)
        _key(
            key,
            [_marker(m, c, rf"$C_\ell^{{{n}}}$", f) for _, _, m, f, c, n in CL_B_ROWS]
            + _bands(),
        )
        return S.save(fig, out, "full")


def cl_ee(s, out, blind, nbin=4, ell_range=(6, 2100)):
    """Blinded ℓ C_ℓ^EE per bin pair on the √ℓ axis, as stored (concealed).

    Drawn by :func:`~sp_validation.cosmo_val.pseudo_cl.plot_pseudo_cl_grid`
    on one y range for every pair; this adds the pair headers and legend.
    """
    with S.figure_style():
        fig, ax_of, key, axs = _grid(nbin, sharey=True)
        mark = {**S.DATA_VECTOR_MARK, "markersize": 2.6}
        mark.pop("ls")
        _cl_grid(
            s,
            axs,
            {"EE": sacc_io.CL_EE},
            {"EE": {"fmt": "o", "color": S.COLOR_E, **mark}},
            ell_range,
        )
        for p, ax in ax_of.items():
            pair_header(ax, p)
            if p[1] == nbin - 1:
                ax.set_xlabel(r"$\ell$")
        key.set_visible(True)
        _key(key, [_marker("o", S.COLOR_E, r"$C_\ell^{EE}$ (blinded)")])
        S.blind_stamp(fig, blind, where="bottom")
        return S.save(fig, out, "full")


def xi_pm(s, out, blind, nbin=4, theta_range=(0.9, 280)):
    """Blinded θξ+ (filled) and θξ− (open) per bin pair, as stored (concealed)."""
    err = sigma(s)
    rows = [
        (sacc_io.XI_PLUS, "o", True, -0.025, r"$\xi_+$"),
        (sacc_io.XI_MINUS, "s", False, 0.025, r"$\xi_-$"),
    ]
    with S.figure_style():
        fig, ax_of, key = triangle(nbin)
        for p, ax in ax_of.items():
            S.zero_line(ax)
            for dtype, mk, filled, off, _ in rows:
                th, v, e = series(s, dtype, p, "theta", err)
                ax.errorbar(
                    th * 10**off,
                    th * v / XI_SCALE,
                    yerr=th * e / XI_SCALE,
                    marker=mk,
                    color=S.COLOR_E,
                    markerfacecolor=S.COLOR_E if filled else S.RAG,
                    **{**S.DATA_VECTOR_MARK, "markersize": 2.8},
                )
            S.theta_axis(ax, *theta_range)
            S.theta_ticks(ax)
            ax.tick_params(axis="y", labelsize=8)
            pair_header(ax, p)
        _label_edges(
            ax_of, nbin, r"$\theta$ [arcmin]", r"$\theta\,\xi_\pm \times 10^{4}$"
        )
        _key(
            key,
            [
                _marker("o", S.COLOR_E, r"$\theta\xi_+$ (blinded)"),
                _marker("s", S.COLOR_E, r"$\theta\xi_-$ (blinded)", filled=False),
            ],
        )
        S.blind_stamp(fig, blind, where="bottom")
        return S.save(fig, out, "full")


def xi_pure_b(s, out, scale_cut, ptes=None, nbin=4, theta_range=(0.9, 280)):
    """Pure-mode θξ+^B and θξ−^B per bin pair (Paper II Fig. 2 hues), scales
    outside ``scale_cut`` shaded. ``ptes``: ``{pair: {"xip_B": .., "xim_B": ..}}``."""
    err = sigma(s)
    ptes = ptes or {}
    pt = sacc_io.PURE_TYPES
    rows = [
        (pt["xip_B"], "xip_B", "o", S.COLOR_B, -0.025, "+"),
        (pt["xim_B"], "xim_B", "s", S.COLOR_XI_B_MINUS, 0.025, "-"),
    ]
    with S.figure_style():
        fig, ax_of, key = triangle(nbin)
        for p, ax in ax_of.items():
            S.zero_line(ax)
            for dtype, _, mk, col, off, _ in rows:
                th, v, e = series(s, dtype, p, "theta", err)
                ax.errorbar(
                    th * 10**off,
                    th * v / XI_SCALE,
                    yerr=th * e / XI_SCALE,
                    marker=mk,
                    color=col,
                    **{**S.DATA_VECTOR_MARK, "markersize": 2.8},
                )
            S.theta_axis(ax, *theta_range)
            S.theta_ticks(ax)
            S.shade_outside(ax, *scale_cut)
            ax.tick_params(axis="y", labelsize=8)
            pair_header(
                ax,
                p,
                [
                    (rf"$\xi_{sg}^B$", ptes.get(p, {}).get(k))
                    for _, k, _, _, _, sg in rows
                ],
            )
        _label_edges(
            ax_of, nbin, r"$\theta$ [arcmin]", r"$\theta\,\xi^B_\pm \times 10^{4}$"
        )
        _key(
            key,
            [_marker(m, c, rf"$\xi_{sg}^B$") for _, _, m, c, _, sg in rows],
            title=rf"PTE at ${scale_cut[0]:g}'\!-\!{scale_cut[1]:g}'$" if ptes else None,
        )
        return S.save(fig, out, "full")


def cosebis_b(s, out, ptes=None, nbin=4):
    """COSEBIs B_n per bin pair in units of their σ (Paper II Fig. 3 hue).
    ``ptes``: ``{pair: pte}`` over every mode shown."""
    err = sigma(s)
    ptes = ptes or {}
    with S.figure_style():
        fig, ax_of, key = triangle(nbin, sharey=True)
        extent = 0
        for p, ax in ax_of.items():
            n, b, e = series(s, sacc_io.COSEBI_BB, p, "n", err)
            y = b / e
            extent = max(extent, np.max(np.abs(y)))
            S.sigma_bands(ax)
            S.zero_line(ax)
            _sigma_units(ax, n, y, "o", S.COLOR_COSEBIS, True)
            ax.set_xlim(0.3, n[-1] + 0.7)
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.set_xticks(np.arange(1, n[-1] + 1), minor=True)
            pair_header(ax, p, [("PTE", ptes.get(p))])
        _sigma_ylim(ax, extent)
        _label_edges(ax_of, nbin, r"COSEBI mode $n$", r"$B_n / \sigma_n$")
        _key(key, [_marker("o", S.COLOR_COSEBIS, r"$B_n$")] + _bands())
        return S.save(fig, out, "full")


def pte_maps(table, joint, stats, out, nbin=4):
    """One lower-triangle PTE map per statistic, the joint PTE in its header.

    ``table``: ``{key: {pair: pte}}``; ``joint``: ``{key: pte}``;
    ``stats``: ``[(key, mathtext title)]``.
    """
    cmap, norm = S.pte_cmap(), S.pte_norm()
    cmap.set_bad((0, 0, 0, 0))  # the empty upper triangle shows the card
    ticks = [f"${i + 1}$" for i in range(nbin)]
    with S.figure_style():
        fig, axes = plt.subplots(1, len(stats), figsize=(7.2, 1.75))
        for ax, (k, title) in zip(axes, stats):
            M = np.full((nbin, nbin), np.nan)
            for p, v in table.get(k, {}).items():
                if v is not None:
                    M[p[1], p[0]] = v
            im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm)
            for p in pairs(nbin):
                v = M[p[1], p[0]]
                if np.isfinite(v):
                    ax.text(
                        p[0],
                        p[1],
                        f"{v:.2f}".lstrip("0"),
                        ha="center",
                        va="center",
                        fontsize=6.3,
                        color="white" if (v < 0.1 or v > 0.9) else S.INK,
                    )
            ax.set_xticks(range(nbin), ticks, fontsize=7.5)
            ax.set_yticks(range(nbin), ticks, fontsize=7.5)
            ax.tick_params(length=0, pad=1.5)
            for sp in ax.spines.values():
                sp.set_visible(False)
            j = joint.get(k)
            bad = j is not None and (j < S.PTE_THRESHOLD or j > 1 - S.PTE_THRESHOLD)
            ax.set_title(title, fontsize=9.5, pad=13)
            ax.text(
                0.5,
                1.02,
                f"joint {S.fmt_pte(j)}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=S.CINNABAR if bad else S.INK_MUTED,
            )
            ax.set_facecolor(S.RAG)
        S.pte_colorbar(fig, im, list(axes), fraction=0.025, pad=0.02, aspect=12)
        return S.save(fig, out, "full")


def nz(curves, out):
    """Normalised n_i(z) per bin; ``curves`` is ``[(z, n)]`` in bin order."""
    with S.figure_style():
        fig, ax = plt.subplots(figsize=(3.6, 2.3))
        for i, (z, n) in enumerate(curves):
            n = n / np.trapezoid(n, z)
            zm = np.trapezoid(z * n, z)
            ax.plot(
                z,
                n,
                color=BIN_COLORS[i],
                lw=1.2,
                label=rf"bin {i + 1}, $\langle z\rangle = {zm:.2f}$",
            )
            ax.fill_between(z, n, color=BIN_COLORS[i], alpha=0.12, lw=0)
        ax.set_xlim(0, 2.0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("$z$")
        ax.set_ylabel(r"$n_i(z)$")
        ax.legend(fontsize=7.5, handlelength=1.4)
        return S.save(fig, out, "half")


def tau_bins(theta, tau, model, out):
    """τ_{0,2,5}+ per source bin with each bin's fitted leakage model.

    ``tau[i][k] = (value, σ)`` and ``model[i][k]`` on ``theta``, bin ``i``,
    ``k`` in (0, 2, 5). Drawn by
    :func:`~sp_validation.cosmo_val.psf_systematics.plot_tau_stats_tomo`
    (one row per bin, θτ for τ2 and τ5); this adds the model lines.
    """
    from sp_validation.cosmo_val.psf_systematics import plot_tau_stats_tomo

    nbin = len(tau)
    data = {
        i + 1: {"theta": theta, **{(str(k), "p"): tau[i][k] for k in (0, 2, 5)}}
        for i in range(nbin)
    }
    mark = {**S.DATA_VECTOR_MARK, "markersize": 2.6}
    mark.pop("ls")
    with S.figure_style():
        fig = plt.figure(figsize=(7.2, 1.45 * nbin + 0.9))
        fig.get_layout_engine().set(hspace=0.04)
        _, axs = plot_tau_stats_tomo(
            {"data": data},
            {"data": {"ids": list(data)}},
            ["data"],
            [S.INK],
            {"data": "measured"},
            savefig=None,
            show=False,
            close=False,
            plot_theta_times_tau=True,
            fmt="o",
            capsize=mark.pop("capsize", 1.6),
            fig=fig,
            errorbar_kw=mark,
            row_label=lambda b, _: f"bin {b}",
            titles=[r"$\tau_{0,+}$", r"$\theta\,\tau_{2,+}$", r"$\theta\,\tau_{5,+}$"],
            legend=False,
        )
        for i in range(nbin):
            for col, k in enumerate((0, 2, 5)):
                ax = axs[i][col]
                f = 1.0 if k == 0 else theta
                ax.plot(theta, f * model[i][k], color=PSF_MODEL, lw=1.0, zorder=1)
                S.zero_line(ax)
                ax.set_xticks([1, 10, 100])
                ax.set_xticklabels(["$1$", "$10$", "$100$"])
        fig.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=S.INK,
                    marker="o",
                    ms=3,
                    ls="none",
                    label=r"measured $\tau_{k,+}$",
                ),
                Line2D(
                    [],
                    [],
                    color=PSF_MODEL,
                    lw=1.0,
                    label=r"fitted $\alpha, \beta, \eta$ model",
                ),
            ],
            loc="outside upper center",
            ncol=2,
        )
        return S.save(fig, out, "full")


def leakage_bins(fits, out):
    """PSF-leakage coefficients α, β, η (± 1σ) per source bin.

    ``fits``: ``[{"alpha": .., "sig_alpha": .., ...}]`` in bin order.
    """
    names = (("alpha", r"$\alpha$"), ("beta", r"$\beta$"), ("eta", r"$\eta$"))
    x = np.arange(1, len(fits) + 1)
    with S.figure_style():
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.9))
        for ax, (k, lab) in zip(axes, names):
            S.zero_line(ax)
            for i, f in enumerate(fits):
                ax.errorbar(
                    x[i],
                    f[k],
                    f["sig_" + k],
                    color=BIN_COLORS[i],
                    marker="o",
                    **{**S.DATA_VECTOR_MARK, "markersize": 4},
                )
            ax.set_xticks(x)
            ax.set_xlim(0.5, len(fits) + 0.5)
            ax.tick_params(axis="x", which="both", top=False)
            ax.set_xlabel("source bin")
            S.header(ax, lab)
        return S.save(fig, out, "full")


def variance_ratio(ratios, out, band=0.14):
    """Analytic over jackknife variance of ξ± per bin pair against θ.

    ``ratios[name][pair] = (θ, ratio)`` for ``name`` in ("+", "-"); auto-pairs
    in their bin's colour, cross-pairs in muted ink; ``band`` is the
    jackknife's own scatter about 1.
    """
    with S.figure_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
        for ax, (sign, by_pair) in zip(axes, ratios.items()):
            ax.axhspan(1 - band, 1 + band, color=S.INK, alpha=0.06, lw=0)
            ax.axhline(1, color=S.INK, lw=0.8)
            for p, (th, r) in by_pair.items():
                auto = p[0] == p[1]
                ax.plot(
                    th,
                    r,
                    color=BIN_COLORS[p[0]] if auto else S.NEUTRAL_LIGHT,
                    lw=1.1 if auto else 0.7,
                    zorder=3 if auto else 2,
                    label=f"{p[0] + 1}" + r"$\times$" + f"{p[1] + 1}" if auto else None,
                )
            S.theta_axis(ax)
            S.theta_ticks(ax)
            ax.set_xlabel(r"$\theta$ [arcmin]")
            S.header(ax, rf"$\xi_{sign}$")
        axes[0].set_ylabel(r"$\sigma^2_{\rm analytic} / \sigma^2_{\rm jackknife}$")
        handles, labels = axes[0].get_legend_handles_labels()
        handles.append(Line2D([], [], color=S.NEUTRAL_LIGHT, lw=0.7))
        labels.append("cross-pairs")
        fig.legend(handles, labels, loc="outside upper center", ncol=len(labels))
        return S.save(fig, out, "full")
