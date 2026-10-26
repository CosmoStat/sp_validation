"""What the galaxy weights are.

The shape weight is built per cell of a log-spaced SNR x size-ratio grid
(:func:`sp_validation.calibration.get_w_des`): every galaxy in a cell gets
that cell's R^2 / sigma_e^2. The weights therefore take at most one value per
occupied cell. The figure shows the weight over the grid, with each cell's
share of galaxies, and the resulting discrete distribution.
"""

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import style as S
from .sky import SEQUENTIAL


def summary(cells):
    """Distinct values, grid size and whether every cell holds one weight."""
    count = cells["count"]
    occupied = count > 0
    spread = np.nanmax(np.abs(cells["w_max"] - cells["w_min"])[occupied])
    return {
        "n_values": int(len(cells["w_values"])),
        "n_cells": int(count.size),
        "n_occupied": int(occupied.sum()),
        "max_within_cell_spread": float(spread),
        "w_min": float(cells["w_values"].min()),
        "w_max": float(cells["w_values"].max()),
        "out_of_range": json.loads(str(cells["out_of_range"])),
    }


def weights(cells, out):
    """Weight on the SNR x size-ratio grid, and the distinct-weight spectrum."""
    snr, size = cells["snr_edges"], cells["size_ratio_edges"]
    count = cells["count"].astype(float)
    frac = count / count.sum()
    w = np.where(count > 0, cells["w_min"], np.nan)
    values, mult = cells["w_values"], cells["w_multiplicity"]
    log_fmt = mpl.ticker.FuncFormatter(lambda v, _: f"${v:g}$")
    with S.figure_style():
        fig, ax = plt.subplot_mosaic(
            [["top", ".", "spec"], ["grid", "side", "spec"]],
            figsize=(7.2, 3.5), width_ratios=(4, 0.7, 4.2), height_ratios=(0.8, 4),
        )
        g = ax["grid"]
        mesh = g.pcolormesh(snr, size, w.T, cmap=SEQUENTIAL, edgecolors=S.RAG,
                            linewidth=0.25)
        g.set_xscale("log")
        g.set_yscale("log")
        g.set_xlim(snr[0], snr[-1])
        g.set_ylim(size[0], size[-1])
        g.set_xlabel("signal-to-noise")
        g.set_ylabel(r"size ratio $T_{\rm gal} / T_{\rm PSF}$")
        g.xaxis.set_major_formatter(log_fmt)
        g.xaxis.set_minor_formatter(mpl.ticker.FuncFormatter(
            lambda v, _: f"${v:g}$" if v in (30, 300) else ""))
        g.yaxis.set_major_formatter(log_fmt)
        g.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(
            lambda v, _: f"${v:g}$" if v in (0.8, 1.5, 2) else ""))
        g.tick_params(which="both", top=False, right=False)
        cbar = fig.colorbar(mesh, ax=g, location="bottom", aspect=30, pad=0.02,
                            fraction=0.06)
        cbar.set_label(r"cell weight $w = R^2 / \sigma_e^2$")
        cbar.outline.set_linewidth(0.5)

        # Marginal shares of galaxies along each grid axis.
        bar = dict(color=S.NEUTRAL_LIGHT, edgecolor=S.RAG, linewidth=0.3)
        ax["top"].bar(snr[:-1], 100 * frac.sum(1), width=np.diff(snr), align="edge",
                      **bar)
        ax["top"].set_xscale("log")
        ax["top"].set_xlim(g.get_xlim())
        ax["side"].barh(size[:-1], 100 * frac.sum(0), height=np.diff(size),
                        align="edge", **bar)
        ax["side"].set_yscale("log")
        ax["side"].set_ylim(g.get_ylim())
        for name, axis in (("top", "x"), ("side", "y")):
            m = ax[name]
            m.tick_params(axis=axis, which="both", labelbottom=False,
                          labelleft=False, bottom=False, left=False, top=False,
                          right=False)
            for side in ("top", "right"):
                m.spines[side].set_visible(False)
            m.tick_params(which="both", top=False, right=False, labelsize=7)
            m.locator_params(axis="y" if name == "top" else "x", nbins=2)
        ax["top"].set_ylabel("%", fontsize=8)
        ax["side"].set_xlabel("%", fontsize=8)

        spec = ax["spec"]
        share = mult / mult.sum()
        spec.vlines(values, 0, share * 100, color=S.INK, lw=0.5, alpha=0.85)
        spec.set_xlabel(r"weight $w$")
        spec.set_ylabel("galaxies at this weight [%]")
        spec.set_ylim(0, None)
        spec.set_xlim(0, values.max() * 1.03)
        spec.tick_params(which="both", top=False, right=False)
        spec.set_title(f"{len(values)} distinct values, one per cell", loc="left",
                       fontsize=9)
        return S.save(fig, out, "full")
