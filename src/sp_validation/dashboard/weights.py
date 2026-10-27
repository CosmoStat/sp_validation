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


def weights(cells, out, layout="wide"):
    """Counts, weights, response and uncalibrated shape noise per cell."""
    count = cells["count"]
    with np.errstate(divide="ignore", invalid="ignore"):
        response = np.where(count > 0, cells["sum_response"] / count, np.nan)
        noise = np.where(count > 0, np.sqrt(cells["sum_uncal_ee"] / count), np.nan)
    panels = [
        (np.where(count > 0, count, np.nan), "Number of galaxies"),
        (cells["w_min"], "Weight"),
        (response, "Mean response"),
        (noise, r"$\sqrt{\langle e_1^2+e_2^2\rangle/2}$"),
    ]
    with S.figure_style():
        fig, axes = plt.subplots(
            2, 2, figsize=(7.2, 5.8) if layout == "wide" else (3.6, 4.5)
        )
        for ax, (values, label) in zip(axes.flat, panels):
            norm = mpl.colors.LogNorm() if label == "Number of galaxies" else None
            mesh = ax.pcolormesh(
                cells["snr_edges"],
                cells["size_ratio_edges"],
                values.T,
                cmap=SEQUENTIAL,
                norm=norm,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("SNR")
            ax.set_ylabel("Galaxy / PSF T")
            S.colorbar(fig, mesh, ax, label, pad=0.02)
        return S.save(fig, out, "full" if layout == "wide" else "half")


def response_histogram(cells, out):
    """Selected galaxies' metacal response components."""
    edges = cells["response_edges"]
    with S.figure_style():
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        for hist, label in zip(
            cells["response_hist"], (r"$R_{11}$", r"$R_{22}$", r"$R_{12}$", r"$R_{21}$")
        ):
            ax.stairs(hist, edges, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Galaxy shear response")
        ax.set_ylabel("Number of galaxies")
        ax.legend(ncol=2)
        return S.save(fig, out, "half")


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


def _tt(text):
    """Catalogue column names in typewriter mathtext."""
    return r"$\mathtt{" + text.replace("_", r"\_") + "}$"


def _legacy_weights(cells, out, layout="wide"):
    """Galaxy weight over the SNR x size-ratio grid it is defined on.

    Colour is each cell's weight; the axis labels name the catalogue columns
    binned and the colour-bar label defines the weight.
    """
    snr, size = cells["snr_edges"], cells["size_ratio_edges"]
    count = cells["count"]
    w = np.where(count > 0, cells["w_min"], np.nan)
    log_fmt = mpl.ticker.FuncFormatter(lambda v, _: f"${v:g}$")
    with S.figure_style():
        wide = layout == "wide"
        fig, g = plt.subplots(figsize=(7.2, 3.7) if wide else (3.6, 4.4))
        mesh = g.pcolormesh(
            snr, size, w.T, cmap=SEQUENTIAL, edgecolors=S.RAG, linewidth=0.3
        )
        g.set_xscale("log")
        g.set_yscale("log")
        g.set_xlim(snr[0], snr[-1])
        g.set_ylim(size[0], size[-1])
        sep = ":  " if wide else "\n"
        g.set_xlabel(
            "signal-to-noise ratio"
            + sep
            + _tt("snr = NGMIX_FLUX_NOSHEAR / NGMIX_FLUX_ERR_NOSHEAR"),
            fontsize=10 if wide else 7.5,
        )
        g.set_ylabel(
            "size ratio" + sep + _tt("NGMIX_T_NOSHEAR / NGMIX_T_PSF_RECONV_NOSHEAR"),
            fontsize=8.5 if wide else 7.5,
        )
        g.xaxis.set_major_formatter(log_fmt)
        g.xaxis.set_minor_formatter(
            mpl.ticker.FuncFormatter(lambda v, _: f"${v:g}$" if v in (30, 300) else "")
        )
        g.yaxis.set_major_formatter(log_fmt)
        g.yaxis.set_minor_formatter(
            mpl.ticker.FuncFormatter(
                lambda v, _: f"${v:g}$" if v in (0.8, 1.5, 2, 3) else ""
            )
        )
        S.frame_only(g)
        S.header(
            g,
            rf"{_tt('w_des')}: one weight per cell"
            + (
                rf", ${count.shape[0]} \times {count.shape[1]}$ log-spaced cells"
                if wide
                else ""
            ),
        )
        S.colorbar(
            fig,
            mesh,
            g,
            r"$w = \bar R^{2} / \sigma_e^{2}$" + "\n"
            r"$\bar R = \frac{1}{2}\langle R_{11} + R_{22}\rangle$ (metacal),"
            "\n"
            r"$\sigma_e^{2} = \frac{1}{2}\langle e_1^{2} + e_2^{2}\rangle$"
            " (uncalibrated), per cell",
            aspect=22,
            pad=0.02,
            location="right" if wide else "bottom",
        )
        return S.save(fig, out, "full" if wide else "half")
