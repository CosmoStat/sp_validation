"""SNR–size weight-cell diagnostics and shear-response distributions."""

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
        if layout == "wide":
            fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
        else:
            fig, axes = plt.subplots(4, 1, figsize=(3.6, 9.5))
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
