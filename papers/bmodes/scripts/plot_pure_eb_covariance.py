# %%
"""Visualize Pure E/B covariance matrix structure."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from plotting_utils import PAPER_MPLSTYLE

# Unbuffered output for Snakemake log streaming
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

# Apply paper style
plt.style.use(PAPER_MPLSTYLE)


def _load_snakemake():
    if hasattr(sys, "ps1"):
        from snakemake_helpers import snakemake_interactive

        return snakemake_interactive(
            "results/paper_plots/pure_eb_covariance.png",
            str(Path.cwd()),
        )
    from snakemake.script import snakemake

    return snakemake


snakemake = _load_snakemake()

params = snakemake.params


def _correlation_from_covariance(cov):
    """Convert covariance to correlation matrix."""
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return corr


def main():
    # Load precomputed data
    dataset = np.load(snakemake.input["pure_eb_data"])

    cov_pure_eb = dataset["cov_pure_eb"]
    theta = dataset["theta"]
    nbins = len(theta)

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 9))
    vlag_cmap = sns.color_palette("vlag", as_cmap=True)
    im = ax.matshow(_correlation_from_covariance(cov_pure_eb), cmap=vlag_cmap)

    # Configure tick labels for E/B modes
    tick_positions = np.arange(nbins / 2, nbins * 6, nbins)
    tick_labels = [
        r"$\xi_+^{\mathrm{E}}$", r"$\xi_-^{\mathrm{E}}$", r"$\xi_+^{\mathrm{B}}$",
        r"$\xi_-^{\mathrm{B}}$", r"$\xi_+^{\mathrm{amb}}$", r"$\xi_-^{\mathrm{amb}}$"
    ]

    for ticks in (plt.xticks, plt.yticks):
        ticks(tick_positions, tick_labels)
        ticks(np.arange(0, nbins * 6 + 1, 20), minor=True)

    ax.tick_params(axis="both", which="major", length=0)
    im.set_clim(-1, 1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    ax.set_title(f"{params['version']}: semi-analytic correlation matrix")

    # Save
    output_path = Path(snakemake.output[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
