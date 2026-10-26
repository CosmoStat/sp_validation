"""Correlation matrices of the covariances behind each statistic.

One square figure per statistic, drawn as the correlation coefficient with
Paper II's diverging ``icefire`` map, block labels at the block centres and
hairline block separators.
"""

import matplotlib.pyplot as plt
import numpy as np

from sp_validation.b_modes import correlation_from_covariance

from . import style as S


def correlation(cov, blocks, out, label):
    """Draw ``cov`` as a correlation matrix; ``blocks`` is ``[(name, size)]``."""
    corr = correlation_from_covariance(np.asarray(cov))
    n = corr.shape[0]
    sizes = [size for _, size in blocks]
    if sum(sizes) != n:
        raise ValueError(f"blocks cover {sum(sizes)} rows of a {n}-row matrix")
    starts = np.concatenate(([0], np.cumsum(sizes)))
    with S.figure_style():
        fig, ax = plt.subplots(figsize=(3.6, 3.25))
        image = ax.imshow(
            corr,
            cmap=S.correlation_cmap(),
            vmin=-1,
            vmax=1,
            interpolation="nearest",
            origin="upper",
        )
        for edge in starts[1:-1]:
            ax.axhline(edge - 0.5, color=S.RAG, lw=0.5, alpha=0.8)
            ax.axvline(edge - 0.5, color=S.RAG, lw=0.5, alpha=0.8)
        centres = (starts[:-1] + starts[1:]) / 2 - 0.5
        names = [name for name, _ in blocks]
        ax.set_xticks(centres)
        ax.set_xticklabels(names, fontsize=8.5)
        ax.set_yticks(centres)
        ax.set_yticklabels(names, fontsize=8.5)
        ax.tick_params(which="both", length=0, top=False, right=False)
        S.header(ax, label)
        cbar = S.colorbar(
            fig, image, ax, "correlation coefficient", shrink=0.92, aspect=22, pad=0.03
        )
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        return S.save(fig, out, "half")


def xi_blocks(n):
    return [(r"$\xi_+$", n), (r"$\xi_-$", n)]


def pure_eb_blocks(n):
    return [(rf"$\xi_{s}^{{{m}}}$", n) for m in ("E", "B", r"\rm amb") for s in "+-"]


def cosebis_blocks(n):
    return [(r"$E_n$", n), (r"$B_n$", n)]


def cl_matrix(cov, keys=("EE", "EB", "BB")):
    """Assemble the NaMaster blocks for ``keys`` into one matrix."""
    return np.block([[cov[f"{a}_{b}"] for b in keys] for a in keys])


def cl_blocks(n, keys=("EE", "EB", "BB")):
    return [(rf"$C_\ell^{{{k}}}$", n) for k in keys]
