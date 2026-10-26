"""Shared look of the dashboard figures.

Typography and colour semantics follow the UNIONS Paper II (B-modes) figures:
Computer Modern via mathtext (no TeX dependency), one hue per estimator
family in the B-mode data vectors, grey shading over excluded scales, a boxed
in-panel annotation naming the statistic, and the threshold-anchored PTE
colormap. Surfaces and ink come from the production page's parchment
palette so the PNGs sit on its cards without a white slab.

Sizes are in inches at print-like font sizes; :func:`save` fixes the tight
saved width, and the resolution is chosen per size class so a figure stays
crisp at the width the page shows it.
"""

from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

# Page palette (production/status.html)
RAG = "#FBF6E9"  # card surface; figure background
INK = "#1B1611"
INK_MUTED = "#7A6F5C"
RULE = "#C5B89E"
CINNABAR = "#BC4538"

# Paper II mode / family hues (Reproductions UNIONS/pure_eb plotting_utils)
COLOR_TOTAL = INK
COLOR_E = "#2a78d6"
COLOR_B = "crimson"
COLOR_AMB = "#9a9a9a"
COLOR_XI_B_MINUS = "#e8792b"
COLOR_COSEBIS = "#7b3fa0"
COLOR_COSEBIS_ALL = "#b48cce"
COLOR_CL_BB = "#008080"
COLOR_CL_EB = "#4d4d4d"

# Catalogue versions, only in version-comparison figures (Paper II Figs 7-9)
VERSION_MARKERS = ["o", "s", "D", "^"]

# Neutral series inside one figure where no mode semantics apply
NEUTRAL = INK
NEUTRAL_LIGHT = "#a89f8c"

DATA_VECTOR_MARK = {
    "ls": "none",
    "markersize": 3.4,
    "markeredgewidth": 0.8,
    "elinewidth": 0.75,
    "capsize": 1.6,
    "capthick": 0.75,
}
EXCLUDED_SHADE = {"color": INK, "alpha": 0.07, "lw": 0, "zorder": 0}

# Width classes (inches) and their resolutions: about 1600 px for the full
# column, about 2x the displayed size for the row of three and phones.
WIDTH = {"full": 7.2, "half": 3.6, "third": 3.0}
DPI = {"full": 220, "half": 220, "third": 250}

RC = {
    "figure.facecolor": RAG,
    "axes.facecolor": RAG,
    "savefig.facecolor": RAG,
    "text.color": INK,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": True,
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "lines.linewidth": 1.2,
    "figure.constrained_layout.use": True,
    "savefig.pad_inches": 0.04,
}


@contextmanager
def figure_style():
    """Apply the dashboard rcParams for the duration of a render."""
    with mpl.rc_context(RC):
        yield


def save(fig, path, size="full", width=None):
    """Save ``fig`` with a tight bounding box exactly the class width wide.

    Scales the canvas (keeping its aspect) until the cropped width equals the
    target, so font sizes in the style are the sizes on the page at that width
    (the Paper II ``savefig_at_width`` recipe).
    """
    width = width or WIDTH[size]
    pad = mpl.rcParams["savefig.pad_inches"]
    for _ in range(10):
        fig.canvas.draw()
        saved = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * pad
        if abs(saved / width - 1.0) < 1e-3:
            break
        fig.set_size_inches(fig.get_size_inches() * (width / saved))
    fig.savefig(path, dpi=DPI.get(size, 220), bbox_inches="tight")
    plt.close(fig)
    return path


def label_box(ax, text, loc="upper left", fontsize=None):
    """Boxed in-panel annotation naming the statistic (Paper II convention)."""
    x, ha = (0.04, "left") if "left" in loc else (0.96, "right")
    y, va = (0.95, "top") if "upper" in loc else (0.05, "bottom")
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize or mpl.rcParams["font.size"],
        zorder=10,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor=RAG,
            edgecolor=INK,
            linewidth=0.7,
        ),
    )


def note(ax, text, loc="lower right", color=INK_MUTED, fontsize=8):
    """Small unboxed annotation, e.g. a PTE, in muted ink."""
    x, ha = (0.04, "left") if "left" in loc else (0.96, "right")
    y, va = (0.95, "top") if "upper" in loc else (0.05, "bottom")
    ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va=va, color=color,
        fontsize=fontsize, zorder=10,
    )


def blind_stamp(fig, blind, where="top"):
    """Mark a figure that shows blinded data vectors, outside the axes."""
    y, va = (1.0, "top") if where == "top" else (0.0, "bottom")
    x, ha = (1.0, "right") if where == "top" else (0.0, "left")
    fig.text(
        x, y, f"BLINDED · {blind}", ha=ha, va=va, fontsize=6.5,
        family="monospace", color=CINNABAR,
    )


def zero_line(ax):
    ax.axhline(0, color=INK, ls=(0, (4, 3)), lw=0.7, alpha=0.6, zorder=1)


def shade_outside(ax, lo, hi):
    """Grey-shade the parts of the x range outside ``[lo, hi]``."""
    x0, x1 = ax.get_xlim()
    if lo is not None and lo > x0:
        ax.axvspan(x0, lo, **EXCLUDED_SHADE)
    if hi is not None and hi < x1:
        ax.axvspan(hi, x1, **EXCLUDED_SHADE)
    ax.set_xlim(x0, x1)


def theta_axis(ax, lo=0.9, hi=280):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f"${v:g}$" if v >= 1 else "")
    )


def fmt_pte(value):
    """PTE to two significant figures, in mathtext."""
    if value is None or not np.isfinite(value):
        return "n/a"
    if value < 1e-3:
        return rf"{value:.1e}".replace("e-0", r"\times10^{-") + "}"
    return f"{value:#.2g}".rstrip(".")


# Threshold-anchored PTE colormap (Paper II): solid poles below the threshold
# and above one minus it, a vlag gradient between.
PTE_THRESHOLD = 0.05
_PTE_LEVELS = 512


def pte_cmap(threshold=PTE_THRESHOLD):
    import seaborn as sns

    vlag = sns.color_palette("vlag", as_cmap=True)
    colors = (
        [vlag(0.0)]
        + [vlag(v) for v in np.linspace(0.15, 0.85, _PTE_LEVELS)]
        + [vlag(1.0)]
    )
    cmap = ListedColormap(colors, name="pte_threshold")
    cmap.set_bad(color="#e4dccb")
    return cmap


def pte_norm(threshold=PTE_THRESHOLD):
    bounds = np.concatenate(
        ([0.0], np.linspace(threshold, 1 - threshold, _PTE_LEVELS + 1), [1.0])
    )
    return BoundaryNorm(bounds, ncolors=_PTE_LEVELS + 2, clip=True)


def pte_colorbar(fig, mappable, ax, threshold=PTE_THRESHOLD, **kwargs):
    cbar = fig.colorbar(mappable, ax=ax, spacing="proportional", **kwargs)
    ticks = [threshold, 0.5, 1 - threshold]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"${t:g}$" for t in ticks])
    cbar.minorticks_off()
    cbar.outline.set_linewidth(0.6)
    cbar.solids.set_edgecolor("face")
    cbar.set_label("PTE")
    return cbar


def correlation_cmap():
    import seaborn as sns

    return sns.color_palette("icefire", as_cmap=True)
