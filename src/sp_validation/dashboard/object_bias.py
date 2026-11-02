"""Object-wise PSF leakage and binned additive-bias regressions."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from . import style as S
from .sky import DIVERGING


def slopes(sums):
    """Weighted least-squares slopes with an intercept, from raw sums."""
    sw, sx, sy, sxx, sxy = sums[:5]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (sxy - sx * sy / sw) / (sxx - sx * sx / sw)


def leakage_fit(sums):
    """Component WLS slopes and conventional residual-based fit errors."""
    values = np.array([slopes(s) for s in sums])
    sw, sx, sy, sxx, sxy, syy, count = np.moveaxis(sums, 1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sxxc = sxx - sx * sx / sw
        rss = syy - sy * sy / sw - values * (sxy - sx * sy / sw)
        errors = np.sqrt(np.maximum(rss, 0) / (count - 2) / sxxc)
    return values, errors


def leakage(cells, out):
    """Weighted component-wise PSF-ellipticity regressions in each cell."""
    values, errors = leakage_fit(cells["leakage_sums"])
    redges, sedges = cells["resolution_edges"], cells["leakage_snr_edges"]
    limit = np.nanpercentile(np.abs(values), 95)
    with S.figure_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3))
        for ax, val, err, title in zip(
            axes, values, errors, (r"$\alpha_1$", r"$\alpha_2$")
        ):
            polygons = []
            for i in range(len(redges) - 1):
                for j in range(sedges.shape[1] - 1):
                    x0, x1 = sedges[i, j : j + 2]
                    y0, y1 = redges[i : i + 2]
                    polygons.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                    if not np.isfinite(err[i, j]) or abs(val[i, j]) < 2 * err[i, j]:
                        ax.add_patch(
                            Rectangle(
                                (x0, y0),
                                x1 - x0,
                                y1 - y0,
                                fill=False,
                                hatch="///",
                                edgecolor="#777777",
                                lw=0,
                                zorder=3,
                            )
                        )
            mesh = PolyCollection(
                polygons,
                array=val.ravel(),
                cmap=DIVERGING,
                clim=(-limit, limit),
                edgecolors="none",
            )
            ax.add_collection(mesh)
            ax.set_xscale("log")
            ax.set_xlim(sedges.min(), sedges.max())
            ax.set_ylim(redges[0], redges[-1])
            ax.set_xlabel("SNR")
            ax.set_ylabel(r"$r_{\rm PSF}/(r_{\rm PSF}+r_{\rm gal})$")
            S.colorbar(fig, mesh, ax, title)
        return S.save(fig, out, "full")


def bias_fit(s):
    """Fit weighted mean ellipticity versus property across populated bins."""
    sw, sx, sy, sxx, _, _, sw2, count, sw2y, sw2yy = s
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = sx / sw, sy / sw
        var = np.maximum(sw2yy - 2 * y * sw2y + y * y * sw2, 0) / (sw * sw)
    ok = (count > 1) & np.isfinite(x) & np.isfinite(y) & (var > 0)
    if ok.sum() < 3:
        return np.nan, np.nan
    mean = np.sum(sx[ok]) / np.sum(sw[ok])
    std = np.sqrt(max(np.sum(sxx[ok]) / np.sum(sw[ok]) - mean**2, 0))
    x, y, weight = x[ok], y[ok], 1 / var[ok]
    if len(x) < 3:
        return np.nan, np.nan
    x -= np.average(x, weights=weight)
    den = np.sum(weight * x * x)
    return std * np.sum(weight * x * y) / den, std * np.sqrt(1 / den)


def bias(cells, out):
    """Slopes and fit significances for weighted binned means."""
    names = [
        r"galaxy $\sqrt{T}$" if name == "galaxy size" else name
        for name in cells["bias_names"]
    ]
    fits = np.array([[bias_fit(s) for s in row] for row in cells["bias_sums"]])
    with S.figure_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.5), sharey=True)
        y = np.arange(len(names))
        for component, color in enumerate(("#3D5BA0", "#BC4538")):
            value, err = fits[:, component, 0], fits[:, component, 1]
            axes[0].errorbar(
                value,
                y + (component - 0.5) * 0.18,
                xerr=err,
                fmt="o",
                ms=3,
                color=color,
                label=rf"$e_{component + 1}$",
            )
            axes[1].plot(
                value / err, y + (component - 0.5) * 0.18, "o", ms=3, color=color
            )
        for ax in axes:
            ax.axvline(0, color=S.INK_MUTED, lw=0.7)
        axes[0].set_yticks(y, names)
        axes[0].set_xlabel(r"$\Delta e$ per $1\sigma$ of property")
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        axes[0].xaxis.set_major_formatter(formatter)
        axes[0].xaxis.set_major_locator(MaxNLocator(5))
        axes[1].set_xlabel("Slope / fit uncertainty")
        axes[0].legend()
        return S.save(fig, out, "full")
