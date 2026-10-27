"""Object-wise PSF leakage and binned additive-bias regressions."""

import matplotlib.pyplot as plt
import numpy as np

from . import style as S
from .sky import DIVERGING


def slopes(sums):
    """Weighted least-squares slopes with an intercept, from raw sums."""
    sw, sx, sy, sxx, sxy = sums[:5]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (sxy - sx * sy / sw) / (sxx - sx * sx / sw)


def leakage(cells, out):
    """Weighted component-wise PSF-ellipticity regressions in each cell."""
    slopes_by_component = np.array([slopes(s) for s in cells["leakage_sums"]])
    with S.figure_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3))
        for ax, values, title in zip(
            axes, slopes_by_component, (r"$\alpha_1$", r"$\alpha_2$")
        ):
            limit = np.nanpercentile(np.abs(values), 98)
            mesh = ax.pcolormesh(
                cells["snr_edges"],
                cells["resolution_edges"],
                values.T,
                cmap=DIVERGING,
                vmin=-limit,
                vmax=limit,
            )
            ax.set_xscale("log")
            ax.set_xlabel("SNR")
            ax.set_ylabel(r"$r_{\rm PSF}/(r_{\rm PSF}+r_{\rm gal})$")
            S.colorbar(fig, mesh, ax, title)
        return S.save(fig, out, "full")


def bias_fit(s):
    """Fit weighted mean ellipticity versus property across populated bins."""
    sw, sx, sy, _, _, syy, sw2, count = s
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = sx / sw, sy / sw
        var = np.maximum(syy / sw - y * y, 0) * sw2 / (sw * sw)
    ok = (count > 1) & np.isfinite(x) & np.isfinite(y) & (var > 0)
    x, y, weight = x[ok], y[ok], 1 / var[ok]
    if len(x) < 3:
        return np.nan, np.nan
    x -= np.average(x, weights=weight)
    den = np.sum(weight * x * x)
    return np.sum(weight * x * y) / den, np.sqrt(1 / den)


def bias(cells, out):
    """Slopes and fit significances for weighted binned means."""
    names = [r"galaxy $\sqrt{T}$" if name == "galaxy size" else name for name in cells["bias_names"]]
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
        axes[0].set_xlabel("Fitted slope")
        axes[1].set_xlabel("Slope / fit uncertainty")
        axes[0].legend()
        return S.save(fig, out, "full")
