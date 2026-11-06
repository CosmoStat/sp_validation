"""Tangential-shear null tests and scale-dependent leakage in magnitude bins.

Renderers for the products of :mod:`sp_validation.dashboard.correlations`
(loaded by :func:`sp_validation.dashboard.products.load_correlation`): galaxy
shear around PSF stars, tile centres and Gaia stars; the PSF field around
galaxies; and α(θ) = τ₀/ρ₀ by galaxy magnitude. Each panel quotes χ²/ν
against zero from the jackknife covariance (Hartlap-corrected).
"""

import matplotlib.pyplot as plt
import numpy as np

from . import style as S

ALL = S.INK
BINS = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots (psf.py)


def chi2_null(values, cov, npatch):
    """χ² of ``values`` against zero, with the Hartlap factor for ``npatch``."""
    p = len(values)
    if npatch <= p + 2:
        return np.nan, p
    hartlap = (npatch - p - 2) / (npatch - 1)
    return float(values @ np.linalg.solve(cov, values) * hartlap), p


def _chi2_label(chi2, nu):
    value = "n/a" if not np.isfinite(chi2) else f"{chi2 / nu:.2f}"
    return rf"$\chi^2/\nu = {value}$"


def _blocks(result):
    n = len(result["theta"])
    cov = result["covariance"]
    return {
        "t": (result["xi_t"], result["error_t"], cov[:n, :n]),
        "x": (result["xi_x"], result["error_x"], cov[n:, n:]),
    }


def _offsets(n, spread=0.035):
    return 10 ** (spread * (np.arange(n) - (n - 1) / 2))


def _theta_panel(ax):
    S.zero_line(ax)
    S.theta_axis(ax)
    S.theta_ticks(ax)
    ax.set_xlabel(r"$\theta$ [arcmin]")


def _gamma_panels(
    series, npatch, out, scale=1e4, theta_weight=False, size="full", header=None
):
    """γ_t and γ_× panels for ``series`` = [(label, colour, result)]."""
    with S.figure_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
        offsets = _offsets(len(series))
        ncol = 2 if len(series) > 2 else 1
        for ax, comp, sym in zip(axes, "tx", ("t", r"\times")):
            _theta_panel(ax)
            for (label, colour, result), off in zip(series, offsets):
                theta = result["theta"]
                value, error, cov = _blocks(result)[comp]
                chi2, nu = chi2_null(value, cov, npatch)
                factor = theta if theta_weight else 1.0
                ax.errorbar(
                    theta * off,
                    value * factor * scale,
                    yerr=error * factor * scale,
                    color=colour,
                    marker="o",
                    label=f"{label}  {_chi2_label(chi2, nu)}",
                    **S.DATA_VECTOR_MARK,
                )
            exponent = int(round(np.log10(scale)))
            if theta_weight:
                ylabel = (
                    rf"$\theta\,\gamma_{{{sym}}}\ [10^{{-{exponent}}}\,\rm arcmin]$"
                )
            else:
                ylabel = rf"$\gamma_{{{sym}}}\ [10^{{-{exponent}}}]$"
            ax.set_ylabel(ylabel)
            S.label_box(ax, rf"$\gamma_{{{sym}}}$", loc="upper right")
            ax.legend(
                loc="lower center",
                ncol=ncol,
                fontsize=6.5,
                handletextpad=0.3,
                columnspacing=0.8,
                borderaxespad=0.3,
                frameon=False,
            )
        # Room below the data for the legend (the y axis is shared).
        lo, hi = axes[0].get_ylim()
        rows = -(-len(series) // ncol)
        axes[0].set_ylim(lo - 0.16 * rows * (hi - lo), hi)
        axes[1].yaxis.set_tick_params(labelleft=True)
        if header:
            S.header(axes[0], header)
        return S.save(fig, out, size)


def _bin_labels(edges, fmt, name):
    return [
        rf"${lo:{fmt}} \leq {name} < {hi:{fmt}}$"
        for lo, hi in zip(edges[:-1], edges[1:])
    ]


def gamma_stars(product, out):
    """Galaxy γ_t, γ_× around PSF validation stars, all and by star magnitude."""
    prov = product["provenance"]
    labels = _bin_labels(prov["star_mag_edges"], ".1f", "m")
    series = [("all", ALL, product["all"])] + [
        (lab, c, product[f"mag{i}"]) for i, (lab, c) in enumerate(zip(labels, BINS))
    ]
    return _gamma_panels(
        series,
        prov["npatch"],
        out,
        header=f"{prov['n_lenses']:,} PSF validation stars (valid HSM)",
    )


def gamma_tiles(product, out):
    """θ γ_t and θ γ_× of galaxies around tile centres."""
    prov = product["provenance"]
    series = [("tile centres", ALL, product["all"])]
    return _gamma_panels(
        series,
        prov["npatch"],
        out,
        scale=1e3,
        theta_weight=True,
        header=f"{prov['n_lenses']:,} tiles",
    )


def gamma_gaia(product, out):
    """Galaxy γ_t, γ_× around Gaia DR3 stars in three G bins, randoms subtracted."""
    prov = product["provenance"]
    labels = _bin_labels(prov["g_edges"], "g", "G")
    series = [
        (lab, c, product[f"G{i}"]) for i, (lab, c) in enumerate(zip(labels, BINS))
    ]
    n = sum(prov["n_lenses"].values())
    return _gamma_panels(
        series, prov["npatch"], out, header=f"{n:,} Gaia DR3 stars, randoms subtracted"
    )


def lambda_stars(product, out):
    """PSF-model (λ₁) and residual (λ₃) ellipticity around galaxy positions."""
    prov = product["provenance"]
    rows = [
        ("psf", r"\lambda_1", r"$e^{\rm PSF}$", 1e4),
        ("residual", r"\lambda_3", r"$e^{\star} - e^{\rm PSF}$", 1e5),
    ]
    with S.figure_style():
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True)
        for (key, sym, what, scale), row in zip(rows, axes):
            result = product[key]
            exponent = int(round(np.log10(scale)))
            for ax, comp, csym in zip(row, "tx", ("t", r"\times")):
                _theta_panel(ax)
                value, error, cov = _blocks(result)[comp]
                chi2, nu = chi2_null(value, cov, prov["npatch"])
                ax.errorbar(
                    result["theta"],
                    value * scale,
                    yerr=error * scale,
                    color=ALL,
                    marker="o",
                    **S.DATA_VECTOR_MARK,
                )
                ax.set_ylabel(rf"${sym},{{\,{csym}}}\ [10^{{-{exponent}}}]$")
                S.label_box(ax, rf"{what}, ${csym}$", loc="upper right")
                S.note(ax, _chi2_label(chi2, nu), loc="lower right")
                if ax in axes[0]:
                    ax.set_xlabel("")
        S.header(axes[0, 0], "PSF validation stars around galaxies, randoms subtracted")
        return S.save(fig, out, "full")


def alpha_mag(product, out):
    """α(θ) = τ₀/ρ₀ for all galaxies and in magnitude tertiles."""
    prov = product["provenance"]
    labels = _bin_labels(prov["galaxy_mag_edges"], ".1f", "m")
    series = [("all", ALL, product["all"])] + [
        (lab, c, product[f"mag{i}"]) for i, (lab, c) in enumerate(zip(labels, BINS))
    ]
    with S.figure_style():
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        _theta_panel(ax)
        for (label, colour, result), off in zip(series, _offsets(len(series))):
            ax.errorbar(
                result["theta"] * off,
                result["alpha"],
                yerr=result["alpha_error"],
                color=colour,
                marker="o",
                label=label,
                **S.DATA_VECTOR_MARK,
            )
        ax.set_ylabel(r"$\alpha(\theta) = \tau_{0,+}/\rho_{0,+}$")
        ax.legend(
            loc="upper left",
            fontsize=7.5,
            ncol=4,
            frameon=False,
            handletextpad=0.3,
            columnspacing=1.0,
        )
        top = ax.get_ylim()
        ax.set_ylim(top[0], top[1] + 0.25 * (top[1] - top[0]))
        return S.save(fig, out, "full")
