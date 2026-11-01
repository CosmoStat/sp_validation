"""Catalogue-free renderers for cached validation diagnostics.

Inputs are arrays, summary dictionaries or a sparse count map, never a
CosmologyValidation instance. No renderer skips an existing output.
"""

import matplotlib.pyplot as plt
import numpy as np
from cs_util.plots import FootprintPlotter


def plot_footprint(count_map, outputs):
    """Render every requested region from one cached HealSparse count map."""
    plotter = FootprintPlotter()
    for region, path in outputs.items():
        plotter.plot_region(
            count_map,
            plotter._regions[region],
            outpath=str(path),
            colorbar_label=f"Galaxies per pixel (NSIDE={count_map.nside_sparse})",
        )
        plt.close()


def plot_weights(histograms, styles, output):
    """Draw weight-weighted bin sums; binning is fixed in the compute step."""
    fig, ax = plt.subplots(figsize=(10, 7))
    for ver, (counts, edges) in histograms.items():
        ax.stairs(counts, edges, label=ver, color=styles[ver]["colour"])
    ax.set(xlabel="$w$", ylabel="frequency", yscale="log")
    ax.legend()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_objectwise_leakage(coefficients, styles, output):
    """Draw cached regression and scale-dependent alpha summaries.

    Each coefficient is a (nominal value, standard deviation) pair.
    """
    fig, ax = plt.subplots(figsize=(15, 15))
    summaries = {
        "alpha_mean": (r"$\bar\alpha$", "-", "full"),
        "alpha_1": (r"$\alpha(\theta_{\rm min})$", "--", "none"),
        "alpha_0": (r"$\alpha(0)$", ":", "left"),
    }
    handles = []
    if not coefficients:
        ax.text(
            0.5,
            0.5,
            "No object-wise leakage coefficients available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    for ver, bins in coefficients.items():
        style = styles[ver]
        for coeff in bins.values():
            x, dx = coeff["aii_mean"]
            for key, (_, ls, fillstyle) in summaries.items():
                y, dy = coeff[key]
                eb = ax.errorbar(
                    x,
                    y,
                    xerr=dx,
                    yerr=dy,
                    fmt=style["marker"],
                    color=style["colour"],
                    fillstyle=fillstyle,
                )
                eb[-1][0].set_linestyle(ls)
        handles.append(
            plt.Line2D(
                [],
                [],
                marker=style["marker"],
                ls="",
                color=style["colour"],
                label=style.get("label", ver),
            )
        )
    handles += [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls=ls,
            color="grey",
            fillstyle=fs,
            label=label,
        )
        for label, ls, fs in summaries.values()
    ]
    ax.plot([-0.02, 0.02], [-0.02, 0.02], "k:", linewidth=0.5)
    ax.legend(handles=handles)
    ax.set(
        xlabel=r"$\langle a_{ii} \rangle$ (object-wise)",
        ylabel=r"$\alpha$ (scale-dependent, $\tau_0 / \rho_0$)",
    )
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_rho_tau_fits(products, styles, contours, samples, summary):
    """Render posterior contours and propagated PSF errors from cached arrays."""
    from shear_psf_leakage import plots as psf_plots

    versions = list(products)
    psf_plots.plot_contours(
        [products[v]["flat_samples"] for v in versions],
        names=["x0", "x1", "x2"],
        labels=[r"\alpha", r"\beta", r"\eta"],
        savefig=str(contours),
        legend_labels=versions,
        legend_loc="upper right",
        contour_colors=[styles[v]["colour"] for v in versions],
        markers={"x0": 0, "x1": 1, "x2": 1},
        show=False,
        close=True,
    )
    for path, draw_samples in ((samples, True), (summary, False)):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
        for ver, data in products.items():
            theta, colour = data["theta"], styles[ver]["colour"]
            for ax, component, sign in zip(axes, ("plus", "minus"), ("+", "-")):
                if draw_samples:
                    ax.plot(
                        theta,
                        data[f"samples_{component}"][-100:].T,
                        color=colour,
                        alpha=0.3,
                    )
                else:
                    ax.plot(theta, data[f"mean_{component}"], color=colour, label=ver)
                    lo, hi = data[f"quantiles_{component}"]
                    ax.plot(theta, lo, color=colour)
                    ax.plot(theta, hi, color=colour)
                    ax.fill_between(theta, lo, hi, color=colour, alpha=0.3)
                ax.set(
                    xscale="log",
                    yscale="linear",
                    xlabel=r"$\theta$ [arcmin]",
                    ylabel=rf"$\xi^{{\rm PSF,sys}}_{sign}(\theta)$",
                )
        if not draw_samples:
            axes[0].legend()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def plot_ratio_xi_sys_xi(
    products, correlations, styles, output, offset=0.1, threshold=0.1
):
    """Draw PSF/signal ratios directly from sealed reporting correlations.

    Ratios aren't persisted separately: the only cosmological input stays the
    existing sealed SACC. Propagating cached variances is just presentation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (ver, data) in enumerate(products.items()):
        gg = correlations[ver]
        theta = gg.meanr * (1 + offset * (idx - (len(products) - 1) / 2))
        for ax, xi, var_xi, component, sign in zip(
            axes,
            (gg.xip, gg.xim),
            (gg.varxip, gg.varxim),
            ("plus", "minus"),
            ("+", "-"),
        ):
            mean, var = data[f"mean_{component}"], data[f"var_{component}"]
            ax.errorbar(
                theta,
                mean / xi,
                yerr=np.sqrt(var / xi**2 + mean**2 * var_xi / xi**4),
                fmt=styles[ver]["marker"],
                color=styles[ver]["colour"],
                markersize=3,
                capsize=2,
                label=ver,
            )
            ax.set(
                xscale="log",
                xlabel=r"$\theta$ [arcmin]",
                ylabel=rf"$\xi^{{\rm PSF,sys}}_{sign}/\xi_{sign}$",
            )
    for ax in axes:
        ax.axhspan(-threshold, threshold, color="black", alpha=0.1)
        ax.legend()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_2pcf(correlations, styles, outputs):
    """Draw sealed reporting xi±, both plain and theta-weighted."""
    for times_theta, output in enumerate(outputs):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
        for ver, gg in correlations.items():
            factor = gg.meanr if times_theta else 1
            for ax, xi, var, sign in zip(
                axes, (gg.xip, gg.xim), (gg.varxip, gg.varxim), ("+", "-")
            ):
                ax.errorbar(
                    gg.meanr,
                    xi * factor,
                    yerr=np.sqrt(var) * factor,
                    fmt=styles[ver]["marker"],
                    color=styles[ver]["colour"],
                    label=ver,
                )
                ax.set(
                    xscale="log",
                    xlabel=r"$\theta$ [arcmin]",
                    ylabel=(r"$\theta$ " if times_theta else "")
                    + rf"$\xi_{sign}(\theta)$",
                )
        axes[0].legend()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)


def plot_psf_stats(tables, styles, output, kind, covariances=()):
    """Overlay cached rho/tau FITS columns, with optional tau covariance errors."""
    indices = (0, 1, 2, 3, 4, 5) if kind == "rho" else (0, 2, 5)
    fig, axes = plt.subplots(
        2 if kind == "rho" else 1,
        3,
        figsize=(15, 10 if kind == "rho" else 6),
        squeeze=False,
        layout="constrained",
    )
    for i, stat in enumerate(indices):
        ax = axes.flat[i]
        for j, (ver, table) in enumerate(tables.items()):
            theta = table["theta"]
            key = f"{kind}_{stat}_p"
            variance = table[f"var{key}"]
            if covariances:
                n = len(theta)
                variance = np.diag(covariances[j])[i * n : (i + 1) * n]
            ax.errorbar(
                theta,
                table[key],
                yerr=np.sqrt(variance),
                color=styles[ver]["colour"],
                label=ver,
                capsize=2,
            )
        ax.set(
            xscale="log",
            xlabel=r"$\theta$ [arcmin]",
            ylabel=rf"$\{kind}_{{{stat},+}}(\theta)$",
        )
        ax.legend()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_covariance(matrix, output):
    """Draw a correlation matrix from the cached analytic covariance."""
    from sp_validation.statistics import corr_from_cov

    fig, ax = plt.subplots(figsize=(9, 9))
    image = ax.imshow(corr_from_cov(matrix), vmin=-1, vmax=1, cmap="RdBu_r")
    fig.colorbar(image, ax=ax, label="correlation")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
