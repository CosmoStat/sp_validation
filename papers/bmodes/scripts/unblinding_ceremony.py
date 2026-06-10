#!/usr/bin/env python3
"""Run the UNIONS unblinding ceremony figure sequence.

Usage (standalone)
------------------
app python workflow/scripts/unblinding_ceremony.py A
app python workflow/scripts/unblinding_ceremony.py A --output-dir /path/to/output

Usage (snakemake)
-----------------
snakemake unblinding_ceremony --config ceremony_blind=A
"""

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from getdist import plots
from matplotlib import scale as mscale
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d

from plotting_utils import PAPER_MPLSTYLE, SquareRootScale

# ── Plotting environment ────────────────────────────────────────────────────
mscale.register_scale(SquareRootScale)
plt.style.use(PAPER_MPLSTYLE)
plt.rc("text", usetex=True)


# ── Data types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChainSpec:
    root: str
    label: str
    color: str
    base_dir: Path
    alpha: float = 1.0


FULL_PARAMS = ["OMEGA_M", "ombh2", "h0", "n_s", "SIGMA_8", "s_8_input", "logt_agn", "a", "m1", "bias_1"]
COSMO_PARAMS = ["OMEGA_M", "s_8_input", "SIGMA_8", "a"]

MUTED_ALPHA = 0.25


@dataclass
class CeremonyConfig:
    """All paths and parameters needed by the ceremony, sourced from snakemake or CLI."""

    blind: str
    chain_version: str
    chain_prefix: str
    chain_root_dir: Path
    external_root_dir: Path
    results_dir: Path
    evidence_dir: Path

    xi_data_path: Path
    pure_eb_path: Path
    pseudo_cl_path: Path
    pseudo_cl_cov_path: Path
    cosmosis_cell_fits: Path
    bestfit_dir: Path
    bestfit_root_fid_cell: str
    bestfit_root_halofit_cell: str
    bestfit_root_config: str


def _save_path(cfg: CeremonyConfig, index: int, slug: str) -> Path:
    return cfg.results_dir / f"{index:02d}_{slug}.pdf"


# ── Chain loading (extracted from Sasha's notebooks) ────────────────────────

def _load_xi_table(path: Path | str, nrows: int = 20) -> np.ndarray:
    rows: list[np.ndarray] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = np.fromstring(stripped, sep=" ")
            if values.size < 9:
                continue
            rows.append(values)
            if len(rows) >= nrows:
                break
    if len(rows) < nrows:
        raise ValueError(f"Expected at least {nrows} xi rows in {path}, found {len(rows)}")
    return np.vstack(rows)


def ensure_getdist_chain(base_dir: Path, root: str) -> Path:
    """MAKE PARAMNAMES FILE + READ CHAIN conversion (from notebooks)."""
    chain_dir = base_dir / root
    samples_path = chain_dir / f"samples_{root}.txt"
    gd_samples_path = chain_dir / f"getdist_{root}.txt"
    paramnames_path = chain_dir / f"getdist_{root}.paramnames"

    if not samples_path.exists():
        return gd_samples_path

    with samples_path.open("r", encoding="utf-8") as file:
        params = file.readline()[1:].split("\t")[:-4]

    with paramnames_path.open("w", encoding="utf-8") as file:
        for param in params:
            if len(param.split("--")) > 1:
                file.write(param.split("--")[1] + "\n")
            else:
                file.write(param.split("--")[0] + "\n")

    samples = np.loadtxt(samples_path)
    if "nautilus" in root:
        samples = np.column_stack((np.exp(samples[:, -3]), samples[:, -1] - samples[:, -2], samples[:, 0:-3]))
    else:
        samples = np.column_stack((samples[:, -1], samples[:, -3], samples[:, 0:-4]))
    np.savetxt(gd_samples_path, samples)
    return gd_samples_path


def _build_plotter(
    width_inch: float,
    axes_fontsize: float,
    axes_labelsize: float,
    legend_fontsize: float,
):
    g = plots.get_subplot_plotter(width_inch=width_inch)
    g.settings.axes_fontsize = axes_fontsize
    g.settings.axes_labelsize = axes_labelsize
    g.settings.alpha_filled_add = 0.7
    g.settings.legend_fontsize = legend_fontsize
    return g


def _set_param_labels(chain) -> None:
    name_list = ["OMEGA_M", "ombh2", "h0", "n_s", "SIGMA_8", "S_8", "s_8_input", "logt_agn", "a", "m1", "bias_1"]
    label_list = [
        r"\Omega_{\rm m}",
        r"\omega_{\rm b} h^2",
        r"h_0",
        r"n_{\rm s}",
        r"\sigma_8",
        r"S_8",
        r"S_8",
        r"\log T_{\rm AGN}",
        r"A_{\rm IA}",
        r"m_1",
        r"\Delta z_1",
    ]

    param_names = chain.getParamNames()
    for name, label in zip(name_list, label_list):
        try:
            param_names.parWithName(name).label = label
        except Exception:
            pass

    try:
        param_names.parWithName("S_8")
    except Exception:
        try:
            s8_input = chain.getParams().s_8_input
            chain.addDerived(s8_input, name="S_8", label=r"S_8")
        except Exception:
            pass


def _adjust_paramname_chain(chain, current_name: str, target_name: str, label: str) -> None:
    try:
        param_names = chain.getParamNames()
        par = param_names.parWithName(current_name)
        par.label = label
        par.name = target_name
        chain.setParamNames(param_names)
    except Exception:
        pass


def _derive_parameter_s8(chain):
    if "S_8" in chain.getParamNames().list():
        return chain
    omega_m = chain.getParams().OMEGA_M
    sigma_8 = chain.getParams().SIGMA_8
    s_8 = sigma_8 * (omega_m / 0.3) ** 0.5
    chain.addDerived(s_8, name="S_8", label=r"S_8")
    return chain


def _harmonize_external_chain(chain, root: str) -> None:
    if root in {"Planck18", "KiDS-1000", "HSC_Y3", "HSC_Y3_cell", "DES+KiDS", "DES_Y3"}:
        _adjust_paramname_chain(chain, "omega_m", "OMEGA_M", r"\Omega_{\rm m}")
    if root == "DES_Y3":
        _derive_parameter_s8(chain)


def load_getdist_chains(
    chain_specs: list[ChainSpec],
    width_inch: float,
    axes_fontsize: float,
    axes_labelsize: float,
    legend_fontsize: float,
):
    g = _build_plotter(width_inch=width_inch, axes_fontsize=axes_fontsize, axes_labelsize=axes_labelsize, legend_fontsize=legend_fontsize)

    chains = []
    for spec in chain_specs:
        ensure_getdist_chain(spec.base_dir, spec.root)
        chain = g.samples_for_root(
            str(spec.base_dir / spec.root / f"getdist_{spec.root}"),
            cache=False,
            settings={"ignore_rows": 0, "smooth_scale_2D": 0.5, "smooth_scale_1D": 0.5},
        )
        _set_param_labels(chain)
        if spec.base_dir.name == "ext_data" or spec.root in {"Planck18", "DES_Y3", "KiDS-1000", "DES+KiDS", "HSC_Y3", "HSC_Y3_cell"}:
            _harmonize_external_chain(chain, spec.root)
        chains.append(chain)

    return g, chains


# ── Plot functions (extracted from Sasha's notebooks) ───────────────────────

def plot_triangle(
    chain_specs: list[ChainSpec],
    param_names: list[str],
    output_path: Path,
    width_inch: float = 20.0,
    axes_fontsize: float = 35.0,
    axes_labelsize: float = 50.0,
    legend_fontsize: float = 40.0,
    legend_loc: str = "upper right",
) -> None:
    """Extracted triangle_plot pattern from contour notebooks."""
    g, chains = load_getdist_chains(chain_specs, width_inch, axes_fontsize, axes_labelsize, legend_fontsize)

    colours = [spec.color for spec in chain_specs]
    linestyle = ["solid" for _ in chain_specs]
    line_args = [dict(color=col, ls=ls) for col, ls in zip(colours, linestyle)]

    g.triangle_plot(
        chains,
        param_names,
        legend_labels=[spec.label for spec in chain_specs],
        line_args=line_args,
        contour_colors=colours,
        legend_loc=legend_loc,
        filled=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.export(str(output_path))
    plt.close(g.fig)


def plot_xipm_data_vector(xipm_path: str, output_path: Path) -> None:
    """Extracted from 2D_cosmic_shear_paper_plots/corr_func.ipynb (xi+ / xi- blocks)."""
    xipm = _load_xi_table(xipm_path, nrows=20)
    theta = xipm[:, 1]
    xip = xipm[:, 3]
    xim = xipm[:, 4]
    varxip = xipm[:, 7]
    varxim = xipm[:, 8]

    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 4.5))

    ax1.tick_params(axis="both", which="both", direction="in", length=6, width=1, top=True, bottom=True, left=True, right=True)
    ax1.yaxis.minorticks_on()
    ax1.plot(theta, xip * 1e4, marker="o", markersize=4, ls="solid", lw=1.8, color="royalblue")
    ax1.fill_between(theta, (xip - varxip) * 1e4, (xip + varxip) * 1e4, color="powderblue", alpha=0.7)
    ax1.text(0.85, 0.88, "1-1", transform=ax1.transAxes, bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", pad=0.5))
    ax1.axvspan(0, 10, color="gray", alpha=0.3)
    ax1.axvspan(150, 200, color="gray", alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\theta$ [arcmin]")
    ax1.set_ylabel(r"$\xi_+\times 10^4$")

    ax2.tick_params(axis="both", which="both", direction="in", length=6, width=1, top=True, bottom=True, left=True, right=True)
    ax2.yaxis.minorticks_on()
    ax2.plot(theta, xim * 1e4, marker="o", markersize=4, ls="solid", lw=1.8, color="orangered")
    ax2.fill_between(theta, (xim - varxim) * 1e4, (xim + varxim) * 1e4, color="pink", alpha=0.7)
    ax2.text(0.85, 0.88, "1-1", transform=ax2.transAxes, bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", pad=0.5))
    ax2.axvspan(0, 10, color="gray", alpha=0.3)
    ax2.axvspan(150, 200, color="gray", alpha=0.3)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\theta$ [arcmin]")
    ax2.set_ylabel(r"$\xi_-\times 10^4$")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cell_ee_data_vector(pseudo_cl_path: str, pseudo_cl_cov_path: str, output_path: Path) -> None:
    """Extracted from 2025_10_08_plot_data_vectors.py (EE panel logic)."""
    cell = fits.getdata(pseudo_cl_path)
    cov_cell = fits.open(pseudo_cl_cov_path)

    ell = cell["ell"]
    cl_ee = cell["EE"]
    cov_cl_ee = cov_cell["COVAR_EE_EE"].data
    cov_cell.close()

    fig, ax0 = plt.subplots(ncols=1, nrows=1, figsize=(7, 5))
    ax0.errorbar(
        ell,
        cl_ee * ell,
        yerr=np.sqrt(np.diag(cov_cl_ee)) * ell,
        label=r"$C_\ell^{EE}$",
        color="royalblue",
        fmt="o",
        capsize=2,
    )

    ax0.set_xscale("squareroot")
    ax0.set_xticks(np.array([100, 400, 900, 1600]))
    ax0.minorticks_on()
    ax0.tick_params(axis="x", which="minor", length=2, width=0.8)
    minor_ticks = [i * 10 for i in range(1, 10)] + [i * 100 for i in range(1, 21)]
    ax0.set_xticks(minor_ticks, minor=True)
    ax0.legend()

    ax0.set_xlabel(r"$\ell$")
    ax0.set_ylabel(r"$\ell \, C_\ell^{EE}$")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_xipm_bestfit_with_bmodes(
    xi_data_path: Path | str,
    pure_eb_data_path: Path,
    bestfit_dir: Path | None,
    output_path: Path,
    scale_cut_xip: tuple[float, float] = (12.0, 83.0),
    scale_cut_xim: tuple[float, float] = (12.0, 83.0),
) -> None:
    """Extracted from 2D_cosmic_shear_paper_plots/workflow/scripts/plot_xi_bestfit.py.

    If bestfit_dir is None, plots data + B-modes only (no theory curves).

    xi_data_path can be:
    - A CosmoSIS FITS file with XI_PLUS/XI_MINUS HDUs and COVMAT (preferred)
    - A plain-text TreeCorr output table (legacy)
    """
    xi_path = Path(xi_data_path)
    if xi_path.suffix == ".fits":
        xip_hdu = fits.getdata(str(xi_path), "XI_PLUS")
        xim_hdu = fits.getdata(str(xi_path), "XI_MINUS")
        cov = fits.getdata(str(xi_path), "COVMAT")
        theta_data = xip_hdu["ANG"]
        xip_data = xip_hdu["VALUE"]
        xim_data = xim_hdu["VALUE"]
        n = len(xip_data)
        sigma_xip = np.sqrt(np.diag(cov[:n, :n]))
        sigma_xim = np.sqrt(np.diag(cov[n : 2 * n, n : 2 * n]))
    else:
        data = _load_xi_table(xi_path, nrows=20)
        theta_data = data[:, 1]
        xip_data = data[:, 3]
        xim_data = data[:, 4]
        sigma_xip = data[:, 7]
        sigma_xim = data[:, 8]

    eb_data = np.load(pure_eb_data_path)
    theta_eb = eb_data["theta"]
    xip_B = eb_data["xip_B"]
    xim_B = eb_data["xim_B"]
    cov_pure_eb = eb_data["cov_pure_eb"]

    nbins = len(theta_eb)
    sigma_xip_B = np.sqrt(np.diag(cov_pure_eb[2 * nbins : 3 * nbins, 2 * nbins : 3 * nbins]))
    sigma_xim_B = np.sqrt(np.diag(cov_pure_eb[3 * nbins : 4 * nbins, 3 * nbins : 4 * nbins]))

    min_sep, max_sep = 1.0, 250.0
    bin_edges = np.geomspace(min_sep, max_sep, nbins + 1)
    bin_centers_nominal = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    def get_bin_edge_cuts(centers, edges, scale_cut):
        mask = (centers >= scale_cut[0]) & (centers <= scale_cut[1])
        idx_first = np.where(mask)[0][0]
        idx_last = np.where(mask)[0][-1]
        return edges[idx_first], edges[idx_last + 1]

    edge_cut_xip = get_bin_edge_cuts(bin_centers_nominal, bin_edges, scale_cut_xip)
    edge_cut_xim = get_bin_edge_cuts(bin_centers_nominal, bin_edges, scale_cut_xim)

    has_theory = bestfit_dir is not None
    if has_theory:
        theta_theory_rad = np.loadtxt(bestfit_dir / "shear_xi_plus" / "theta.txt", comments="#")
        theta_theory = np.rad2deg(theta_theory_rad) * 60
        xip_theory = np.loadtxt(bestfit_dir / "shear_xi_plus" / "bin_1_1.txt", comments="#")
        xim_theory = np.loadtxt(bestfit_dir / "shear_xi_minus" / "bin_1_1.txt", comments="#")

        theta_sys_rad = np.loadtxt(bestfit_dir / "xi_sys" / "theta.txt", comments="#")
        theta_sys = np.rad2deg(theta_sys_rad) * 60
        xip_sys = np.loadtxt(bestfit_dir / "xi_sys" / "shear_xi_plus.txt", comments="#")
        xim_sys = np.loadtxt(bestfit_dir / "xi_sys" / "shear_xi_minus.txt", comments="#")

    theta_fine = np.geomspace(0.5, 300, 500)
    if has_theory:
        xip_th_interp = interp1d(theta_theory, xip_theory, kind="cubic", fill_value="extrapolate")(theta_fine)
        xim_th_interp = interp1d(theta_theory, xim_theory, kind="cubic", fill_value="extrapolate")(theta_fine)
        xip_sys_interp = interp1d(theta_sys, xip_sys, kind="cubic", fill_value="extrapolate")(theta_fine)
        xim_sys_interp = interp1d(theta_sys, xim_sys, kind="cubic", fill_value="extrapolate")(theta_fine)

    scale_factor = 1e-4
    xlim = [1, 250]
    ylim = [-0.15, 1.25]

    ms_data = 3
    ms_bmode = 3
    capsize = 1.5
    elinewidth = 0.8

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    plot_configs = [
        (axes[0], xip_data, sigma_xip, xip_B, sigma_xip_B,
         xip_th_interp if has_theory else None, xip_sys_interp if has_theory else None,
         edge_cut_xip, r"$\xi_+$", "+"),
        (axes[1], xim_data, sigma_xim, xim_B, sigma_xim_B,
         xim_th_interp if has_theory else None, xim_sys_interp if has_theory else None,
         edge_cut_xim, r"$\xi_-$", "-"),
    ]

    for idx, (ax, xi_data_arr, sigma_xi, xi_B, sigma_B, xi_th, xi_sys_arr, edge_cut, label, _pm) in enumerate(plot_configs):
        show_legend = idx == 1

        ax.axvspan(xlim[0], edge_cut[0], color="0.90", zorder=0, alpha=0.7)
        ax.axvspan(edge_cut[1], xlim[1], color="0.90", zorder=0, alpha=0.7)

        if has_theory:
            ax.plot(
                theta_fine,
                theta_fine * (xi_th + xi_sys_arr) / scale_factor,
                "-",
                color="k",
                lw=1.5,
                label=r"Best-fit $\xi^{\mathrm{th}}_\pm + \xi^{\mathrm{sys}}_\pm$" if show_legend else None,
                zorder=2,
            )
            ax.plot(
                theta_fine,
                theta_fine * xi_sys_arr / scale_factor,
                "-",
                color="C0",
                lw=1.2,
                label=r"Best-fit $\xi^{\mathrm{sys}}_\pm$" if show_legend else None,
                zorder=2,
            )

        ax.errorbar(
            theta_data,
            theta_data * xi_data_arr / scale_factor,
            yerr=theta_data * sigma_xi / scale_factor,
            fmt="o",
            color="k",
            markersize=ms_data,
            capsize=capsize,
            elinewidth=elinewidth,
            label=r"$\xi_\pm$" if show_legend else None,
            zorder=3,
        )

        theta_eb_offset = theta_eb * 1.03
        ax.errorbar(
            theta_eb_offset,
            theta_eb_offset * xi_B / scale_factor,
            yerr=theta_eb_offset * sigma_B / scale_factor,
            fmt="o",
            color="C3",
            markersize=ms_bmode,
            capsize=capsize,
            elinewidth=elinewidth,
            alpha=0.85,
            label=r"$\xi^B_\pm$" if show_legend else None,
            zorder=3,
        )

        ax.axhline(0, color="gray", linestyle="--", alpha=0.8, linewidth=0.8, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$\theta$ (arcmin)")
        ax.set_title(label)
        if show_legend:
            ax.legend(loc="upper left")

    axes[0].set_ylabel(r"$\theta\xi \times 10^4$")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _get_stats_row(chain, label: str, color: str) -> list[str | float]:
    margestats = chain.getMargeStats()
    try:
        s8_stats = margestats.parWithName("S_8")
    except Exception:
        s8_stats = margestats.parWithName("s_8_input")
    sigma8_stats = margestats.parWithName("SIGMA_8")
    omegam_stats = margestats.parWithName("OMEGA_M")
    return [
        label,
        color,
        s8_stats.mean,
        s8_stats.mean - s8_stats.limits[0].lower,
        s8_stats.limits[0].upper - s8_stats.mean,
        sigma8_stats.mean,
        sigma8_stats.mean - sigma8_stats.limits[0].lower,
        sigma8_stats.limits[0].upper - sigma8_stats.mean,
        omegam_stats.mean,
        omegam_stats.mean - omegam_stats.limits[0].lower,
        omegam_stats.limits[0].upper - omegam_stats.mean,
    ]


def get_sigma_tension(mean1, low1, high1, mean2, low2, high2):
    sigma1 = 0.5 * (high1 + low1)
    sigma2 = 0.5 * (high2 + low2)
    delta_mean = np.abs(mean1 - mean2)
    sigma_tension = delta_mean / np.sqrt(sigma1**2 + sigma2**2)
    sign = 1 if mean1 > mean2 else -1
    return sigma_tension * sign


def plot_cell_ee_with_bestfit(
    cosmosis_data_path: str,
    bestfit_specs: list[tuple[str, str, dict]],
    output_folder: str,
    output_path: Path,
    ell_min: float = 10.0,
    ell_max: float = 2048.0,
    label_data: str = "Fiducial data",
) -> None:
    """Extracted from get_chi2_cell.ipynb plot_best_fit() (cell 21-22).

    Parameters
    ----------
    cosmosis_data_path : str
        Path to CosmoSIS FITS file with CELL_EE and COVMAT HDUs.
    bestfit_specs : list of (label, root, line_args_dict)
        Each entry is (legend label, chain root name, dict of plot kwargs).
    output_folder : str
        Base path to best_fit directories (e.g. /n09data/guerrini/output_chains/).
    output_path : Path
        Where to save the figure.
    """
    data = fits.getdata(cosmosis_data_path, "CELL_EE")
    cov_mat = fits.getdata(cosmosis_data_path, "COVMAT")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ell = data["ANG"]
    cell = data["VALUE"]
    ax.errorbar(
        ell,
        ell * cell,
        yerr=ell * np.sqrt(np.diag(cov_mat)),
        fmt="o",
        label=label_data,
        color="black",
        capsize=2,
    )

    for label, root, line_kw in bestfit_specs:
        ell_th = np.loadtxt(f"{output_folder}/best_fit/{root}/shear_cl/ell.txt")
        shear_cl = np.loadtxt(f"{output_folder}/best_fit/{root}/shear_cl/bin_1_1.txt")
        mask = (ell_th > ell_min) & (ell_th < ell_max)
        ax.plot(ell_th[mask], ell_th[mask] * shear_cl[mask], label=label, **line_kw)

    ax.axvline(x=1800, color="black", linestyle="--", alpha=0.5)
    ax.axvline(x=2048, color="black", linestyle="--", alpha=1.0)
    ax.axvline(x=500, color="black", linestyle="--", alpha=0.3)

    ax.text(
        1740, 0.90, r"$k_\mathrm{max} = 3 h$ Mpc$^{-1}$",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=10, rotation=90,
    )
    ax.text(
        1978, 0.90, r"$k_\mathrm{max} = 5 h$ Mpc$^{-1}$",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=10, rotation=90,
    )
    ax.text(
        470, 0.90, r"$k_\mathrm{max} = 1 h$ Mpc$^{-1}$",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=10, rotation=90,
    )

    ax.set_ylabel(r"$\ell C_\ell$", fontsize=16)
    ax.set_xlabel(r"$\ell$", fontsize=16)
    ax.set_xlim(ell.min() - 10, ell.max() + 100)
    ax.set_xscale("squareroot")
    ax.set_xticks(np.array([100, 400, 900, 1600]))
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", length=2, width=0.8)
    minor_ticks = [i * 10 for i in range(1, 10)] + [i * 100 for i in range(1, 21)]
    ax.xaxis.set_ticks(minor_ticks, minor=True)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="both", which="minor", labelsize=10)
    ax.yaxis.get_offset_text().set_fontsize(14)

    plt.legend(loc="lower center", bbox_to_anchor=(0.685, 0.70), fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_s8_whisker(
    chain_specs: list[ChainSpec],
    output_path: Path,
    reference_labels: list[str] | None = None,
    reference_colors: list[str] | None = None,
    reference_label: str | None = None,
) -> None:
    """Extracted from 2025_10_28_plot_whisker.ipynb (cells 5-11).

    Supports multiple reference bands: pass reference_labels and
    reference_colors as parallel lists. Each reference gets its own
    shaded band in the corresponding color. For backwards compatibility,
    a single reference_label still works.
    """
    if reference_labels is None and reference_label is not None:
        reference_labels = [reference_label]
        reference_colors = reference_colors or [None]

    g, chains = load_getdist_chains(chain_specs, width_inch=30, axes_fontsize=60, axes_labelsize=60, legend_fontsize=60)
    plt.close(g.fig)

    labels = [spec.label for spec in chain_specs]
    colours = [spec.color for spec in chain_specs]
    alphas = [spec.alpha for spec in chain_specs]

    param_values = np.array(
        [["# Expt", "Colour", "S8_Mean", "S8_low", "S8_high", "sigma_8_Mean", "sigma_8_low", "sigma_8_high", "Omega_m_Mean", "Omega_m_low", "Omega_m_high"]],
        dtype=object,
    )

    escaped_labels = np.char.replace(np.array(labels), "\\", "\\\\")
    for i, chain in enumerate(chains):
        row = _get_stats_row(chain, escaped_labels[i], colours[i])
        param_values = np.vstack((param_values, row))

    expt = np.char.replace(param_values[1:, 0].astype(str), "\\\\", "\\")
    colours_arr = param_values[1:, 1].astype(str)
    s8_mean = param_values[1:, 2].astype(np.float64)
    s8_low = param_values[1:, 3].astype(np.float64)
    s8_high = param_values[1:, 4].astype(np.float64)
    sigma8_mean = param_values[1:, 5].astype(np.float64)
    sigma8_low = param_values[1:, 6].astype(np.float64)
    sigma8_high = param_values[1:, 7].astype(np.float64)
    omegam_mean = param_values[1:, 8].astype(np.float64)
    omegam_low = param_values[1:, 9].astype(np.float64)
    omegam_high = param_values[1:, 10].astype(np.float64)

    ref_indices = []
    for rl in (reference_labels or []):
        matches = np.where(expt == rl)[0]
        if len(matches):
            ref_indices.append(matches[0])
    ref_label_set = set(reference_labels or [])

    n_rows = len(expt)
    fig_height = max(6, 0.5 * n_rows + 1)
    fig = plt.figure(figsize=(10, fig_height))
    gs = GridSpec(1, 3, width_ratios=[1, 0.5, 0.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    ax3 = fig.add_subplot(gs[2], sharey=ax1)

    axs = [ax1, ax2, ax3]

    params = [
        (s8_mean, s8_low, s8_high, r"$S_8$"),
        (sigma8_mean, sigma8_low, sigma8_high, r"$\sigma_8$"),
        (omegam_mean, omegam_low, omegam_high, r"$\Omega_{\rm m}$"),
    ]

    row_spacing = 0.1
    y = np.arange(len(expt))

    for ax, param in zip(axs, params):
        means, lows, highs, label = param
        for i, mean, low, high, color, alpha in zip(y, means, lows, highs, colours_arr, alphas):
            ax.errorbar(mean, 0.05 + i * row_spacing, xerr=np.array([low, high])[:, None], fmt="o", color=color, ecolor=color, elinewidth=2, capsize=3, alpha=alpha)
        ax.set_xlabel(label, fontsize=14)

        for ri, ref_idx in enumerate(ref_indices):
            band_color = (reference_colors[ri] if reference_colors and ri < len(reference_colors) else colours_arr[ref_idx]) or colours_arr[ref_idx]
            ax.axvspan(means[ref_idx] - lows[ref_idx], means[ref_idx] + highs[ref_idx], color=band_color, alpha=0.15, zorder=0)

        ax.grid(False)
        ax.tick_params(axis="y", left=False, labelleft=False)
        if label == r"$S_8$":
            ax.set_xlim(0.25, 1.05)
        elif label == r"$\sigma_8$":
            ax.set_xlim(0.5, 1.2)
        elif label == r"$\Omega_{\rm m}$":
            ax.set_xlim(0.1, 0.5)

    axs[0].set_yticks(0.05 + y * row_spacing)
    axs[0].set_yticklabels([])
    for label, color, alpha in zip(expt, colours_arr, alphas):
        idx = np.where(expt == label)[0][0]
        yloc = 0.05 + row_spacing * idx
        axs[0].text(0.26, yloc, label, fontsize=12, ha="left", va="center", color=color, alpha=alpha)
        if label not in ref_label_set and ref_indices:
            ri0 = ref_indices[0]
            s8_tension = get_sigma_tension(s8_mean[idx], s8_low[idx], s8_high[idx], s8_mean[ri0], s8_low[ri0], s8_high[ri0])
            sign_str = "+" if s8_tension > 0 else "-"
            axs[0].text(1.045, yloc, rf"${sign_str}{np.abs(s8_tension):.2f}" + r"\, \sigma$", fontsize=10, ha="right", va="center", color=color, alpha=alpha)

    plt.gca().invert_yaxis()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Snakemake entry ──────────────────────────────────────────────────────────

def _config_from_snakemake(smk) -> CeremonyConfig:
    """Build config from snakemake.input / snakemake.output / snakemake.params."""
    chain_root_dir = Path(smk.params.chain_root_dir)

    return CeremonyConfig(
        blind=smk.params.blind,
        chain_version=smk.params.chain_version,
        chain_prefix=smk.params.chain_prefix,
        chain_root_dir=chain_root_dir,
        external_root_dir=chain_root_dir / "ext_data",
        results_dir=Path(smk.params.results_dir),
        evidence_dir=Path(smk.output.evidence).parent,
        xi_data_path=Path(smk.input.xi_data),
        pure_eb_path=Path(smk.input.pure_eb),
        pseudo_cl_path=Path(smk.input.pseudo_cl),
        pseudo_cl_cov_path=Path(smk.input.pseudo_cl_cov),
        cosmosis_cell_fits=Path(smk.input.cosmosis_cell_fits),
        bestfit_dir=Path(smk.input.bestfit_dir).parent.parent,
        bestfit_root_fid_cell=smk.params.bestfit_root_fid_cell,
        bestfit_root_halofit_cell=smk.params.bestfit_root_halofit_cell,
        bestfit_root_config=smk.params.bestfit_root_config,
    )


# ── CLI entry ────────────────────────────────────────────────────────────────

_CHAIN_ROOT_DIR = Path("/n09data/guerrini/output_chains")
_COSMOSIS_DATA_DIR = Path("/home/guerrini/sp_validation/cosmo_inference/data")
_DEFAULT_CHAIN_VERSION = "v1.4.6"


def _require_path(path: Path, label: str) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(f"{label}: {path}")


def _require_bestfit_root(chain_root_dir: Path, root: str) -> str:
    if (chain_root_dir / "best_fit" / root / "shear_cl" / "ell.txt").exists():
        return root
    raise FileNotFoundError(f"No best-fit shear_cl for {root}")


def _config_from_cli() -> CeremonyConfig:
    """Build config from command-line arguments + path resolution.

    Uses the exact data vectors from inference (Lisa's xi_pm, Sasha's pseudo-Cl
    and CosmoSIS FITS) — the same files the chains were fit to.
    """
    _PROJECT_ROOT = _SCRIPT_DIR.parent.parent

    parser = argparse.ArgumentParser(description="Run the UNIONS unblinding ceremony plot sequence.")
    parser.add_argument("blind", choices=["A", "B", "C"], help="Revealed blind letter")
    parser.add_argument("--chain-version", default=_DEFAULT_CHAIN_VERSION, help="Chain version (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for results (default: <project_root>/results/unblinding)")
    args = parser.parse_args()

    blind = args.blind
    chain_version = args.chain_version
    chain_prefix = f"SP_{chain_version}_leak_corr"
    output_dir = args.output_dir or (_PROJECT_ROOT / "results" / "unblinding")

    xi_data_path = _require_path(
        _COSMOSIS_DATA_DIR / f"{chain_prefix}_{blind}" / f"cosmosis_{chain_prefix}_{blind}.fits",
        f"CosmoSIS xi FITS for blind {blind}",
    )

    pseudo_cl_path = _require_path(
        Path(f"/home/guerrini/sp_validation/notebooks/cosmo_val/output/pseudo_cl_{chain_prefix}.fits"),
        f"pseudo-Cl for {chain_prefix} (Guerrini)",
    )
    pseudo_cl_cov_path = _require_path(
        Path(f"/home/guerrini/sp_validation/notebooks/cosmo_val/output/pseudo_cl_cov_{chain_prefix}.fits"),
        f"pseudo-Cl covariance for {chain_prefix} (Guerrini)",
    )

    cosmosis_cell_fits = _require_path(
        _COSMOSIS_DATA_DIR / f"{chain_prefix}_{blind}_fid" / f"cosmosis_{chain_prefix}_{blind}_fid_cell.fits",
        f"CosmoSIS C_ell FITS for blind {blind}",
    )

    bestfit_root_config = f"{chain_prefix}_{blind}_10_80"
    bestfit_dir = _require_path(
        _CHAIN_ROOT_DIR / "best_fit" / bestfit_root_config,
        f"Best-fit directory for blind {blind}",
    )

    return CeremonyConfig(
        blind=blind,
        chain_version=chain_version,
        chain_prefix=chain_prefix,
        chain_root_dir=_CHAIN_ROOT_DIR,
        external_root_dir=_CHAIN_ROOT_DIR / "ext_data",
        results_dir=output_dir,
        evidence_dir=output_dir / "claims" / "unblinding_ceremony",
        xi_data_path=xi_data_path,
        pure_eb_path=_require_path(
            _PROJECT_ROOT / "results" / "paper_plots" / "intermediate" / f"{chain_prefix}_{blind}_pure_eb_semianalytic.npz",
            f"Pure E/B file for blind {blind}",
        ),
        pseudo_cl_path=pseudo_cl_path,
        pseudo_cl_cov_path=pseudo_cl_cov_path,
        cosmosis_cell_fits=cosmosis_cell_fits,
        bestfit_dir=bestfit_dir,
        bestfit_root_fid_cell=_require_bestfit_root(_CHAIN_ROOT_DIR, f"{chain_prefix}_{blind}_fid_cell"),
        bestfit_root_halofit_cell=_require_bestfit_root(_CHAIN_ROOT_DIR, f"{chain_prefix}_{blind}_halofit_cell"),
        bestfit_root_config=bestfit_root_config,
    )


# ── Main ceremony ────────────────────────────────────────────────────────────

def run_ceremony(cfg: CeremonyConfig) -> None:
    blind = cfg.blind
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    reveal_harmonic_root = f"{cfg.chain_prefix}_{blind}_lmin=300_lmax=1600_cell"
    reveal_config_root = f"{cfg.chain_prefix}_{blind}_10_80"

    # ── Act 1 — The Data (naked, then with fits) ──────────────────────

    # 01: xi+/- data + B-modes, no theory
    plot_xipm_bestfit_with_bmodes(
        xi_data_path=cfg.xi_data_path,
        pure_eb_data_path=cfg.pure_eb_path,
        bestfit_dir=None,
        output_path=_save_path(cfg, 1, "xi_pm_data"),
    )

    # 02: C_ell^EE data, no theory
    plot_cell_ee_data_vector(
        str(cfg.pseudo_cl_path), str(cfg.pseudo_cl_cov_path), _save_path(cfg, 2, "cell_ee_data")
    )

    # 03: xi+/- with config-space best-fit (Paper IV Fig 1)
    plot_xipm_bestfit_with_bmodes(
        xi_data_path=cfg.xi_data_path,
        pure_eb_data_path=cfg.pure_eb_path,
        bestfit_dir=cfg.bestfit_dir,
        output_path=_save_path(cfg, 3, f"xi_bestfit_blind_{blind}"),
    )

    # 04: C_ell^EE with harmonic + config best-fit (Paper V Fig 2)
    plot_cell_ee_with_bestfit(
        cosmosis_data_path=str(cfg.cosmosis_cell_fits),
        bestfit_specs=[
            (
                rf"UNIONS $C_\ell$, Blind {blind}",
                cfg.bestfit_root_fid_cell,
                {"color": "royalblue", "linestyle": "-"},
            ),
            (
                r"UNIONS $C_\ell$, Halofit",
                cfg.bestfit_root_halofit_cell,
                {"color": "royalblue", "linestyle": "--"},
            ),
            (
                rf"UNIONS $\xi_\pm(\vartheta)$ (Goh et al., 2026)",
                cfg.bestfit_root_config,
                {"color": "orange", "linestyle": "-"},
            ),
        ],
        output_folder=str(cfg.chain_root_dir),
        output_path=_save_path(cfg, 4, f"cell_ee_bestfit_blind_{blind}"),
    )

    # ── Act 2 — The Reveal ────────────────────────────────────────────

    # 05: Consistency — Omega_m-S8, harmonic vs config vs Planck
    plot_triangle(
        [
            ChainSpec(
                root=reveal_harmonic_root,
                label=rf"UNIONS $C_\ell$, Blind {blind}",
                color="royalblue",
                base_dir=cfg.chain_root_dir,
            ),
            ChainSpec(
                root=reveal_config_root,
                label=rf"UNIONS $\xi_\pm(\vartheta)$, Blind {blind}",
                color="orange",
                base_dir=cfg.chain_root_dir,
            ),
            ChainSpec(
                root="Planck18",
                label=r"\textit{Planck} 2018",
                color="violet",
                base_dir=cfg.external_root_dir,
            ),
        ],
        ["OMEGA_M", "S_8"],
        _save_path(cfg, 5, f"consistency_blind_{blind}"),
        width_inch=12,
        axes_fontsize=24,
        axes_labelsize=28,
    )

    # 06: 4-param triangle — revealed blind, harmonic + config overlaid
    plot_triangle(
        [
            ChainSpec(
                root=reveal_harmonic_root,
                label=rf"UNIONS $C_\ell$, Blind {blind}",
                color="royalblue",
                base_dir=cfg.chain_root_dir,
            ),
            ChainSpec(
                root=reveal_config_root,
                label=rf"UNIONS $\xi_\pm(\vartheta)$, Blind {blind}",
                color="orange",
                base_dir=cfg.chain_root_dir,
            ),
        ],
        COSMO_PARAMS,
        _save_path(cfg, 6, f"triangle_cosmo_blind_{blind}"),
        width_inch=20,
    )

    # 07: Full-param triangle — revealed blind, harmonic + config overlaid
    plot_triangle(
        [
            ChainSpec(
                root=reveal_harmonic_root,
                label=rf"UNIONS $C_\ell$, Blind {blind}",
                color="royalblue",
                base_dir=cfg.chain_root_dir,
            ),
            ChainSpec(
                root=reveal_config_root,
                label=rf"UNIONS $\xi_\pm(\vartheta)$, Blind {blind}",
                color="orange",
                base_dir=cfg.chain_root_dir,
            ),
        ],
        FULL_PARAMS,
        _save_path(cfg, 7, f"triangle_full_blind_{blind}"),
        width_inch=30,
        axes_fontsize=26,
        axes_labelsize=28,
    )

    # ── Act 3 — In Context ────────────────────────────────────────────

    # 08: S8 whisker — all 6 UNIONS blinds (3 harmonic + 3 config),
    #     revealed blind highlighted, rest muted. Plus external surveys.
    whisker_specs = []

    for b in ("A", "B", "C"):
        is_revealed = b == blind
        alpha = 1.0 if is_revealed else MUTED_ALPHA
        whisker_specs.append(
            ChainSpec(
                root=f"{cfg.chain_prefix}_{b}_lmin=300_lmax=1600_cell",
                label=rf"UNIONS $C_\ell$, Blind {b}",
                color="royalblue",
                base_dir=cfg.chain_root_dir,
                alpha=alpha,
            )
        )
        whisker_specs.append(
            ChainSpec(
                root=f"{cfg.chain_prefix}_{b}_10_80",
                label=rf"UNIONS $\xi_\pm$, Blind {b}",
                color="orange",
                base_dir=cfg.chain_root_dir,
                alpha=alpha,
            )
        )

    whisker_specs.extend([
        ChainSpec(root="Planck18", label=r"\textit{Planck} 2018", color="black", base_dir=cfg.external_root_dir),
        ChainSpec(root="DES_Y3", label=r"DES Y3 $\xi_\pm$", color="black", base_dir=cfg.external_root_dir),
        ChainSpec(root="KiDS-1000", label=r"KiDS-1000 $\xi_\pm$", color="black", base_dir=cfg.external_root_dir),
        ChainSpec(root="HSC_Y3", label=r"HSC Y3 $\xi_\pm$", color="black", base_dir=cfg.external_root_dir),
    ])

    plot_s8_whisker(
        whisker_specs,
        reference_labels=[
            rf"UNIONS $C_\ell$, Blind {blind}",
            rf"UNIONS $\xi_\pm$, Blind {blind}",
        ],
        reference_colors=["royalblue", "orange"],
        output_path=_save_path(cfg, 8, f"s8_whisker_blind_{blind}"),
    )

    # ── Write evidence ───────────────────────────────────────────────

    produced_figures = [
        _save_path(cfg, 1, "xi_pm_data"),
        _save_path(cfg, 2, "cell_ee_data"),
        _save_path(cfg, 3, f"xi_bestfit_blind_{blind}"),
        _save_path(cfg, 4, f"cell_ee_bestfit_blind_{blind}"),
        _save_path(cfg, 5, f"consistency_blind_{blind}"),
        _save_path(cfg, 6, f"triangle_cosmo_blind_{blind}"),
        _save_path(cfg, 7, f"triangle_full_blind_{blind}"),
        _save_path(cfg, 8, f"s8_whisker_blind_{blind}"),
    ]
    output_dict = {}
    for fig_path in produced_figures:
        output_dict[fig_path.stem] = fig_path.name

    cfg.evidence_dir.mkdir(parents=True, exist_ok=True)

    evidence = {
        "id": "unblinding_ceremony",
        "spec_id": "unblinding_ceremony",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": {
            "xi_data": str(cfg.xi_data_path),
            "pure_eb_data": str(cfg.pure_eb_path),
            "pseudo_cl": str(cfg.pseudo_cl_path),
            "pseudo_cl_cov": str(cfg.pseudo_cl_cov_path),
            "cosmosis_cell_fits": str(cfg.cosmosis_cell_fits),
            "harmonic_chains": str(cfg.chain_root_dir / f"{cfg.chain_prefix}_{{A,B,C}}_lmin=300_lmax=1600_cell"),
            "config_chains": str(cfg.chain_root_dir / f"{cfg.chain_prefix}_{{A,B,C}}_10_80"),
            "external_chains": str(cfg.external_root_dir / "{Planck18,DES_Y3,KiDS-1000,HSC_Y3}"),
        },
        "output": output_dict,
        "params": {
            "chain_version": cfg.chain_version,
            "blind": blind,
            "scale_cut_arcmin": "12-83",
            "ell_range": "300-1600",
            "n_figures": len(output_dict),
        },
        "evidence": {
            "ceremony_date": "2026-02-27",
            "script": "workflow/scripts/unblinding_ceremony.py",
        },
    }

    evidence_path = cfg.evidence_dir / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")

    for fig_path in produced_figures:
        shutil.copy2(fig_path, cfg.evidence_dir / fig_path.name)

    print(f"Saved ceremony figures to {cfg.results_dir.resolve()}")
    print(f"Wrote evidence to {evidence_path}")


# ── Entry point dispatch ─────────────────────────────────────────────────────

try:
    snakemake  # injected by snakemake's script: directive
except NameError:
    snakemake = None

if snakemake is not None:
    run_ceremony(_config_from_snakemake(snakemake))
elif __name__ == "__main__":
    run_ceremony(_config_from_cli())
