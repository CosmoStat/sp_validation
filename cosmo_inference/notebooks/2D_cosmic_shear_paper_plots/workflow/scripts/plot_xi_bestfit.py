"""
Plot ξ± data with best-fit theory curve from fiducial cosmology inference.

Style matches the original paper_plots.py figure:
- Black line: Best-fit ξ_th + ξ_sys
- Blue line: Best-fit ξ_sys alone
- Black points: Data ξ±
- Red points: B-mode ξ_B
- Gray shaded region: excluded by fiducial scale cuts
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from snakemake.script import snakemake

# Load plot style
plt.style.use(snakemake.config["plot_style"])

# Fiducial scale cuts from Paper IV (main.tex Sect. 5.5)
# Based on B-modes, PSF leakage, and nonlinear modelling
# [12, 83] for both xi+/xi- with v1.4.6.3 fiducial
scale_cut_xip = [12, 83]  # arcmin
scale_cut_xim = [12, 83]  # arcmin

# --- Load data ξ± ---
data = np.loadtxt(snakemake.input.xi_data, comments="#")
theta_data = data[:, 1]  # meanr in arcmin
xip_data = data[:, 3]
xim_data = data[:, 4]
sigma_xip = data[:, 7]
sigma_xim = data[:, 8]

# --- Load B-mode data ---
eb_data = np.load(snakemake.input.pure_eb_data)
theta_eb = eb_data["theta"]
xip_B = eb_data["xip_B"]
xim_B = eb_data["xim_B"]
cov_pure_eb = eb_data["cov_pure_eb"]

# Extract B-mode uncertainties from covariance (blocks 2 and 3)
nbins = len(theta_eb)
sigma_xip_B = np.sqrt(np.diag(cov_pure_eb[2*nbins:3*nbins, 2*nbins:3*nbins]))
sigma_xim_B = np.sqrt(np.diag(cov_pure_eb[3*nbins:4*nbins, 3*nbins:4*nbins]))

# TreeCorr bin edges: log-spaced from min_sep to max_sep
# These are the actual bin boundaries, not derived from meanr
min_sep, max_sep = 1.0, 250.0
bin_edges = np.geomspace(min_sep, max_sep, nbins + 1)

# Nominal bin centers (geometric mean of edges) for matching scale cuts
bin_centers_nominal = np.sqrt(bin_edges[:-1] * bin_edges[1:])

# Compute actual bin edge boundaries for scale cuts
def get_bin_edge_cuts(centers, edges, scale_cut):
    """Get bin edges that bound the included bins based on nominal centers."""
    mask = (centers >= scale_cut[0]) & (centers <= scale_cut[1])
    idx_first = np.where(mask)[0][0]
    idx_last = np.where(mask)[0][-1]
    return edges[idx_first], edges[idx_last + 1]

edge_cut_xip = get_bin_edge_cuts(bin_centers_nominal, bin_edges, scale_cut_xip)
edge_cut_xim = get_bin_edge_cuts(bin_centers_nominal, bin_edges, scale_cut_xim)

# --- Load best-fit theory ---
bestfit_dir = snakemake.params.bestfit_dir

# Theory theta in radians, convert to arcmin
theta_theory_rad = np.loadtxt(f"{bestfit_dir}/shear_xi_plus/theta.txt", comments="#")
theta_theory = np.rad2deg(theta_theory_rad) * 60  # radians -> arcmin

xip_theory = np.loadtxt(f"{bestfit_dir}/shear_xi_plus/bin_1_1.txt", comments="#")
xim_theory = np.loadtxt(f"{bestfit_dir}/shear_xi_minus/bin_1_1.txt", comments="#")

# Load xi_sys (PSF leakage contribution)
theta_sys_rad = np.loadtxt(f"{bestfit_dir}/xi_sys/theta.txt", comments="#")
theta_sys = np.rad2deg(theta_sys_rad) * 60
xip_sys = np.loadtxt(f"{bestfit_dir}/xi_sys/shear_xi_plus.txt", comments="#")
xim_sys = np.loadtxt(f"{bestfit_dir}/xi_sys/shear_xi_minus.txt", comments="#")

# Interpolate to common fine grid
theta_fine = np.geomspace(0.5, 300, 500)
xip_th_interp = interp1d(theta_theory, xip_theory, kind="cubic", fill_value="extrapolate")(theta_fine)
xim_th_interp = interp1d(theta_theory, xim_theory, kind="cubic", fill_value="extrapolate")(theta_fine)
xip_sys_interp = interp1d(theta_sys, xip_sys, kind="cubic", fill_value="extrapolate")(theta_fine)
xim_sys_interp = interp1d(theta_sys, xim_sys, kind="cubic", fill_value="extrapolate")(theta_fine)

# --- Plotting parameters ---
scale_factor = 1e-4
xlim = [1, 250]
ylim = [-0.15, 1.25]

# Smaller markers
ms_data = 3
ms_bmode = 3
capsize = 1.5
elinewidth = 0.8

# --- Create figure ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

plot_configs = [
    (axes[0], xip_data, sigma_xip, xip_B, sigma_xip_B, xip_th_interp, xip_sys_interp,
     edge_cut_xip, r"$\xi_+$", "+"),
    (axes[1], xim_data, sigma_xim, xim_B, sigma_xim_B, xim_th_interp, xim_sys_interp,
     edge_cut_xim, r"$\xi_-$", "-"),
]

for idx, (ax, xi_data_arr, sigma_xi, xi_B, sigma_B, xi_th, xi_sys, edge_cut, label, pm) in enumerate(plot_configs):

    show_legend = (idx == 1)  # Only right panel

    # Gray out excluded regions (outside fiducial scale cuts, using bin edges)
    ax.axvspan(xlim[0], edge_cut[0], color="0.90", zorder=0, alpha=0.7)
    ax.axvspan(edge_cut[1], xlim[1], color="0.90", zorder=0, alpha=0.7)

    # Best-fit theory + systematics (black line)
    ax.plot(
        theta_fine,
        theta_fine * (xi_th + xi_sys) / scale_factor,
        "-",
        color="k",
        lw=1.5,
        label=r"Best-fit $\xi^{\mathrm{th}}_\pm + \xi^{\mathrm{sys}}_\pm$" if show_legend else None,
        zorder=2,
    )

    # Best-fit systematics alone (blue line)
    ax.plot(
        theta_fine,
        theta_fine * xi_sys / scale_factor,
        "-",
        color="C0",
        lw=1.2,
        label=r"Best-fit $\xi^{\mathrm{sys}}_\pm$" if show_legend else None,
        zorder=2,
    )

    # Data ξ± (black points) - plot at meanr
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

    # B-mode (red points) - slight x-offset for visibility
    theta_eb_offset = theta_eb * 1.03  # 3% offset to right
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

    # Zero line
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
fig.savefig(snakemake.output[0], dpi=150, bbox_inches="tight")
print(f"Saved: {snakemake.output[0]}")
