"""COSEBIS angular binning convergence test.

Computes COSEBIS B-modes from both 1,000-bin and 10,000-bin ξ± integration
grids and compares. Tests whether the numerical integration is converged:
if B_n values agree, the 1,000-bin results are reliable; if they differ,
integration error may contaminate the anomalous PTE.

Reference: Asgari et al. 2017 — ≥10,000 bins for E_7 at 0.5% accuracy.
"""

import argparse
import json
import types
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import treecorr
from cosmo_numba.B_modes.cosebis import COSEBIS
from plotting_utils import (
    FIG_WIDTH_SINGLE,
    PAPER_MPLSTYLE,
    compute_chi2_pte,
)

from sp_validation.b_modes import calculate_cosebis, scale_cut_to_bins

plt.style.use(PAPER_MPLSTYLE)


def _load_gg(xi_path, min_sep, max_sep, nbins, columns_only=False):
    """Load a TreeCorr GGCorrelation from a text file.

    If columns_only=True, read only the per-bin columns (meanr, xip, xim)
    and skip the patch covariance. This avoids loading a 20000x20000
    covariance matrix for high-nbins files.
    """
    gg = treecorr.GGCorrelation(
        min_sep=min_sep,
        max_sep=max_sep,
        nbins=nbins,
        sep_units="arcmin",
    )
    if columns_only:
        # TreeCorr ASCII: ## comment, # col_names, then data rows.
        # Read only per-bin columns, skip the huge patch covariance.
        # cols: r_nom meanr meanlogr xip xim xip_im xim_im sigma_xip sigma_xim weight npairs
        data = np.loadtxt(xi_path, max_rows=nbins)
        # Compute bin edges from log-spaced binning (matching TreeCorr)
        bin_edges = np.exp(np.linspace(np.log(min_sep), np.log(max_sep), nbins + 1))
        return types.SimpleNamespace(
            meanr=data[:, 1],
            xip=data[:, 3],
            xim=data[:, 4],
            left_edges=bin_edges[:-1],
            right_edges=bin_edges[1:],
        )
    else:
        gg.read(xi_path)
    return gg


def _compute_Bn_only(gg, nmodes, scale_cut):
    """Compute COSEBIS B_n from a GGCorrelation without covariance."""
    min_theta, max_theta = scale_cut
    start_bin, stop_bin = scale_cut_to_bins(gg, min_theta, max_theta)
    inds = np.arange(start_bin, stop_bin)
    theta_cut = gg.meanr[inds]
    xip_cut = gg.xip[inds]
    xim_cut = gg.xim[inds]

    cosebis = COSEBIS(
        theta_min=np.min(theta_cut),
        theta_max=np.max(theta_cut),
        N_max=nmodes,
        precision=120,
    )
    En, Bn = cosebis.cosebis_from_xipm(theta_cut, xip_cut, xim_cut, parallel=True)
    return En, Bn


def main(config, xi_1k_path, xi_10k_path, cov_1k_path, out_dir):
    nmodes = config["fiducial"]["nmodes"]

    # Scale cuts
    fiducial_scale_cut = (
        float(config["fiducial"]["fiducial_min_scale"]),
        float(config["fiducial"]["fiducial_max_scale"]),
    )
    full_scale_cut = (
        float(config["cosebis"]["theta_min"]),
        float(config["cosebis"]["theta_max"]),
    )
    scale_cuts = {"fiducial": fiducial_scale_cut, "full": full_scale_cut}

    # Integration parameters
    min_sep_int = float(config["fiducial"]["min_sep_int"])
    max_sep_int = float(config["fiducial"]["max_sep_int"])
    nbins_1k = int(config["fiducial"]["nbins_int"])
    nbins_10k = 10_000

    # Load both ξ± grids
    print("Loading 1,000-bin ξ±...")
    gg_1k = _load_gg(xi_1k_path, min_sep_int, max_sep_int, nbins_1k)
    print("Loading 10,000-bin ξ±...")
    gg_10k = _load_gg(
        xi_10k_path, min_sep_int, max_sep_int, nbins_10k, columns_only=True
    )

    # Compute COSEBIS from 1,000-bin ξ± (with covariance for PTE baseline)
    print("\nComputing COSEBIS from 1,000-bin ξ±...")
    results_1k = calculate_cosebis(
        gg_1k,
        nmodes=nmodes,
        scale_cuts=list(scale_cuts.values()),
        cov_path=cov_1k_path,
    )

    # Compute COSEBIS from 10,000-bin ξ± (B_n only — no matching covariance)
    print("\nComputing COSEBIS from 10,000-bin ξ±...")
    results_10k = {}
    for scale_key, scale_cut in scale_cuts.items():
        En, Bn = _compute_Bn_only(gg_10k, nmodes, scale_cut)
        results_10k[scale_key] = {"En": En, "Bn": Bn}
        print(f"  {scale_key} {scale_cut}: done")

    # Compare and compute PTEs using 1k covariance for both
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence = {}
    for scale_key, scale_cut in scale_cuts.items():
        r1k = results_1k[scale_cut]
        r10k = results_10k[scale_key]

        Bn_1k = r1k["Bn"]
        Bn_10k = r10k["Bn"]
        cov_B = r1k["cov"][nmodes:, nmodes:]
        sigma_B = np.sqrt(np.diag(cov_B))

        # Fractional difference
        delta_Bn = Bn_10k - Bn_1k
        delta_Bn_sigma = delta_Bn / sigma_B

        # PTEs using the same covariance
        chi2_1k, pte_1k, dof = compute_chi2_pte(Bn_1k, cov_B)
        chi2_10k, pte_10k, _ = compute_chi2_pte(Bn_10k, cov_B)

        # Also for first 6 modes
        cov_B_6 = cov_B[:6, :6]
        chi2_1k_6, pte_1k_6, _ = compute_chi2_pte(Bn_1k[:6], cov_B_6)
        chi2_10k_6, pte_10k_6, _ = compute_chi2_pte(Bn_10k[:6], cov_B_6)

        prefix = scale_key
        evidence[f"{prefix}_max_delta_Bn_over_sigma"] = float(
            np.max(np.abs(delta_Bn_sigma))
        )
        evidence[f"{prefix}_rms_delta_Bn_over_sigma"] = float(
            np.sqrt(np.mean(delta_Bn_sigma**2))
        )
        evidence[f"{prefix}_pte_1k_20"] = float(pte_1k)
        evidence[f"{prefix}_pte_10k_20"] = float(pte_10k)
        evidence[f"{prefix}_chi2_1k_20"] = float(chi2_1k)
        evidence[f"{prefix}_chi2_10k_20"] = float(chi2_10k)
        evidence[f"{prefix}_pte_1k_6"] = float(pte_1k_6)
        evidence[f"{prefix}_pte_10k_6"] = float(pte_10k_6)
        evidence[f"{prefix}_chi2_1k_6"] = float(chi2_1k_6)
        evidence[f"{prefix}_chi2_10k_6"] = float(chi2_10k_6)

        print(f"\n{scale_key} [{scale_cut[0]:.0f}–{scale_cut[1]:.0f}']:")
        print(f"  max |ΔB_n/σ| = {np.max(np.abs(delta_Bn_sigma)):.4f}")
        print(f"  RMS  ΔB_n/σ  = {np.sqrt(np.mean(delta_Bn_sigma**2)):.4f}")
        print(f"  PTE (20 modes): 1k={pte_1k:.4e}  10k={pte_10k:.4e}")
        print(f"  PTE ( 6 modes): 1k={pte_1k_6:.4e}  10k={pte_10k_6:.4e}")

    # Figure: side-by-side comparison
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_WIDTH_SINGLE * 2, FIG_WIDTH_SINGLE * 0.6), sharey=True
    )

    colors = sns.color_palette("colorblind", 4)
    modes = np.arange(1, nmodes + 1)

    for ax, (scale_key, scale_cut) in zip(axes, scale_cuts.items()):
        r1k = results_1k[scale_cut]
        r10k = results_10k[scale_key]

        cov_B = r1k["cov"][nmodes:, nmodes:]
        sigma_B = np.sqrt(np.diag(cov_B))

        Bn_1k_norm = r1k["Bn"] / sigma_B
        Bn_10k_norm = r10k["Bn"] / sigma_B

        ax.errorbar(
            modes - 0.15,
            Bn_1k_norm,
            yerr=np.ones(nmodes),
            fmt="o",
            color=colors[0],
            markerfacecolor=colors[0],
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5,
            capsize=2,
            capthick=0.8,
            linewidth=0.8,
            elinewidth=0.8,
            label=r"1,000 bins",
        )
        ax.errorbar(
            modes + 0.15,
            Bn_10k_norm,
            yerr=np.ones(nmodes),
            fmt="s",
            color=colors[1],
            markerfacecolor=colors[1],
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5,
            capsize=2,
            capthick=0.8,
            linewidth=0.8,
            elinewidth=0.8,
            label=r"10,000 bins",
        )

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        ax.axvspan(0.5, 6.5, color="0.95", alpha=0.5, zorder=0)

        # Annotate PTEs
        pte_1k = evidence[f"{scale_key}_pte_1k_20"]
        pte_10k = evidence[f"{scale_key}_pte_10k_20"]
        ax.text(
            0.97,
            0.97,
            f"PTE(20): {pte_1k:.2e} → {pte_10k:.2e}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("COSEBIS mode $n$")
        ax.set_xlim(0.5, nmodes + 0.5)
        ax.set_xticks(np.arange(1, nmodes + 1))
        ax.set_title(
            rf"$\theta = {scale_cut[0]:.0f}$--${scale_cut[1]:.0f}'$", fontsize=10
        )

    axes[0].set_ylabel(r"$B_n / \sigma_n$")
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=8)

    plt.tight_layout()
    fig_path = output_dir / "figure.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved {fig_path}")
    plt.close(fig)

    # Write evidence
    evidence_data = {
        "id": "cosebis_binning_comparison",
        "generated": datetime.now().isoformat(),
        "input": {
            "xi_1k": str(xi_1k_path),
            "xi_10k": str(xi_10k_path),
            "cov_1k": str(cov_1k_path),
        },
        "output": {"figure": "figure.png"},
        "params": {
            "nbins_1k": nbins_1k,
            "nbins_10k": nbins_10k,
            "nmodes": nmodes,
            "fiducial_scale_cut": list(fiducial_scale_cut),
            "full_scale_cut": list(full_scale_cut),
        },
        "evidence": evidence,
    }

    evidence_path = output_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"Saved evidence to {evidence_path}")


def _from_cli(argv=None):
    import yaml

    ap = argparse.ArgumentParser(
        description="COSEBI angular-binning convergence (1,000 vs 10,000-bin xi_pm) figure."
    )
    ap.add_argument(
        "--config", required=True, help="Absolute path to bmodes config.yaml"
    )
    ap.add_argument(
        "--xi-1k",
        required=True,
        help="Fiducial 1000-bin integration-grid TreeCorr xi_pm .txt",
    )
    ap.add_argument(
        "--xi-10k",
        required=True,
        help="Fiducial 10000-bin integration-grid TreeCorr xi_pm .txt",
    )
    ap.add_argument(
        "--cov-1k",
        required=True,
        help="Fiducial 1000-bin Gaussian covariance (processed .txt) used for both grids",
    )
    ap.add_argument("--out", required=True, help="Output directory (lc {output})")
    a = ap.parse_args(argv)
    with open(a.config) as f:
        config = yaml.safe_load(f)
    main(config, a.xi_1k, a.xi_10k, a.cov_1k, a.out)


if __name__ == "__main__":
    _from_cli()
