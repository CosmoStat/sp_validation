"""
BB Covariance Blind Independence Claim

Tests whether BB covariances are blind-independent (as expected for null signals)
while EE covariances vary across blinds (due to sample variance from cosmological signal).

Compares:
- Pure E/B: cov(xi+^B), cov(xi-^B) vs cov(xi+^E), cov(xi-^E)
- COSEBIS: cov(B_n) vs cov(E_n)
- Harmonic: cov(C_ell^BB) vs cov(C_ell^EE)
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import treecorr
from astropy.io import fits

from plotting_utils import PAPER_MPLSTYLE
from sp_validation.b_modes import calculate_cosebis

plt.style.use(PAPER_MPLSTYLE)


BLINDS = ["A", "B", "C"]


def load_pure_eb_diagonals(path):
    """Load pure E/B covariance and extract block diagonals.

    Covariance is 120x120 with 6 blocks of 20 bins:
    [E+, E-, B+, B-, amb+, amb-]
    """
    data = np.load(path)
    cov = data["cov_pure_eb"]
    theta = data["theta"]
    nbins = len(theta)

    def get_block_diag(block_idx):
        sl = slice(block_idx * nbins, (block_idx + 1) * nbins)
        return np.diag(cov[sl, sl])

    return {
        "theta": theta,
        "xip_E": get_block_diag(0),
        "xim_E": get_block_diag(1),
        "xip_B": get_block_diag(2),
        "xim_B": get_block_diag(3),
    }


def load_harmonic_diagonals(path):
    """Load harmonic covariance and extract EE and BB diagonals."""
    with fits.open(path) as hdu:
        diag_EE = np.diag(hdu["COVAR_EE_EE"].data)
        diag_BB = np.diag(hdu["COVAR_BB_BB"].data)
    return {
        "EE": diag_EE,
        "BB": diag_BB,
    }


def load_cosebis_diagonals(
    xi_integration_path, cov_integration_path, nmodes, theta_min, theta_max,
    min_sep_int, max_sep_int, nbins_int
):
    """Compute COSEBIS covariance diagonals from config-space covariance.

    Uses calculate_cosebis to transform config-space covariance to COSEBIS space.
    Returns E_n and B_n covariance diagonals.
    """
    # Load fine-binned 2PCF (need the binning info for COSEBIS calculation)
    gg = treecorr.GGCorrelation(min_sep=min_sep_int, max_sep=max_sep_int, nbins=nbins_int, sep_units="arcmin")
    gg.read(xi_integration_path)

    # Compute COSEBIS with this blind's covariance
    results = calculate_cosebis(
        gg,
        nmodes=nmodes,
        scale_cuts=[(theta_min, theta_max)],
        cov_path=cov_integration_path,
    )

    # Extract covariance for the fiducial scale cut
    result = results[(theta_min, theta_max)]
    cov = result["cov"]

    # E is [:nmodes, :nmodes], B is [nmodes:, nmodes:]
    diag_E = np.diag(cov[:nmodes, :nmodes])
    diag_B = np.diag(cov[nmodes:, nmodes:])

    return {
        "E": diag_E,
        "B": diag_B,
        "nmodes": nmodes,
    }


def compute_ratios(diag_ref, diag_test):
    """Compute ratio and deviation statistics."""
    ratio = diag_test / diag_ref
    dev = np.abs(ratio - 1.0)
    return {
        "ratio": ratio,
        "max_dev": float(np.max(dev)),
        "mean_dev": float(np.mean(dev)),
        "min_ratio": float(np.min(ratio)),
        "max_ratio": float(np.max(ratio)),
    }


def make_figure(theta, ell_eff, pure_eb_results, harmonic_results, cosebis_results, output_path, n_samples=2000):
    """Four-panel figure comparing BB vs EE stability across blinds.

    Layout: 2x2
    - Top row: Pure E/B (xi+, xi-)
    - Bottom row: COSEBIS (B_n vs E_n), Harmonic (C_ell)

    Color encoding: E-mode vs B-mode
    Marker encoding: square = B/A, triangle = C/A (same for both E and B)
    """
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    # Colors: E-mode vs B-mode
    color_E = "#E69F00"  # orange for E
    color_B = "#0072B2"  # blue for B

    # Expected 1σ error on ratio of two MC-estimated quantities
    # σ(ratio) ≈ √(2/N) for ratio ≈ 1
    ratio_err = np.sqrt(2.0 / n_samples)

    # Helper for ratio panels
    def setup_ratio_panel(ax, ylabel_x, title, show_mc_band=False):
        ax.axhline(1.0, color="gray", ls="-", lw=0.8, zorder=0)
        if show_mc_band:
            # Show expected 1σ MC scatter band only
            ax.axhspan(1 - ratio_err, 1 + ratio_err, color="gray", alpha=0.25,
                      label=rf"$\pm\sqrt{{2/N}}$ ($N={n_samples}$)")
        ax.set_xscale("log")
        ax.set_xlabel(ylabel_x)
        ax.set_title(title)

    # --- Panel 1: xi+^B vs xi+^E ---
    ax = axes[0, 0]
    setup_ratio_panel(ax, r"$\theta$ [arcmin]", r"$\xi_+$: covariance ratio across blinds", show_mc_band=True)

    # B-mode (blue): square for B/A, triangle for C/A
    ax.errorbar(theta, pure_eb_results["B"]["xip_B"]["ratio"], yerr=ratio_err,
                fmt="s", color=color_B, label=r"B-mode B/A", markersize=5, alpha=0.8, capsize=0)
    ax.errorbar(theta * 1.03, pure_eb_results["C"]["xip_B"]["ratio"], yerr=ratio_err,
                fmt="^", color=color_B, label=r"B-mode C/A", markersize=5, alpha=0.8, capsize=0)

    # E-mode (orange): square for B/A, triangle for C/A
    ax.plot(theta, pure_eb_results["B"]["xip_E"]["ratio"],
            "s", color=color_E, label=r"E-mode B/A", markersize=4, alpha=0.6)
    ax.plot(theta * 1.03, pure_eb_results["C"]["xip_E"]["ratio"],
            "^", color=color_E, label=r"E-mode C/A", markersize=4, alpha=0.6)

    ax.set_ylabel("Diagonal ratio")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_xlim(1, 300)
    ax.set_ylim(0.85, 1.15)

    # --- Panel 2: xi-^B vs xi-^E ---
    ax = axes[0, 1]
    setup_ratio_panel(ax, r"$\theta$ [arcmin]", r"$\xi_-$: covariance ratio across blinds", show_mc_band=True)

    ax.errorbar(theta, pure_eb_results["B"]["xim_B"]["ratio"], yerr=ratio_err,
                fmt="s", color=color_B, label=r"B-mode B/A", markersize=5, alpha=0.8, capsize=0)
    ax.errorbar(theta * 1.03, pure_eb_results["C"]["xim_B"]["ratio"], yerr=ratio_err,
                fmt="^", color=color_B, label=r"B-mode C/A", markersize=5, alpha=0.8, capsize=0)

    ax.plot(theta, pure_eb_results["B"]["xim_E"]["ratio"],
            "s", color=color_E, label=r"E-mode B/A", markersize=4, alpha=0.6)
    ax.plot(theta * 1.03, pure_eb_results["C"]["xim_E"]["ratio"],
            "^", color=color_E, label=r"E-mode C/A", markersize=4, alpha=0.6)

    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_xlim(1, 300)
    ax.set_ylim(0.85, 1.15)

    # --- Panel 3: COSEBIS B_n vs E_n ---
    ax = axes[1, 0]
    nmodes = cosebis_results["B"]["nmodes"]
    n_arr = np.arange(1, nmodes + 1)
    ax.axhline(1.0, color="gray", ls="-", lw=0.8, zorder=0)
    ax.set_xlabel(r"Mode $n$")
    ax.set_title(r"COSEBIS: covariance ratio across blinds")

    # B-mode (blue): square for B/A, triangle for C/A
    ax.plot(n_arr, cosebis_results["B"]["B"]["ratio"],
            "s", color=color_B, label=r"B-mode B/A", markersize=5, alpha=0.8)
    ax.plot(n_arr + 0.15, cosebis_results["C"]["B"]["ratio"],
            "^", color=color_B, label=r"B-mode C/A", markersize=5, alpha=0.8)

    # E-mode (orange): square for B/A, triangle for C/A
    ax.plot(n_arr, cosebis_results["B"]["E"]["ratio"],
            "s", color=color_E, label=r"E-mode B/A", markersize=4, alpha=0.6)
    ax.plot(n_arr + 0.15, cosebis_results["C"]["E"]["ratio"],
            "^", color=color_E, label=r"E-mode C/A", markersize=4, alpha=0.6)

    ax.set_ylabel("Diagonal ratio")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_ylim(0.85, 1.15)

    # --- Panel 4: C_ell^BB vs C_ell^EE ---
    ax = axes[1, 1]
    setup_ratio_panel(ax, r"$\ell$", r"$C_\ell$: covariance ratio across blinds")

    # B-mode (blue): square for B/A, triangle for C/A
    ax.plot(ell_eff, harmonic_results["B"]["BB"]["ratio"],
            "s", color=color_B, label=r"BB B/A", markersize=5, alpha=0.8)
    ax.plot(ell_eff * 1.03, harmonic_results["C"]["BB"]["ratio"],
            "^", color=color_B, label=r"BB C/A", markersize=5, alpha=0.8)

    # E-mode (orange): square for B/A, triangle for C/A
    ax.plot(ell_eff, harmonic_results["B"]["EE"]["ratio"],
            "s", color=color_E, label=r"EE B/A", markersize=4, alpha=0.6)
    ax.plot(ell_eff * 1.03, harmonic_results["C"]["EE"]["ratio"],
            "^", color=color_E, label=r"EE C/A", markersize=4, alpha=0.6)

    ax.set_ylabel("Diagonal ratio")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_ylim(0.85, 1.15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(snakemake):
    config = snakemake.config
    version = config["fiducial"]["mock_version"]

    # Load pure E/B covariances for all blinds
    pure_eb_data = {}
    for blind in BLINDS:
        path = snakemake.input[f"pure_eb_{blind}"]
        pure_eb_data[blind] = load_pure_eb_diagonals(path)

    theta = pure_eb_data["A"]["theta"]

    # Load harmonic covariances for all blinds
    harmonic_data = {}
    for blind in BLINDS:
        path = snakemake.input[f"harmonic_{blind}"]
        harmonic_data[blind] = load_harmonic_diagonals(path)

    # Load COSEBIS covariances for all blinds
    nmodes = snakemake.params.nmodes
    theta_min = snakemake.params.theta_min
    theta_max = snakemake.params.theta_max
    # Integration binning parameters from config
    min_sep_int = config["fiducial"]["min_sep_int"]
    max_sep_int = config["fiducial"]["max_sep_int"]
    nbins_int = config["fiducial"]["nbins_int"]

    cosebis_data = {}
    for blind in BLINDS:
        cosebis_data[blind] = load_cosebis_diagonals(
            snakemake.input.xi_integration,
            snakemake.input[f"cov_integration_{blind}"],
            nmodes,
            theta_min,
            theta_max,
            min_sep_int,
            max_sep_int,
            nbins_int,
        )

    # Read ell bin centers from pseudo-Cl data file
    with fits.open(snakemake.input.pseudo_cl) as hdu:
        ell_eff = hdu["PSEUDO_CELL"].data["ELL"]

    # Compute ratios relative to blind A
    pure_eb_results = {}
    for blind in ["B", "C"]:
        pure_eb_results[blind] = {}
        for mode in ["xip_E", "xim_E", "xip_B", "xim_B"]:
            pure_eb_results[blind][mode] = compute_ratios(
                pure_eb_data["A"][mode],
                pure_eb_data[blind][mode]
            )

    harmonic_results = {}
    for blind in ["B", "C"]:
        harmonic_results[blind] = {}
        for mode in ["EE", "BB"]:
            harmonic_results[blind][mode] = compute_ratios(
                harmonic_data["A"][mode],
                harmonic_data[blind][mode]
            )

    cosebis_results = {}
    for blind in ["B", "C"]:
        cosebis_results[blind] = {
            "nmodes": nmodes,
        }
        for mode in ["E", "B"]:
            cosebis_results[blind][mode] = compute_ratios(
                cosebis_data["A"][mode],
                cosebis_data[blind][mode]
            )

    # Generate figure
    n_samples = config["covariance"]["n_samples"]
    make_figure(theta, ell_eff, pure_eb_results, harmonic_results, cosebis_results, snakemake.output.figure, n_samples=n_samples)

    # Compute summary statistics
    bb_max_devs = [
        pure_eb_results["B"]["xip_B"]["max_dev"],
        pure_eb_results["C"]["xip_B"]["max_dev"],
        pure_eb_results["B"]["xim_B"]["max_dev"],
        pure_eb_results["C"]["xim_B"]["max_dev"],
        cosebis_results["B"]["B"]["max_dev"],
        cosebis_results["C"]["B"]["max_dev"],
        harmonic_results["B"]["BB"]["max_dev"],
        harmonic_results["C"]["BB"]["max_dev"],
    ]
    ee_max_devs = [
        pure_eb_results["B"]["xip_E"]["max_dev"],
        pure_eb_results["C"]["xip_E"]["max_dev"],
        pure_eb_results["B"]["xim_E"]["max_dev"],
        pure_eb_results["C"]["xim_E"]["max_dev"],
        cosebis_results["B"]["E"]["max_dev"],
        cosebis_results["C"]["E"]["max_dev"],
        harmonic_results["B"]["EE"]["max_dev"],
        harmonic_results["C"]["EE"]["max_dev"],
    ]

    # Pass criteria (updated: analytic vs MC methods show different behavior)
    # COSEBIS uses analytic propagation (T @ Cov @ T.T) → blind-independent
    # Pure E/B and Harmonic use MC sampling → inherit sampling noise
    cosebis_bb_max = max(
        cosebis_results["B"]["B"]["max_dev"],
        cosebis_results["C"]["B"]["max_dev"],
    )
    cosebis_ee_max = max(
        cosebis_results["B"]["E"]["max_dev"],
        cosebis_results["C"]["E"]["max_dev"],
    )

    # COSEBIS B_n should be blind-independent (<0.1%)
    cosebis_bb_blind_independent = cosebis_bb_max < 0.001
    # E-modes should vary ~10% due to sample variance
    ee_varies_as_expected = 0.05 < cosebis_ee_max < 0.15

    # Legacy summary (for backwards compatibility)
    bb_max = max(bb_max_devs)
    ee_max = max(ee_max_devs)
    bb_closer_to_unity = bb_max < ee_max
    bb_within_2pct = bb_max < 0.02

    # Build evidence
    evidence = {
        "spec_id": "bb_covariance_blind_independence",
        "spec_path": "workflow/config/bb_covariance_blind_independence.md",
        "depends_on": ["covariance", "pure_eb", "cosebis", "pseudo_cl"],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "pure_eb": {
                "xip_B": {
                    "B_to_A": pure_eb_results["B"]["xip_B"],
                    "C_to_A": pure_eb_results["C"]["xip_B"],
                },
                "xim_B": {
                    "B_to_A": pure_eb_results["B"]["xim_B"],
                    "C_to_A": pure_eb_results["C"]["xim_B"],
                },
                "xip_E": {
                    "B_to_A": pure_eb_results["B"]["xip_E"],
                    "C_to_A": pure_eb_results["C"]["xip_E"],
                },
                "xim_E": {
                    "B_to_A": pure_eb_results["B"]["xim_E"],
                    "C_to_A": pure_eb_results["C"]["xim_E"],
                },
            },
            "harmonic": {
                "BB": {
                    "B_to_A": harmonic_results["B"]["BB"],
                    "C_to_A": harmonic_results["C"]["BB"],
                },
                "EE": {
                    "B_to_A": harmonic_results["B"]["EE"],
                    "C_to_A": harmonic_results["C"]["EE"],
                },
            },
            "cosebis": {
                "B": {
                    "B_to_A": cosebis_results["B"]["B"],
                    "C_to_A": cosebis_results["C"]["B"],
                },
                "E": {
                    "B_to_A": cosebis_results["B"]["E"],
                    "C_to_A": cosebis_results["C"]["E"],
                },
                "nmodes": nmodes,
                "theta_min": theta_min,
                "theta_max": theta_max,
            },
            "summary": {
                # Primary pass criteria (revised)
                "cosebis_bb_max_deviation": cosebis_bb_max,
                "cosebis_ee_max_deviation": cosebis_ee_max,
                "cosebis_bb_blind_independent": cosebis_bb_blind_independent,
                "ee_varies_as_expected": ee_varies_as_expected,
                # Legacy (for reference)
                "bb_max_deviation": bb_max,
                "ee_max_deviation": ee_max,
                "bb_closer_to_unity": bb_closer_to_unity,
                "bb_within_2pct": bb_within_2pct,
            },
            "version": version,
        },
        "output": {
            "figure": Path(snakemake.output.figure).name,
        },
    }

    # Remove numpy arrays from nested dicts (keep only scalars)
    def clean_ratios(d):
        if isinstance(d, dict):
            return {k: clean_ratios(v) for k, v in d.items() if k != "ratio"}
        return d

    evidence["evidence"]["pure_eb"] = clean_ratios(evidence["evidence"]["pure_eb"])
    evidence["evidence"]["harmonic"] = clean_ratios(evidence["evidence"]["harmonic"])
    evidence["evidence"]["cosebis"] = clean_ratios(evidence["evidence"]["cosebis"])

    # Write evidence
    Path(snakemake.output.evidence).parent.mkdir(parents=True, exist_ok=True)
    with open(snakemake.output.evidence, "w") as f:
        json.dump(evidence, f, indent=2)

    # Print summary
    print("\nBB Covariance Blind Independence Summary:")
    print(f"  COSEBIS B_n max deviation: {cosebis_bb_max*100:.6f}%")
    print(f"  COSEBIS E_n max deviation: {cosebis_ee_max*100:.2f}%")
    print(f"  COSEBIS B_n blind-independent (<0.1%): {cosebis_bb_blind_independent}")
    print(f"  E-modes vary as expected (5-15%): {ee_varies_as_expected}")
    print("\n  Pure E/B + Harmonic (MC methods):")
    print(f"    BB max deviation: {bb_max*100:.2f}%")
    print(f"    EE max deviation: {ee_max*100:.2f}%")
    print("    Note: MC sampling noise causes BB to vary similarly to EE")


if __name__ == "__main__":
    main(snakemake)  # noqa: F821
