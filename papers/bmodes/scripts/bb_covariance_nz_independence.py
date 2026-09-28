"""
BB Covariance n(z) Independence Claim

Tests whether BB covariances are independent of the n(z) realisation (as
expected for null signals) while EE covariances vary across realisations (due to
sample variance from cosmological signal). Each realisation is a catalogue entry
of its own (config fiducial.nz_realisations), identical but for its n(z).

Compares:
- Pure E/B: cov(xi+^B), cov(xi-^B) vs cov(xi+^E), cov(xi-^E)
- COSEBIS: cov(B_n) vs cov(E_n)
- Harmonic: cov(C_ell^BB) vs cov(C_ell^EE)
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import treecorr
import yaml
from astropy.io import fits
from plotting_utils import PAPER_MPLSTYLE
from pseudo_cl_io import load_pseudo_cl_data

from sp_validation.b_modes import calculate_cosebis

plt.style.use(PAPER_MPLSTYLE)


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
    xi_integration_path,
    cov_integration_path,
    nmodes,
    theta_min,
    theta_max,
    min_sep_int,
    max_sep_int,
    nbins_int,
):
    """Compute COSEBIS covariance diagonals from config-space covariance.

    Uses calculate_cosebis to transform config-space covariance to COSEBIS space.
    Returns E_n and B_n covariance diagonals.
    """
    # Load fine-binned 2PCF (need the binning info for COSEBIS calculation)
    gg = treecorr.GGCorrelation(
        min_sep=min_sep_int, max_sep=max_sep_int, nbins=nbins_int, sep_units="arcmin"
    )
    gg.read(xi_integration_path)

    # Compute COSEBIS with this realisation's covariance
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


# Marker per compared realisation, and the multiplicative x-offset that keeps
# their points from overlapping.
MARKERS = ["s", "^", "v", "D", "o"]


def make_figure(
    theta,
    ell_eff,
    pure_eb_results,
    harmonic_results,
    cosebis_results,
    reference,
    output_path,
    n_samples=2000,
):
    """Four-panel figure comparing BB vs EE stability across n(z) realisations.

    Layout: 2x2
    - Top row: Pure E/B (xi+, xi-)
    - Bottom row: COSEBIS (B_n vs E_n), Harmonic (C_ell)

    Color encodes E- vs B-mode; marker encodes the compared realisation, each
    shown as its ratio to ``reference``.
    """
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    color_E = "#E69F00"  # orange for E
    color_B = "#0072B2"  # blue for B

    # Expected 1σ error on ratio of two MC-estimated quantities
    # σ(ratio) ≈ √(2/N) for ratio ≈ 1
    ratio_err = np.sqrt(2.0 / n_samples)

    def setup_ratio_panel(ax, xlabel, title, show_mc_band=False):
        ax.axhline(1.0, color="gray", ls="-", lw=0.8, zorder=0)
        if show_mc_band:
            ax.axhspan(
                1 - ratio_err,
                1 + ratio_err,
                color="gray",
                alpha=0.25,
                label=rf"$\pm\sqrt{{2/N}}$ ($N={n_samples}$)",
            )
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(title)

    def plot_ratios(ax, x, results, b_key, e_key, b_name, e_name, shift, b_err):
        """B (with optional MC error bar) and E ratios for every realisation."""
        for i, (label, res) in enumerate(results.items()):
            xi = shift(x, i)
            marker = MARKERS[i % len(MARKERS)]
            pair = f"{label}/{reference}"
            ax.errorbar(
                xi,
                res[b_key]["ratio"],
                yerr=b_err,
                fmt=marker,
                color=color_B,
                label=f"{b_name} {pair}",
                markersize=5,
                alpha=0.8,
                capsize=0,
            )
            ax.plot(
                xi,
                res[e_key]["ratio"],
                marker,
                color=color_E,
                label=f"{e_name} {pair}",
                markersize=4,
                alpha=0.6,
            )

    def log_shift(x, i):
        return x * 1.03**i

    def lin_shift(x, i):
        return x + 0.15 * i

    # --- Panels 1–2: pure E/B xi± ---
    for ax, comp, name in (
        (axes[0, 0], "xip", r"\xi_+"),
        (axes[0, 1], "xim", r"\xi_-"),
    ):
        setup_ratio_panel(
            ax,
            r"$\theta$ [arcmin]",
            rf"${name}$: covariance ratio across n(z)",
            show_mc_band=True,
        )
        plot_ratios(
            ax,
            theta,
            pure_eb_results,
            f"{comp}_B",
            f"{comp}_E",
            "B-mode",
            "E-mode",
            log_shift,
            ratio_err,
        )
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.set_xlim(1, 300)
        ax.set_ylim(0.85, 1.15)
    axes[0, 0].set_ylabel("Diagonal ratio")

    # --- Panel 3: COSEBIS B_n vs E_n ---
    ax = axes[1, 0]
    nmodes = next(iter(cosebis_results.values()))["nmodes"]
    n_arr = np.arange(1, nmodes + 1)
    ax.axhline(1.0, color="gray", ls="-", lw=0.8, zorder=0)
    ax.set_xlabel(r"Mode $n$")
    ax.set_title(r"COSEBIS: covariance ratio across n(z)")
    plot_ratios(
        ax, n_arr, cosebis_results, "B", "E", "B-mode", "E-mode", lin_shift, None
    )
    ax.set_ylabel("Diagonal ratio")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_ylim(0.85, 1.15)

    # --- Panel 4: C_ell^BB vs C_ell^EE ---
    ax = axes[1, 1]
    setup_ratio_panel(ax, r"$\ell$", r"$C_\ell$: covariance ratio across n(z)")
    plot_ratios(ax, ell_eff, harmonic_results, "BB", "EE", "BB", "EE", log_shift, None)
    ax.set_ylabel("Diagonal ratio")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_ylim(0.85, 1.15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(
    config,
    pure_eb_paths,
    harmonic_paths,
    xi_integration_path,
    cov_integration_paths,
    pseudo_cl_path,
    nmodes,
    theta_min,
    theta_max,
    figure_path,
    evidence_path,
):
    """Run the BB-covariance n(z)-independence cross-check.

    ``pure_eb_paths`` / ``harmonic_paths`` / ``cov_integration_paths`` are dicts
    keyed by n(z) realisation label, in the order of config
    ``fiducial.nz_realisations``: the first label is the reference, every other
    realisation is compared against it. All paths are absolute; the
    per-realisation covariances are read directly, so the check is
    self-contained.
    """
    version = config["fiducial"]["mock_version"]
    reference, *compared = pure_eb_paths

    pure_eb_data = {
        label: load_pure_eb_diagonals(path) for label, path in pure_eb_paths.items()
    }
    theta = pure_eb_data[reference]["theta"]

    harmonic_data = {
        label: load_harmonic_diagonals(path) for label, path in harmonic_paths.items()
    }

    # Integration binning parameters from config
    min_sep_int = config["fiducial"]["min_sep_int"]
    max_sep_int = config["fiducial"]["max_sep_int"]
    nbins_int = config["fiducial"]["nbins_int"]

    cosebis_data = {
        label: load_cosebis_diagonals(
            xi_integration_path,
            path,
            nmodes,
            theta_min,
            theta_max,
            min_sep_int,
            max_sep_int,
            nbins_int,
        )
        for label, path in cov_integration_paths.items()
    }

    # Ell bin centers from the pseudo-Cl SACC part
    ell_eff = load_pseudo_cl_data(pseudo_cl_path)["ELL"]

    def ratios_to_reference(data, modes):
        return {
            label: {
                mode: compute_ratios(data[reference][mode], data[label][mode])
                for mode in modes
            }
            for label in compared
        }

    pure_eb_results = ratios_to_reference(
        pure_eb_data, ["xip_E", "xim_E", "xip_B", "xim_B"]
    )
    harmonic_results = ratios_to_reference(harmonic_data, ["EE", "BB"])
    cosebis_results = ratios_to_reference(cosebis_data, ["E", "B"])
    for res in cosebis_results.values():
        res["nmodes"] = nmodes

    n_samples = config["covariance"]["n_samples"]
    make_figure(
        theta,
        ell_eff,
        pure_eb_results,
        harmonic_results,
        cosebis_results,
        reference,
        figure_path,
        n_samples=n_samples,
    )

    def max_dev(results, mode):
        return max(res[mode]["max_dev"] for res in results.values())

    bb_max_devs = [
        max_dev(pure_eb_results, "xip_B"),
        max_dev(pure_eb_results, "xim_B"),
        max_dev(cosebis_results, "B"),
        max_dev(harmonic_results, "BB"),
    ]
    ee_max_devs = [
        max_dev(pure_eb_results, "xip_E"),
        max_dev(pure_eb_results, "xim_E"),
        max_dev(cosebis_results, "E"),
        max_dev(harmonic_results, "EE"),
    ]

    # Pass criteria: COSEBIS uses analytic propagation (T @ Cov @ T.T), so its
    # B_n covariance is n(z)-independent; pure E/B and harmonic use MC sampling
    # and inherit its noise.
    cosebis_bb_max = max_dev(cosebis_results, "B")
    cosebis_ee_max = max_dev(cosebis_results, "E")

    # COSEBIS B_n should be n(z)-independent (<0.1%)
    cosebis_bb_nz_independent = cosebis_bb_max < 0.001
    # E-modes should vary ~10% due to sample variance
    ee_varies_as_expected = 0.05 < cosebis_ee_max < 0.15

    # Summary over all three spaces, MC methods included
    bb_max = max(bb_max_devs)
    ee_max = max(ee_max_devs)
    bb_closer_to_unity = bb_max < ee_max
    bb_within_2pct = bb_max < 0.02

    def by_pair(results, mode):
        return {f"{label}_to_{reference}": res[mode] for label, res in results.items()}

    evidence = {
        "depends_on": ["covariance", "pure_eb", "cosebis", "pseudo_cl"],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "reference_realisation": reference,
            "pure_eb": {
                mode: by_pair(pure_eb_results, mode)
                for mode in ["xip_B", "xim_B", "xip_E", "xim_E"]
            },
            "harmonic": {
                mode: by_pair(harmonic_results, mode) for mode in ["BB", "EE"]
            },
            "cosebis": {
                **{mode: by_pair(cosebis_results, mode) for mode in ["B", "E"]},
                "nmodes": nmodes,
                "theta_min": theta_min,
                "theta_max": theta_max,
            },
            "summary": {
                # Primary pass criteria (COSEBIS, analytic)
                "cosebis_bb_max_deviation": cosebis_bb_max,
                "cosebis_ee_max_deviation": cosebis_ee_max,
                "cosebis_bb_nz_independent": cosebis_bb_nz_independent,
                "ee_varies_as_expected": ee_varies_as_expected,
                # All spaces, MC methods included
                "bb_max_deviation": bb_max,
                "ee_max_deviation": ee_max,
                "bb_closer_to_unity": bb_closer_to_unity,
                "bb_within_2pct": bb_within_2pct,
            },
            "version": version,
        },
        "output": {
            "figure": Path(figure_path).name,
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
    Path(evidence_path).parent.mkdir(parents=True, exist_ok=True)
    with open(evidence_path, "w") as f:
        json.dump(evidence, f, indent=2)

    # Print summary
    print("\nBB Covariance n(z) Independence Summary:")
    print(f"  COSEBIS B_n max deviation: {cosebis_bb_max * 100:.6f}%")
    print(f"  COSEBIS E_n max deviation: {cosebis_ee_max * 100:.2f}%")
    print(f"  COSEBIS B_n n(z)-independent (<0.1%): {cosebis_bb_nz_independent}")
    print(f"  E-modes vary as expected (5-15%): {ee_varies_as_expected}")
    print("\n  Pure E/B + Harmonic (MC methods):")
    print(f"    BB max deviation: {bb_max * 100:.2f}%")
    print(f"    EE max deviation: {ee_max * 100:.2f}%")
    print("    Note: MC sampling noise causes BB to vary similarly to EE")


def _from_snakemake(smk):
    config = smk.config
    labels = list(config["fiducial"]["nz_realisations"])
    pure_eb_paths = {b: smk.input[f"pure_eb_{b}"] for b in labels}
    harmonic_paths = {b: smk.input[f"harmonic_{b}"] for b in labels}
    cov_integration_paths = {b: smk.input[f"cov_integration_{b}"] for b in labels}
    main(
        config=config,
        pure_eb_paths=pure_eb_paths,
        harmonic_paths=harmonic_paths,
        xi_integration_path=smk.input.xi_integration,
        cov_integration_paths=cov_integration_paths,
        pseudo_cl_path=smk.input.pseudo_cl,
        nmodes=smk.params.nmodes,
        theta_min=smk.params.theta_min,
        theta_max=smk.params.theta_max,
        figure_path=smk.output.figure,
        evidence_path=smk.output.evidence,
    )


def _cov_integration_path(cov_dir, version, min_sep, max_sep, nbins):
    """Reproduce common.covariance_path for the Gaussian integration-grid,
    masked covariance (suffix _processed.txt)."""
    base = (
        f"covariance_{version}_g_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_masked"
    )
    return os.path.join(cov_dir, base, f"{base}_processed.txt")


def _from_cli(argv=None):
    ap = argparse.ArgumentParser(
        description="BB-covariance n(z)-independence cross-check (mock version)."
    )
    ap.add_argument("--config", required=True, help="Absolute path to config.yaml")
    ap.add_argument(
        "--version",
        default=None,
        help="Mock (leak-corrected) version; default <fiducial.mock_version>_leak_corr",
    )
    ap.add_argument(
        "--pure-eb-dir",
        required=True,
        help="Dir with {version}_pure_eb_semianalytic.npz per n(z) realisation",
    )
    ap.add_argument(
        "--cosmo-val-dir",
        required=True,
        help=(
            "COSMO_VAL output dir (pseudo_cl SACC parts, pseudo_cl_cov FITS "
            "+ xi_integration txt)"
        ),
    )
    ap.add_argument(
        "--covariance-dir",
        required=True,
        help="COSMO_INFERENCE data/covariance dir (Gaussian integration covariances)",
    )
    ap.add_argument("--out", required=True, help="Output directory (lc {output})")
    a = ap.parse_args(argv)

    with open(a.config) as f:
        config = yaml.safe_load(f)

    version = a.version or f"{config['fiducial']['mock_version']}_leak_corr"
    fid = config["fiducial"]
    min_sep_int, max_sep_int, nbins_int = (
        fid["min_sep_int"],
        fid["max_sep_int"],
        fid["nbins_int"],
    )
    npatch = fid["npatch"]

    realisations = fid["nz_realisations"]
    pure_eb_paths = {
        b: os.path.join(a.pure_eb_dir, f"{ver}_pure_eb_semianalytic.npz")
        for b, ver in realisations.items()
    }
    harmonic_paths = {
        b: os.path.join(a.cosmo_val_dir, f"pseudo_cl_cov_{ver}_powspace_nbins=32.fits")
        for b, ver in realisations.items()
    }
    cov_integration_paths = {
        b: _cov_integration_path(
            a.covariance_dir, ver, min_sep_int, max_sep_int, nbins_int
        )
        for b, ver in realisations.items()
    }
    xi_integration_path = os.path.join(
        a.cosmo_val_dir,
        f"{version}_xi_minsep={min_sep_int}_maxsep={max_sep_int}"
        f"_nbins={nbins_int}_npatch={npatch}.txt",
    )
    pseudo_cl_path = os.path.join(
        a.cosmo_val_dir, f"pseudo_cl_{version}_powspace_nbins=32.sacc"
    )

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    main(
        config=config,
        pure_eb_paths=pure_eb_paths,
        harmonic_paths=harmonic_paths,
        xi_integration_path=xi_integration_path,
        cov_integration_paths=cov_integration_paths,
        pseudo_cl_path=pseudo_cl_path,
        nmodes=fid["nmodes"],
        theta_min=config["cosebis"]["theta_min"],
        theta_max=config["cosebis"]["theta_max"],
        figure_path=str(out_dir / "figure.png"),
        evidence_path=str(out_dir / "evidence.json"),
    )


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821 — injected by Snakemake's script: directive
    except NameError:
        _from_cli()
    else:
        _from_snakemake(snakemake)  # noqa: F821
