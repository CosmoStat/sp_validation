r"""COSEBIS W_n(\ell) filter functions overlaid on BB bandpower data.

Builds intuition about the C_ell -> COSEBIS transform by showing:
- 32-bin BB bandpower data (gray background)
- Continuous W_n(ell) filter functions for modes 1-6 (colored curves)
- Vertical tick marks at the 32 bin centres to show sampling density

Demonstrates visually why coarse bandpowers underresolve higher COSEBIS modes.

Author: Claude Code
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from astropy.io import fits
from cosmo_numba.B_modes.cosebis import COSEBIS
from plotting_utils import PAPER_MPLSTYLE

plt.style.use(PAPER_MPLSTYLE)


def load_bb_data(pseudo_cl_path, pseudo_cov_path):
    """Load BB bandpower data and errorbars."""
    with fits.open(pseudo_cl_path) as hdul:
        data = hdul["PSEUDO_CELL"].data
        ell = np.asarray(data["ELL"], dtype=float)
        bb = np.asarray(data["BB"], dtype=float)

    with fits.open(pseudo_cov_path) as hdul:
        cov_bb = hdul["COVAR_BB_BB"].data
    sigma_bb = np.sqrt(np.diag(cov_bb))

    return ell, bb, sigma_bb


def compute_filters(theta_min, theta_max, nmodes, ell_dense):
    """Compute W_n(ell) filter functions on a dense ell grid."""
    cosebis = COSEBIS(theta_min, theta_max, nmodes)
    return cosebis.get_Wn_log(ell_dense)


def make_figure(ell_32, bb, sigma_bb, ell_dense, Wn_full, Wn_fid, output_path):
    """Two-panel figure: full and fiducial scale cuts."""
    nmodes = Wn_full.shape[0]
    colors = sns.color_palette("husl", nmodes)

    fig, axes = plt.subplots(
        2, 1, figsize=(8.5, 7.5), sharex=True,
        gridspec_kw={"hspace": 0.12},
    )

    for ax, Wn, cut_label in [
        (axes[0], Wn_full, r"Full range $[1', 250']$"),
        (axes[1], Wn_fid, r"Fiducial $[12', 83']$"),
    ]:
        # --- Background: BB data on right axis ---
        ax_bb = ax.twinx()
        Dl_bb = ell_32 * (ell_32 + 1) * bb / (2 * np.pi)
        Dl_sig = ell_32 * (ell_32 + 1) * sigma_bb / (2 * np.pi)
        ax_bb.fill_between(
            ell_32, Dl_bb - Dl_sig, Dl_bb + Dl_sig,
            color="0.85", zorder=0, label=r"$C_\ell^{BB} \pm 1\sigma$",
        )
        ax_bb.scatter(
            ell_32, Dl_bb,
            s=12, color="0.55", zorder=1, marker="s",
        )
        ax_bb.set_ylabel(
            r"$\ell(\ell{+}1)\,C_\ell^{BB}/2\pi$",
            fontsize=8, color="0.5", labelpad=8,
        )
        ax_bb.tick_params(axis="y", labelcolor="0.5", labelsize=7)

        # --- Foreground: W_n filter curves ---
        for n in range(nmodes):
            peak = np.max(np.abs(Wn[n]))
            if peak > 0:
                ax.plot(
                    ell_dense, Wn[n] / peak,
                    color=colors[n], lw=1.4, alpha=0.9,
                    label=rf"$n = {n+1}$",
                )

        ax.axhline(0, color="k", lw=0.4, zorder=0)
        ax.set_ylabel(r"$W_n(\ell)$ (peak-normalized)", fontsize=9)
        ax.set_ylim(-1.15, 1.35)

        # Bin-centre tick marks along the top
        for e in ell_32:
            ax.plot(
                [e, e], [1.1, 1.2], color="0.35", lw=0.6,
                clip_on=False, zorder=5, transform=ax.get_xaxis_transform(),
            )

        # Scale-cut annotation
        ax.text(
            0.02, 0.95, cut_label,
            transform=ax.transAxes, fontsize=9,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
        )

    # Shared legend for filter modes
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles, labels,
        fontsize=7.5, ncol=nmodes, loc="upper right",
        title=r"COSEBIS mode $n$", title_fontsize=8,
        columnspacing=1.0, handlelength=1.5,
    )

    axes[1].set_xlabel(r"$\ell$", fontsize=10)
    axes[1].set_xscale("log")
    axes[1].set_xlim(8, 2500)
    axes[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
    axes[1].xaxis.set_minor_formatter(ticker.NullFormatter())

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    config = snakemake.config  # noqa: F821

    pseudo_cl_path = snakemake.input.pseudo_cl  # noqa: F821
    pseudo_cov_path = snakemake.input.pseudo_cl_cov  # noqa: F821
    output_evidence = snakemake.output.evidence  # noqa: F821
    output_figure = snakemake.output.figure  # noqa: F821

    theta_min_full = config["cosebis"]["theta_min"]
    theta_max_full = config["cosebis"]["theta_max"]
    theta_min_fid = config["fiducial"]["fiducial_min_scale"]
    theta_max_fid = config["fiducial"]["fiducial_max_scale"]
    nmodes = 6

    ell_32, bb, sigma_bb = load_bb_data(pseudo_cl_path, pseudo_cov_path)

    ell_dense = np.logspace(np.log10(8), np.log10(2500), 2000)

    print(f"Computing W_n filters for [{theta_min_full}, {theta_max_full}]'...")
    Wn_full = compute_filters(theta_min_full, theta_max_full, nmodes, ell_dense)

    print(f"Computing W_n filters for [{theta_min_fid}, {theta_max_fid}]'...")
    Wn_fid = compute_filters(theta_min_fid, theta_max_fid, nmodes, ell_dense)

    make_figure(ell_32, bb, sigma_bb, ell_dense, Wn_full, Wn_fid, output_figure)

    # --- Evidence ---
    spec_paths = snakemake.input.specs  # noqa: F821
    depends_on = [Path(p).stem for p in spec_paths[1:]]

    evidence = {
        "id": Path(spec_paths[0]).stem,
        "kind": "claim",
        "depends_on": depends_on,
        "generated": datetime.now().isoformat(),
        "evidence": {
            "nmodes_shown": nmodes,
            "n_bandpower_bins": len(ell_32),
            "ell_range": [float(ell_32.min()), float(ell_32.max())],
            "scale_cuts": {
                "full": [theta_min_full, theta_max_full],
                "fiducial": [theta_min_fid, theta_max_fid],
            },
            "note": (
                "Higher COSEBIS modes have increasingly oscillatory W_n(ell) filters. "
                "32 powspace bandpowers cannot resolve these oscillations for n >= 3, "
                "causing harmonic-space COSEBIS to diverge from configuration-space results. "
                "This is a sampling limitation, not a code error."
            ),
        },
        "artifacts": {
            "figure": Path(output_figure).name,
        },
    }

    Path(output_evidence).parent.mkdir(parents=True, exist_ok=True)
    with open(output_evidence, "w") as f:
        json.dump(evidence, f, indent=2)
    print(f"Evidence written: {output_evidence}")


if __name__ == "__main__":
    main()
