"""
Covariance Blind Consistency Claim

Compare reporting covariance diagonals across blinds A, B, C.
Claim: diagonals agree at percent level.
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use(
    "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/"
    "2D_cosmic_shear_paper_plots/config/paper.mplstyle"
)


def load_covariance_diagonal(path, nbins=20):
    """Load covariance and extract ξ+ and ξ- diagonals."""
    cov = np.loadtxt(path)
    # Covariance is 2*nbins x 2*nbins: [ξ+, ξ-]
    diag = np.diag(cov)
    return {
        "xip": diag[:nbins],
        "xim": diag[nbins:],
    }


def compute_ratios(diag_ref, diag_test):
    """Compute ratio and deviation statistics."""
    ratios = {}
    for key in ["xip", "xim"]:
        ratio = diag_test[key] / diag_ref[key]
        dev = np.abs(ratio - 1.0)
        ratios[key] = {
            "ratio": ratio,
            "max_dev": float(np.max(dev)),
            "mean_dev": float(np.mean(dev)),
        }
    return ratios


def make_figure(theta, ratios_B, ratios_C, output_path):
    """Two-panel ratio plot with threshold bands."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    colors = sns.color_palette("colorblind", 2)

    for ax, key, label in zip(axes, ["xip", "xim"], [r"$\xi_+$", r"$\xi_-$"]):
        ax.axhline(1.0, color="gray", ls="-", lw=0.8, zorder=0)
        ax.axhspan(0.99, 1.01, color="teal", alpha=0.15, label=r"$\pm 1\%$")
        ax.axhspan(0.90, 1.10, color="orange", alpha=0.10, label=r"$\pm 10\%$")

        ax.plot(
            theta,
            ratios_B[key]["ratio"],
            "o-",
            color=colors[0],
            label="B / A",
            markersize=5,
        )
        ax.plot(
            theta,
            ratios_C[key]["ratio"],
            "s--",
            color=colors[1],
            label="C / A",
            markersize=5,
        )

        ax.set_xscale("log")
        ax.set_xlabel(r"$\theta$ [arcmin]")
        ax.set_title(label)

    axes[0].set_ylabel("Diagonal ratio")
    axes[0].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(snakemake):
    # Load covariances (lowercase names match rule definition)
    diag_A = load_covariance_diagonal(snakemake.input.cov_a)
    diag_B = load_covariance_diagonal(snakemake.input.cov_b)
    diag_C = load_covariance_diagonal(snakemake.input.cov_c)

    # Compute ratios
    ratios_B = compute_ratios(diag_A, diag_B)
    ratios_C = compute_ratios(diag_A, diag_C)

    # Angular bins (log-spaced)
    min_sep = snakemake.config["fiducial"]["min_sep"]
    max_sep = snakemake.config["fiducial"]["max_sep"]
    nbins = snakemake.config["fiducial"]["nbins"]
    theta = np.logspace(np.log10(min_sep), np.log10(max_sep), nbins)

    # Generate figure
    make_figure(theta, ratios_B, ratios_C, snakemake.output.figure)

    # Determine pass/fail
    all_max_devs = [
        ratios_B["xip"]["max_dev"],
        ratios_B["xim"]["max_dev"],
        ratios_C["xip"]["max_dev"],
        ratios_C["xim"]["max_dev"],
    ]
    pass_1pct = all(d < 0.01 for d in all_max_devs)
    pass_10pct = all(d < 0.10 for d in all_max_devs)

    # Build evidence
    evidence = {
        "spec_id": "covariance_blind_consistency",
        "spec_path": "workflow/config/covariance_blind_consistency.md",
        "depends_on": ["covariance"],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "xip": {
                "B_to_A": {
                    "max_dev": ratios_B["xip"]["max_dev"],
                    "mean_dev": ratios_B["xip"]["mean_dev"],
                },
                "C_to_A": {
                    "max_dev": ratios_C["xip"]["max_dev"],
                    "mean_dev": ratios_C["xip"]["mean_dev"],
                },
            },
            "xim": {
                "B_to_A": {
                    "max_dev": ratios_B["xim"]["max_dev"],
                    "mean_dev": ratios_B["xim"]["mean_dev"],
                },
                "C_to_A": {
                    "max_dev": ratios_C["xim"]["max_dev"],
                    "mean_dev": ratios_C["xim"]["mean_dev"],
                },
            },
            "pass_1pct": pass_1pct,
            "pass_10pct": pass_10pct,
        },
        "output": {
            "figure": Path(snakemake.output.figure).name,
        },
    }

    # Write evidence
    Path(snakemake.output.evidence).parent.mkdir(parents=True, exist_ok=True)
    with open(snakemake.output.evidence, "w") as f:
        json.dump(evidence, f, indent=2)


if __name__ == "__main__":
    main(snakemake)  # noqa: F821
