"""Pure E/B covariance structure claim.

Validates covariance matrix for B-mode tests by analyzing block structure:
- 6 blocks: E+, E-, B+, B-, amb+, amb-
- Block-wise condition numbers
- Correlation matrix heatmap

Shows that E and B blocks are well-conditioned while ill-conditioning
is localized to ambiguous modes (expected from finite angular range).
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Registers seaborn colormaps (icefire, etc.) with matplotlib

from plotting_utils import PAPER_MPLSTYLE


plt.style.use(PAPER_MPLSTYLE)


def _analyze_block(covariance, start_idx, end_idx, name):
    """Compute condition number and definiteness for a covariance block.

    Parameters
    ----------
    covariance : array
        Full covariance matrix
    start_idx : int
        Start index of block
    end_idx : int
        End index of block (exclusive)
    name : str
        Block name for reporting

    Returns
    -------
    dict
        Block statistics: condition number, positive definiteness, eigenvalue bounds
    """
    block = covariance[start_idx:end_idx, start_idx:end_idx]
    eigenvalues = np.linalg.eigvalsh(block)

    min_eig = float(eigenvalues[0])
    max_eig = float(eigenvalues[-1])
    is_positive_definite = bool(min_eig > 0)
    condition_number = max_eig / min_eig if min_eig > 0 else float('inf')

    return {
        "condition_number": condition_number,
        "is_positive_definite": is_positive_definite,
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "block_size": list(block.shape),
        "description": name,
    }


def _cov_to_corr(covariance):
    """Convert covariance matrix to correlation matrix.

    Parameters
    ----------
    covariance : array (N, N)
        Covariance matrix

    Returns
    -------
    array (N, N)
        Correlation matrix with diagonal = 1
    """
    std = np.sqrt(np.diag(covariance))
    std_inv = 1.0 / np.where(std > 0, std, 1.0)
    correlation = covariance * np.outer(std_inv, std_inv)
    return correlation


def main():
    config = snakemake.config
    version = config["fiducial"]["version"]

    # Load precomputed pure E/B data
    dataset = np.load(snakemake.input["pure_eb_data"])

    theta = dataset["theta"]
    nbins = len(theta)
    cov_pure_eb = dataset["cov_pure_eb"]

    # Matrix is 6*nbins x 6*nbins:
    # Block structure: [E+, E-, B+, B-, amb+, amb-]
    # Each block is nbins x nbins
    assert cov_pure_eb.shape == (6*nbins, 6*nbins), f"Expected ({6*nbins}, {6*nbins}), got {cov_pure_eb.shape}"

    # Analyze full matrix
    eigenvalues = np.linalg.eigvalsh(cov_pure_eb)
    min_eig = float(eigenvalues[0])
    max_eig = float(eigenvalues[-1])
    is_positive_definite = bool(min_eig > 0)
    condition_number = max_eig / min_eig if min_eig > 0 else float('inf')

    # Analyze blocks
    block_analysis = {
        "xi_E": _analyze_block(
            cov_pure_eb, 0, 2*nbins, "xi_+^E and xi_-^E combined (40x40)"
        ),
        "xi_B": _analyze_block(
            cov_pure_eb, 2*nbins, 4*nbins, "xi_+^B and xi_-^B combined (40x40)"
        ),
        "xi_amb": _analyze_block(
            cov_pure_eb, 4*nbins, 6*nbins, "xi_+^amb and xi_-^amb combined (40x40)"
        ),
    }

    # Convert to correlation matrix for visualization
    correlation = _cov_to_corr(cov_pure_eb)

    # Create figure (matching v1 epistemics style)
    fig, ax = plt.subplots(figsize=(7.24, 5.79))

    # Plot correlation matrix (no origin="lower" so y increases downward like v1)
    im = ax.imshow(
        correlation,
        cmap="icefire",
        vmin=-1.0,
        vmax=1.0,
        aspect="equal",
    )

    # Add block boundaries at all 6 block edges
    for i in range(1, 6):
        ax.axhline(i * nbins - 0.5, color="black", linewidth=1.5, alpha=0.7)
        ax.axvline(i * nbins - 0.5, color="black", linewidth=1.5, alpha=0.7)

    # 6 block labels (matching v1 epistemics style)
    block_labels = [
        r"$\xi_+^{\mathrm{E}}$",
        r"$\xi_-^{\mathrm{E}}$",
        r"$\xi_+^{\mathrm{B}}$",
        r"$\xi_-^{\mathrm{B}}$",
        r"$\xi_+^{\mathrm{amb}}$",
        r"$\xi_-^{\mathrm{amb}}$",
    ]

    # Compute block boundaries and centers
    block_sizes = [nbins] * 6
    boundaries = np.cumsum(block_sizes)
    centers = [0.5 * (start + end) - 0.5 for start, end in zip([0] + list(boundaries[:-1]), boundaries)]

    # Set ticks at block centers with labels (no axis labels, tick labels only)
    ax.set_xticks(centers)
    ax.set_xticklabels(block_labels, rotation=45, ha="right")
    ax.set_yticks(centers)
    ax.set_yticklabels(block_labels, rotation=0)

    # Colorbar — match height to matrix
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation coefficient")

    fig.tight_layout()

    # Save outputs
    output_dir = Path(snakemake.output["evidence"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = Path(snakemake.output["figure"])
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_path}")

    # Copy to paper figures directory
    paper_fig_path = Path(snakemake.output["paper_figure"])
    paper_fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved {paper_fig_path}")

    plt.close(fig)

    # Write evidence.json
    spec_paths = snakemake.input["specs"]

    evidence_data = {
        "spec_id": "pure_eb_covariance",
        "spec_path": spec_paths[0],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "condition_number": condition_number,
            "is_positive_definite": is_positive_definite,
            "min_eigenvalue": min_eig,
            "max_eigenvalue": max_eig,
            "matrix_shape": list(cov_pure_eb.shape),
            "n_bins": int(nbins),
            "n_blocks": 6,
            "block_analysis": block_analysis,
        },
        "parameters": {
            "version": version,
            "blind": config["fiducial"]["blind"],
        },
        "artifacts": {
            "figure": "figure.png",
        },
    }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"Saved evidence to {evidence_path}")

    # Print summary
    print(f"\nCovariance Matrix Summary:")
    print(f"  Shape: {cov_pure_eb.shape}")
    print(f"  Full matrix condition number: {condition_number:.2e}")
    print(f"  Positive definite: {is_positive_definite}")
    print(f"\nBlock Analysis:")
    for block_name, stats in block_analysis.items():
        print(f"  {block_name}:")
        print(f"    Condition number: {stats['condition_number']:.2e}")
        print(f"    Positive definite: {stats['is_positive_definite']}")
        print(f"    Min eigenvalue: {stats['min_eigenvalue']:.2e}")
        print(f"    Max eigenvalue: {stats['max_eigenvalue']:.2e}")


if __name__ == "__main__":
    main()
