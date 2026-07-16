#!/usr/bin/env python3
# %%
"""
Visual comparison of CosmoCov covariance matrices with and without mask window.

This script loads processed covariance matrices from the Snakemake exploration
and produces diagnostic plots highlighting the impact of applying the CosmoCov
mask power spectrum file.
"""

# %%
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython import get_ipython

# %%
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

plt.style.use(
    "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/"
    "notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle"
)


# %%
def load_covariance_pair(
    mask_path: Path, nomask_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """Load covariance matrices for mask/no-mask comparison."""
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask covariance not found: {mask_path}")
    if not nomask_path.exists():
        raise FileNotFoundError(f"No-mask covariance not found: {nomask_path}")

    mask_cov = np.loadtxt(mask_path)
    nomask_cov = np.loadtxt(nomask_path)

    if mask_cov.shape != nomask_cov.shape:
        raise ValueError(
            f"Covariance shapes differ: mask {mask_cov.shape}, no-mask {nomask_cov.shape}"
        )

    return mask_cov, nomask_cov


# %%
def create_diagnostics(
    mask_cov: np.ndarray, nomask_cov: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute standard deviations, variance ratios, and covariance differences."""
    mask_std = np.sqrt(np.diag(mask_cov))
    nomask_std = np.sqrt(np.diag(nomask_cov))

    # Protect against divide-by-zero in variance ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        variance_ratio = np.square(mask_std) / np.square(nomask_std)
        variance_ratio[~np.isfinite(variance_ratio)] = np.nan

    covariance_delta = mask_cov - nomask_cov

    return mask_std, nomask_std, variance_ratio, covariance_delta


# %%
def plot_comparison(
    mask_std: np.ndarray,
    nomask_std: np.ndarray,
    variance_ratio: np.ndarray,
    covariance_delta: np.ndarray,
    output_path: Path,
) -> None:
    """Generate diagnostic plots comparing masked vs nominal covariance."""
    sns.set_palette("husl", 2)

    n_points = mask_std.size
    indices = np.arange(n_points)

    fig = plt.figure(figsize=(12, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2])

    ax_std = fig.add_subplot(grid[0, 0])
    ax_ratio = fig.add_subplot(grid[0, 1])
    ax_delta = fig.add_subplot(grid[1, :])

    ax_std.plot(indices, nomask_std, label="No mask", linewidth=2)
    ax_std.plot(indices, mask_std, label="Mask applied", linewidth=2)
    ax_std.set_xlabel("Data vector index")
    ax_std.set_ylabel("Standard deviation")
    ax_std.set_title("Marginal uncertainties")
    ax_std.grid(alpha=0.3)
    ax_std.legend()

    ax_ratio.plot(indices, variance_ratio, linewidth=2, color="#ef8a62")
    ax_ratio.set_xlabel("Data vector index")
    ax_ratio.set_ylabel(
        "$\\sigma^2_{\\mathrm{mask}} / \\sigma^2_{\\mathrm{no\\ mask}}$"
    )
    ax_ratio.set_title("Variance ratio")
    ax_ratio.grid(alpha=0.3)

    im = ax_delta.imshow(
        covariance_delta,
        cmap="PRGn",
        vmin=-np.nanmax(np.abs(covariance_delta)),
        vmax=np.nanmax(np.abs(covariance_delta)),
        origin="upper",
    )
    ax_delta.set_title("Covariance difference (mask - no mask)")
    ax_delta.set_xlabel("Data vector index")
    ax_delta.set_ylabel("Data vector index")
    fig.colorbar(im, ax=ax_delta, fraction=0.046, pad=0.04, label="$\\Delta C$")

    variance_reduction = np.nanmedian(variance_ratio)
    ax_ratio.text(
        0.02,
        0.95,
        f"Median variance ratio: {variance_reduction:.3f}",
        transform=ax_ratio.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# %%
def main() -> None:
    try:
        mask_path = Path(snakemake.input["mask"])  # type: ignore[name-defined]
        nomask_path = Path(snakemake.input["nomask"])  # type: ignore[name-defined]
        output_path = Path(snakemake.output["plot"])  # type: ignore[name-defined]
    except NameError as exc:  # pragma: no cover - executed only outside Snakemake
        raise RuntimeError(
            "This script is intended to be executed via Snakemake."
        ) from exc

    mask_cov, nomask_cov = load_covariance_pair(mask_path, nomask_path)
    mask_std, nomask_std, variance_ratio, covariance_delta = create_diagnostics(
        mask_cov, nomask_cov
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(mask_std, nomask_std, variance_ratio, covariance_delta, output_path)


# %%
if __name__ == "__main__":
    main()
