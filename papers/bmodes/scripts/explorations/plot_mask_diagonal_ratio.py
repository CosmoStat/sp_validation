import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


STYLE_PATH = Path(
    "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle"
)


def compute_ratio(mask_path: Path, nomask_path: Path, data_out: Path, plot_out: Path) -> None:
    mask_cov = np.loadtxt(mask_path)
    nomask_cov = np.loadtxt(nomask_path)

    if mask_cov.shape != nomask_cov.shape:
        raise ValueError(f"Mask and no-mask covariances have mismatched shapes {mask_cov.shape} vs {nomask_cov.shape}")

    nomask_diag = np.diag(nomask_cov)
    mask_diag = np.diag(mask_cov)

    if np.any(nomask_diag == 0):
        zero_indices = np.where(nomask_diag == 0)[0]
        raise ValueError(f"Zero variance entries at indices {zero_indices.tolist()} prevent ratio evaluation.")

    ratio = mask_diag / nomask_diag

    data_out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(data_out, ratio)

    plt.style.use(STYLE_PATH)
    sns.set_palette("husl", 1)

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    x = np.arange(ratio.size)

    ax.plot(x, ratio, linewidth=1.5)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Covariance diagonal element")
    ax.set_ylabel(r"$C_{\mathrm{mask}} / C_{\mathrm{nomask}}$")
    ax.set_title("Gaussian covariance diagonal ratio")
    ax.set_xlim(0, ratio.size - 1)

    plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_out, dpi=300)


def run_from_snakemake() -> None:
    mask_path = Path(snakemake.input["mask"])
    nomask_path = Path(snakemake.input["nomask"])
    data_out = Path(snakemake.output["data"])
    plot_out = Path(snakemake.output["plot"])
    compute_ratio(mask_path, nomask_path, data_out, plot_out)


def run_from_cli() -> None:
    parser = argparse.ArgumentParser(description="Plot ratio of covariance diagonals with and without mask.")
    parser.add_argument("mask", type=Path, help="Path to Gaussian covariance matrix generated with mask.")
    parser.add_argument("nomask", type=Path, help="Path to Gaussian covariance matrix generated without mask.")
    parser.add_argument("data", type=Path, help="Output path for diagonal ratio text file.")
    parser.add_argument("plot", type=Path, help="Output path for diagonal ratio plot.")
    args = parser.parse_args()

    compute_ratio(args.mask, args.nomask, args.data, args.plot)


if "snakemake" in globals():
    run_from_snakemake()
else:
    run_from_cli()
