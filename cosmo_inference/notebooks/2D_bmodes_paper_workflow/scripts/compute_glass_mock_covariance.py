"""Compute mock covariance for xi and C_ell data vectors."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits


def extract_mock_id(path: Path) -> str:
    """Return the zero-padded mock identifier embedded in file names."""

    stem = path.stem
    parts = stem.split("_")
    for token in parts:
        if token.isdigit() and len(token) == 5:
            return token
    raise ValueError(f"Cannot determine mock id from {path}")


def load_xi_components(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load xi+ and xi- vectors from a TreeCorr FITS catalog."""

    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        xip = np.asarray(data["xip"], dtype=np.float64)
        xim = np.asarray(data["xim"], dtype=np.float64)
    return xip, xim


def load_cl_components(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load C_ell^EE and C_ell^BB from GLASS mock numpy output."""

    cl_block = np.load(path)
    if cl_block.shape[0] < 5:
        raise ValueError(f"Unexpected C_ell array shape {cl_block.shape} for {path}")
    clee = np.asarray(cl_block[1], dtype=np.float64)
    clbb = np.asarray(cl_block[4], dtype=np.float64)
    return clee, clbb


def build_mock_matrix(
    xi_files: dict[str, Path],
    cl_files: dict[str, Path],
) -> tuple[np.ndarray, list[int]]:
    """Stack data vectors for mocks present in both xi and C_ell collections."""

    mock_ids = sorted(set(xi_files) & set(cl_files))
    if not mock_ids:
        raise ValueError("No overlapping mocks between xi and C_ell selections")

    vectors = []
    block_lengths: list[int] | None = None
    for mock_id in mock_ids:
        xip, xim = load_xi_components(xi_files[mock_id])
        clee, clbb = load_cl_components(cl_files[mock_id])
        vectors.append(np.concatenate([xip, xim, clee, clbb]))

        if block_lengths is None:
            block_lengths = [len(xip), len(xim), len(clee), len(clbb)]

    data_matrix = np.vstack(vectors)

    # Dimension checks guard against silent shape changes across mocks.
    assert block_lengths is not None  # for type checkers
    return data_matrix, block_lengths


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    """Convert covariance matrix into correlation matrix."""

    diag = np.sqrt(np.diag(covariance))
    if np.any(diag == 0):
        raise ValueError("Encountered zero variance while constructing correlation matrix")
    corr = covariance / np.outer(diag, diag)
    np.clip(corr, -1.0, 1.0, out=corr)
    return corr


def plot_correlation(
    correlation: np.ndarray,
    block_sizes: list[int],
    output_png: Path,
) -> None:
    """Generate a correlation heatmap with block annotations."""

    sns.set_theme(style="white")
    sns.set_palette("husl", len(block_sizes))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation,
        ax=ax,
        cmap="icefire",
        center=0.0,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        vmin=-1.0,
        vmax=1.0,
    )

    boundaries = np.cumsum(block_sizes)
    for boundary in boundaries[:-1]:
        ax.axhline(boundary, color="white", linewidth=1.2)
        ax.axvline(boundary, color="white", linewidth=1.2)

    centers = [0.5 * (start + end) for start, end in zip([0] + list(boundaries[:-1]), boundaries)]
    labels = [r"$\xi_+$", r"$\xi_-$", r"$C_\ell^{EE}$", r"$C_\ell^{BB}$"]
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(centers)
    ax.set_yticklabels(labels, rotation=0)
    ax.set_xlabel("Data vector block")
    ax.set_ylabel("Data vector block")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def collect_mock_files(paths: list[Path]) -> dict[str, Path]:
    """Return dictionary mapping mock id to supplied file paths."""

    files: dict[str, Path] = {}
    for path in sorted(paths):
        mock_id = extract_mock_id(path)
        if mock_id in files:
            raise ValueError(f"Duplicate mock id {mock_id}: {path} and {files[mock_id]}")
        files[mock_id] = path
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xi-files", type=Path, nargs="+", required=True, help="Paths to xi mocks")
    parser.add_argument("--cl-files", type=Path, nargs="+", required=True, help="Paths to C_ell mocks")
    parser.add_argument("--xi-covariance", type=Path, required=True, help="Output path for xi-only covariance (npy)")
    parser.add_argument("--cl-covariance", type=Path, required=True, help="Output path for cl-only covariance (npy)")
    parser.add_argument("--combined-covariance", type=Path, required=True, help="Output path for combined covariance (npy)")
    parser.add_argument("--combined-correlation-plot", type=Path, required=True, help="Output path for combined correlation heatmap (png)")
    parser.add_argument("--xi-mean", type=Path, required=True, help="Output path for xi mean vector (npy)")
    parser.add_argument("--cl-mean", type=Path, required=True, help="Output path for cl mean vector (npy)")
    parser.add_argument("--combined-mean", type=Path, required=True, help="Output path for combined mean vector (npy)")
    return parser.parse_args()


def main() -> None:
    # Try to use Snakemake's injected snakemake object, fall back to argparse
    try:
        from snakemake.script import snakemake
        xi_files = collect_mock_files([Path(f) for f in snakemake.input.xi])
        cl_files = collect_mock_files([Path(f) for f in snakemake.input.cl])
        xi_covariance_path = Path(snakemake.output.xi_covariance)
        cl_covariance_path = Path(snakemake.output.cl_covariance)
        combined_covariance_path = Path(snakemake.output.combined_covariance)
        correlation_plot_path = Path(snakemake.output.correlation_plot)
        xi_mean_path = Path(snakemake.output.xi_mean)
        cl_mean_path = Path(snakemake.output.cl_mean)
        combined_mean_path = Path(snakemake.output.combined_mean)
    except NameError:
        # Standalone execution
        args = parse_args()
        xi_files = collect_mock_files(args.xi_files)
        cl_files = collect_mock_files(args.cl_files)
        xi_covariance_path = args.xi_covariance
        cl_covariance_path = args.cl_covariance
        combined_covariance_path = args.combined_covariance
        correlation_plot_path = args.combined_correlation_plot
        xi_mean_path = args.xi_mean
        cl_mean_path = args.cl_mean
        combined_mean_path = args.combined_mean

    data_matrix, block_lengths = build_mock_matrix(xi_files, cl_files)

    # Extract data vector blocks
    xi_split = sum(block_lengths[:2])
    xi_data = data_matrix[:, :xi_split]
    cl_data = data_matrix[:, xi_split:]

    # Compute covariances for each type
    xi_covariance = np.cov(xi_data, rowvar=False, ddof=1)
    cl_covariance = np.cov(cl_data, rowvar=False, ddof=1)
    combined_covariance = np.cov(data_matrix, rowvar=False, ddof=1)

    # Compute means for each type
    xi_mean = np.mean(xi_data, axis=0)
    cl_mean = np.mean(cl_data, axis=0)
    combined_mean = np.mean(data_matrix, axis=0)

    # Save covariances
    np.save(xi_covariance_path, xi_covariance)
    np.save(cl_covariance_path, cl_covariance)
    np.save(combined_covariance_path, combined_covariance)

    # Save means
    np.save(xi_mean_path, xi_mean)
    np.save(cl_mean_path, cl_mean)
    np.save(combined_mean_path, combined_mean)

    # Plot combined correlation matrix only
    correlation = covariance_to_correlation(combined_covariance)
    plot_correlation(correlation, block_lengths, correlation_plot_path)

if __name__ == "__main__":
    main()
