"""Assemble a raw CosmoCov block dump into an analysis-ready covariance matrix.

CosmoCov writes one row per (i, j) element with the Gaussian term in column 8
and the non-Gaussian term in column 9; this rebuilds the symmetric matrices,
checks positive-definiteness, and plots the correlation matrix.
"""

import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def get_cov(filename):
    """Return (gaussian, non-gaussian, ndata) from a CosmoCov element list."""

    data = np.loadtxt(filename)
    ndata = int(np.max(data[:, 0])) + 1

    cov_g = np.zeros((ndata, ndata))
    cov_ng = np.zeros((ndata, ndata))
    for i in range(data.shape[0]):
        row, col = int(data[i, 0]), int(data[i, 1])
        cov_g[row, col] = cov_g[col, row] = data[i, 8]
        cov_ng[row, col] = cov_ng[col, row] = data[i, 9]

    return cov_g, cov_ng, ndata


def plot_correlation(cov, ndata, plot_path):
    """Save the correlation matrix with xi+/xi- block annotations."""

    diag = np.sqrt(np.diag(cov))
    correlation = cov / np.outer(diag, diag)

    fig, ax = plt.subplots()
    extent = (0, ndata, ndata, 0)
    image = ax.imshow(correlation, cmap="seismic", vmin=-1, vmax=1, extent=extent)

    ax.axvline(x=ndata // 2, color="black", linewidth=1.0)
    ax.axhline(y=ndata // 2, color="black", linewidth=1.0)

    fig.colorbar(image, orientation="vertical")

    ax.text(ndata // 4, ndata + 5, r"$\xi_+^{ij}(\theta)$", fontsize=12)
    ax.text(3 * (ndata // 4), ndata + 5, r"$\xi_-^{ij}(\theta)$", fontsize=12)
    ax.text(-9, ndata // 4, r"$\xi_+^{ij}(\theta)$", fontsize=12)
    ax.text(-9, 3 * (ndata // 4), r"$\xi_-^{ij}(\theta)$", fontsize=12)

    fig.savefig(plot_path, dpi=300)
    plt.close(fig)


def main(covfile, matrix_path, gaussian_path, plot_path):
    cov_g, cov_ng, ndata = get_cov(covfile)
    print(f"Dimension of cov: {ndata}x{ndata}")

    cov = cov_g + cov_ng

    eigenvalues = np.linalg.eigvalsh(cov)
    print(f"min+max eigenvalues cov: {eigenvalues.min():e}, {eigenvalues.max():e}")
    if eigenvalues.min() <= 0.0:
        sys.exit("non-positive eigenvalue encountered! Covariance invalid!")

    np.savetxt(matrix_path, cov)
    np.savetxt(gaussian_path, cov_g)
    plot_correlation(cov, ndata, plot_path)


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821 - injected by snakemake at runtime
    except NameError:
        if len(sys.argv) != 3:
            print("Usage: python cosmocov_process.py <input_file> <output_stub>")
            sys.exit(1)
        stub = sys.argv[2]
        main(sys.argv[1], f"{stub}.txt", f"{stub}_g.txt", f"{stub}_plot.pdf")
    else:
        main(
            snakemake.input[0],  # noqa: F821
            snakemake.output.matrix,  # noqa: F821
            snakemake.output.gaussian,  # noqa: F821
            snakemake.output.plot,  # noqa: F821
        )
