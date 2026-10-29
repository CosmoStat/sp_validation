"""Combine independent SNR-resolved simulation grids into a FITS product."""

import argparse
from pathlib import Path

import inputs
import numpy as np

from sp_validation import sim_m_snr


def main(directory, output):
    """Combine grid products with component-wise inverse-variance weights."""
    products = []
    for grid in range(1, 5):
        with np.load(directory / f"snr_grid_{grid}.npz") as f:
            products.append({k: f[k].copy() for k in f.files})
    combined = sim_m_snr.combine_grids(products)
    sim_m_snr.write_product(output, combined, inputs.PRODUCT, overwrite=True)
    error = np.sqrt(np.diag(combined["covariance"])).reshape(-1, 4)
    for i, row in enumerate(combined["value"]):
        print(
            f"{combined['edges'][i]:g}–{combined['edges'][i + 1]:g}: "
            f"m1={row[0] * 1000:.3f} ± {error[i, 0] * 1000:.3f}, "
            f"m2={row[2] * 1000:.3f} ± {error[i, 2] * 1000:.3f} (×10⁻³)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs.add_arguments(parser)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    inputs.configure(args)
    main(args.directory, args.output)
