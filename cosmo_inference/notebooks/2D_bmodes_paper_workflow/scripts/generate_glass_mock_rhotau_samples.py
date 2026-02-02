#!/usr/bin/env python
"""Generate zero-mean tau realizations for GLASS mocks.

Samples tau statistics from the measured covariance and writes FITS files
matching the structure of real tau stats. For GLASS mocks without PSF
leakage, tau should be consistent with zero, so we sample around mean zero.

Note: Only tau is sampled; inference uses real rho data (rho statistics
measure PSF shape correlations which exist in real data but not in mocks).
"""

import argparse
import numpy as np
from pathlib import Path
from astropy.io import fits


def load_reference_fits(filename):
    """Load reference FITS file to get structure and theta values."""
    with fits.open(filename) as hdul:
        data = hdul[1].data.copy()
        header = hdul[1].header.copy()
    return data, header


def create_tau_fits(sampled_values, theta, header=None):
    """Create tau statistics FITS file from sampled values."""
    nbins = len(theta)

    # sampled_values should be shape (60,) representing:
    # [tau_0_p (20 bins), tau_2_p (20 bins), tau_5_p (20 bins)]
    # (only the three tau statistics used in inference)

    # Reshape to (3, 20)
    tau_values = sampled_values.reshape(3, nbins)
    tau_0_p = tau_values[0]
    tau_2_p = tau_values[1]
    tau_5_p = tau_values[2]

    # Create columns
    columns = []
    columns.append(fits.Column(name='theta', format='D', array=theta))

    # tau_0_p
    columns.append(fits.Column(name='tau_0_p', format='D', array=tau_0_p))
    columns.append(fits.Column(name='vartau_0_p', format='D',
                              array=np.abs(tau_0_p) * 0.1))
    # tau_0_m
    columns.append(fits.Column(name='tau_0_m', format='D',
                              array=np.zeros(nbins)))
    columns.append(fits.Column(name='vartau_0_m', format='D',
                              array=np.ones(nbins) * 1e-15))

    # tau_2_p
    columns.append(fits.Column(name='tau_2_p', format='D', array=tau_2_p))
    columns.append(fits.Column(name='vartau_2_p', format='D',
                              array=np.abs(tau_2_p) * 0.1))
    # tau_2_m
    columns.append(fits.Column(name='tau_2_m', format='D',
                              array=np.zeros(nbins)))
    columns.append(fits.Column(name='vartau_2_m', format='D',
                              array=np.ones(nbins) * 1e-15))

    # tau_5_p
    columns.append(fits.Column(name='tau_5_p', format='D', array=tau_5_p))
    columns.append(fits.Column(name='vartau_5_p', format='D',
                              array=np.abs(tau_5_p) * 0.1))
    # tau_5_m
    columns.append(fits.Column(name='tau_5_m', format='D',
                              array=np.zeros(nbins)))
    columns.append(fits.Column(name='vartau_5_m', format='D',
                              array=np.ones(nbins) * 1e-15))

    # Create HDU
    coldefs = fits.ColDefs(columns)
    hdu = fits.BinTableHDU.from_columns(coldefs)

    # Copy header if provided
    if header is not None:
        for key, value in header.items():
            if key not in hdu.header:
                try:
                    hdu.header[key] = value
                except (ValueError, KeyError):
                    pass  # Skip invalid FITS header keys

    return hdu


def generate_samples_for_mock(mock_id, cov_tau, theta, ref_tau_header, output_dir):
    """Generate sampled tau statistics for a single mock.

    Only tau is needed; inference_prep_glass_mock uses real rho data.
    """
    # Deterministic seeding based on mock_id
    rng = np.random.default_rng(int(mock_id))

    # Sample tau statistics from N(0, Cov_tau)
    tau_samples = rng.multivariate_normal(
        mean=np.zeros(cov_tau.shape[0]),
        cov=cov_tau
    )

    # Create FITS file
    tau_hdu = create_tau_fits(tau_samples, theta, ref_tau_header)

    # Create output directory and write
    output_path = Path(output_dir) / f"{mock_id:05d}"
    output_path.mkdir(parents=True, exist_ok=True)
    tau_fits_path = output_path / "tau_stats_sampled.fits"

    tau_hdul = fits.HDUList([fits.PrimaryHDU(), tau_hdu])
    tau_hdul.writeto(tau_fits_path, overwrite=True)

    return f"Generated tau samples for mock {mock_id:05d}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate zero-mean tau samples for GLASS mocks"
    )
    parser.add_argument("--cov-tau", type=str, required=True,
                        help="Path to tau covariance matrix (.npy file)")
    parser.add_argument("--ref-tau", type=str, required=True,
                        help="Reference tau FITS file for structure")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for sampled FITS files")
    parser.add_argument("--mock-ids", type=str, default="00001-00350",
                        help="Range of mock IDs (format: 00001-00350)")
    args = parser.parse_args()

    # Load tau covariance and reference
    print("Loading tau covariance...")
    cov_tau = np.load(args.cov_tau)
    print(f"  cov_tau shape: {cov_tau.shape}")

    print("Loading reference FITS...")
    ref_tau_data, ref_tau_header = load_reference_fits(args.ref_tau)
    theta = ref_tau_data['theta']
    print(f"  theta range: {theta.min():.3f} - {theta.max():.3f} arcmin, nbins: {len(theta)}")

    # Parse mock ID range
    mock_ids = (range(int(args.mock_ids.split("-")[0]), int(args.mock_ids.split("-")[1]) + 1)
                if "-" in args.mock_ids else [int(args.mock_ids)])

    print(f"Generating tau samples for {len(list(mock_ids))} mocks...")
    for mock_id in mock_ids:
        msg = generate_samples_for_mock(mock_id, cov_tau, theta, ref_tau_header, args.output_dir)
        print(msg)
    print("Done!")


if __name__ == "__main__":
    main()
