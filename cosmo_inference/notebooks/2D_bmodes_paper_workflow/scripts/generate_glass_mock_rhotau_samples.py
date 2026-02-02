#!/usr/bin/env python
"""
Generate zero-mean rho/tau realizations for GLASS mocks by sampling from
measured covariance matrices.

This script samples rho and tau statistics from the measured covariances
and writes them to FITS files matching the structure of real rho/tau stats.
For GLASS mocks without PSF leakage, rho/tau should be consistent with zero,
so we sample around a mean of zero.
"""

import argparse
import numpy as np
from pathlib import Path
from astropy.io import fits
from multiprocessing import Pool


def load_reference_fits(filename):
    """Load reference FITS file to get structure and theta values."""
    with fits.open(filename) as hdul:
        data = hdul[1].data.copy()
        header = hdul[1].header.copy()
    return data, header


def create_rho_fits(sampled_values, theta, header=None):
    """Create rho statistics FITS file from sampled values."""
    nbins = len(theta)

    # sampled_values should be shape (120,) representing:
    # [rho_0_p (20 bins), rho_1_p (20 bins), ..., rho_5_p (20 bins)]

    # Reshape to (6, 20) for easier indexing
    rho_p_values = sampled_values.reshape(6, nbins)

    # Create columns: theta, then each rho stat
    columns = []
    columns.append(fits.Column(name='theta', format='D', array=theta))

    # For each rho statistic
    stat_names = ['rho_0', 'rho_1', 'rho_2', 'rho_3', 'rho_4', 'rho_5']
    for stat_idx, stat_name in enumerate(stat_names):
        # Positive values
        columns.append(fits.Column(
            name=f'{stat_name}_p',
            format='D',
            array=rho_p_values[stat_idx]
        ))
        # Variance (always positive)
        columns.append(fits.Column(
            name=f'var{stat_name}_p',
            format='D',
            array=np.abs(rho_p_values[stat_idx]) * 0.1  # Dummy variance
        ))
        # Negative values (set to small values for zero-mean case)
        columns.append(fits.Column(
            name=f'{stat_name}_m',
            format='D',
            array=np.zeros(nbins)
        ))
        # Variance for negative
        columns.append(fits.Column(
            name=f'var{stat_name}_m',
            format='D',
            array=np.ones(nbins) * 1e-15
        ))

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


def generate_samples_for_mock(mock_id, cov_rho, cov_tau, theta,
                             ref_rho_header, ref_tau_header, output_dir):
    """Generate sampled rho/tau for a single mock."""

    # Deterministic seeding based on mock_id
    seed = int(mock_id)
    np.random.seed(seed)

    # Sample rho statistics from N(0, Cov_rho)
    rho_samples = np.random.multivariate_normal(
        mean=np.zeros(cov_rho.shape[0]),
        cov=cov_rho
    )

    # Sample tau statistics from N(0, Cov_tau)
    tau_samples = np.random.multivariate_normal(
        mean=np.zeros(cov_tau.shape[0]),
        cov=cov_tau
    )

    # Create FITS files
    rho_hdu = create_rho_fits(rho_samples, theta, ref_rho_header)
    tau_hdu = create_tau_fits(tau_samples, theta, ref_tau_header)

    # Create output directory
    output_path = Path(output_dir) / f"{mock_id:05d}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Write FITS files
    rho_fits_path = output_path / "rho_stats_sampled.fits"
    tau_fits_path = output_path / "tau_stats_sampled.fits"

    # Write with PRIMARY HDU
    rho_hdul = fits.HDUList([fits.PrimaryHDU(), rho_hdu])
    rho_hdul.writeto(rho_fits_path, overwrite=True)

    tau_hdul = fits.HDUList([fits.PrimaryHDU(), tau_hdu])
    tau_hdul.writeto(tau_fits_path, overwrite=True)

    return f"Generated samples for mock {mock_id:05d}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate zero-mean rho/tau samples for GLASS mocks"
    )
    parser.add_argument(
        "--cov-rho",
        type=str,
        required=True,
        help="Path to rho covariance matrix (.npy file)"
    )
    parser.add_argument(
        "--cov-tau",
        type=str,
        required=True,
        help="Path to tau covariance matrix (.npy file)"
    )
    parser.add_argument(
        "--ref-rho",
        type=str,
        required=True,
        help="Reference rho FITS file for structure"
    )
    parser.add_argument(
        "--ref-tau",
        type=str,
        required=True,
        help="Reference tau FITS file for structure"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for sampled FITS files"
    )
    parser.add_argument(
        "--mock-ids",
        type=str,
        default="00001-00350",
        help="Range of mock IDs (format: 00001-00350)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads for parallel processing"
    )

    args = parser.parse_args()

    # Load covariance matrices
    print("Loading covariance matrices...")
    cov_rho = np.load(args.cov_rho)
    cov_tau = np.load(args.cov_tau)
    print(f"  cov_rho shape: {cov_rho.shape}")
    print(f"  cov_tau shape: {cov_tau.shape}")

    # Load reference FITS files
    print("Loading reference FITS files...")
    ref_rho_data, ref_rho_header = load_reference_fits(args.ref_rho)
    ref_tau_data, ref_tau_header = load_reference_fits(args.ref_tau)
    theta = ref_rho_data['theta']
    print(f"  theta range: {theta.min():.3f} - {theta.max():.3f} arcmin")
    print(f"  nbins: {len(theta)}")

    # Parse mock ID range
    if "-" in args.mock_ids:
        start_id, end_id = args.mock_ids.split("-")
        mock_ids = range(int(start_id), int(end_id) + 1)
    else:
        mock_ids = [int(args.mock_ids)]

    print(f"Generating samples for {len(list(mock_ids))} mocks...")

    # Generate samples
    if args.threads > 1:
        with Pool(args.threads) as pool:
            results = []
            for mock_id in mock_ids:
                result = pool.apply_async(
                    generate_samples_for_mock,
                    (mock_id, cov_rho, cov_tau, theta,
                     ref_rho_header, ref_tau_header, args.output_dir)
                )
                results.append(result)

            for result in results:
                print(result.get())
    else:
        for mock_id in mock_ids:
            msg = generate_samples_for_mock(
                mock_id, cov_rho, cov_tau, theta,
                ref_rho_header, ref_tau_header, args.output_dir
            )
            print(msg)

    print("Done!")


if __name__ == "__main__":
    main()
