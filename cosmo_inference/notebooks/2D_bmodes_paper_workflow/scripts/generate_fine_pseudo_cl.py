"""Generate finely-binned pseudo-Cls for COSEBIS comparison.

Uses linear binning with configurable ell_step for accurate C_ℓ → COSEBIS conversion.
Designed to be called by Snakemake or standalone.
"""

import os
import sys

from astropy.io import fits
import numpy as np

from sp_validation.cosmo_val import CosmologyValidation


def generate_pseudo_cl(
    version: str,
    ell_step: int,
    output_cl: str,
    output_cov: str,
    cat_config: str,
    nside: int = 1024,
    npatch: int = 1,
):
    """Generate pseudo-Cl and covariance with linear binning.

    Parameters
    ----------
    version : str
        Catalog version (e.g., "SP_v1.4.6_leak_corr")
    ell_step : int
        Width of ell bins (1 for single-ell bins)
    output_cl : str
        Output path for pseudo-Cl FITS file
    output_cov : str
        Output path for covariance FITS file
    cat_config : str
        Path to catalog configuration YAML
    nside : int
        HEALPix nside for map-based estimation
    npatch : int
        Number of jackknife patches
    """
    output_dir = os.path.dirname(output_cl)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating pseudo-Cl for {version} with ell_step={ell_step}")
    print(f"{'='*60}\n")

    # Create CosmologyValidation with fine ell binning
    cv = CosmologyValidation(
        versions=[version],
        catalog_config=cat_config,
        output_dir=output_dir,
        binning="linear",
        ell_step=ell_step,
        nside=nside,
        cell_method="map",
        nrandom_cell=10,
        # Standard params (not used for pseudo-Cl but required)
        npatch=npatch,
        theta_min=1.0,
        theta_max=250.0,
        nbins=20,
    )

    # Calculate pseudo-Cls
    cv.calculate_pseudo_cl()

    # Rename to final output
    src_cl = os.path.join(output_dir, f"pseudo_cl_{version}.fits")
    if os.path.exists(src_cl) and src_cl != output_cl:
        with fits.open(src_cl) as hdul:
            data = hdul["PSEUDO_CELL"].data
            n_ell = len(data["ELL"])
            print(f"Generated pseudo-Cl with {n_ell} ell bins")
            print(f"ell range: [{data['ELL'].min():.1f}, {data['ELL'].max():.1f}]")
        os.rename(src_cl, output_cl)
        print(f"Saved to: {output_cl}")

    # Calculate covariance
    print(f"\nCalculating covariance...")
    cv.calculate_pseudo_cl_eb_cov()

    src_cov = os.path.join(output_dir, f"pseudo_cl_cov_{version}.fits")
    if os.path.exists(src_cov) and src_cov != output_cov:
        os.rename(src_cov, output_cov)
        print(f"Saved covariance to: {output_cov}")


if __name__ == "__main__":
    # Check if running under Snakemake
    try:
        snakemake  # noqa: F821
    except NameError:
        # Standalone usage
        print("Usage: Run via Snakemake rule 'fine_pseudo_cl'")
        sys.exit(1)

    generate_pseudo_cl(
        version=snakemake.params.version,  # noqa: F821
        ell_step=snakemake.params.ell_step,  # noqa: F821
        output_cl=snakemake.output.pseudo_cl,  # noqa: F821
        output_cov=snakemake.output.pseudo_cl_cov,  # noqa: F821
        cat_config=snakemake.params.cat_config,  # noqa: F821
        nside=snakemake.params.nside,  # noqa: F821
        npatch=snakemake.params.npatch,  # noqa: F821
    )
