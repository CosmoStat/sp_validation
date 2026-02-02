"""Generate fine-binned xi± for GLASS mock using TreeCorr.

Standalone script for computing correlation functions from GLASS mocks.
"""

import os
import sys

import numpy as np
import treecorr
from astropy.io import fits


def compute_xi(catalog_path: str, output_path: str, nbins: int = 1000,
               min_sep: float = 1.0, max_sep: float = 250.0):
    """Compute xi± from mock catalog using TreeCorr."""
    print(f"Loading catalog from {catalog_path}...")
    with fits.open(catalog_path) as hdul:
        data = hdul[1].data
        ra = data["RA"]
        dec = data["Dec"]
        e1 = data["e1"]
        e2 = data["e2"]
        w = data["w"]

    print(f"  {len(ra):,} galaxies")

    # Create TreeCorr catalog
    cat = treecorr.Catalog(
        ra=ra, dec=dec, g1=e1, g2=e2, w=w,
        ra_units='degrees', dec_units='degrees'
    )

    print(f"Computing xi± with {nbins} bins...")
    print(f"  theta range: [{min_sep}, {max_sep}] arcmin")

    gg = treecorr.GGCorrelation(
        min_sep=min_sep, max_sep=max_sep, nbins=nbins,
        sep_units='arcmin', bin_type='Log'
    )

    gg.process(cat, num_threads=12)

    print(f"  Done. npairs range: [{gg.npairs.min():.0f}, {gg.npairs.max():.0f}]")
    print(f"  xi+(theta~10'): {gg.xip[np.argmin(np.abs(gg.meanr - 10))]:.3e}")
    print(f"  xi-(theta~10'): {gg.xim[np.argmin(np.abs(gg.meanr - 10))]:.3e}")

    # Save to FITS
    print(f"Saving to {output_path}...")

    col_rnom = fits.Column(name='r_nom', format='D', array=gg.rnom)
    col_meanr = fits.Column(name='meanr', format='D', array=gg.meanr)
    col_meanlogr = fits.Column(name='meanlogr', format='D', array=gg.meanlogr)
    col_xip = fits.Column(name='xip', format='D', array=gg.xip)
    col_xim = fits.Column(name='xim', format='D', array=gg.xim)
    col_xip_im = fits.Column(name='xip_im', format='D', array=gg.xip_im)
    col_xim_im = fits.Column(name='xim_im', format='D', array=gg.xim_im)
    col_sigma_xip = fits.Column(name='sigma_xip', format='D', array=np.sqrt(gg.varxip))
    col_sigma_xim = fits.Column(name='sigma_xim', format='D', array=np.sqrt(gg.varxim))
    col_weight = fits.Column(name='weight', format='D', array=gg.weight)
    col_npairs = fits.Column(name='npairs', format='D', array=gg.npairs)

    cols = fits.ColDefs([col_rnom, col_meanr, col_meanlogr, col_xip, col_xim,
                         col_xip_im, col_xim_im, col_sigma_xip, col_sigma_xim,
                         col_weight, col_npairs])

    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.header['MIN_SEP'] = min_sep
    hdu.header['MAX_SEP'] = max_sep
    hdu.header['NBINS'] = nbins
    hdu.header['BIN_TYPE'] = 'Log'

    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    hdul.writeto(output_path, overwrite=True)

    print("  Done")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate xi± for GLASS mock")
    parser.add_argument("--mock-id", type=str, default="00001", help="Mock ID (5 digits)")
    parser.add_argument("--nbins", type=int, default=1000, help="Number of theta bins")
    parser.add_argument("--min-sep", type=float, default=1.0, help="Min separation (arcmin)")
    parser.add_argument("--max-sep", type=float, default=250.0, help="Max separation (arcmin)")
    parser.add_argument("--output", type=str, help="Output FITS path")
    args = parser.parse_args()

    mock_dir = "/n09data/guerrini/glass_mock_v1.4.6/results"
    catalog_path = f"{mock_dir}/unions_glass_sim_{args.mock_id}_4096.fits"

    if args.output is None:
        output_path = f"results/glass_mock/xi_glass_mock_{args.mock_id}_nbins={args.nbins}.fits"
    else:
        output_path = args.output

    compute_xi(catalog_path, output_path, args.nbins, args.min_sep, args.max_sep)

    print(f"\nSummary:")
    print(f"  Mock: {args.mock_id}")
    print(f"  Output: {output_path}")
    print(f"  Bins: {args.nbins}")
    print(f"  theta range: [{args.min_sep}, {args.max_sep}] arcmin")


if __name__ == "__main__":
    main()
