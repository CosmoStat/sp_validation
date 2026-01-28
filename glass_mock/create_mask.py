import os
import argparse

import numpy as np
import healpy as hp
from astropy.io import fits

def get_parser():
    """
    Creates the parser
    """
    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description='''
        Creates a UNIONS simulations using GLASS
        ''',
        fromfile_prefix_chars='@'
    )

    parser.add_argument("-n", "--nside", help="Nside for the simulation. Nside=Lmax", type=int, default=32)
    parser.add_argument("-p", "--path", help="Path to the galaxy catalog used to generate the mask", type=str, default='./')
    parser.add_argument("-o", "--output", help="Output path for the mask", type=str, default='./')
    parser.add_argument("--ra_col", help="Column name for the right ascension", type=str, default='RA')
    parser.add_argument("--dec_col", help="Column name for the declination", type=str, default='DEC')
    
    return parser

def create_mask(nside, path, output, ra_col, dec_col):
    """
    Create a mask using the galaxy catalog
    """
    # Load the catalog
    cat_gal = fits.getdata(path)

    # Create a mask using n_gal map
    ra = cat_gal[ra_col]
    dec = cat_gal[dec_col]

    theta = (90. - dec) * np.pi / 180.
    phi = ra * np.pi / 180.
    pix = hp.ang2pix(nside, theta, phi)

    unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
    n_gal = np.zeros(hp.nside2npix(nside))
    n_gal[unique_pix] = np.bincount(idx_rep)
    mask = n_gal != 0
    

    # Save the mask
    hp.write_map(output, mask, dtype=np.float32, overwrite=True)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print('------------------------------------------------------------------')
    print("Creating the mask with resolution nside={}".format(args.nside))
    print("Using the catalog in path: {}".format(args.path))
    print("Output path: {}".format(args.output))
    print('------------------------------------------------------------------')

    create_mask(args.nside, args.path, args.output, args.ra_col, args.dec_col)