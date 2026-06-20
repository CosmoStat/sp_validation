# Name file: get_matched_catalogue.py

# @author: Antonin Corinaldi

# Python file to match UNIONS shape catalogue with a spectroscopic catalogue and save the matched catalogue with redshift information.


import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.coordinates import concatenate
from astropy import units as u


def read_fits_to_pandas(path):
    """
    Read a FITS catalogue and return a pandas DataFrame
    keeping only 1D columns (i.e. scalar quantities).

    Parameters
    ----------
    path : str
        Path to the FITS file.

    Returns
    -------
    df : pandas.DataFrame
        Cleaned catalogue.
    """
    table = Table.read(path)

    names = [name for name in table.colnames if len(table[name].shape) <= 1]
    table = table[names]

    return table.to_pandas()


def load_and_merge_catalogues(paths):
    """
    Load multiple FITS catalogues and merge them into a single DataFrame.

    Parameters
    ----------
    paths : list of str
        List of FITS file paths.

    Returns
    -------
    df : pandas.DataFrame
        Merged catalogue.
    """
    dfs = [read_fits_to_pandas(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


def load_unions_catalogue(path):
    """
    Load UNIONS FITS catalogue handling endian issues.

    Parameters
    ----------
    path : str
        Path to UNIONS FITS file.

    Returns
    -------
    df : pandas.DataFrame
        UNIONS catalogue.
    """
    with fits.open(path, memmap=True, ignore_missing_simple=True) as hdul:
        data = hdul[1].data

        df = pd.DataFrame(
            {
                col: data[col].byteswap().view(data[col].dtype.newbyteorder())
                for col in data.names
            }
        )

    return df


def cross_match_catalogues(
    cat1,
    cat2,
    ra1="RA",
    dec1="Dec",
    ra2="RA",
    dec2="DEC",
    z_col="Z",
    max_sep_deg=0.00028,
):
    """
    Cross-match two catalogues on the sky and attach redshift from cat2 to cat1.

    Parameters
    ----------
    cat1 : pandas.DataFrame
        First catalogue (e.g. photometric catalogue).
    cat2 : pandas.DataFrame
        Second catalogue (e.g. spectroscopic catalogue).
    ra1, dec1 : str
        RA/DEC column names for cat1.
    ra2, dec2 : str
        RA/DEC column names for cat2.
    z_col : str
        Redshift column name in cat2.
    max_sep_deg : float
        Maximum angular separation in degrees.

    Returns
    -------
    merged : pandas.DataFrame
        Matched catalogue with redshift added.
    """

    # Building SkyCoord objects
    coords1 = SkyCoord(ra=cat1[ra1].values * u.deg, dec=cat1[dec1].values * u.deg)

    coords2 = SkyCoord(ra=cat2[ra2].values * u.deg, dec=cat2[dec2].values * u.deg)

    # Nearest-neighbour match
    idx, d2d, _ = match_coordinates_sky(coords1, coords2)

    # Angular separation cut
    mask = d2d.deg < max_sep_deg

    # Selecting matched objects
    cat1_match = cat1.loc[mask].reset_index(drop=True)
    cat2_match = cat2.loc[idx[mask]].reset_index(drop=True)

    # Merging redshift
    merged = cat1_match.copy()
    merged[z_col] = cat2_match[z_col].values

    return merged


# Example of usage (change the path to use it):


# Loading UNIONS shape catalogue
unions = load_unions_catalogue(
    "/n17data/UNIONS/WL/v1.6.x/v1.6.9/unions_shapepipe_cut_struc_2024_v1.6.9.fits"
)

# Loading a generic spectroscopic catalogue and merging NGC and SGC parts if needed
spec_catalogue = load_and_merge_catalogues(
    [
        "/n17data/corinaldi/DESI/clustering_data/LRG/LRG_NGC_clustering.dat.fits",
        "/n17data/corinaldi/DESI/clustering_data/LRG/LRG_SGC_clustering.dat.fits",
    ]
)

# Cross-matching
matched = cross_match_catalogues(
    cat1=unions,
    cat2=spec_catalogue,
    ra1="RA",
    dec1="Dec",
    ra2="RA",
    dec2="DEC",
    z_col="Z",
    max_sep_deg=0.00028,
)

# Saving result
Table.from_pandas(matched).write(
    "/n17data/corinaldi/matched_catalogues/unions1.6.9/UNIONS1.6.9_lrg_cross.fits",
    format="fits",
    overwrite=True,
)
