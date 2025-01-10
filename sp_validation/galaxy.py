"""GALAXY.

:Name: galaxy.py

:Description: This script contains methods to deal with
    galaxy and star images.

:Author: Martin Kilbinger <martin.kilbinger@cea.fr>
         Axel Guinot

"""


import re
import numpy as np

from joblib import Parallel, delayed
from tqdm import tqdm

import regions
from astropy import units
from astropy import coordinates as coords
from astropy.wcs import WCS
from astropy.nddata import bitmask

from cs_util import cfis

from sp_validation import io


def T_to_fwhm(T):
    """T to fwhm.

    Transform from size T to FWHM.
    This interprets T as the RMS (``sigma'') of a Gaussian.

    Parameters
    ----------
    T : (array of) float
        input size(s)

    Returns
    -------
    fwhm : (array of) float
        output fwhm(s)
    """

    # MKDEBUG: Check this equation. Why is FWHM not quadratic in T???
    return T / 1.17741 * 2.355


def sigma_to_fwhm(sigma, pixel_size=1):
    """Sigma to fwhm.

    Transform from size sigma to FWHM.

    Parameters
    ----------
    sigma : (array of) float
        input size(s)
    pixel_size : float, optional, default=1
        pixel size in arcsec, set to 1 if no scaling
        required

    Returns
    -------
    fwhm : (array of) float
        output fwhm(s)
    """
    return sigma * 2.355 * pixel_size


def classification_galaxy_overlap_ra_dec(
    dd,
    ra_key='XWIN_WORLD',
    dec_key='YWIN_WORLD'
):
    """Classification Galaxy Overlap Ra Dec.

    Return mask corresponding to non-overlapping tile areas using
    simple cuts in RA and Dec.

    Parameters
    ----------
    dd : FITS.record
        input data
    ra_key : str, optional
        key name for right ascension column, default is 'XWIN_WORLD'
    dec_key : str, optional
        key name for declination column, default is 'YWIN_WORLD'

    Returns
    -------
    list of bool
        mask where `True` indicates galaxy to retain, and
        `False` being in overlapping region, to remove

    """
    # Unique set of tile IDs in data
    tile_ID_list = set(dd['TILE_ID'])

    # Transform to string format
    tile_ID_str_list = []
    for ID in tile_ID_list:
        tile_ID_str_list.append(f'{ID:07.3f}')

    # Extract integer numbers from tile IDs
    nix = []
    niy = []
    for ID in tile_ID_str_list:
        x, y = cfis.get_tile_number(ID)
        nix.append(x)
        niy.append(y)

    # Get RA and Dec coordinates of tile centers
    ra_cen, dec_cen = cfis.get_tile_coord_from_nixy(nix, niy)

    # Create limits on Dec by adding/subtracting half of the tile size
    delta_dec = cfis.Cfis().size['tile'] / 2
    dec_upper = dec_cen + delta_dec
    dec_lower = dec_cen - delta_dec

    # Create mask for Dec coordinates

    # Initialise global Dec mask
    mask_dec = np.full(len(dd), True)

    # Loop over tiles and mask objects outside the Dec limits
    for idx, tile_ID in enumerate(tile_ID_list):

        # Get indices of galaxies on this tile ID
        idx_ID = (dd['TILE_ID'] == tile_ID)

        # Set mask for this tile ID
        mask_dec_ID = (
            (dd[idx_ID][dec_key] < dec_upper[idx].value)
            & (dd[idx_ID][dec_key] >= dec_lower[idx].value)
        )

        # Apply to global mask
        mask_dec[idx_ID] = mask_dec_ID

    # Create limits on RA by finding half-point to neigbouring tiles
    ra_upper_list = []
    ra_lower_list = []

    for idx, tile_ID in enumerate(tile_ID_str_list):

        # Find tile ID towards increasing RA
        ID_upper = f'{int(nix[idx]) + 1:03d}.{int(niy[idx]):03d}'
        if ID_upper in tile_ID_str_list:
            # If found: compute halfway RA
            idx_upper = tile_ID_str_list.index(ID_upper)
            ra_upper = (ra_cen[idx] + ra_cen[idx_upper]) / 2
        else:
            # If not: no cut desired, set to large value
            ra_upper = 370 * units.deg

        # Add to list of cuts
        ra_upper_list.append(ra_upper)

        # Repeat towards decreasing RA
        ID_lower = f'{int(nix[idx]) - 1:03d}.{int(niy[idx]):03d}'
        if ID_lower in tile_ID_str_list:
            idx_lower = tile_ID_str_list.index(ID_lower)
            ra_lower = (ra_cen[idx] + ra_cen[idx_lower]) / 2
        else:
            ra_lower = -1 * units.deg

        ra_lower_list.append(ra_lower)

    # See above for Dec: Repeat for RA
    mask_ra = np.full(len(dd), True)

    for idx, tile_ID in enumerate(tile_ID_list):
        idx_ID = (dd['TILE_ID'] == tile_ID)
        mask_ra_ID = (
            (dd[idx_ID][ra_key] < ra_upper_list[idx].value)
            & (dd[idx_ID][ra_key] >= ra_lower_list[idx].value)
        )
        mask_ra[idx_ID] = mask_ra_ID

    return mask_dec & mask_ra


def classification_galaxy_base(
    dd,
    cut_overlap,
    gal_mag_bright=20,
    gal_mag_faint=26,
    flags_keep=None,
    n_epoch_min=1,
    do_spread_model=True
):
    """Classification Galaxy Base.

    Return mask corresponding to basic classification for galaxies.

    """
    if do_spread_model:
        # spread model class, add two times the uncertainty to be conservative
        sm_classif = dd['SPREAD_MODEL'] + 2 * dd['SPREADERR_MODEL']
        cut_sm = sm_classif > 0.0035

        cut_sm_all = (
            cut_sm
            & (dd['SPREAD_MODEL'] > 0)
            & (dd['SPREAD_MODEL'] < 0.03)
        )
    else:
        # Do not use spread model
        cut_sm_all = True

    # SExtractor flags
    # Keep some flags if specified
    if flags_keep:

        # Check whether flags are powers of 2
        if not all([bin(flag).count('1') == 1 for flag in flags_keep]):
            raise ValueError('Flag values in "flags_keep" not powers of 2')

        cut_flags = bitmask.bitfield_to_boolean_mask(
            dd['FLAGS'],
            good_mask_value=True,
            ignore_flags=flags_keep,
            dtype=bool,
        )
    else:
        cut_flags = bitmask.bitfield_to_boolean_mask(
            dd['FLAGS'],
            good_mask_value=True,
            dtype=bool,
        )

    cut_common = (
        cut_overlap
        & cut_flags
        & cut_sm_all
        & (dd['MAG_AUTO'] <= gal_mag_faint)
        & (dd['MAG_AUTO'] >= gal_mag_bright)
        & (dd['IMAFLAGS_ISO'] == 0)
        & (dd['N_EPOCH'] >= n_epoch_min)
    )

    return cut_common


def classification_galaxy_ngmix(
    dd,
    cut_common,
    stats_file=None,
    verbose=False,
):
    """Classification Galaxy Ngmx.

    Return mask corresponding to ngmix classification of galaxies
    """
    m_gal_ngmix = (
        cut_common
        & (dd['NGMIX_MCAL_FLAGS'] == 0)
        & (dd['NGMIX_ELL_PSFo_NOSHEAR'][:, 0] != -10)
        & (dd['NGMIX_MOM_FAIL'] == 0)
    )

    n_gal_ngmix = len(np.where(m_gal_ngmix)[0])
    n_tot = len(dd)

    if stats_file:
        io.print_ratio(
            'ngmix: Objects selected as galaxies',
            n_gal_ngmix,
            n_tot,
            stats_file,
            verbose=verbose)

    return m_gal_ngmix


def classification_galaxy_galsim(dd, cut_common, stats_file, verbose=False):
    """Classification Galaxy Galsim.

    Return mask corresponding to galsim classification of galaxies

    """
    m_gal_galsim = (
        cut_common
        & (dd['GALSIM_PSF_ELL_ORIGINAL_PSF'][:, 0] != -10)
    )

    n_gal_galsim = len(np.where(m_gal_galsim)[0])
    n_tot = len(dd)

    io.print_ratio(
        'galsim: Objects selected as galaxies',
        n_gal_galsim,
        n_tot,
        stats_file,
        verbose=verbose)

    return m_gal_galsim


def mask_overlap(ra, dec, tile_id_in, region_file_path, n_jobs=-1):
    """Mask Overlap.

    ...

    """

    def get_tile_wcs_new(xxx, yyy):
        """Get tile WCS.

        Create an astropy.wcs.WCS object from the name of the tile.

        Parameters
        ----------
        xxx : int
            First 3 numbers in the tile name.
        yyy : int
            Last 3 numbers in the tile name.

        Returns
        -------
        astropy.wcs.WCS
            WCS for the tile.

        """
        ra, dec = cfis.get_tile_coord_from_nixy(xxx, yyy)

        w = WCS(naxis=2)
        w.wcs.crval = np.array([ra.deg, dec.deg])
        w.wcs.crpix = np.array([5000, 5000])
        w.wcs.cd = np.array(
            [[0.187 / 3600, 0], [0, 0.187 / 3600]]
        )
        w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        w.wcs.cunit = ['deg', 'deg']
        w._naxis = [10000, 10000]

        return w

    def get_tile_wcs(xxx, yyy):
        dec = float(yyy) / 2 - 90
        ra = float(xxx) / 2 / np.cos(np.deg2rad(dec))

        new_wcs = WCS(naxis=2)
        new_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        new_wcs.wcs.cunit = ['deg     ', 'deg     ']
        new_wcs.wcs.crpix = [5.000000000000E+03, 5.000000000000E+03]
        new_wcs.wcs.crval = [ra, dec]
        new_wcs.wcs.cd = [
            [-5.160234650248E-05, 0.],
            [0., 5.160234650248E-05]
        ]

        return new_wcs

    def runner(r, all_tiles_id, all_tiles_ra, all_tiles_dec):
        xxx, yyy = re.findall(r'\d+', re.split(r'\s', r.meta['text'])[1])
        idx = np.where(all_tiles_id == float(xxx) + float(yyy) / 1000)
        tile_points = coords.SkyCoord(
            all_tiles_ra[idx],
            all_tiles_dec[idx],
            unit='deg',
        )
        m_cont = r.contains(tile_points, get_tile_wcs(xxx, yyy))
        m_not_cont = np.invert(m_cont)

        return m_not_cont, idx

    tile_id = np.copy(tile_id_in)
    tile_ra = np.copy(ra)
    tile_dec = np.copy(dec)

    all_regions = regions.Regions.read(region_file_path)

    res = Parallel(n_jobs=n_jobs, backend='loky')(delayed(runner)(
        r,
        tile_id,
        tile_ra,
        tile_dec
    ) for r in tqdm(all_regions, total=len(all_regions)))

    m_over = np.ones(len(tile_id), dtype=bool)
    for m_not_cont, idx in res:
        m_over[idx] = m_not_cont

    return m_over
