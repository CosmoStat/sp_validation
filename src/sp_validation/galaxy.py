"""GALAXY.

:Name: galaxy.py

:Description: This script contains methods to deal with
    galaxy and star images.

:Author: Martin Kilbinger <martin.kilbinger@cea.fr>
         Axel Guinot

"""

import re
import warnings

import numpy as np
import regions
from astropy import coordinates as coords
from astropy import units
from astropy.nddata import bitmask
from astropy.wcs import WCS
from cs_util import cfis
from cs_util.size import T_to_fwhm, sigma_to_fwhm  # noqa: F401  re-exported;
from joblib import Parallel, delayed
from tqdm import tqdm

# the previous local T_to_fwhm (T / 1.17741 * 2.355) treated the area
# T = 2 sigma^2 as if it were sigma; the cs_util version carries the
# required square root: FWHM = 2.35482 sqrt(T / 2)
from sp_validation import io

#: All mask columns written by ShapePipe v2 (bool, ``True`` = masked).
#: They replace the single IMAFLAGS_ISO bitmask of ShapePipe v1.
#:
#: Reason bits making up the r-band default bitmask: n1/n2 star halos
#: (which of the two is faint and which bright is unconfirmed for the
#: Aug-2026 products), n4 stars, n8 manual galaxy mask, n64 (an
#: undocumented reason bit), n1024 MaxiMask.
#:
#: Per-band coverage flags: n16 (u), n32 (g), n128 (i), n256 (z). There is
#: no r coverage flag because the catalogue is r-selected. n2048 is ``True``
#: where Pan-STARRS z2 coverage is absent.
MASK_COLUMNS = (
    "MASK_n1",
    "MASK_n2",
    "MASK_n4",
    "MASK_n8",
    "MASK_n16",
    "MASK_n32",
    "MASK_n64",
    "MASK_n128",
    "MASK_n256",
    "MASK_n1024",
    "MASK_n2048",
)

#: Mask columns OR'd together for the default galaxy selection. This set is
#: exactly the reason bits of the ShapePipe r-band default bitmask: their OR
#: reproduces ``mask_r``, the v1 r-band mask, on the P3 region. Deliberately
#: not a blanket OR over MASK_COLUMNS: the per-band coverage flags
#: (n16, n32, n128, n256) and n2048 would mask essentially the whole
#: catalogue.
DEFAULT_MASK_COLUMNS = (
    "MASK_n4",
    "MASK_n1",
    "MASK_n2",
    "MASK_n8",
    "MASK_n64",
    "MASK_n1024",
)


def _column_names(dd):
    """Return the column names of a structured array or mapping."""
    dtype = getattr(dd, "dtype", None)
    if dtype is not None and dtype.names is not None:
        return tuple(dtype.names)
    return tuple(dd.keys())


def mask_cut(dd, mask_columns=None):
    """Mask Cut.

    Return a boolean mask that is ``True`` for objects *not* flagged by any
    of the requested ShapePipe mask columns.

    Parameters
    ----------
    dd : numpy.ndarray or dict
        input catalogue
    mask_columns : list of str, optional
        mask columns to OR together; default is ``DEFAULT_MASK_COLUMNS``

    Returns
    -------
    numpy.ndarray
        boolean mask, ``True`` = keep

    Raises
    ------
    KeyError
        if any requested mask column is absent from the catalogue

    """
    columns = list(DEFAULT_MASK_COLUMNS if mask_columns is None else mask_columns)
    if not columns:
        # No masking requested (e.g. the image simulations, which run no
        # imaging-flag masking stage and carry no mask columns).
        return np.ones(len(dd[_column_names(dd)[0]]), dtype=bool)

    available = _column_names(dd)
    missing = [col for col in columns if col not in available]
    if missing:
        raise KeyError(
            f"Mask column(s) {missing} not found in catalogue."
            + " ShapePipe v2 catalogues carry the boolean columns"
            + f" {list(MASK_COLUMNS)}; ShapePipe v1 catalogues carry"
            + " IMAFLAGS_ISO instead and are not supported."
            + f" Available columns: {sorted(available)}"
        )

    masked = np.zeros(len(dd[columns[0]]), dtype=bool)
    n_undefined = 0
    for col in columns:
        values = np.asarray(dd[col])
        if values.dtype == bool:
            flagged = values
        else:
            # ShapePipe's writer emits the MASK_n* columns as float64 {0, 1}
            # rather than bool (being fixed upstream), so decide on the value
            # rather than on truthiness: a bare astype(bool) would silently
            # read NaN as True, i.e. masked. Threshold at 0.5 so an integer,
            # a float and a bool column all behave identically.
            values = values.astype(float)
            undefined = np.isnan(values)
            n_undefined += int(undefined.sum())
            flagged = np.where(undefined, False, values > 0.5)
        masked |= flagged

    if n_undefined:
        # NaN means the masking stage recorded no verdict for this object.
        # Treat it as un-masked (keep the object) so an incomplete mask
        # column cannot silently delete sky, but say so loudly: a nonzero
        # count here means the input product is defective.
        warnings.warn(
            f"{n_undefined} NaN value(s) in mask column(s) {columns};"
            + " treated as not masked. The mask columns of a complete"
            + " ShapePipe product hold only 0 and 1.",
            RuntimeWarning,
            stacklevel=2,
        )

    return ~masked


def classification_galaxy_overlap_ra_dec(dd, ra_key="XWIN_WORLD", dec_key="YWIN_WORLD"):
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
    tile_ID_list = set(dd["TILE_ID"])

    # Transform to string format
    tile_ID_str_list = []
    for ID in tile_ID_list:
        tile_ID_str_list.append(f"{ID:07.3f}")

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
    delta_dec = cfis.Cfis().size["tile"] / 2
    dec_upper = dec_cen + delta_dec
    dec_lower = dec_cen - delta_dec

    # Create mask for Dec coordinates

    # Initialise global Dec mask
    mask_dec = np.full(len(dd), True)

    # Loop over tiles and mask objects outside the Dec limits
    for idx, tile_ID in enumerate(tile_ID_list):
        # Get indices of galaxies on this tile ID
        idx_ID = dd["TILE_ID"] == tile_ID

        # Set mask for this tile ID
        mask_dec_ID = (dd[idx_ID][dec_key] < dec_upper[idx].value) & (
            dd[idx_ID][dec_key] >= dec_lower[idx].value
        )

        # Apply to global mask
        mask_dec[idx_ID] = mask_dec_ID

    # Create limits on RA by finding half-point to neigbouring tiles
    ra_upper_list = []
    ra_lower_list = []

    for idx, tile_ID in enumerate(tile_ID_str_list):
        # Find tile ID towards increasing RA
        ID_upper = f"{int(nix[idx]) + 1:03d}.{int(niy[idx]):03d}"
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
        ID_lower = f"{int(nix[idx]) - 1:03d}.{int(niy[idx]):03d}"
        if ID_lower in tile_ID_str_list:
            idx_lower = tile_ID_str_list.index(ID_lower)
            ra_lower = (ra_cen[idx] + ra_cen[idx_lower]) / 2
        else:
            ra_lower = -1 * units.deg

        ra_lower_list.append(ra_lower)

    # See above for Dec: Repeat for RA
    mask_ra = np.full(len(dd), True)

    for idx, tile_ID in enumerate(tile_ID_list):
        idx_ID = dd["TILE_ID"] == tile_ID
        mask_ra_ID = (dd[idx_ID][ra_key] < ra_upper_list[idx].value) & (
            dd[idx_ID][ra_key] >= ra_lower_list[idx].value
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
    mask_columns=None,
):
    """Classification Galaxy Base.

    Return mask corresponding to basic classification for galaxies.

    Parameters
    ----------
    mask_columns : list of str, optional
        ShapePipe mask columns OR'd together to reject masked objects;
        default is ``DEFAULT_MASK_COLUMNS``

    """
    # SExtractor flags
    # Keep some flags if specified
    if flags_keep:
        # Check whether flags are powers of 2
        if not all([bin(flag).count("1") == 1 for flag in flags_keep]):
            raise ValueError('Flag values in "flags_keep" not powers of 2')

        cut_flags = bitmask.bitfield_to_boolean_mask(
            dd["FLAGS"],
            good_mask_value=True,
            ignore_flags=flags_keep,
            dtype=bool,
        )
    else:
        cut_flags = bitmask.bitfield_to_boolean_mask(
            dd["FLAGS"],
            good_mask_value=True,
            dtype=bool,
        )

    cut_common = (
        cut_overlap
        & cut_flags
        & (dd["MAG_AUTO"] <= gal_mag_faint)
        & (dd["MAG_AUTO"] >= gal_mag_bright)
        & mask_cut(dd, mask_columns)
        & (dd["N_EPOCH"] >= n_epoch_min)
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
    # NGMIX_N_EPOCH == 0 marks objects ngmix never fit: ShapePipe's make_cat
    # pre-fills every NGMIX_* column with sentinels (G1/G2 = -10, T/FLUX = 0)
    # and only overwrites them for objects present in the ngmix output, so a
    # never-fit object keeps NGMIX_MCAL_FLAGS == 0 and passes a flag-only cut.
    # In final_cat_smk-g7.hdf5 this is 18,983 / 1,851,100 objects (1.03%);
    # admitting them drags mean e1 to -0.096 (std 0.98) from +0.0001 (std
    # 0.24). The coadd N_EPOCH cut in classification_galaxy_base does not
    # catch them (18,750 of the 18,983 have N_EPOCH >= 1). Cut on N_EPOCH
    # explicitly rather than relying on the -10 sentinel comparison below,
    # which is an exact float equality against a value ShapePipe may change.
    m_gal_ngmix = (
        cut_common
        & (dd["NGMIX_N_EPOCH"] > 0)
        & (dd["NGMIX_MCAL_FLAGS"] == 0)
        & (dd["NGMIX_G1_PSF_ORIG_NOSHEAR"] != -10)
        & (dd["NGMIX_MCAL_TYPES_FAIL"] == 0)
    )

    n_gal_ngmix = len(np.where(m_gal_ngmix)[0])
    n_tot = len(dd)

    if stats_file:
        io.print_ratio(
            "ngmix: Objects selected as galaxies",
            n_gal_ngmix,
            n_tot,
            stats_file,
            verbose=verbose,
        )

    return m_gal_ngmix


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
        w.wcs.cd = np.array([[0.187 / 3600, 0], [0, 0.187 / 3600]])
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.cunit = ["deg", "deg"]
        w._naxis = [10000, 10000]

        return w

    def get_tile_wcs(xxx, yyy):
        dec = float(yyy) / 2 - 90
        ra = float(xxx) / 2 / np.cos(np.deg2rad(dec))

        new_wcs = WCS(naxis=2)
        new_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        new_wcs.wcs.cunit = ["deg     ", "deg     "]
        new_wcs.wcs.crpix = [5.000000000000e03, 5.000000000000e03]
        new_wcs.wcs.crval = [ra, dec]
        new_wcs.wcs.cd = [[-5.160234650248e-05, 0.0], [0.0, 5.160234650248e-05]]

        return new_wcs

    def runner(r, all_tiles_id, all_tiles_ra, all_tiles_dec):
        xxx, yyy = re.findall(r"\d+", re.split(r"\s", r.meta["text"])[1])
        idx = np.where(all_tiles_id == float(xxx) + float(yyy) / 1000)
        tile_points = coords.SkyCoord(
            all_tiles_ra[idx],
            all_tiles_dec[idx],
            unit="deg",
        )
        m_cont = r.contains(tile_points, get_tile_wcs(xxx, yyy))
        m_not_cont = np.invert(m_cont)

        return m_not_cont, idx

    tile_id = np.copy(tile_id_in)
    tile_ra = np.copy(ra)
    tile_dec = np.copy(dec)

    all_regions = regions.Regions.read(region_file_path)

    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(runner)(r, tile_id, tile_ra, tile_dec)
        for r in tqdm(all_regions, total=len(all_regions))
    )

    m_over = np.ones(len(tile_id), dtype=bool)
    for m_not_cont, idx in res:
        m_over[idx] = m_not_cont

    return m_over
