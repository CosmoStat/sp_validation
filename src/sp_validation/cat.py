"""CAT.

:Name: cat.py

:Description: This script contains methods to deal with catalogues.

:Author: Martin Kilbinger

"""

import os
import re
import numpy as np
import getpass

import h5py
import tqdm

from datetime import datetime

from astropy.io import fits
from astropy import coordinates as coords
from astropy import units as u

from cs_util import cat

from sp_validation import util
from sp_validation import io
from sp_validation import basic
from sp_validation.survey import get_footprint
from sp_validation import __version__, __name__


def print_mean_ellipticity(
    dd,
    ell_col_name,
    ell_n_comp,
    n_tot,
    stats_file,
    invalid=-10,
    verbose=False,
):
    """Print Mean Ellipticity.

    Output mean ellipticity from a catalogue.

    Parameters
    ----------
    dd : dict
        galaxy catalog
    ell_col_name : array of string
        ellipticity column name(s)
    ell_n_comp : int
        dimension (= number of components) of ellipticity column.
        Should be 1 or 2.
    n_tot : int
        number of total objects
    stats_file : file handler
        summary statistics output file handler
    invalid : float, optional, default -10
        flag objects with ellipticity value = invalid
    verbose : bool, optional, default=False
        verbose output if True
    """
    # Get ellipticity columns
    all_ell = []
    ell_str = ""
    if ell_n_comp == 1:
        for i in (0, 1):
            all_ell.append(dd[ell_col_name[i]])
            ell_str = f"{ell_str}{ell_col_name[i]} "
    elif ell_n_comp == 2:
        for i in (0, 1):
            all_ell.append(dd[ell_col_name][:, i])
        ell_str = ell_col_name

    # Tolerance around invalid value
    EPS = 0.0001

    # Index list of valid objects
    ind_val = np.zeros(shape=(2, n_tot), dtype=bool)
    for i in (0, 1):
        ind_val[i] = np.abs(all_ell[i] - invalid) > EPS
    # Valid objects = those for which both ellipticity
    # are valid
    ind_v = ind_val[0] & ind_val[1]

    n_tot_val = len(np.where(ind_v)[0])
    n_tot_mil = util.millify(n_tot_val)
    msg = (
        f"Total number of valid objects ({ell_col_name}0,1 != {invalid})"
        + f" = {n_tot_val} = {n_tot_mil}"
    )
    io.print_stats(msg, stats_file, verbose=verbose)

    msg = (
        f"Fraction of invalid objects = {n_tot-n_tot_val}/{n_tot}"
        + f" = {(n_tot-n_tot_val)/n_tot:.3%}"
    )
    io.print_stats(msg, stats_file, verbose=verbose)

    # Select valid objects
    for i in (0, 1):
        all_ell[i] = all_ell[i][ind_v]

    # Mean ellipticity
    ell = np.zeros(2)
    for i in (0, 1):
        ell[i] = all_ell[i].mean()

    io.print_stats(
        f"Mean ellipticity of valid objects ({ell_str}):",
        stats_file,
        verbose=verbose,
    )
    for i in (0, 1):
        msg = "<e_{}> = {:.3g}".format(i + 1, ell[i])
        io.print_stats(msg, stats_file, verbose=verbose)


def print_some_quantities(dd, stats_file, verbose=False):
    """Print some quantities.

    Output some summary statistics from a catalogue.

    Parameters
    ----------
    dd : dict
        galaxy catalog
    stats_file : file handler
        summary statistics output file handler
    verbose : bool, optional, default=False
        verbose output if True

    Returns
    -------
    n_tot : int
        number of objects
    """
    # Print all column names
    if verbose:
        print("Column names:")
        print(dd.dtype.names)
        print("")

    n_tot = len(dd)
    n_mil = util.millify(n_tot)
    msg = f"Total number of objects = {n_tot} = {n_mil}"
    io.print_stats(msg, stats_file, verbose=verbose)

    return n_tot


def check_matching(
    d1,
    d2,
    keys_1,
    keys_2,
    thresh,
    stats_file,
    name=None,
    verbose=False,
):
    """Check matching.

    Check matching between two catalogues.

    Parameters
    ----------
    d1, d2 : dict
        catalogs
    keys_1, keys_2 : list
        column keys for d1, d2, corresponding to x, y
    thres : float
        threshold for matching, in deg
    stats_file : file handler
        summary statistics output file handler
    verbose : bool, optional, default=False
        verbose output if True

    Returns
    -------
    ind : array of int
        index list of d2 of objects that were matched to d1
    mask_area_tiles : array of int
        index list of tiles in footprint

    """
    if name is not None:
        # Filter stars outside footprint for efficiency
        mask_area_tiles = get_footprint(name, d1[keys_1[0]], d1[keys_1[1]])
        if len(np.where(mask_area_tiles)[0]) == 0:
            raise ValueError(f"Error: no object found in field '{name}'")
    else:
        mask_area_tiles = np.arange(len(d1))

    # Match stars from exposure (PSF) catalogue to total catalogue
    ind = match_stars2(
        d2[keys_2[0]],
        d2[keys_2[1]],
        d1[keys_1[0]][mask_area_tiles],
        d1[keys_1[1]][mask_area_tiles],
        thresh=thresh,
    )

    n_tot = len(d1[keys_1[0]][mask_area_tiles])
    msg = (
        "Number of matched stars from exposures to total catalogue = "
        + f"{len(ind)}/{n_tot} = {len(ind) / n_tot:.1%}"
    )
    io.print_stats(msg, stats_file, verbose=verbose)

    # Remove stars matched to more than one target object
    ind = np.array(list(set(ind)))

    msg = (
        "Number of matched stars after removing multiple matches = "
        + f"{len(ind)}/{n_tot} = {len(ind) / n_tot:.1%}"
    )
    io.print_stats(msg, stats_file, verbose=verbose)

    return ind, mask_area_tiles, n_tot


def check_invalid(dd, key, comp, val, stats_file, name=None, verbose=False):
    """Check invalid objects.

    Check whether objects have invalid values.

    Parameters
    ----------
    dd : dict
        catalog
    key : list
        key names of columns to check
    comp : array of int
        components for above columns
    val : array of float
        values for above columns indicating invalid entries
    stats_file : file handler
        summary statistics output file handler
    name : list, optional, default=None
        for output message. If None, key strings are used
    verbose : bool, optional, default=False
        verbose output if True
    """
    if name is None:
        name = key

    n_all = len(dd)

    for i in range(len(key)):

        w = dd[key[i]][:, comp[i]] == val[i]
        n_inv_psf = len(np.where(w)[0])
        msg = "Invalid {} found for {}/{} = {:.1g}% objects" "".format(
            name[i], n_inv_psf, n_all, n_inv_psf / n_all
        )
        io.print_stats(msg, stats_file, verbose=verbose)


def match_subsample(
    dd,
    ind,
    mask,
    pos_key,
    ell_key,
    n_ref,
    stats_file,
    verbose=False,
):
    """Match subsample.

    Match subsamples of catalogues.

    Parameters
    ----------
    dd : dict
        catalog
    ind : array of int
        index list of d2 of objects that were matched to d1
    mask : array of bool
        boolean mask
    pos_key : list
        key names for position columns
    ell_key : str
        key name for ellipticity column
    n_ref : int
        reference number of objects
    stats_file : file handler
        summary statistics output file handler
    verbose : bool, optional, default=False
        verbose output if True

    Returns
    -------
    ra, dec : array of float
        positions
    g : array(2) of float
        ellipticities
    """
    msg = (
        "Number of stars matched to valid sample = "
        + f"{len(dd[pos_key[0]][ind][mask])}/{n_ref} = "
        + f"{len(dd[pos_key[0]][ind][mask]) / n_ref * 100:.1f}%"
    )
    io.print_stats(msg, stats_file, verbose=verbose)

    ra = dd[pos_key[0]][ind][mask]
    dec = dd[pos_key[1]][ind][mask]
    g1 = dd[ell_key][:, 0][ind][mask]
    g2 = dd[ell_key][:, 1][ind][mask]
    g = np.array([g1, g2])

    return ra, dec, g


def match_spread_class(dd, ind, mask, stats_file, n_ref, verbose=False):
    """Match spread class.

    Match
    """
    tot_star = n_ref
    tot_as_star = len(np.where(dd["SPREAD_CLASS"][ind][mask] == 0)[0])
    tot_as_gal = len(np.where(dd["SPREAD_CLASS"][ind][mask] == 1)[0])
    tot_as_other = len(np.where(dd["SPREAD_CLASS"][ind][mask] == 2)[0])

    msg = (
        "Number of stars selected as star (SPREAD_CLASS=0)   = "
        + f"{tot_as_star}/{tot_star} = {tot_as_star / tot_star * 100:.1f}%"
    )
    io.print_stats(msg, stats_file, verbose=verbose)

    msg = (
        "Number of stars selected as galaxy (SPREAD_CLASS=1) = "
        + f"{tot_as_gal}/{tot_star} = {tot_as_gal / tot_star * 100:.1f}%"
    )
    io.print_stats(msg, stats_file, verbose=verbose)

    msg = (
        "Number of stars selected as other (SPREAD_CLASS=2)  = "
        + f"{tot_as_other}/{tot_star} = {tot_as_other / tot_star * 100:.1f}%"
    )
    io.print_stats(msg, stats_file, verbose=verbose)


def match_stars2(ra_gal, dec_gal, ra_star, dec_star, thresh=0.0002):
    """Add docstring.

    ...

    """
    gal_coord = coords.SkyCoord(ra=ra_gal * u.degree, dec=dec_gal * u.degree)
    star_coord = coords.SkyCoord(
        ra=ra_star * u.degree,
        dec=dec_star * u.degree,
    )

    res_coord = coords.match_coordinates_sky(star_coord, gal_coord)

    ind_stars = res_coord[0][np.where(res_coord[1].value < thresh)]

    return ind_stars


def read_shape_catalog(
    input_path,
    w_name="w",
):
    """Read Shape Catalog.

    Read catalogue with galaxy shapes = shear estimates.

    Parameters
    ----------
    input_path : str
        input file path
    w_name : str, optional
        name of weight column, default is "w"

    Returns
    -------
    ra : array of float
        right ascension in degrees
    dec : array of float
        declination in degrees
    g1 : array of float
        uncalibrated shear estimate component 1
    g2 : array of float
        uncalibrated shear estimate component 2
    w : array of float
        weight
    mag : array of float
        magnitude
    snr : array of float
        signal-to-noise ratio
    """
    dat = fits.open(input_path)

    hdu_no = 1

    ra = dat[hdu_no].data["RA"]
    dec = dat[hdu_no].data["Dec"]

    g = [np.empty_like(ra), np.empty_like(ra)]
    g1 = dat[hdu_no].data["e1_uncal"]
    g2 = dat[hdu_no].data["e2_uncal"]
    w = dat[hdu_no].data[w_name]
    mag = dat[hdu_no].data["mag"]

    if "snr" in dat[hdu_no].data.dtype.names:
        snr = dat[hdu_no].data["snr"]
    else:
        snr = None

    return ra, dec, g1, g2, w, mag, snr


def write_shape_catalog(
    output_path,
    ra,
    dec,
    w,
    mag=None,
    snr=None,
    g=None,
    g1_uncal=None,
    g2_uncal=None,
    R_g11=None,
    R_g22=None,
    R_g12=None,
    R_g21=None,
    R=None,
    R_shear=None,
    R_select=None,
    c=None,
    c_err=None,
    alpha_leakage=None,
    sigma_epsilon=None,
    w_type="iv",
    add_cols=None,
    add_cols_format=None,
    add_header=None,
):
    """Write Shape Catalog.

    Write catalogue with galaxy shapes = shear estimates.

    Parameters
    ----------
    output_path : str
        output file path
    ra, dec : arrays(ngal) of float
        coordinates in deg
    w : np.ndarray
        inverse-variance weights
    mag : array(ngal) of float, optional
        magnitude, signal-to-noise ratio
    snr : array(ngal) of float, optional
        signal-to-noise ratio, default is `None`
    g : np.ndarray, optional
        calibrated reduced shear estimate components, corrected for
        multiplicative and additive bias, g = R^-1 g_uncal - c;
        expected type is arrays(2, ngal) of float;
        default is ``None`` (no calibrated shears written)
    g1_uncal, g2_uncal : np.ndarray, optional
        uncalibrated shear estimates;
        expected types are arrays(ngal) of float
        default is ``None`` (no uncalibrated shears written)
    R_g11, R_g22, R_g12, R_g21 : np.ndarray, optional
        shear response matrix elements per galaxy;
        expected format is arrays(ngal) of float;
         default is ``None``
    R : 2x2 matrix of float, optional
        Mean full response matrix, default is ``None``
    R_shear : 2x2 matrix of float, optional
        Mean shear response matrix, default is ``None``
    R_select : 2x2 matrix of float, optional
        Global selection response matrix, default is ``None``
    c : array(2) of float, optional, default is ``None``
        additive shear bias
    c_err : array(2) of float, optional, default is ``None``
        error of c
    alpha_leakage : float, optional
        Mean scale-dependent PSF leakage, default is ``None``
    sigma_epsilon: float, optional
        shape noise, default is ``None``
    w_type : str, optional
        weight type, allowed are "iv" (default), "des"
    add_cols : dict, optional, default is ``None``
        data for n additional columns to add
    add_cols_format : dict, optional
        format for n additional columns to add, default is ``None``, for which
        ``float`` format is used
    add_header : fits.header.Header, optional
        additional header information; default is ``None``

    """
    col_info_arr = []

    # Principal columns: coordinates and weights
    col_info_arr.append(
        (
            fits.Column(name="RA", array=ra, format="D", unit="deg"),
            "Right Ascension",
        )
    )
    col_info_arr.append(
        (
            fits.Column(name="Dec", array=dec, format="D", unit="deg"),
            "Declination",
        )
    )
    if w_type == "iv":
        descr = "Inverse-variance weight"
    elif w_type == "des":
        descr = "DES-like weight"
    else:
        raise ValueError(f"Invalid weight type {w_type}")

    name = f"w_{w_type}"
    col_info_arr.append((fits.Column(name=name, array=w, format="D"), descr))

    # Additional columns
    ## Magnitude
    if mag is not None:
        col_info_arr.append(
            (
                fits.Column(name="mag", array=mag, format="D"),
                "MAG_AUTO magnitude",
            )
        )
    ## Calibrated shear estimates
    if g is not None:
        for idx in (0, 1):
            col_info_arr.append(
                (
                    fits.Column(name=f"e{idx+1}", array=g[idx], format="D"),
                    f"Calibrated reduced shear estimate comp {idx+1}",
                )
            )
    # Signal-to-noise ratio
    if snr is not None:
        col_info_arr.append(
            (
                fits.Column(name="snr", array=snr, format="D"),
                "Signal-to-noise ratio",
            )
        )
    for x, name, descr in zip(
        [g1_uncal, g2_uncal, R_g11, R_g22, R_g12, R_g21],
        ["e1_uncal", "e2_uncal", "R_g11", "R_g22", "R_g12", "R_g21"],
        [
            "Uncalibrated shear comp 1",
            "Uncalibrated shear comp 2",
            "Shear response matrix comp 1 1",
            "Shear response matrix comp 2 2",
            "Shear response matrix comp 1 2",
            "Shear response matrix comp 2 1",
        ],
    ):
        if x is not None:
            col_info_arr.append(
                (fits.Column(name=name, array=x, format="D"), descr)
            )

    if add_cols is not None:
        for idx, name in enumerate(add_cols):
            if add_cols_format is not None and name in add_cols_format:
                my_format = add_cols_format[name]
            else:
                shape = add_cols[name].shape
                if len(shape) == 1:
                    my_format = "D"
                else:
                    my_format = f"{shape[1]}D"
            col_info_arr.append(
                (
                    fits.Column(
                        name=name, array=add_cols[name], format=my_format
                    ),
                    name,
                )
            )

    # Write columns to FITS file
    cols = []
    for col, _ in col_info_arr:
        cols.append(col)
    table_hdu = fits.BinTableHDU.from_columns(cols)

    # Add human-readable descriptions
    for idx, col_info in enumerate(col_info_arr):
        table_hdu.header[f"TTYPE{idx+1}"] = (
            col_info[0].name,
            col_info[1],
        )

    # Primary HDU with information in header
    primary_header = fits.Header()
    
    if add_header:
        primary_header.update(add_header)

    primary_header = cat.write_header_info_sp(
        primary_header,
        software_name="sp_validation",
        software_version=__version__,
        author=getpass.getuser(),
    )

    if all(v is not None for v in (R, R_shear, R_select, c)):
        cat.add_shear_bias_to_header(primary_header, R, R_shear, R_select, c)
    if c_err is not None:
        primary_header["c1_err"] = (c_err[0], "Standard deviation of c_1")
        primary_header["c2_err"] = (c_err[1], "Standard deviation of c_2")

    primary_header["w"] = "DES weight"

    if sigma_epsilon is not None:
        primary_header["sig_eps"] = (sigma_epsilon, "Shape noise RMS")

    if alpha_leakage:
        primary_header["alpha"] = (
            alpha_leakage,
            "Mean scale-dependent PSF leakage",
        )

    primary_hdu = fits.PrimaryHDU(header=primary_header)

    # Final file
    hdu_list = fits.HDUList([primary_hdu, table_hdu])

    hdu_list.writeto(output_path, overwrite=True)


def write_galaxy_cat(output_path, ra, dec, tile_id):
    """Write Galaxy Cat.

    Write catalogue with  position information only, no shapes. E.g.
    random object catalogue.

    Parameters
    ----------
    output_path : string
        output file path
    ra, dec : arrays(ngal) of float
        coordinates in deg
    tile_id : array(ngal) of float
        tile ID of objects
    """
    c_ra = fits.Column(name="ra", array=ra, format="E", unit="deg")
    c_dec = fits.Column(name="dec", array=dec, format="E", unit="deg")
    c_id = fits.Column(name="tile_id", array=tile_id, format="E")
    cols = [c_ra, c_dec, c_id]

    cat.write_fits_BinTable_file(cols, output_path)


def write_PSF_cat(output_path, ra, dec, e1, e2):
    """Write PSF Cat.

    Write PSF catalogue to file.

    Parameters
    ----------
    output_path : string
        output file path
    ra, dec : list of float
        coordinates in deg
    e1 : list of float
        first ellipticity  component
    e2 : list of float
        second ellipticity  component
    """
    c_ra = fits.Column(name="RA", array=ra, format="D", unit="deg")
    c_dec = fits.Column(name="Dec", array=dec, format="D", unit="deg")
    c_e1 = fits.Column(name="e1", array=e1, format="D")
    c_e2 = fits.Column(name="e2", array=e2, format="D")
    cols = [c_ra, c_dec, c_e1, c_e2]

    cat.write_fits_BinTable_file(cols, output_path)


def read_param_file(path, verbose=False):
    """Read Param File.
    MKDEBUG TODO: Move to cs_util. Also used in sp/create_final_cat.

    Return parameter list read from file.

    Parameters
    ----------
    path: str
        input file name
    verbose: bool, optional, default=False
        verbose output if True

    Returns
    -------
    list of str
        parameter names

    """
    param_list = []

    if path:
        if verbose:
            print(f"Reading parameter file: {path}")

        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                entry = line.rstrip()
                if not entry or entry == "":
                    continue
                param_list.append(entry)

    if verbose:
        if len(param_list) > 0:
            print(f"Read {len(param_list)} columns", end="")
        else:
            print("No parameters read", end="")
        print(" into merged catalogue")

    param_list_unique = list(set(param_list))

    if verbose:
        n = len(param_list) - len(param_list_unique)
        if n > 1:
            print("Removed {n} duplicate entries")

    return param_list_unique


def read_hdf5_file(
    file_path, name, stats_file, check_only=False, param_path=None
):
    """Read HDF5 File.

    Read hdf5 file and return contained data.

    Parameters
    ----------
    file_path : str
        input file path
    name : str
        patch name
    stats_file : file handler
        summary statistics output file handler
    check_only : bool, optional
        If True only check, not return data

    Returns
    -------
    dict
        data

    """
    param_list = (
        read_param_file(param_path, verbose=True) if param_path else None
    )

    with h5py.File(file_path, "r") as hdf5_file:
        # Find patch group in hierarchical structure
        if not f"patches/{name}" in hdf5_file:
            raise KeyError(
                f"Entry patches/{name} not found in file {file_path}"
            )
        patch_group = hdf5_file[f"patches/{name}"]

        # Get size of data array
        num_rows = sum(patch_group[ID].shape[0] for ID in patch_group)
        # num_cols = patch_group[next(iter(patch_group))].shape[1]
        num_cols = len(param_list)

        print(
            f"Estimating {num_cols * num_rows * 8 / 1024**3:.1f}"
            + f" Gb memory for the ({num_cols} x {num_rows}) data array ..."
        )
        # data_comb = np.memmap(output_file, dtype=patch_group[next(iter(patch_group))].dtype,
        #              mode="w+", shape=(num_rows, num_cols))

        data_list = []
        ID_pbl = set()
        for ID in tqdm.tqdm(patch_group):

            # Get data for this ID from file
            data = patch_group[ID][()]

            # Restrict to parameter list if given
            data = data[param_list] if param_list is not None else data

            if not check_only:
                # Add new to existing data
                data_list.append(data)

        print("Combine tile catalogues")
        data_comb = np.concatenate(data_list, axis=0)
        print("Done")

    # Print problematic tile IDs
    for ID in ID_pbl:
        print("Tile IDs with missing keys:", file=stats_file)
        print(ID, file=stats_file)

    return data_comb


def get_maked_col(dat, col, mask):
    """Get Masked Col.

    Retrieve a specific column from the data with a mask.

    Parameters
    ----------
    dat: dict
        Input data
    col: str
        Key of the column to be returned
    mask: array-like
        Boolean mask used for selection

    Returns
    -------
    array-like
        Requested column from the data, filtered by the mask.

    """
    
    return dat[col][mask]


def get_col(dat, col, m_sel=None, m_flg=None):
    """Get Col.

    Retrieve a specific column from the data with optional selection and flag masks.

    Parameters
    ----------
    dat: dict
        Input data
    col: str
        Key of the column to be returned
    m_sel: array-like, optional
        Boolean mask used for selection. If specified, m_flg must also be specified;
        default is ``None``
    m_flg: array-like, optional
        Boolean mask used as a flag. If specified, m_sel must also be specified.
        default is ``None``

    Returns
    -------
    array-like
        Requested column from the data, optionally filtered by the selection and flag masks.

    See Also
    --------
        get_maked_col : More efficient if masks have been combined beforehand.

    Raises
    ------
    ValueError
        If only one of m_sel or m_flg is specified without the other.
    """

    if bool(m_sel is None) != bool(m_flg is None):
        raise ValueError("Specify both or none of selection and flag masks")

    if m_sel is None and m_flg is None:
        return dat[col][:]
    else:
        return dat[col][m_sel][m_flg]
        # The following does not work
        #return get_maked_col(dat, col, mask_combined)


def get_snr(sh, dat, m_sel, m_flg):
    """Get SNR.

    Return signal-to-noise ratio.

    Parameters
    ----------
    sh: str
        shape method identified, e.g. "ngmix"
    dat: dict
        Input data
    m_sel: array-like, optional
        Boolean mask used for selection. If specified, m_flg must also be specified;
        default is ``None``
    m_flg: array-like, optional
        Boolean mask used as a flag. If specified, m_sel must also be specified.
        default is ``None``

    Returns
    -------
    array-like
        signal-to-noise ratios

    """
    if sh == "ngmix":
        my_snr = get_col(dat, "NGMIX_FLUX_NOSHEAR", m_sel, m_flg) / get_col(
            dat, "NGMIX_FLUX_ERR_NOSHEAR", m_sel, m_flg
        )
    elif sh == "galsim":
        my_snr = get_col(dat, "SNR_WIN", m_sel, m_flg)

    return my_snr
