#!/usr/bin/env python

"""Script apply_alpha.py

Add PSF leakage to galaxy ellipticity using fitted model of alpha.
Write FITS file with added columns.

:Authors: Martin Kilbinger

"""

import sys
import pickle

from optparse import OptionParser

from lmfit import Parameters
from astropy.io import fits
from astropy.table import Table, Column

from cs_util import logging

from shear_psf_leakage.leakage import func_bias_2d

from sp_validation import cat
from sp_validation import util


def params_default():
    """Params Default

    Set default parameter values.

    Returns
    -------
    dict
        default parameter values

    """
    param = {
        "order": "lin",
        "mix": True,
        "e1_col": "e1",
        "e2_col": "e2",
        "e1_PSF_col": "e1_PSF",
        "e2_PSF_col": "e2_PSF",
        "input_path_shear": "SP/unions_shapepipe_extended_2022_v1.0.fits",
        "output_path": "shape_cat_cor_alpha.fits",
    }

    param["alpha_leakage"] = (
        f"./leakage/PSF_e_vs_e_gal_order-{param['order']}_mix-{param['mix']}.json"
    )

    return param


def parse_options(p_def):
    """Parse Options

    Parse command line options and update parameter values accordingly.

    Parameters
    ----------
    p_def: dict
        default parameter values

    Returns
    -------
    tuple
        Command line options
    string
        Command line string

    """
    usage = "%prog [OPTIONS]"
    parser = OptionParser(usage=usage)

    parser.add_option(
        "-i",
        "--input_path_shear",
        dest="input_path_shear",
        default=p_def["input_path_shear"],
        type="string",
        help="input path of the extended shear catalogue",
    )
    parser.add_option(
        "",
        "--e1_col",
        dest="e1_col",
        default=p_def["e1_col"],
        type="string",
        help=(
            "e1 column name in galaxy catalogue,"
            + f" default='{p_def['e1_col']}'"
        ),
    )
    parser.add_option(
        "",
        "--e2_col",
        dest="e2_col",
        default=p_def["e2_col"],
        type="string",
        help=(
            "e2 column name in galaxy catalogue,"
            + f" default='{p_def['e2_col']}'"
        ),
    )
    parser.add_option(
        "",
        "--e1_PSF_col",
        dest="e1_PSF_col",
        default=p_def["e1_PSF_col"],
        type="string",
        help=f"PSF e1 column name, default='{p_def['e1_PSF_col']}'",
    )
    parser.add_option(
        "",
        "--e2_PSF_col",
        dest="e2_PSF_col",
        default=p_def["e2_PSF_col"],
        type="string",
        help=f"PSF e2 column name, default='{p_def['e2_PSF_col']}'",
    )
    parser.add_option(
        "",
        "--alpha_leakage",
        dest="alpha_leakage",
        default=p_def["alpha_leakage"],
        type="string",
        help=f"File containing the alpha leakage fitted parameters, default='{p_def['alpha_leakage']}'",
    )
    parser.add_option(
        "-o",
        "--output_path",
        dest="output_path",
        default=p_def["output_path"],
        type="string",
        help=(
            "input path of the extended shear catalogue,"
            + f" default='{p_def['output_path']}'"
        ),
    )

    options, args = parser.parse_args()

    return options, args


def main(argv=None):

    # Get default parameter values
    params = params_default()

    # Parse command line options
    options, args = parse_options(params)

    # Update parameter values
    for key in vars(options):
        params[key] = getattr(options, key)

    logging.log_command(argv)

    # Load best-fit parameters for alpha leakage
    fname = params["alpha_leakage"]
    with open(fname, "rb") as fp:
        par_best_fit = pickle.load(fp)

    # Set additive component to zero to avoid double accounting
    for c in ("c1", "c2"):
        par_best_fit[c].value = 0

    for key in par_best_fit:
        print(key, par_best_fit[key])

    # Load extended shear catalogue
    hdu_list = fits.open(params["input_path_shear"])
    dat_shear = hdu_list[1].data
    e1_PSF = dat_shear[params["e1_PSF_col"]]
    e2_PSF = dat_shear[params["e2_PSF_col"]]
    e1 = dat_shear[params["e1_col"]]
    e2 = dat_shear[params["e2_col"]]

    # Get ellipticity correction de = alpha e
    de1, de2 = func_bias_2d(
        par_best_fit,
        e1_PSF,
        e2_PSF,
        order=params["order"],
        mix=params["mix"],
    )

    # If the PSF leakage term has contaminated the data, correct
    # for it by subtracting this term
    e1_cor = e1 - de1
    e2_cor = e2 - de2

    t = Table(dat_shear)
    t.add_column(Column(name="e1_cor", data=e1_cor))
    t.add_column(Column(name="e2_cor", data=e2_cor))
    print(
        "Saving alpha-corrected ellipticities to" + f' {params["output_path"]}'
    )
    t.write(params["output_path"], overwrite=True, format="fits")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
