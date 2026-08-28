# Name file: shear_calibration+leakage_correction_ellipticities_.py

# @author: Antonin Corinaldi, antonin.corinaldi@cea.fr

# Python file to perform metacalibration and PSF leakage correction of the ellipticities of matched catalogues
# (cf file get_matched_catalogue.py in src/sp_validation for the creation of the matched catalogues)


import numpy as np
from astropy.table import Table
from calibration import fill_cat_gal, get_alpha_leakage_per_object


class ResponseMatrix:
    def __init__(self, R11, R12, R21, R22):
        self.R11 = R11
        self.R12 = R12
        self.R21 = R21
        self.R22 = R22


def get_R(cat, weights=None):
    """
    Compute weighted mean response matrix.
    """
    R = np.zeros((2, 2))

    for i in range(2):
        for j in range(2):
            R[i, j] = np.average(cat[f"R_g{i + 1}{j + 1}"], weights=weights)

    return R


def process_matched_catalogue(
    input_path, output_path=None, weight_column="w_iv", num_bins=20, weight_type="iv"
):
    """
    Function to:
    - read matched catalogue
    - compute metacalibration correction
    - apply PSF leakage correction
    - save corrected matched catalogue

    Parameters
    ----------
    input_path : str
        Path to input FITS catalogue
    output_path : str or None
        Path to save output catalogue (if None, overwrite input)
    weight_column : str
        Column used for weighting (default: w_iv)
    num_bins : int
        Number of bins for leakage estimation
    weight_type : str
        Weight type for alpha leakage computation
    """

    # Reading catalogue
    table = Table.read(input_path)

    names = [name for name in table.colnames if len(table[name].shape) <= 1]
    table = table[names]

    df = table.to_pandas()

    # Computing response matrix
    g_uncorr = np.array([df["e1_uncal"], df["e2_uncal"]])

    gal_metacal = ResponseMatrix(df["R_g11"], df["R_g12"], df["R_g21"], df["R_g22"])

    cat = {}
    fill_cat_gal(
        cat, df, g_uncorr, gal_metacal, mask1=None, mask2=None, purpose="weights"
    )

    weights = df[weight_column]
    R = get_R(cat, weights=weights)

    # Metacalibration correction
    R_inv = np.linalg.inv(R)
    e_cal = R_inv.dot(np.array(df[["e1_uncal", "e2_uncal"]].T))

    df["e1"] = e_cal[0, :]
    df["e2"] = e_cal[1, :]

    # PSF leakage correction
    alpha_1, alpha_2 = get_alpha_leakage_per_object(
        df, num_bins=num_bins, weight_type=weight_type
    )

    df["e1_leak_corrected"] = df["e1"] - alpha_1 * df["e1_PSF"]
    df["e2_leak_corrected"] = df["e2"] - alpha_2 * df["e2_PSF"]

    # Save catalogue
    output_path = output_path or input_path

    output_table = Table.from_pandas(df)
    output_table.write(output_path, overwrite=True)

    print(f"Catalogue processed and saved to: {output_path}")

    return output_table


# Example of usage

process_matched_catalogue(
    "/n17data/corinaldi/matched_catalogues/unions1.6.9/UNIONS1.6.9_lrg_cross.fits"
)
