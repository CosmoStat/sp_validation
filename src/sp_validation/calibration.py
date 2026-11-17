"""CALIBRATION.

:Name: calibration.py

:Description: Per-object PSF leakage of calibrated shapes. The metacal
    response, weights and additive bias are in :mod:`sp_validation.response`.

:Author: Martin Kilbinger

"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def get_alpha_leakage_per_object(cat_gal, num_bins, weight_type="des"):
    """
    Compute the leakage per object (Li et al. 2024)
    Return an array of leakage coefficients obtained by binning in
    SNR and size.

    Parameters
    ----------
    cat_gal : dict
        A catalog of galaxies containing galaxy ellipticity, PSF ellipticity, SNR and size of the galaxy and the PSF.
    num_bins : int
        Number of bins

    Returns
    -------
    alpha_1 : np.array
        Array containing the correction coefficient for the PSF leakage
        per object for the first component.
    alpha_2 : np.array
        Array containing the correction coefficient for the PSF leakage
        per object for the second component.
    """
    assert weight_type in ["des", "iv"], "weight_type must be either 'des' or 'iv'"
    # Compute the size ratio
    size_ratio = cat_gal["NGMIX_T_PSF_RECONV_NOSHEAR"] / (
        cat_gal["NGMIX_T_NOSHEAR"] + cat_gal["NGMIX_T_PSF_RECONV_NOSHEAR"]
    )

    df_gal = pd.DataFrame(
        np.array(
            [
                np.array(cat_gal["e1"], dtype=np.float64),
                np.array(cat_gal["e2"], dtype=np.float64),
                np.array(cat_gal["e1_PSF"], dtype=np.float64),
                np.array(cat_gal["e2_PSF"], dtype=np.float64),
                np.array(cat_gal["snr"], dtype=np.float64),
                np.array(cat_gal[f"w_{weight_type}"], dtype=np.float64),
                np.array(size_ratio, dtype=np.float64),
            ]
        ).T,
        columns=["e1", "e2", "e1_PSF", "e2_PSF", "snr", "w", "size_ratio"],
    )

    del size_ratio

    n_bins_snr = num_bins
    n_bins_r = num_bins

    # Create logarithmic bins in size and SNR
    df_gal.loc[:, "bin_R"] = pd.qcut(
        df_gal["size_ratio"], n_bins_r, labels=False, retbins=False
    )

    # initialize bin snr
    df_gal.loc[:, "bin_snr"] = -999

    for ibin_r in range(n_bins_r):
        # select galaxies in the bin
        mask_binr = df_gal["bin_R"].values == ibin_r

        # bin in snr
        df_gal.loc[mask_binr, "bin_snr"] = pd.qcut(
            df_gal.loc[mask_binr, "snr"], n_bins_snr, labels=False, retbins=False
        )

    # group by bin
    df_gal_grouped = df_gal.groupby(["bin_R", "bin_snr"])
    ngroups = df_gal_grouped.ngroups

    # Performing first round calibration
    alpha_df = pd.DataFrame(
        0.0,
        index=np.arange(ngroups),
        columns=["R", "SNR", "alpha_1", "alpha_2", "alpha_1_err", "alpha_2_err"],
    )

    i_group = 0
    for name, group in df_gal_grouped:
        # Save weighted average
        alpha_df.loc[i_group, "R"] = np.average(group["size_ratio"], weights=group["w"])
        alpha_df.loc[i_group, "SNR"] = np.average(group["snr"], weights=group["w"])

        # Fit linear model to compute alpha
        e1_out = np.array(group["e1"])
        e2_out = np.array(group["e2"])
        weight_out = np.array(group["w"])
        e1_PSF = np.array(group["e1_PSF"])
        e2_PSF = np.array(group["e2_PSF"])
        del group

        # Fit e1
        mod_wls = sm.WLS(e1_out, sm.add_constant(e1_PSF), weights=weight_out)
        try:
            res_wls = mod_wls.fit()
        except Exception as err:
            raise RuntimeError("Linear regression fit for PSF leakage failed") from err
        alpha_df.loc[i_group, "alpha_1"] = res_wls.params[1]
        alpha_df.loc[i_group, "alpha_1_err"] = np.sqrt(res_wls.cov_params()[1, 1])
        del res_wls, mod_wls

        # Fit e2
        mod_wls = sm.WLS(e2_out, sm.add_constant(e2_PSF), weights=weight_out)
        res_wls = mod_wls.fit()
        alpha_df.loc[i_group, "alpha_2"] = res_wls.params[1]
        alpha_df.loc[i_group, "alpha_2_err"] = np.sqrt(res_wls.cov_params()[1, 1])
        del weight_out, res_wls, mod_wls

        i_group += 1

        # Fit polynomial to remove general trend
    fitting_weight = 1.0 / np.square(alpha_df["alpha_1_err"].values)
    A = np.vstack(
        [
            fitting_weight * 1,
            fitting_weight * np.power(alpha_df["SNR"].values, -2),
            fitting_weight * np.power(alpha_df["SNR"].values, -3),
            fitting_weight * alpha_df["R"].values,
            fitting_weight
            * alpha_df["R"].values
            * np.power(alpha_df["SNR"].values, -2),
        ]
    ).T
    poly1 = np.linalg.lstsq(A, fitting_weight * alpha_df["alpha_1"].values, rcond=None)[
        0
    ]
    del fitting_weight, A

    fitting_weight = 1.0 / np.square(alpha_df["alpha_2_err"].values)
    A = np.vstack(
        [
            fitting_weight * 1,
            fitting_weight * np.power(alpha_df["SNR"].values, -2),
            fitting_weight * np.power(alpha_df["SNR"].values, -3),
            fitting_weight * alpha_df["R"].values,
            fitting_weight
            * alpha_df["R"].values
            * np.power(alpha_df["SNR"].values, -2),
        ]
    ).T
    poly2 = np.linalg.lstsq(A, fitting_weight * alpha_df["alpha_2"].values, rcond=None)[
        0
    ]
    del fitting_weight, A

    # Compute alpha to remove the general trend
    # e1
    alpha_1 = (
        poly1[0]
        + poly1[1] * np.power(df_gal["snr"], -2)
        + poly1[2] * np.power(df_gal["snr"], -3)
        + poly1[3] * df_gal["size_ratio"]
        + poly1[4] * df_gal["size_ratio"] * np.power(df_gal["snr"], -2)
    )

    df_gal.loc[:, "e1_cor"] = df_gal["e1"].values - alpha_1 * df_gal["e1_PSF"].values
    del poly1

    # e2
    alpha_2 = (
        poly2[0]
        + poly2[1] * np.power(df_gal["snr"], -2)
        + poly2[2] * np.power(df_gal["snr"], -3)
        + poly2[3] * df_gal["size_ratio"]
        + poly2[4] * df_gal["size_ratio"] * np.power(df_gal["snr"], -2)
    )
    df_gal.loc[:, "e2_cor"] = df_gal["e2"].values - alpha_2 * df_gal["e2_PSF"].values
    del poly2

    # Initialise second run of calibration
    df_gal.loc[:, "alpha_1_cor"] = -999.0
    df_gal.loc[:, "alpha_2_cor"] = -999.0
    alpha_df.loc[:, "alpha_1_corr"] = -999.0
    alpha_df.loc[:, "alpha_2_corr"] = -999.0
    alpha_df.loc[:, "alpha_1_corr_err"] = -999.0
    alpha_df.loc[:, "alpha_2_corr_err"] = -999.0

    # Second run of calibration
    for i_group in range(n_bins_r * n_bins_snr):
        # Get the mask
        mask_group = (df_gal["bin_R"].values == i_group // n_bins_snr) & (
            df_gal["bin_snr"].values == i_group % n_bins_snr
        )
        # Fit linear model to compute alpha
        e1_out = np.array(df_gal.loc[mask_group, "e1_cor"])
        e2_out = np.array(df_gal.loc[mask_group, "e2_cor"])
        weight_out = np.array(df_gal.loc[mask_group, "w"])
        e1_PSF = np.array(df_gal.loc[mask_group, "e1_PSF"])
        e2_PSF = np.array(df_gal.loc[mask_group, "e2_PSF"])

        # Fit e1
        mod_wls = sm.WLS(e1_out, sm.add_constant(e1_PSF), weights=weight_out)
        res_wls = mod_wls.fit()
        alpha_df.loc[i_group, "alpha_1_corr"] = res_wls.params[1]
        alpha_df.loc[i_group, "alpha_1_corr_err"] = np.sqrt(res_wls.cov_params()[1, 1])
        del res_wls, mod_wls

        # Fit e2
        mod_wls = sm.WLS(e2_out, sm.add_constant(e2_PSF), weights=weight_out)
        res_wls = mod_wls.fit()
        alpha_df.loc[i_group, "alpha_2_corr"] = res_wls.params[1]
        alpha_df.loc[i_group, "alpha_2_corr_err"] = np.sqrt(res_wls.cov_params()[1, 1])
        del weight_out, res_wls, mod_wls

        df_gal.loc[mask_group, "alpha_1_corr"] = alpha_df.loc[i_group, "alpha_1_corr"]
        df_gal.loc[mask_group, "alpha_2_corr"] = alpha_df.loc[i_group, "alpha_2_corr"]

    alpha_1 += df_gal["alpha_1_corr"].values
    alpha_2 += df_gal["alpha_2_corr"].values

    return alpha_1, alpha_2
