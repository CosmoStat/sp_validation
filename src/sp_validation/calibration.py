"""CALIBRATION.

:Name: calibration.py

:Description: This script contains methods for shear calibration.

:Author: Martin Kilbinger

"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import tqdm
from astropy.io import fits
from shear_psf_leakage import leakage, run_object

from sp_validation import catalog as sp_cat
from sp_validation import response
from sp_validation.io import read_catalogue
from sp_validation.statistics import jackknif_weighted_average2


def metacal_from_config(data, mask, cm, verbose=False):
    """One calibration entry point; old release configs are explicitly legacy.

    Missing weight_scheme preserves their historical R/c/unsquared IV policy.
    New production configurations must declare des or iv and matching w_col.
    """
    modern = "weight_scheme" in cm
    return metacal(
        data,
        mask,
        snr_min=cm["gal_snr_min"],
        snr_max=cm["gal_snr_max"],
        rel_size_min=cm["gal_rel_size_min"],
        rel_size_max=cm["gal_rel_size_max"],
        size_corr_ell=cm["gal_size_corr_ell"],
        sigma_eps=cm["sigma_eps_prior"],
        global_R_weight=None if modern else cm.get("global_R_weight"),
        weight_scheme=cm.get("weight_scheme"),
        legacy_response=not modern,
        des_num_bins=cm.get("des_num_bins", 20),
        des_min_count=cm.get("des_min_count", 3),
        des_weight_product=cm.get("des_weight_product"),
        w_col=cm.get("w_col"),
        verbose=verbose,
    )


def load_des_weight_product(product):
    """Read a named, independent calibration product, never refitting its W.

    Use modern response products, whose headers declare the coordinate
    definitions. Historical released weights alone do not specify such a table.
    The product's edges and pooling rule are authoritative, not the sim cuts.
    """
    if (
        not isinstance(product, dict)
        or not product.get("name")
        or not product.get("path")
    ):
        raise ValueError("des_weight_product needs a name and a FITS path")
    with fits.open(product["path"]) as hdus:
        hdr = hdus[0].header
        if (
            hdr.get("WGHTSCH") != "des"
            or hdr.get("WSNRCOL") != "FLUX/FLUX_ERR"
            or hdr.get("WSIZDEF") != "T/T_PSF_RECONV"
        ):
            raise ValueError(
                "Incompatible DES product: regenerate with coordinate metadata"
            )
        es, er = np.array(hdus["W_EDGES"].data, dtype=float)
        w = np.array(hdus["W_DES"].data, dtype=float)
        count = np.array(hdus["W_COUNT"].data)
        if (
            w.shape != (len(es) - 1, len(er) - 1)
            or count.shape != w.shape
            or not np.all(np.diff(es) > 0)
            or not np.all(np.diff(er) > 0)
            or not np.isfinite(w).all()
            or np.any(w < 0)
            or not np.any(w > 0)
        ):
            raise ValueError("Invalid DES table or edges in calibration product")
        return (es, er, w, count), hdr["DESNMIN"]


def save_response_product(path, gm, groups=None):
    """Attach matrix definitions, fitted W, and spatial response/c covariance.

    groups are aligned to the preselected parent, never only the ns selection.
    """
    if gm._legacy_response:
        with fits.open(path, mode="update") as hdus:
            hdus[0].header["RLEGACY"] = (True, "Historical R/c and unsquared IV policy")
        return
    with fits.open(path, mode="update") as hdus:
        hdr = hdus[0].header
        hdr["MEANSUB"] = getattr(gm, "subtract_mean_shear", False)
        hdr["WGHTSCH"] = gm.weight_scheme
        hdr["RAPPLIED"] = True
        hdr["MCSTEP"] = gm._step
        hdr["WPOLICY"] = (
            "fixed-external"
            if gm._des_weight_product is not None
            else "refit"
            if gm.weight_scheme == "des"
            else "not applicable"
        )
        hdr["IVPOLICY"] = "branch variances"
        hdr["CMUNITS"] = "raw ellipticity"
        for i in range(2):
            hdr[f"CE_{i + 1}"] = gm.c_e[i]
            hdr[f"CG_{i + 1}"] = gm.c_g[i]
            for j in range(2):
                hdr[f"RJ_{i + 1}{j + 1}"] = gm.R_joint[i, j]
        for b in response.BRANCHES:
            hdr[f"N_{b}"] = gm.branch_counts[b]
            hdr[f"SW_{b}"] = gm.branch_weight_sums[b]
            if hasattr(gm, "branch_migration"):
                hdr[f"MIG_{b}"] = gm.branch_migration[b]
        if hasattr(gm, "W"):
            hdr["DESNMIN"] = (gm._des_min_count, "Minimum DES training count")
            hdr["DESMERGE"] = "concentric square; pooled moments"
            hdr["WSNRCOL"] = "FLUX/FLUX_ERR"
            hdr["WSIZDEF"] = "T/T_PSF_RECONV"
            hdr["WSOURCE"] = "external" if gm._des_weight_product else "self"
            if gm._des_weight_product:
                hdr["WPRODUCT"] = gm._des_weight_product["name"]
                hdr["WPRODPTH"] = gm._des_weight_product["path"]
            es, er, w, count = gm.W
            hdus.append(fits.ImageHDU(w, name="W_DES"))
            hdus.append(fits.ImageHDU(count, name="W_COUNT"))
            hdus.append(fits.ImageHDU(np.array([es, er]), name="W_EDGES"))
        if groups is not None:
            labels, estimates, cov = gm.response_jackknife(
                groups, refit_w=gm.weight_scheme == "des"
            )
            labels = np.asarray(labels).astype(str)
            columns = [
                fits.Column(
                    name="TILE_ID", array=labels, format=f"{max(map(len, labels))}A"
                )
            ]
            for key, prefix in [
                ("total", "R"),
                ("joint", "RJ"),
                ("shear", "RG"),
                ("selection", "RS"),
            ]:
                for i in range(2):
                    for j in range(2):
                        columns.append(
                            fits.Column(
                                name=f"{prefix}_{i + 1}{j + 1}",
                                array=estimates[key][:, i, j],
                                format="D",
                            )
                        )
            for key in ("c_e", "c_g"):
                for i in range(2):
                    columns.append(
                        fits.Column(
                            name=f"{key}{i + 1}", array=estimates[key][:, i], format="D"
                        )
                    )
            hdus.append(fits.BinTableHDU.from_columns(columns, name="R_JK"))
            cov_hdu = fits.ImageHDU(cov, name="R_COV")
            cov_hdu.header["ORDER"] = "shear,selection,total,joint,c_e,c_g; row-major"
            hdus.append(cov_hdu)


def get_calibrated_quantities(gal_metacal):
    """Get Calibrated Quantities.

    Return catalogue quantities for objects calibrated for multiplicative
    bias.

    Parameters
    ----------
    gal_metacal : dict
        galaxy metacalibration catalogue

    Returns
    -------
    g_corr : array(2, ngal) of float
        shear estimates calibrated for multiplicative bias
    g_uncorr : array(2, ngal) of float
        uncalibrated shear estimates
    w : array of float
        weights
    mask : array of bool
        mask to indicate valid objects in "no-shear" sample
    """
    # mask for 'no shear' images
    mask = gal_metacal.mask_dict["ns"]

    # uncalibrated shear estimates
    g_uncorr = np.array([gal_metacal.ns["g1"][mask], gal_metacal.ns["g2"][mask]])

    # calibratied shear estimates: multiply with inverse response matrix
    g_corr = np.linalg.inv(gal_metacal.R).dot(g_uncorr)

    # weights
    w = gal_metacal.ns["w"][mask]

    return g_corr, g_uncorr, w, mask


def get_calibrated_m_c(gal_metacal, additive_correction=True):
    """Get Calibrated C.

    Return catalogue quantities for objects calibrated for multiplicative and
    additive bias.

    Parameters
    ----------
    gal_metacal : dict
        galaxy metacalibration catalogue
    additive_correction : bool, optional, default=True
        if False, do not subtract the additive bias c from the shear
        estimates; use for constant-shear image sims, where the mean
        shear is the signal (see issue #226). c and c_err are still
        computed and returned

    Returns
    -------
    numpy.ndarray :
        shear estimates calibrated for multiplicative and additive bias;
        array(2, ngal) of float
    numpy.ndarray :
        uncalibrated shear estimates; array(2, ngal) of float
    numpy.ndarray :
        weights; array of float
    numpy.ndarray :
        mask to indicate valid objects in "no-shear" sample; array of bool
    numpy.ndarray :
        additive bias for both components;
    numpy.ndarray :
        error on the additive bias for both components

    """
    # Get m-calibrated quantities
    g_corr, g_uncorr, w, mask_metacal = get_calibrated_quantities(gal_metacal)

    # Additive bias
    c = np.zeros(2)
    c_err = np.zeros(2)

    for comp in (0, 1):
        if getattr(gal_metacal, "_legacy_response", False):
            c[comp] = np.mean(g_uncorr[comp])
            c_err[comp] = np.std(g_uncorr[comp])
        else:
            c[comp] = np.average(g_uncorr[comp], weights=w)
            c_err[comp] = (
                np.sqrt(np.sum(w**2 * (g_uncorr[comp] - c[comp]) ** 2)) / w.sum()
            )

    gal_metacal.subtract_mean_shear = additive_correction
    gal_metacal.c_e = c
    gal_metacal.c_g = np.linalg.solve(gal_metacal.R, c)

    # Shear estimate corrected for additive bias
    g_corr_mc = np.copy(g_corr)
    if additive_correction:
        c_corr = np.linalg.inv(gal_metacal.R).dot(c)
        for comp in (0, 1):
            g_corr_mc[comp] = g_corr[comp] - c_corr[comp]

    return g_corr_mc, g_uncorr, w, mask_metacal, c, c_err


def create_bins(x, num_bins, type="log", x_min=None, x_max=None):
    """Create Bins.
    Create bins for a given array. The bins are logarithmic by default.

    Parameters
    ----------
    x : array
        Array to bin
    num_bins : int
        Number of bins
    type : str, optional
        Type of binning. Options are 'log' (defaults)
    x_min : float, optional
        Minimum value of the bins. If None, the minimum value of x is used.
    x_max : float, optional
        Maximum value of the bins. If None, the maximum value of x is used.

    """
    if type == "log":
        xmin = x.min() if not x_min else x_min
        xmax = x.max() if not x_max else x_max
        return np.logspace(np.log10(xmin), np.log10(xmax), num_bins + 1)
    else:
        raise ValueError("Type not supported")


def cut_to_bins(df, key, num_bins, type="log", x_min=None, x_max=None):
    """Cut To Bins.

    Cut a given array into bins. Create a new column in the DataFrame with the binning.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to cut
    key : str
        Key to cut
    num_bins : int
        Number of bins
    type : str, optional
        Type of binning. Options are 'log' (default)
    x_min : float, optional
        Minimum value of the bins. If None, the minimum value of x is used.
    x_max : float, optional
        Maximum value of the bins. If None, the maximum value of x is used.

    Returns
    -------
    numpy.ndarray
        bin edges
    """
    key_cut = f"{key}_{type}_bins"

    bin_edges = create_bins(df[key], num_bins, type=type, x_min=x_min, x_max=x_max)
    df[key_cut] = pd.cut(df[key], bin_edges, labels=False)

    df.loc[np.isnan(df[key_cut]), key_cut] = num_bins - 1

    return bin_edges


def fill_cat_gal(cat_gal, dat, g_uncorr, gal_metacal, mask1, mask2, purpose="weights"):
    cat_gal["e1_uncal"] = g_uncorr[0]
    cat_gal["e2_uncal"] = g_uncorr[1]
    cat_gal["R_g11"] = gal_metacal.R11
    cat_gal["R_g12"] = gal_metacal.R12
    cat_gal["R_g21"] = gal_metacal.R21
    cat_gal["R_g22"] = gal_metacal.R22

    cat_gal["NGMIX_T_NOSHEAR"] = sp_cat.get_col(dat, "NGMIX_T_NOSHEAR", mask1, mask2)
    cat_gal["NGMIX_T_PSF_RECONV_NOSHEAR"] = sp_cat.get_col(
        dat, "NGMIX_T_PSF_RECONV_NOSHEAR", mask1, mask2
    )
    cat_gal["size_ratio"] = (
        cat_gal["NGMIX_T_NOSHEAR"] / cat_gal["NGMIX_T_PSF_RECONV_NOSHEAR"]
    )

    cat_gal["snr"] = sp_cat.get_col(
        dat, "NGMIX_FLUX_NOSHEAR", mask1, mask2
    ) / sp_cat.get_col(dat, "NGMIX_FLUX_ERR_NOSHEAR", mask1, mask2)

    if purpose == "weights":
        pass
    elif purpose == "leakage":
        cat_gal["w"] = sp_cat.get_col(dat, "w_iv", mask1, mask2)
        for idx in (1, 2):
            cat_gal[f"e{idx}_PSF"] = sp_cat.get_col(dat, f"e{idx}_PSF", mask1, mask2)
        cat_gal["fwhm_PSF"] = sp_cat.get_col(dat, "fwhm_PSF", mask1, mask2)


def build_df(cat_gal):
    """Build DF.

    Build pandas dataframe.

    Parameters
    ----------
    cat_gal : dict
        input data

    Returns
    -------
    pd.DataFrame
        collected data

    """
    # Build a pandas dataframe to perform the binning and the computation of the weights
    arr = np.array([np.array(cat_gal[key], dtype=np.float64) for key in cat_gal]).T
    return pd.DataFrame(arr, columns=cat_gal.keys())


def get_w_des(
    cat_gal,
    num_bins,
    snr_min=None,
    snr_max=None,
    size_ratio_min=None,
    size_ratio_max=None,
):
    """
    Get DES weights. (Gatti et al. 2021)
    Return an array of DES weights obtained by binning in SNR and size and computing the ratio between
    the shear response and the shape noise.

    Parameters
    ----------
    cat_gal: dict
        A catalog of galaxies containing the response matrix and the
        uncalibrated ellipticities
    num_bins : int
        Number of bins to use for the binning of the SNR and size.
    snr_min : float, optional
        Minimum SNR, default (`None`): determined by the data
    snr_max : float, optional
        Maximum SNR, default (`None`): determined by the data
    size_ratio_min : float, optional
        Minimum size ratio, default (`None`): determined by the data
    size_ratio_max : float, optional
        Maximum size ratio, default (`None`): determined by the data

    Returns
    -------
    w : array of float
        DES weights

    """
    table = fit_w_des(
        cat_gal, num_bins, snr_min, snr_max, size_ratio_min, size_ratio_max
    )
    return response.evaluate_w(table, cat_gal["snr"], cat_gal["size_ratio"])


def fit_w_des(
    cat_gal,
    num_bins=20,
    snr_min=None,
    snr_max=None,
    size_ratio_min=None,
    size_ratio_max=None,
    min_count=0,
):
    """Fit once on raw no-shear shapes; evaluate via response.evaluate_w."""
    return response.fit_w(
        np.asarray(cat_gal["snr"]),
        np.asarray(cat_gal["size_ratio"]),
        np.array([cat_gal["e1_uncal"], cat_gal["e2_uncal"]]),
        0.5 * (np.asarray(cat_gal["R_g11"]) + np.asarray(cat_gal["R_g22"])),
        create_bins(cat_gal["snr"], num_bins, x_min=snr_min, x_max=snr_max),
        create_bins(
            cat_gal["size_ratio"], num_bins, x_min=size_ratio_min, x_max=size_ratio_max
        ),
        min_count=min_count,
    )


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


def get_quantities_binned(
    cat_gal,
    num_bins_x,
    num_bins_y=None,
    which=["response", "number", "leakage"],
    verbose=True,
):
    if verbose:
        print("Compute binned quantities")

    if num_bins_y is None:
        num_bins_y = num_bins_x

    # Create input dataframe
    df_gal = build_df(cat_gal)

    # Create logarithmic bins in size and SNR
    bin_edges = {}
    bin_edges["snr"] = cut_to_bins(
        df_gal, "snr", num_bins_x, type="log", x_min=2, x_max=700
    )
    bin_edges["size_ratio"] = cut_to_bins(
        df_gal, "size_ratio", num_bins_y, type="log", x_min=0.3, x_max=10
    )

    # Initialize output dict
    quantities = {}
    for key in which:
        if key == "response":
            quantities["response"] = np.zeros((num_bins_x, num_bins_y, 2, 2))
        elif key == "leakage":
            obj = run_object.LeakageObject()
            obj._params["e1_col"] = "e1_uncal"
            obj._params["e2_col"] = "e2_uncal"
            obj._params["size_PSF_col"] = "fwhm_PSF"
            obj._params["verbose"] = False
            obj._params["no_stats_file"] = True
            obj._params["output_dir"] = ""
            obj.prepare_output()
            quantities[key] = np.zeros((num_bins_x, num_bins_y, 2, 2))
        else:
            quantities[key] = np.zeros((num_bins_x, num_bins_y))

    # Iniitialize parameter for minimizations
    params = leakage.init_parameters()

    # Loop over bins
    for i in tqdm.tqdm(
        range(num_bins_x), position=0, disable=not verbose, desc="bins_x"
    ):
        for j in tqdm.tqdm(range(num_bins_y), position=1, leave=False, desc="bins_y"):
            # Get indices for bin (i, j)
            bin_mask = (df_gal["snr_log_bins"] == i) & (
                df_gal["size_ratio_log_bins"] == j
            )
            if any(bin_mask):
                # Compute quantity
                for key in which:
                    if key == "response":
                        for idx in (0, 1):
                            for jdx in (0, 1):
                                quantities["response"][i, j, idx, jdx] = np.mean(
                                    df_gal[bin_mask][f"R_g{idx + 1}{jdx + 1}"]
                                )
                    elif key == "number":
                        quantities["number"][i, j] = np.sum(bin_mask)
                    elif key == "leakage":
                        obj._dat = df_gal[bin_mask]
                        obj.PSF_leakage(params=params, do_plots=False)
                        for idx in (0, 1):
                            for jdx in (0, 1):
                                quantities["leakage"][i, j, idx, jdx] = (
                                    obj.par_best_fit[f"a{idx + 1}{jdx + 1}"].value
                                )

    return quantities, bin_edges


def get_calibrate_e_from_cat(path_cat_gal, weight_type="des", verbose=False):
    """
    Calibrates ellipticities from a galaxy catalog with a certain weight type.

    Parameters
    ----------
    path_cat_gal : str
        Path to the galaxy catalog
    weight_type : str, optional, default='des'
        Type of weight to use. Options are 'des' (DES weight) or 'iv' (inverse variance)
    verbose : bool, optional, default=False
        If True, print intermediate results

    Returns
    -------
    g_cal : np.array
        Calibrated ellipticities
    """
    assert weight_type in ["iv", "des"], (
        "The weight_type is not correct. Options 'iv' (inverse variance) or 'des'."
    )

    hdu_gal = fits.open(path_cat_gal)

    cat_gal = hdu_gal[1].data

    R_select = np.array(
        [
            [hdu_gal[0].header["R_S11"], hdu_gal[0].header["R_S12"]],
            [hdu_gal[0].header["R_S21"], hdu_gal[0].header["R_S22"]],
        ]
    )

    stored_scheme = hdu_gal[0].header.get("WGHTSCH")
    if stored_scheme != weight_type:
        hdu_gal.close()
        raise ValueError(
            "Selection response cannot be reused with unknown/mismatched "
            "weights; recalibrate from the parent five-branch catalogue"
        )
    hdr = hdu_gal[0].header
    weights = cat_gal["w_" + weight_type]
    if hdr.get("N_ns") != len(cat_gal) or not np.isclose(
        hdr.get("SW_ns", np.nan), weights.sum(), rtol=1e-12
    ):
        hdu_gal.close()
        raise ValueError(
            "Stored selection response does not describe this "
            "population/weights; use the parent five-branch catalogue"
        )
    if verbose:
        print("R_select\n", R_select)

    g = np.array([cat_gal["e1_uncal"], cat_gal["e2_uncal"]])

    c = np.average(g, axis=1, weights=cat_gal["w_" + weight_type])

    if verbose:
        print("Additive bias\n", c)

    R_shear = np.zeros((2, 2))
    for iidx in range(2):
        for jidx in range(2):
            R_shear[iidx, jidx] = np.average(
                cat_gal[f"R_g{iidx + 1}{jidx + 1}"], weights=cat_gal["w_" + weight_type]
            )

    if verbose:
        print("R_shear\n", R_shear)

    R = R_shear + R_select

    if verbose:
        print("R\n", R)

    c_corr = np.linalg.inv(R) @ c
    g_cal = np.linalg.inv(R) @ g

    if hdu_gal[0].header.get("MEANSUB", True):
        for comp in (0, 1):
            g_cal[comp] -= c_corr[comp]

    hdu_gal.close()
    return g_cal[0], g_cal[1]


def get_calibrate_no_leakage_e_from_cat(path_cat_gal, weight_type="des", verbose=False):
    """
    Calibrate ellipticities and removes leakage from a galaxy catalog with a certain weight type.

    Parameters
    ----------
    path_cat_gal : str
        Path to the galaxy catalog
    weight_type : str, optional, default='des'
        Type of weight to use. Options are 'des' (DES weight) or 'iv' (inverse variance)
    verbose : bool, optional, default=False
        If True, print intermediate results

    Returns
    -------
    e1_noleak : np.array
        Calibrated ellipticities without leakage for the first component
    e2_noleak : np.array
        Calibrated ellipticities without leakage for the second component
    """
    e1, e2 = get_calibrate_e_from_cat(path_cat_gal, weight_type, verbose)

    cat_gal = read_catalogue(
        path_cat_gal, columns=["alpha_1", "alpha_2", "e1_PSF", "e2_PSF"]
    )
    e1_noleak = e1 - cat_gal["alpha_1"] * cat_gal["e1_PSF"]
    e2_noleak = e2 - cat_gal["alpha_2"] * cat_gal["e2_PSF"]

    return e1_noleak, e2_noleak


class metacal:
    """Metacal.

    Metacalibration.

    Parameters
    ----------
    data :
        input galaxy catalogue
    mask : array of bool
        pre-selection mask, e.g. flag, magnitude and footprint cuts; the
        size-based galaxy selection (``rel_size_min`` < T/Tpsf <
        ``rel_size_max``) is applied here
    masking_type : string, optional, default='gal'
        masking type, one in 'gal', 'gal_mom', 'star'
    step : float, optional, default=0.01
        step h in finite differences
    snr_min : float, optional, default=10
        signal-to-noise minimum
    snr_max; float, optional, default=500
        signal-to-noise maximum
    rel_size_min : float, optional, default=0.5
        relative size minimum
    rel_size_max : float, optional, default=3.0
        relative size maximum
    size_corr_ell : bool, optional, default=True
    global_R_weight : str, optional,
        weight column name for global response matrix; default is ``None``
        (unweighted mean)
    sigma_eps : float, optional
        ellipticity dispersion (one component) for computation
        of weights; default is 0.34
    verbose : bool, optional, default=False
        verbose output if True

    """

    def __init__(
        self,
        data,
        mask,
        masking_type="gal",
        step=0.01,
        snr_min=10,
        snr_max=500,
        rel_size_min=0.5,
        rel_size_max=3.0,
        size_corr_ell=True,
        global_R_weight=None,
        sigma_eps=0.34,
        verbose=False,
        weight_scheme=None,
        legacy_response=False,
        des_num_bins=20,
        des_min_count=3,
        des_weight_product=None,
        w_col=None,
    ):
        if weight_scheme is not None and global_R_weight is not None:
            raise ValueError("Use weight_scheme, not both response weight knobs")
        if weight_scheme is None and not legacy_response:
            if global_R_weight == "w":
                warnings.warn(
                    "global_R_weight=w is deprecated; use weight_scheme=iv",
                    DeprecationWarning,
                    stacklevel=2,
                )
                weight_scheme = "iv"
            elif global_R_weight not in (None, "None"):
                raise ValueError("Use weight_scheme=des or iv")
            else:
                legacy_response = True  # Historical unweighted diagnostic API.
        self.weight_scheme = weight_scheme
        self._legacy_response = legacy_response
        self._des_num_bins = des_num_bins
        self._des_min_count = des_min_count
        self._des_weight_product = des_weight_product
        if des_weight_product is not None:
            if weight_scheme != "des" or legacy_response:
                raise ValueError("des_weight_product requires modern weight_scheme=des")
            self.W, self._des_min_count = load_des_weight_product(des_weight_product)
        if w_col is not None:
            response.validate_weight_scheme(weight_scheme, w_col)
        self._masking_type = masking_type
        self._step = step

        # Cuts
        self._snr_min = snr_min
        self._snr_max = snr_max
        self._rel_size_min = rel_size_min
        self._rel_size_max = rel_size_max
        self._size_corr_ell = size_corr_ell
        if verbose:
            print(
                f"Metacal cuts: {snr_min}<snr<{snr_max}, "
                + f"rel_size_min={rel_size_min}, "
                + f"rel_size_max={rel_size_max}, "
                + f"size_corr_ell={size_corr_ell}"
            )

        self._global_R_weight = global_R_weight

        self._sigma_eps = sigma_eps

        self._verbose = verbose

        self._read_data(data, mask)
        self._compute_calibration()

    def _read_data(self, data, mask):
        """Read Data.

        Read relevant data columns.
        """
        m1 = {}
        p1 = {}
        m2 = {}
        p2 = {}
        ns = {}

        masked_data = data[mask]
        m1, p1, m2, p2, ns = self._read_data_ngmix(masked_data, m1, p1, m2, p2, ns)

        self.m1 = m1
        self.p1 = p1
        self.m2 = m2
        self.p2 = p2
        self.ns = ns

    def _read_data_ngmix(self, masked_data, m1, p1, m2, p2, ns):
        """Read Data Ngmix.

        Read data from ngmix catalogue.

        """

        for name_shear, dict_tmp in zip(
            ["1M", "1P", "2M", "2P", "NOSHEAR"], [m1, p1, m2, p2, ns]
        ):
            if self._verbose:
                print("Extracting {}".format(name_shear))

            dict_tmp["flag"] = masked_data[f"NGMIX_FLAGS_{name_shear}"]

            # Ellipticity in named scalar components (ShapePipe-v2 grammar)
            for comp in (0, 1):
                dict_tmp[f"g{comp + 1}"] = masked_data[
                    f"NGMIX_G{comp + 1}_{name_shear}"
                ]

            for key in ("flux", "flux_err", "T", "T_err"):
                dict_tmp[key] = masked_data[f"NGMIX_{key.upper()}_{name_shear}"]

            dict_tmp["Tpsf"] = masked_data[f"NGMIX_T_PSF_RECONV_{name_shear}"]

        ns["C11"], ns["C22"], ns["w"] = self.get_variance_ivweights(
            masked_data,
            self._sigma_eps,
            mask=None,
            legacy=self._legacy_response,
        )
        ns["w_iv"] = ns["w"]
        if self.weight_scheme == "iv" and not self._legacy_response:
            for name, branch in zip(("1M", "1P", "2M", "2P"), (m1, p1, m2, p2)):
                required = [f"NGMIX_G{i}_ERR_{name}" for i in (1, 2)]
                if not all(key in masked_data.dtype.names for key in required):
                    raise ValueError(
                        "IV response needs per-branch G1_ERR/G2_ERR; regenerate "
                        "the comprehensive catalogue from the tile final catalogues"
                    )
                branch["w_iv"] = self.get_variance_ivweights(
                    masked_data, self._sigma_eps, branch=name
                )[2]

        self._n_input = len(masked_data)
        self._n_after_gal_mask = len(dict_tmp["flag"])
        if self._verbose:
            print(f"Number of objects on metacal input = {self._n_input}")
            print(
                "Number of objects after galaxy selection masking ="
                + f" {self._n_after_gal_mask}"
            )

        return m1, p1, m2, p2, ns

    @staticmethod
    def get_variance_ivweights(
        data, sigma_eps, mask=None, legacy=False, branch="NOSHEAR"
    ):
        """Get Variance IVWEIGHTS.

        Compute variance and inverse-variance weights.

        Parameters
        ----------
        data : numpy.ndarray
            input data
        sigma_eps : float
            ellipticity dispersion
        mask : list, optional
            indicates valid objects with ``True`` values; default is ``None`` = use all objects
            type has to be bool

        Returns
        -------
        float
            variance first component
        float
            variance second component
        float
            weight

        """
        C11 = data[f"NGMIX_G1_ERR_{branch}"]
        C22 = data[f"NGMIX_G2_ERR_{branch}"]
        if not legacy:
            C11 = C11**2
            C22 = C22**2
        if mask is not None:
            C11 = C11[mask]
            C22 = C22[mask]

        iv_w = 1 / (2 * sigma_eps**2 + C11 + C22)

        return C11, C22, iv_w

    def _compute_calibration(self):
        """Compute Calibration.

        Perform masking and compute calibration.
        """
        if self._masking_type == "gal":
            self._masking_gal()
        elif self._masking_type == "galmom":
            self._masking_gal_mom()
        elif self._masking_type == "star":
            self._masking_star()
        else:
            raise ValueError(f"Invalid masking type '{self._masking_type}'")

        self._shear_response()
        if self._legacy_response:
            self._selection_response()
            self._total_response()
        else:
            self._prepare_weights()
            result = self.response_estimates()
            self.R_shear_global = result["shear"]
            self.R_selection = result["selection"]
            self.R = result["total"]
            self.R_joint = result["joint"]
            self.c_e, self.c_g = result["c_e"], result["c_g"]
        # self._shear_response_std(stat_operator=lambda x:
        # jackknif_weighted_average(x, np.ones_like(x)))

    def _prepare_weights(self):
        """Case d: fit W once, evaluate each branch's own SNR/size."""
        ma = self.mask_dict["ns"]
        self.branches = {
            b: (getattr(self, b)["g1"], getattr(self, b)["g2"])
            for b in response.BRANCHES
        }
        self.weight_metadata = dict(
            scheme=self.weight_scheme,
            step=self._step,
            estimator="split",
            W_policy="fixed",
            des_min_count=self._des_min_count,
            iv_policy="branch measurement variances",
            c_units="c_e raw; c_g calibrated",
        )
        if self.weight_scheme == "iv":
            self.branch_weights = {
                b: getattr(self, b)["w_iv"] for b in response.BRANCHES
            }
        else:
            snr = self.ns["flux"] / self.ns["flux_err"]
            size = self.ns["T"] / self.ns["Tpsf"]
            if self._des_weight_product is None:
                self.W = fit_w_des(
                    dict(
                        snr=snr[ma],
                        size_ratio=size[ma],
                        e1_uncal=self.ns["g1"][ma],
                        e2_uncal=self.ns["g2"][ma],
                        R_g11=self.R11,
                        R_g22=self.R22,
                    ),
                    self._des_num_bins,
                    self._snr_min,
                    self._snr_max,
                    self._rel_size_min,
                    self._rel_size_max,
                    min_count=self._des_min_count,
                )
            ns_cell = (
                response.cell_index(snr, self.W[0]),
                response.cell_index(size, self.W[1]),
            )
            self.branch_weights = {}
            self.branch_migration = {}
            for b in response.BRANCHES:
                d = getattr(self, b)
                s, r = d["flux"] / d["flux_err"], d["T"] / d["Tpsf"]
                self.branch_weights[b] = response.evaluate_w(self.W, s, r)
                selected = self.mask_dict[b]
                moved = (response.cell_index(s, self.W[0]) != ns_cell[0]) | (
                    response.cell_index(r, self.W[1]) != ns_cell[1]
                )
                self.branch_migration[b] = float(np.mean(moved[selected]))
            self.ns["w_des"] = self.branch_weights["ns"]
        self.ns["w"] = self.branch_weights["ns"]
        self.branch_counts = {b: len(self.mask_dict[b]) for b in response.BRANCHES}
        self.branch_weight_sums = {
            b: self.branch_weights[b][self.mask_dict[b]].sum()
            for b in response.BRANCHES
        }

    def response_estimates(self, keep=None):
        return response.responses(
            self.branches,
            self.mask_dict,
            self.branch_weights,
            self.R_shear,
            self._step,
            keep,
        )

    def refit_weights(self, keep):
        """Refit DES W on retained S_ns, evaluate it on every parent branch.

        Edges and cuts stay fixed; cell moments and membership of the training
        sample change. Artificial branches never train their own table.
        """
        if self.weight_scheme != "des" or self._des_weight_product is not None:
            return self.branch_weights
        idx = self.mask_dict["ns"]
        retained = keep[idx]
        selected = idx[retained]
        snr = self.ns["flux"] / self.ns["flux_err"]
        size = self.ns["T"] / self.ns["Tpsf"]
        table = response.fit_w(
            snr[selected],
            size[selected],
            np.asarray(self.branches["ns"])[:, selected],
            0.5 * (self.R11[retained] + self.R22[retained]),
            self.W[0],
            self.W[1],
            min_count=self._des_min_count,
        )
        return {
            b: response.evaluate_w(
                table,
                getattr(self, b)["flux"] / getattr(self, b)["flux_err"],
                getattr(self, b)["T"] / getattr(self, b)["Tpsf"],
            )
            for b in response.BRANCHES
        }

    def response_jackknife(self, groups, refit_w=False, *, reference=False):
        """Delete spatial groups, optionally refitting W from retained cell sums.

        reference=True keeps the original K full-parent rescans for validation.
        Both paths use the full-sample edges and retained-count pooling rule.
        """
        if refit_w and self.weight_scheme == "des" and self._des_weight_product is None:
            if not reference:
                from sp_validation.des_jackknife import accumulate

                return accumulate(self, groups).jackknife()
            groups = np.asarray(groups)
            if len(groups) != len(self.ns["g1"]):
                raise ValueError("Spatial groups must align to the common parent rows")
            labels = np.unique(groups)
            if len(labels) < 2:
                raise ValueError(
                    "Response jackknife requires at least two spatial groups"
                )
            samples = []
            for label in labels:
                keep = groups != label
                weights = self.refit_weights(keep)
                samples.append(
                    response.responses(
                        self.branches,
                        self.mask_dict,
                        weights,
                        self.R_shear,
                        self._step,
                        keep,
                    )
                )
            estimates = {key: np.array([s[key] for s in samples]) for key in samples[0]}
            vectors = np.concatenate(
                [x.reshape(len(labels), -1) for x in estimates.values()], axis=1
            )
            delta = vectors - vectors.mean(axis=0)
            cov = (len(labels) - 1) / len(labels) * delta.T @ delta
            return labels, estimates, cov
        return response.jackknife(
            self.branches,
            self.mask_dict,
            self.branch_weights,
            self.R_shear,
            self._step,
            np.asarray(groups),
        )

    def add_cuts(self, snr_min=10, snr_max=500, rel_size_min=0.5, rel_size_max=3.0):
        """Add Cuts.

        Apply additional cuts to metacal galaxy catalogue.
        """
        if (
            snr_min < self._snr_min
            or snr_max > self._snr_max
            or rel_size_min < self._rel_size_min
            or rel_size_max > self._rel_size_max
        ):
            print(
                "At least on cut is less stringend than existing one, " + "skipping..."
            )
            return

        self._snr_min = snr_min
        self._snr_max = snr_max
        self._rel_size_min = rel_size_min
        self._rel_size_max = rel_size_max
        if self._verbose:
            print(
                f"Metacal new cuts: {snr_min}<snr<{snr_max}, "
                + f"rel_size_min={rel_size_min}"
            )

        self._compute_calibration()

    def _masking_gal(self):
        """Masking Gal.

        Mask metacal catalogue, i.e. apply cuts.
        """
        self.mask_dict = {}
        for data, name in zip(
            [self.ns, self.m1, self.p1, self.m2, self.p2],
            ["ns", "m1", "p1", "m2", "p2"],
        ):
            Tr_tmp = data["T"] if self._legacy_response else data["T"].copy()
            if self._size_corr_ell:
                Tr_tmp *= (1 - (data["g1"] ** 2 + data["g2"] ** 2)) / (
                    1 + (data["g1"] ** 2 + data["g2"] ** 2)
                )
            if hasattr(self, "snr_sextractor"):
                snr_flux = self.snr_sextractor
            else:
                snr_flux = data["flux"] / data["flux_err"]

            Tpsf = data["Tpsf"]

            mask_tmp = (
                (data["flag"] == 0)
                & (Tr_tmp / Tpsf > self._rel_size_min)
                & (Tr_tmp / Tpsf < self._rel_size_max)
                & (snr_flux > self._snr_min)
                & (snr_flux < self._snr_max)
            )

            # Take care of rotated version
            ind_masked = np.where(mask_tmp == True)[0]

            self.mask_dict[name] = ind_masked

    def _masking_gal_mom(self):
        """Add docstring.

        ...

        """
        self.mask_dict = {}
        for data, name in zip(
            [self.ns, self.m1, self.p1, self.m2, self.p2],
            ["ns", "m1", "p1", "m2", "p2"],
        ):
            Tr_tmp = data["T"]
            if self._size_corr_ell:
                Tr_tmp *= (1 - (data["g1"] ** 2 + data["g2"] ** 2)) / (
                    1 + (data["g1"] ** 2 + data["g2"] ** 2)
                )

            mask_tmp = (
                (data["flag"] == 0)
                & (1 - data["Tpsf"] / data["T"] > self._rel_size_min)
                & (data["s2n"] > self._snr_min)
                & (data["s2n"] < self._snr_max)
                & (data["g1"] != -10)
                & (data["g1"] != 0)
            )

            # Take care of rotated version
            ind_masked = np.where(mask_tmp == True)[0]

            self.mask_dict[name] = ind_masked

    def _masking_star(self):
        """Add docstring.

        ...

        """
        self.mask_dict = {}
        for data, name in zip(
            [self.ns, self.m1, self.p1, self.m2, self.p2],
            ["ns", "m1", "p1", "m2", "p2"],
        ):
            if hasattr(self, "snr_sextractor"):
                snr_flux = self.snr_sextractor
            else:
                snr_flux = data["flux"] / data["flux_err"]
            mask_tmp = (data["flag"] == 0) & (snr_flux > 10) & (snr_flux < 500)

            # Take care of rotated version
            ind_masked = np.where(mask_tmp == True)[0]

            self.mask_dict[name] = ind_masked

    def _shear_response(self):
        """Shear Response.

        Compute shear response matrix
        """
        ma = self.mask_dict["ns"]
        h2 = 2 * self._step

        self.R11 = (self.p1["g1"][ma] - self.m1["g1"][ma]) / h2
        self.R22 = (self.p2["g2"][ma] - self.m2["g2"][ma]) / h2
        self.R12 = (self.p2["g1"][ma] - self.m2["g1"][ma]) / h2
        self.R21 = (self.p1["g2"][ma] - self.m1["g2"][ma]) / h2

        self.R_shear = np.array([[self.R11, self.R12], [self.R21, self.R22]])

    def _shear_response_std(
        self, stat_operator=lambda x: jackknif_weighted_average2(x, np.ones_like(x))
    ):
        """Shear Response Std.

        Standard deviation of shear response
        """
        ma = self.mask_dict["ns"]
        h2 = 2 * self._step

        if len(self.ns["g1"][ma]) == 0:
            self.R_shear_std = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        else:
            self.R11_stds = stat_operator((self.p1["g1"][ma] - self.m1["g1"][ma]) / h2)[
                1
            ]
            self.R22_stds = stat_operator((self.p2["g2"][ma] - self.m2["g2"][ma]) / h2)[
                1
            ]
            self.R12_stds = stat_operator((self.p2["g1"][ma] - self.m2["g1"][ma]) / h2)[
                1
            ]
            self.R21_stds = stat_operator((self.p1["g2"][ma] - self.m1["g2"][ma]) / h2)[
                1
            ]

            self.R_shear_std = np.array(
                [[self.R11_stds, self.R12_stds], [self.R21_stds, self.R22_stds]]
            )

    def _selection_response(self):
        """Add docstring.

        ...

        """
        ma_p1 = self.mask_dict["p1"]
        ma_m1 = self.mask_dict["m1"]
        ma_p2 = self.mask_dict["p2"]
        ma_m2 = self.mask_dict["m2"]
        h2 = 2 * self._step

        self.R11_s = (
            np.mean(self.ns["g1"][ma_p1]) - np.mean(self.ns["g1"][ma_m1])
        ) / h2
        self.R22_s = (
            np.mean(self.ns["g2"][ma_p2]) - np.mean(self.ns["g2"][ma_m2])
        ) / h2
        self.R12_s = (
            np.mean(self.ns["g1"][ma_p2]) - np.mean(self.ns["g1"][ma_m2])
        ) / h2
        self.R21_s = (
            np.mean(self.ns["g2"][ma_p1]) - np.mean(self.ns["g2"][ma_m1])
        ) / h2

        self.R_selection = np.array(
            [[self.R11_s, self.R12_s], [self.R21_s, self.R22_s]]
        )

    def _total_response(self):
        """Add docstring.

        ...

        """
        if self._global_R_weight is None or self._global_R_weight == "None":
            print("Computing unweighted response")
            self.R_shear_global = np.mean(self.R_shear, axis=2)
        else:
            print("Computing response weighted by", self._global_R_weight)
            # Get weights of masked no-shear objects
            weights = self.ns[self._global_R_weight][self.mask_dict["ns"]]
            self.R_shear_global = np.average(self.R_shear, axis=2, weights=weights)

        self.R = self.R_shear_global + self.R_selection


def mask_gal_size(
    T, Tpsf, rel_size_min, rel_size_max, size_corr_ell=False, g1=None, g2=None
):
    Tr_tmp = T
    if size_corr_ell:
        Tr_tmp *= (1 - g1**2 + g2**2) / (1 + g1**2 + g2**2)

    mask = (Tr_tmp / Tpsf > rel_size_min) & (Tr_tmp / Tpsf < rel_size_max)

    return mask


def mask_gal_SNR(SNR, snr_min, snr_max):
    mask = (SNR > snr_min) & (SNR < snr_max)

    return mask
