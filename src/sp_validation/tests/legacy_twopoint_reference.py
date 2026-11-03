"""Frozen pre-SACC FITS builders; byte-compatibility oracle, never production code."""

import numpy as np
from astropy.io import fits


def _create_2pt_hdu(values, theta, name, quant1, quant2):
    """Create standardized 2-point correlation FITS HDU."""
    nbins = len(values)
    lst = np.arange(1, nbins + 1)

    col1 = fits.Column(name="BIN1", format="K", array=np.ones(nbins))
    col2 = fits.Column(name="BIN2", format="K", array=np.ones(nbins))
    col3 = fits.Column(name="ANGBIN", format="K", array=lst)
    col4 = fits.Column(name="VALUE", format="D", array=values)
    col5 = fits.Column(name="ANG", format="D", unit="arcmin", array=theta)

    coldefs = fits.ColDefs([col1, col2, col3, col4, col5])
    hdu = fits.BinTableHDU.from_columns(coldefs, name=name)

    hdu_dict = {
        "2PTDATA": "T",
        "QUANT1": quant1,
        "QUANT2": quant2,
        "KERNEL_1": "NZ_SOURCE",
        "KERNEL_2": "NZ_SOURCE",
        "WINDOWS": "SAMPLE",
    }
    for key, value in hdu_dict.items():
        hdu.header[key] = value

    return hdu


def nz_to_fits(filename):
    """Convert n(z) text file to FITS format."""
    line = np.loadtxt(filename, max_rows=1)
    nbins = len(line) - 1

    z_mid = np.loadtxt(filename, usecols=0)
    nstep = z_mid[1] - z_mid[0]

    z_low = z_mid - nstep / 2
    z_high = z_mid + nstep / 2

    col1 = fits.Column(name="Z_LOW", format="D", array=z_low)
    col2 = fits.Column(name="Z_MID", format="D", array=z_mid)
    col3 = fits.Column(name="Z_HIGH", format="D", array=z_high)
    cols = [col1, col2, col3]

    for i in range(nbins):
        bin_col = np.loadtxt(filename, usecols=i + 1)
        hdu_col = fits.Column(name="BIN%d" % (i + 1), format="D", array=bin_col)
        cols.append(hdu_col)

    coldefs = fits.ColDefs(cols)
    nz_hdu = fits.BinTableHDU.from_columns(coldefs, name="NZDATA")

    nz_lens_dict = {
        "NZDATA": "T  ",
        "EXTNAME": "NZ_SOURCE",
        "NBIN": nbins,
        "NZ": len(z_low),
    }

    for key, value in nz_lens_dict.items():
        nz_hdu.header[key] = value

    return nz_hdu


def load_pseudo_cl(cl_file):
    """Load pseudo-C_ell spectra from .npy or FITS formats."""
    if cl_file.endswith(".npy"):
        cl_block = np.load(cl_file)
        if cl_block.shape[0] < 5:
            raise ValueError(
                f"Unexpected C_ell array shape {cl_block.shape} for {cl_file}"
            )
        ell = np.asarray(cl_block[0], dtype=np.float64)
        cl_ee = np.asarray(cl_block[1], dtype=np.float64)
        cl_bb = np.asarray(cl_block[4], dtype=np.float64)
        return ell, cl_ee, cl_bb

    if cl_file.endswith(".fits"):
        with fits.open(cl_file) as hdul:
            data = hdul[1].data
            ell = np.asarray(data["ELL"], dtype=np.float64)
            cl_ee = np.asarray(data["EE"], dtype=np.float64)
            cl_bb = np.asarray(data["BB"], dtype=np.float64)
        return ell, cl_ee, cl_bb

    raise NotImplementedError(f"Cl format not supported: {cl_file}")


def cl_to_fits(ell, cl_ee, cl_bb):
    """Convert C_ell arrays to CosmoSIS FITS HDUs for EE and BB."""
    nbins = len(ell)
    lst = np.arange(1, nbins + 1)

    col1 = fits.Column(name="BIN1", format="K", array=np.ones(nbins))
    col2 = fits.Column(name="BIN2", format="K", array=np.ones(nbins))
    col3 = fits.Column(name="ANGBIN", format="K", array=lst)
    col5 = fits.Column(name="ANG", format="D", array=ell)

    col4_ee = fits.Column(name="VALUE", format="D", array=cl_ee)
    coldefs_ee = fits.ColDefs([col1, col2, col3, col4_ee, col5])
    cl_ee_hdu = fits.BinTableHDU.from_columns(coldefs_ee, name="CELL_EE")

    cl_ee_dict = {
        "2PTDATA": "T",
        "QUANT1": "GEF",
        "QUANT2": "GEF",
        "KERNEL_1": "NZ_SOURCE",
        "KERNEL_2": "NZ_SOURCE",
        "WINDOWS": "SAMPLE",
    }
    for key, value in cl_ee_dict.items():
        cl_ee_hdu.header[key] = value

    col4_bb = fits.Column(name="VALUE", format="D", array=cl_bb)
    coldefs_bb = fits.ColDefs([col1, col2, col3, col4_bb, col5])
    cl_bb_hdu = fits.BinTableHDU.from_columns(coldefs_bb, name="CELL_BB")

    cl_bb_dict = {
        "2PTDATA": "T",
        "QUANT1": "GBF",
        "QUANT2": "GBF",
        "KERNEL_1": "NZ_SOURCE",
        "KERNEL_2": "NZ_SOURCE",
        "WINDOWS": "SAMPLE",
    }
    for key, value in cl_bb_dict.items():
        cl_bb_hdu.header[key] = value

    return cl_ee_hdu, cl_bb_hdu


def cov_cl_to_fits(cov_file, cov_hdu="COVAR_FULL"):
    """Convert pseudo-C_ell covariance to a CosmoSIS ImageHDU."""
    if cov_file.endswith(".fits"):
        with fits.open(cov_file) as hdul:
            cov_data = np.asarray(hdul[cov_hdu].data, dtype=np.float64)
    elif cov_file.endswith(".npy"):
        cov_data = np.load(cov_file)
    else:
        raise NotImplementedError(
            f"Unsupported pseudo-Cl covariance format: {cov_file}"
        )

    if cov_data.shape[0] != cov_data.shape[1]:
        raise ValueError("Pseudo-Cl covariance matrix must be square")

    cov_hdu = fits.ImageHDU(cov_data, name="COVMAT_CELL")
    cov_dict = {
        "COVDATA": "True",
        "EXTNAME": "COVMAT_CELL",
        "NAME_0": "CELL_EE",
        "STRT_0": 0,
    }

    for key, value in cov_dict.items():
        cov_hdu.header[key] = value

    return cov_hdu


def tau_to_fits(filename, theta=None):
    """
    Convert tau statistics to FITS format.
    If theta provided, override original values for consistency with xi.
    """
    tau_stats = fits.getdata(filename)

    if theta is not None:
        ang = theta
        print("Using provided theta values for tau statistics (forcing consistency)")
    else:
        ang = tau_stats["theta"]
        print("Using original tau theta values")

    tau_0_p_hdu = _create_2pt_hdu(tau_stats["tau_0_p"], ang, "TAU_0_PLUS", "G+R", "P+R")
    tau_2_p_hdu = _create_2pt_hdu(
        tau_stats["tau_2_p"], ang, "TAU_2_PLUS", "G+R", "SR+R"
    )

    return tau_0_p_hdu, tau_2_p_hdu


def rho_to_fits(filename, theta=None):
    """
    Convert rho statistics to FITS format.
    If theta provided, override original values for consistency with xi.
    """
    rho_stat_hdul = fits.open(filename)
    rho_stat_hdu = rho_stat_hdul[1].copy()
    rho_stat_hdu.name = "RHO_STATS"

    if theta is not None:
        print(
            "Forcing rho statistics to use provided theta values (forcing consistency)"
        )
        rho_stat_hdu.data = rho_stat_hdu.data.copy()
        rho_stat_hdu.data["theta"] = theta
    else:
        print("Using original rho theta values")

    rho_stat_hdul.close()
    return rho_stat_hdu


def covdat_to_fits(
    filename_cov_xi, filename_cov_tau=None, filename_cov_cl=None, cov_hdu=None
):
    """
    Convert CosmoCov covariance matrix to FITS format.

    If tau covariance provided, block with xi covariance.
    """
    covmat_xi = np.loadtxt(filename_cov_xi)

    if filename_cov_tau is not None:
        covmat_tau = np.load(filename_cov_tau)
        nbins = int(len(covmat_tau) / 3)
        covmat_tau = covmat_tau[: 2 * nbins, : 2 * nbins]
        covmat = np.block(
            [
                [covmat_xi, np.zeros((len(covmat_xi), len(covmat_tau)))],
                [np.zeros((len(covmat_tau), len(covmat_xi))), covmat_tau],
            ]
        )
    else:
        covmat = covmat_xi

    if filename_cov_cl is not None:
        if filename_cov_cl.endswith(".fits"):
            with fits.open(filename_cov_cl) as hdul:
                cov_data = np.asarray(hdul[cov_hdu].data, dtype=np.float64)
        elif filename_cov_cl.endswith(".npy"):
            cov_data = np.load(filename_cov_cl)
        else:
            raise NotImplementedError(
                f"Unsupported pseudo-Cl covariance format: {filename_cov_cl}"
            )
        covmat = np.block(
            [
                [covmat, np.zeros((len(covmat), len(cov_data)))],
                [np.zeros((len(cov_data), len(covmat))), cov_data],
            ]
        )

    if len(covmat) != len(covmat[0]):
        raise RuntimeError("Covariance matrix is not square")

    cov_hdu = fits.ImageHDU(covmat)

    cov_dict = {
        "COVDATA": "True",
        "EXTNAME": "COVMAT",
        "NAME_0": "XI_PLUS",
        "STRT_0": 0,
        "NAME_1": "XI_MINUS",
        "STRT_1": int(len(covmat_xi) / 2),
    }

    if filename_cov_tau:
        cov_dict.update(
            {
                "NAME_2": "TAU_0_PLUS",
                "STRT_2": len(covmat_xi),
                "NAME_3": "TAU_2_PLUS",
                "STRT_3": len(covmat_xi) + int(len(covmat_tau) / 2),
            }
        )

    filename_cov_cl and cov_dict.update(
        {
            "NAME_4": "CELL_EE",
            "STRT_4": len(covmat_xi) + (len(covmat_tau) if filename_cov_tau else 0),
        }
    )

    for key, value in cov_dict.items():
        cov_hdu.header[key] = value

    return cov_hdu
