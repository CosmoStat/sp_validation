#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import re
import sys

import numpy as np
from astropy.io import fits

# ============================================================================
# SECTION 1: FITS FORMAT CONVERTERS (data → FITS HDU)
# ============================================================================


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

    z_low = np.loadtxt(filename, usecols=0)
    nstep = z_low[1] - z_low[0]

    z_mid = z_low + nstep / 2
    z_high = np.append(z_low[1:], z_low[-1] + nstep)

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


def treecorr_to_fits(filename1, filename2):
    """Load xi+ and xi- from separate TreeCorr FITS files."""
    xiplus_hdu = fits.open(filename1)
    ximinus_hdu = fits.open(filename2)
    return xiplus_hdu[1], ximinus_hdu[1]


def parse_combined_xi_fits(filepath):
    """Parse combined FITS file with xip/xim columns (mock format)."""
    with fits.open(filepath) as hdul:
        xi_data = hdul[1].data
        xip_vals = xi_data["xip"]
        xim_vals = xi_data["xim"]
        xi_theta = xi_data["ANG"] if "ANG" in xi_data.names else xi_data["meanr"]

    xip_hdu = _create_2pt_hdu(xip_vals, xi_theta, "XI_PLUS", "P+P", "P+P")
    xim_hdu = _create_2pt_hdu(xim_vals, xi_theta, "XI_MINUS", "P+P", "P+P")

    return xip_hdu, xim_hdu


def load_glass_cl(cl_file):
    """
    Load GLASS mock C_ell from .npy file.
    Expected shape: (5, 32) where row 0=ell, row 1=EE, row 4=BB
    """
    cl_block = np.load(cl_file)
    if cl_block.shape[0] < 5:
        raise ValueError(f"Unexpected C_ell array shape {cl_block.shape} for {cl_file}")

    ell = np.asarray(cl_block[0], dtype=np.float64)
    cl_ee = np.asarray(cl_block[1], dtype=np.float64)
    cl_bb = np.asarray(cl_block[4], dtype=np.float64)

    return ell, cl_ee, cl_bb


def glass_cl_to_fits(ell, cl_ee, cl_bb):
    """Convert GLASS C_ell arrays to CosmoSIS FITS HDUs for EE and BB."""
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


def covdat_to_fits(filename_cov_xi, filename_cov_tau=None):
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

    if len(covmat) != len(covmat[0]):
        raise RuntimeError("Covariance matrix is not square")

    cov_hdu = fits.ImageHDU(covmat)

    cov_dict = {
        "COVDATA": "True",
        "EXTNAME": "COVMAT",
        "NAME_0": "XI_PLUS",
        "STRT_0": 0,
        "NAME_1": "XI_MINUS",
        "STRT_1": int(len(covmat) / 2),
    }

    filename_cov_tau and cov_dict.update({
            "NAME_2": "TAU_0_PLUS",
            "STRT_2": len(covmat_xi),
            "NAME_3": "TAU_2_PLUS",
            "STRT_3": len(covmat_xi) + int(len(covmat_tau) / 2),
        })

    for key, value in cov_dict.items():
        cov_hdu.header[key] = value

    return cov_hdu


# ============================================================================
# SECTION 2: VALIDATION & CONSISTENCY CHECKS
# ============================================================================


def check_meanr_consistency(xi_theta, tau_theta, rho_theta, threshold=1.0):
    """Check and report theta consistency across xi, tau, and rho statistics."""
    print("=" * 60)
    print("MEANR CONSISTENCY CHECK")
    print("=" * 60)

    tau_diff = np.abs((tau_theta - xi_theta) / xi_theta) * 100
    rho_diff = np.abs((rho_theta - xi_theta) / xi_theta) * 100

    print(f"Xi theta range: {xi_theta.min():.6f} - {xi_theta.max():.6f} arcmin")
    print(f"Tau theta range: {tau_theta.min():.6f} - {tau_theta.max():.6f} arcmin")
    print(f"Rho theta range: {rho_theta.min():.6f} - {rho_theta.max():.6f} arcmin")
    print()
    print(f"Max tau-xi relative difference: {tau_diff.max():.3f}%")
    print(f"Mean tau-xi relative difference: {tau_diff.mean():.3f}%")
    print(f"Max rho-xi relative difference: {rho_diff.max():.3f}%")
    print(f"Mean rho-xi relative difference: {rho_diff.mean():.3f}%")

    if tau_diff.max() > threshold or rho_diff.max() > threshold:
        print(f"\nWARNING: Meanr differences exceed {threshold}% threshold")
        if tau_diff.max() > threshold:
            print(f"  Tau-xi: max {tau_diff.max():.3f}%")
        if rho_diff.max() > threshold:
            print(f"  Rho-xi: max {rho_diff.max():.3f}%")
    else:
        print(f"✓ All differences below {threshold}% threshold")

    print("=" * 60)
    print()


# ============================================================================
# SECTION 3: COSMOSIS CONFIGURATION GENERATION
# ============================================================================


def _generate_ini_file(
    args,
    template_base,
    priors_file,
    values_file,
    suffix="",
    is_harmonic=False,
):
    """Generate a CosmoSIS INI configuration file from template with modifications."""
    template_path = f"cosmosis_config/{template_base}"
    output_path = f"cosmosis_config/cosmosis_pipeline_{args.cosmosis_root}{suffix}.ini"

    with open(template_path, "r") as f:
        config_content = f.read()

    modifications = []

    relative_fits_file = f"data/{args.cosmosis_root}/cosmosis_{args.cosmosis_root}.fits"
    default_section = (
        f"[DEFAULT]\nSCRATCH = {args.data_dir}\nFITS_FILE = {relative_fits_file}"
    )
    modifications.append((r"^\[DEFAULT\]", default_section))

    output_section = (
        f"[output]\nfilename = %(SCRATCH)s/{args.cosmosis_root}/samples_"
        f"{args.cosmosis_root}{suffix}.txt"
    )
    modifications.append((r"^\[output\]", output_section))

    pipeline_section = (
        f"[pipeline]\nvalues = cosmosis_config/{values_file}\npriors = "
        f"cosmosis_config/{priors_file}"
    )
    modifications.append((r"^\[pipeline\]", pipeline_section))

    if not is_harmonic:
        if args.use_rho_tau:
            like_section = (
                "[2pt_like]\nfile = %(COSMOSIS_DIR)s/likelihood/2pt/2pt_like_xi_sys.py"
                "\ndata_sets=XI_PLUS XI_MINUS TAU_0_PLUS TAU_2_PLUS\nadd_xi_sys=T"
            )
        else:
            like_section = (
                "[2pt_like]\nfile = %(COSMOSIS_DIR)s/likelihood/2pt/2pt_like.py"
                "\ndata_sets=XI_PLUS XI_MINUS"
            )
    else:
        like_section = (
            "[2pt_like]\nfile = %(COSMOSIS_DIR)s/likelihood/2pt/2pt_like.py"
            "\ndata_sets=CELL_EE CELL_BB"
        )

    modifications.append((r"^\[2pt_like\]", like_section))

    poly_section = f"[polychord]\npolychord_outfile_root = {args.cosmosis_root}{suffix}"
    modifications.append((r"^\[polychord\]", poly_section))

    test_section = (
        f"[test]\nsave_dir = %(SCRATCH)s/best_fit/{args.cosmosis_root}{suffix}"
    )
    modifications.append((r"^\[test\]", test_section))

    for pattern, replacement in modifications:
        config_content = re.sub(
            pattern, replacement, config_content, flags=re.MULTILINE
        )

    with open(output_path, "w") as f:
        f.write(config_content)

    config_type = "harmonic-space" if is_harmonic else "real-space"
    print(f"Prepared CosmoSIS {config_type} configuration file in {output_path}")
    print(f"You can now run inference with the command: cosmosis {output_path}")


def generate_cosmosis_config(args):
    """Generate CosmoSIS INI files (real-space and optional harmonic-space)."""
    if args.use_rho_tau:
        template_base_realspace = "cosmosis_pipeline_A_psf.ini"
        priors_file = "priors_psf.ini"
        values_file = "values_psf.ini"
    else:
        template_base_realspace = "cosmosis_pipeline_A_ia.ini"
        priors_file = "priors.ini"
        values_file = "values_ia.ini"

    os.makedirs("cosmosis_config", exist_ok=True)

    _generate_ini_file(
        args,
        template_base_realspace,
        priors_file,
        values_file,
        suffix="",
        is_harmonic=False,
    )

    if args.cl_file:
        template_base_harmonic = "cosmosis_pipeline_A_ia_cell.ini"
        _generate_ini_file(
            args,
            template_base_harmonic,
            priors_file,
            values_file,
            suffix="_cell",
            is_harmonic=True,
        )


# ============================================================================
# SECTION 4: MAIN WORKFLOW
# ============================================================================


def parse_args():
    """Parse command-line arguments for unified data/mock interface."""
    parser = argparse.ArgumentParser(
        description="Prepare CosmoSIS inference FITS files from real or mock data. "
        "Supports multiple xi input formats (separate files, combined FITS)."
    )

    parser.add_argument(
        "--cosmosis-root", type=str, required=True, help="CosmoSIS root identifier"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Output MCMC chain directory"
    )
    parser.add_argument("--nz-file", type=str, required=True, help="Path to n(z) file")
    parser.add_argument(
        "--out-file", type=str, required=True, help="Path to output FITS file"
    )
    parser.add_argument(
        "--xi",
        nargs="+",
        required=True,
        help="Xi files: 1 (mock FITS) or 2 (data: xi_plus.fits, xi_minus.fits)",
    )
    parser.add_argument(
        "--cov-xi", type=str, required=True, help="Xi covariance matrix file"
    )

    parser.add_argument(
        "--use-rho-tau",
        action="store_true",
        help="Include PSF systematics (requires --rho-stats, --tau-stats, --cov-tau)",
    )
    parser.add_argument(
        "--rho-stats",
        type=str,
        required=False,
        help="Path to rho statistics FITS file (required if --use-rho-tau)",
    )
    parser.add_argument(
        "--tau-stats",
        type=str,
        required=False,
        help="Path to tau statistics FITS file (required if --use-rho-tau)",
    )
    parser.add_argument(
        "--cov-tau",
        type=str,
        required=False,
        help="Path to tau covariance matrix (required if --use-rho-tau)",
    )
    parser.add_argument(
        "--cl-file",
        type=str,
        required=False,
        help="Path to C_ell data file (.npy, optional for data and mock)",
    )
    parser.add_argument(
        "--mock", action="store_true", help="Mock data mode"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        print("=" * 60)
        print("COSMOSIS_FITTING.PY")
        print("=" * 60)
        print(f"cosmosis_root: {args.cosmosis_root}")
        print(f"data_dir: {args.data_dir}")
        print(f"nz_file: {args.nz_file}")
        print(f"out_file: {args.out_file}")
        print(f"xi files: {args.xi}")
        print(f"cov_xi: {args.cov_xi}")
        print(f"use_rho_tau: {args.use_rho_tau}")
        if args.use_rho_tau:
            print(f"rho_stats: {args.rho_stats}")
            print(f"tau_stats: {args.tau_stats}")
            print(f"cov_tau: {args.cov_tau}")
        if args.cl_file:
            print(f"cl_file: {args.cl_file}")
        print("=" * 60)
        print()

        if args.use_rho_tau and not all([args.rho_stats, args.tau_stats, args.cov_tau]):
            raise ValueError(
                "--use-rho-tau requires: --rho-stats, --tau-stats, --cov-tau"
            )

        os.makedirs(args.data_dir, exist_ok=True)
        output_dir = os.path.dirname(args.out_file)
        os.makedirs(output_dir, exist_ok=True)

        print("Loading xi correlation functions...")
        if args.mock:
            xip_hdu, xim_hdu = treecorr_to_fits(args.xi[0], args.xi[1])
        else:
            xip_hdu, xim_hdu = parse_combined_xi_fits(args.xi[0])

        xi_theta = xip_hdu.data["ANG"]
        print(f"Loaded xi: {len(xip_hdu.data)} bins")

        print("Loading covariance matrix...")
        cov_hdu = covdat_to_fits(args.cov_xi, filename_cov_tau=None)
        print(f"Loaded covariance: shape {cov_hdu.data.shape}")

        print("Loading n(z)...")
        nz_hdu = nz_to_fits(args.nz_file)
        print("Loaded n(z)")

        cl_ee_hdu = None
        cl_bb_hdu = None
        if args.cl_file:
            print("Loading Cl data...")
            if args.cl_file.endswith(".npy"):
                ell, cl_ee, cl_bb = load_glass_cl(args.cl_file)
                cl_ee_hdu, cl_bb_hdu = glass_cl_to_fits(ell, cl_ee, cl_bb)
                print(f"Loaded Cl: {len(ell)} multipoles")
            else:
                raise NotImplementedError(f"Cl format not supported: {args.cl_file}")

        rho_hdu = None
        tau_0_p_hdu = None
        tau_2_p_hdu = None
        if args.use_rho_tau:
            print("Loading rho/tau statistics...")
            tau_stats = fits.getdata(args.tau_stats)
            rho_stats = fits.getdata(args.rho_stats)
            tau_theta = tau_stats["theta"]
            rho_theta = rho_stats["theta"]

            check_meanr_consistency(xi_theta, tau_theta, rho_theta, threshold=5.0)
            print("✓ Forcing rho and tau to use xi meanr values for consistency")

            rho_hdu = rho_to_fits(args.rho_stats, theta=xi_theta)
            tau_0_p_hdu, tau_2_p_hdu = tau_to_fits(args.tau_stats, theta=xi_theta)
            print("Loaded rho/tau statistics")

            cov_hdu = covdat_to_fits(args.cov_xi, filename_cov_tau=args.cov_tau)

        pri_hdr = fits.Header()
        pri_hdu = fits.PrimaryHDU(header=pri_hdr)

        print("Assembling FITS file...")
        hdu_list = [pri_hdu, cov_hdu, nz_hdu, xip_hdu, xim_hdu]

        if args.cl_file:
            hdu_list.extend([cl_ee_hdu, cl_bb_hdu])

        if args.use_rho_tau:
            hdu_list.extend([tau_0_p_hdu, tau_2_p_hdu, rho_hdu])

        hdul = fits.HDUList(hdu_list)
        hdul.writeto(args.out_file, overwrite=True)
        print(f"✓ FITS file written to {args.out_file}")
        print()

        generate_cosmosis_config(args)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
