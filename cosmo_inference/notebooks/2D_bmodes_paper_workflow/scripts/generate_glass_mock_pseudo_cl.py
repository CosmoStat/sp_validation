"""Generate pseudo-Cl for GLASS mock with sqrt binning and noise subtraction.

Standalone script for GLASS mocks that don't use the full CosmologyValidation
pipeline. Uses NaMaster to compute pseudo-Cl with mask and binning matching
real data analysis.

Key features:
- sqrt-spaced binning (power=0.5) for COSEBIS integration
- Analytic noise bias subtraction
- Deconvolution via MASTER algorithm
"""

import os

import healpy as hp
import numpy as np
import pymaster as nmt
from astropy.io import fits


def create_shear_maps(catalog_path: str, nside: int = 1024):
    """Create shear maps from mock catalog.

    Returns:
        shear_e1, shear_e2: Weighted mean shear in each pixel
        n_gal: Number of galaxies (weights) per pixel
        e1, e2, w: Raw ellipticity and weight arrays (for noise calculation)
        unique_pix, idx_rep: Pixel indices (for noise calculation)
    """
    print(f"Loading catalog from {catalog_path}...")
    with fits.open(catalog_path) as hdul:
        data = hdul[1].data
        ra = data["RA"]
        dec = data["Dec"]
        e1 = data["e1"]
        e2 = data["e2"]
        w = data["w"]

    print(f"  {len(ra):,} galaxies")

    theta = (90.0 - dec) * np.pi / 180.0
    phi = ra * np.pi / 180.0
    pix = hp.ang2pix(nside, theta, phi)

    unique_pix, idx_rep = np.unique(pix, return_inverse=True)
    n_gal = np.zeros(hp.nside2npix(nside))
    n_gal[unique_pix] = np.bincount(idx_rep, weights=w)

    shear_e1 = np.zeros(hp.nside2npix(nside))
    shear_e2 = np.zeros(hp.nside2npix(nside))
    shear_e1[unique_pix] = np.bincount(idx_rep, weights=e1 * w)
    shear_e2[unique_pix] = np.bincount(idx_rep, weights=e2 * w)

    mask = n_gal > 0
    shear_e1[mask] /= n_gal[mask]
    shear_e2[mask] /= n_gal[mask]

    print(f"  Created maps with {mask.sum():,} pixels occupied")
    print(f"  Sky fraction: {mask.mean():.3f}")

    return shear_e1, shear_e2, n_gal, e1, e2, w, unique_pix, idx_rep


def get_sqrt_binning(lmin: int, lmax: int, nbins: int):
    """Create sqrt-spaced (power=0.5) binning."""
    power = 0.5
    ells = np.arange(lmin, lmax + 1)

    start = np.power(lmin, power)
    end = np.power(lmax, power)
    bin_edges = np.power(np.linspace(start, end, nbins + 1), 1 / power)

    bpws = np.digitize(ells.astype(float), bin_edges) - 1
    bpws[0] = 0
    bpws[-1] = nbins - 1

    return nmt.NmtBin(ells=ells, bpws=bpws, lmax=lmax - 1)


def compute_variance_map(nside, e1, e2, w, unique_pix, idx_rep):
    """Compute per-pixel variance map for noise estimation.

    Returns variance map: Var(e) * n_eff / n_eff^2 = Var(e) / n_eff per pixel.
    """
    # Weighted variance per pixel
    w_sum = np.zeros(hp.nside2npix(nside))
    w_sum[unique_pix] = np.bincount(idx_rep, weights=w)

    e1_mean = np.zeros(hp.nside2npix(nside))
    e2_mean = np.zeros(hp.nside2npix(nside))
    e1_mean[unique_pix] = np.bincount(idx_rep, weights=e1 * w)
    e2_mean[unique_pix] = np.bincount(idx_rep, weights=e2 * w)

    mask = w_sum > 0
    e1_mean[mask] /= w_sum[mask]
    e2_mean[mask] /= w_sum[mask]

    # Variance: sum(w * (e - mean)^2) / sum(w)
    var_e1 = np.zeros(hp.nside2npix(nside))
    var_e2 = np.zeros(hp.nside2npix(nside))
    var_e1[unique_pix] = np.bincount(idx_rep, weights=w * (e1 - e1_mean[unique_pix[idx_rep]])**2)
    var_e2[unique_pix] = np.bincount(idx_rep, weights=w * (e2 - e2_mean[unique_pix[idx_rep]])**2)

    var_e1[mask] /= w_sum[mask]
    var_e2[mask] /= w_sum[mask]

    # Shape noise power: (var_e1 + var_e2) / 2 / n_eff_per_pixel
    # For pseudo-Cl, noise power = pixel_area * mean(variance_map)
    variance_map = (var_e1 + var_e2) / 2.0

    return variance_map


def compute_pseudo_cl(shear_e1, shear_e2, n_gal, nside: int, nbins: int = 100,
                      e1=None, e2=None, w=None, unique_pix=None, idx_rep=None,
                      subtract_noise: bool = True):
    """Compute pseudo-Cl with sqrt binning and optional noise subtraction."""
    lmin = 8
    lmax = 3 * nside

    b = get_sqrt_binning(lmin, lmax, nbins)
    ell_eff = b.get_effective_ells()

    print(f"Computing pseudo-Cl with {nbins} sqrt-spaced bins...")
    print(f"  ell range: [{lmin}, {lmax}]")
    print(f"  ell_eff range: [{ell_eff.min():.1f}, {ell_eff.max():.1f}]")

    mask = (n_gal > 0).astype(float)

    # Create NaMaster field
    f = nmt.NmtField(mask, maps=[shear_e1, -shear_e2], lmax=lmax - 1)

    # Get workspace for decoupling
    wsp = nmt.NmtWorkspace()
    wsp.compute_coupling_matrix(f, f, b)

    # Compute deconvolved pseudo-Cl
    cl = nmt.compute_full_master(f, f, b, workspace=wsp)

    # cl array is [EE, EB, BE, BB]
    cl_ee = cl[0]
    cl_eb = cl[1]
    cl_be = cl[2]
    cl_bb = cl[3]

    print(f"  Raw C_EE(ell~100): {cl_ee[np.argmin(np.abs(ell_eff - 100))]:.3e}")
    print(f"  Raw C_BB(ell~100): {cl_bb[np.argmin(np.abs(ell_eff - 100))]:.3e}")

    # Noise subtraction
    if subtract_noise and e1 is not None:
        print("  Computing analytic noise bias...")
        variance_map = compute_variance_map(nside, e1, e2, w, unique_pix, idx_rep)
        noise_power = hp.nside2pixarea(nside) * np.mean(variance_map[n_gal > 0])
        print(f"  Noise power level: {noise_power:.3e}")

        # Create noise bias array and decouple it
        noise_bias_coupled = np.zeros((4, lmax))
        noise_bias_coupled[0, :] = noise_power  # EE
        noise_bias_coupled[3, :] = noise_power  # BB

        # Decouple the noise bias
        noise_bias_decoupled = wsp.decouple_cell(noise_bias_coupled)

        # Subtract noise
        cl_ee = cl_ee - noise_bias_decoupled[0]
        cl_bb = cl_bb - noise_bias_decoupled[3]

        print(f"  Noise-subtracted C_EE(ell~100): {cl_ee[np.argmin(np.abs(ell_eff - 100))]:.3e}")
        print(f"  Noise-subtracted C_BB(ell~100): {cl_bb[np.argmin(np.abs(ell_eff - 100))]:.3e}")

    return ell_eff, cl_ee, cl_bb, cl_eb


def save_pseudo_cl(output_path: str, ell, cl_ee, cl_bb, cl_eb):
    """Save pseudo-Cl to FITS format matching real data structure."""
    print(f"Saving to {output_path}...")

    col_ell = fits.Column(name="ELL", format="D", array=ell)
    col_ee = fits.Column(name="EE", format="D", array=cl_ee)
    col_bb = fits.Column(name="BB", format="D", array=cl_bb)
    col_eb = fits.Column(name="EB", format="D", array=cl_eb)

    cols = fits.ColDefs([col_ell, col_ee, col_bb, col_eb])
    hdu = fits.BinTableHDU.from_columns(cols, name="PSEUDO_CELL")

    hdu.header["BINNING"] = "powspace"
    hdu.header["POWER"] = 0.5
    hdu.header["COMMENT"] = "sqrt-spaced binning for COSEBIS comparison"

    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    hdul.writeto(output_path, overwrite=True)
    print(f"  Done")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate pseudo-Cl for GLASS mock")
    parser.add_argument("--mock-id", type=str, default="00001", help="Mock ID (5 digits)")
    parser.add_argument("--nside", type=int, default=1024, help="HEALPix nside")
    parser.add_argument("--nbins", type=int, default=100, help="Number of ell bins")
    parser.add_argument("--output", type=str, help="Output FITS path")
    args = parser.parse_args()

    mock_dir = "/n09data/guerrini/glass_mock_v1.4.6/results"
    catalog_path = f"{mock_dir}/unions_glass_sim_{args.mock_id}_4096.fits"

    if args.output is None:
        output_path = f"results/glass_mock/pseudo_cl_glass_mock_{args.mock_id}_powspace_nbins={args.nbins}.fits"
    else:
        output_path = args.output

    shear_e1, shear_e2, n_gal, e1, e2, w, unique_pix, idx_rep = create_shear_maps(catalog_path, args.nside)
    ell, cl_ee, cl_bb, cl_eb = compute_pseudo_cl(
        shear_e1, shear_e2, n_gal, args.nside, args.nbins,
        e1=e1, e2=e2, w=w, unique_pix=unique_pix, idx_rep=idx_rep,
        subtract_noise=True
    )
    save_pseudo_cl(output_path, ell, cl_ee, cl_bb, cl_eb)

    print("\nSummary:")
    print(f"  Mock: {args.mock_id}")
    print(f"  Output: {output_path}")
    print(f"  Bins: {args.nbins} (sqrt-spaced)")
    print(f"  ell range: [{ell.min():.1f}, {ell.max():.1f}]")


if __name__ == "__main__":
    main()
