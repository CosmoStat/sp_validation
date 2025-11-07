# %%
import argparse
import numpy as np
import treecorr
from astropy.io import fits
import pymaster as nmt

def get_parser():
    """
    Creates the parser
    """
    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description='''
        Selects the simulation on which to run the two-point statistics.
        ''',
        fromfile_prefix_chars='@'
    )

    parser.add_argument("-N", "--number", help="Mock Number", type=int, default=0)
    parser.add_argument("-n", "--nside", help="Nside for the simulation. Nside=Lmax", type=int, default=32)
    parser.add_argument("-p", "--path", help="Path to the simulation data", type=str, default='/n09data/guerrini/glass_mock_v1.4.6/results/')
    parser.add_argument("-s", "--star_cat_path", help="Path to the star catalog data", type=str, default="/n17data/UNIONS/WL/v1.4.x/unions_shapepipe_psf_2024_v1.4.a.fits")
    return parser


def compute_leakage_harmony(cat, cat_star):
    e1 = cat['e1']
    e2 = cat['e2']
    e1_psf = cat_star['E1_PSF_HSM']
    e2_psf = cat_star['E2_PSF_HSM']

    ra = cat['ra']
    dec = cat['dec']
    ra_star = cat_star['RA']
    dec_star = cat_star['DEC']

    del cat

    ra[ra < 0] += 360
    ra_star[ra_star < 0] += 360

    nside = 1024
    lmin = 8
    lmax = 2*nside
    n_bins = 32
    b_lmax = lmax - 1

    ells = np.arange(lmin, lmax+1)

    start = np.power(lmin, 1/2)
    end = np.power(lmax, 1/2)
    bins_ell = np.power(np.linspace(start, end, n_bins +1), 2)

    bpws = np.digitize(ells.astype(float), bins_ell) - 1
    bpws[0] = 0
    bpws[-1] = n_bins - 1

    b = nmt.NmtBin(ells=ells, bpws=bpws, lmax=b_lmax)

    ell_eff = b.get_effective_ells()

    factor = -1

    f_psf = nmt.NmtFieldCatalog(
        positions=[ra_star, dec_star],
        weights=np.ones_like(ra_star),
        field=[e1_psf, factor * e2_psf],
        lmax=b_lmax,
        lmax_mask=b_lmax,
        spin=2,
        lonlat=True
    )

    f_gal = nmt.NmtFieldCatalog(
        positions=[ra, dec],
        weights=np.ones_like(ra),
        field=[e1, factor * e2],
        lmax=b_lmax,
        lmax_mask=b_lmax,
        spin=2,
        lonlat=True
    )

    # Compute rho_0
    wsp = nmt.NmtWorkspace.from_fields(f_psf, f_psf, b)

    rho_cl = nmt.compute_coupled_cell(f_psf, f_psf)
    rho_cl = wsp.decouple_cell(rho_cl)

    rho_cl = np.concatenate([ell_eff[np.newaxis, ...], rho_cl])

    # Compute tau_0
    wsp = nmt.NmtWorkspace.from_fields(f_gal, f_psf, b)

    tau_cl = nmt.compute_coupled_cell(f_gal, f_psf)
    tau_cl = wsp.decouple_cell(tau_cl)

    tau_cl = np.concatenate([ell_eff[np.newaxis, ...], tau_cl])

    return rho_cl, tau_cl

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Load the catalog data
    print("Loading the catalog data...")
    cat = fits.getdata(f"{args.path}/unions_glass_sim_{str(args.number).zfill(5)}_{args.nside}.fits")
    print("Catalog data loaded successfully.")

    print("Loading the star catalog data...")
    cat_star = fits.getdata(f"{args.star_cat_path}")
    print("Star catalog data loaded successfully.")

    print("Computing leakage in harmonic space...")
    rho_cl, tau_cl = compute_leakage_harmony(cat, cat_star)
    print("Leakage computation completed.")

    np.save(f'{args.path}/rho_cl_glass_mock_{str(args.number).zfill(5)}_{args.nside}.npy', rho_cl)
    np.save(f'{args.path}/tau_cl_glass_mock_{str(args.number).zfill(5)}_{args.nside}.npy', tau_cl)

    print(f"Leakage results computed and saved for mock {args.number}.")
    print(f"Results saved to {args.path}/rho_cl_glass_mock_{str(args.number).zfill(5)}_{args.nside}.npy and {args.path}/tau_cl_glass_mock_{str(args.number).zfill(5)}_{args.nside}.npy")