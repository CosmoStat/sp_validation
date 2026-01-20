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
    return parser

treecorr_config = {
    "ra_units": "degrees",
    "dec_units": "degrees",
    "min_sep": 1.0,
    "max_sep": 250.0,
    "nbins": 20,
    "sep_units": "arcmin",
    "num_threads": 4
}

def compute_two_point_xi(cat):
    e1 = cat['e1']
    e2 = cat['e2']

    treecorr_cat = treecorr.Catalog(
        ra=cat['ra'],
        dec=cat['dec'],
        g1=e1,
        g2=e2,
        ra_units='degrees',
        dec_units='degrees'
    )

    gg = treecorr.GGCorrelation(treecorr_config)

    gg.process(treecorr_cat)
    return gg

def compute_two_point_cl(cat):
    e1 = cat['e1']
    e2 = cat['e2']

    ra = cat['ra']
    dec = cat['dec']

    del cat

    ra[ra < 0] += 360

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

    f_all = nmt.NmtFieldCatalog(
        positions=[ra, dec],
        weights = np.ones_like(e1),
        field=[e1, factor * e2],
        lmax = b_lmax,
        lmax_mask = b_lmax,
        spin=2,
        lonlat=True
    )
    
    wsp = nmt.NmtWorkspace.from_fields(f_all, f_all, b)

    cl_coupled = nmt.compute_coupled_cell(f_all, f_all)
    cl_all = wsp.decouple_cell(cl_coupled)

    cl_all = np.concatenate([ell_eff[np.newaxis, ...], cl_all])
    
    return cl_all

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Load the catalog data
    print("Loading the catalog data...")
    cat = fits.getdata(f"{args.path}/unions_glass_sim_{str(args.number).zfill(5)}_{args.nside}.fits")
    print("Catalog data loaded successfully.")

    # Compute two-point statistics
    print("Computing two-point statistics in configuration space...")
    xi = compute_two_point_xi(cat)
    print("Two-point statistics computed successfully.")
    # Save results
    xi.write(f"{args.path}/xi_glass_mock_{str(args.number).zfill(5)}_{args.nside}_nbins=20.fits")
    print("Computing two-point statistics in harmonic space...")
    cl = compute_two_point_cl(cat)
    print("Two-point statistics in harmonic space computed successfully.")
    #save results
    np.save(f"{args.path}/cl_glass_mock_{str(args.number).zfill(5)}_{args.nside}.npy", cl)

    print(f"Two-point statistics computed and saved for mock {args.number}.")
    print(f"Results saved to {args.path}/xi_glass_mock_{str(args.number).zfill(5)}_{args.nside}.npy and {args.path}/cl_glass_mock_{str(args.number).zfill(5)}_{args.nside}.npy")
