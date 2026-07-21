"""
Executable script to run Gaussian simulations of shear maps and compute their power spectra using the NaMaster library.
The execution is distributed across multiple MPI processes to speed up the computation.

To run the script, use the following command:
    mpirun -np <number_of_processes> python run_cl_gaussian_sims.py --config <path_to_config_file> --output <output_directory> --version <version_name> [options]

Options:
    -c, --config: Path to the configuration file to generate the simulation (required).
    -o, --output: Output directory for the simulation results (required, ideally the same as cosmo_val).
    -v, --version: Version name for the simulation (required).
    -t, --tomo: Whether to run the simulation in tomographic mode (default: False).
    -iw, --ignore-warnings: Ignore warnings during the simulation.
    -n, --n_sims: Number of simulations to run (default: 100).
    -ns, --nside: Nside for the HEALPix maps (default: 1024).
    -b, --binning: Binning scheme for the power spectra (default: powspace).
    -es, --ell_step: Step size for the linear binning (default: 10).
    -eb, --ell_bins: Number of bins for the power spectra (default: 32).
    -p, --power: Power for the powspace binning (default: 0.5).
    -cc, --cosmo-config: Path to the cosmology configuration file (default: None, use Planck18 cosmology by default).
    -f, --force: Force overwrite of existing output files (default: False).

Author: Sacha Guerrini
"""

import argparse
import itertools
import os
import warnings
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt
import yaml
from astropy.io import fits
from cs_util.cosmo import get_cosmo
from mpi4py import MPI
from tqdm import tqdm

import sp_validation.pseudo_cl as spv_pseudo_cl

pol_to_pol_index_dict = {"EE": 0, "EB": 1, "BE": 2, "BB": 3}


def get_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate Gaussian simulations of shear maps and compute their power spectra.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to the configuration file to generate the simulation",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory for the simulation results (Ideally the same than cosmo_val)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-v", "--version", help="Run some validation checks", type=str, required=True
    )
    parser.add_argument(
        "-t",
        "--tomo",
        help="Whether to run the simulation in tomographic mode (default: False)",
        action="store_true",
    )
    parser.add_argument(
        "-iw",
        "--ignore-warnings",
        help="Ignore warnings during the simulation",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--n_sims",
        help="Number of simulations to run (default: 100)",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-ns",
        "--nside",
        help="Nside for the HEALPix maps (default: 1024)",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "-b",
        "--binning",
        help="Binning scheme for the power spectra (default: powspace)",
        type=str,
        default="powspace",
    )
    parser.add_argument(
        "-es",
        "--ell_step",
        help="Step size for the linear binning (default: 10)",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-eb",
        "--ell_bins",
        help="Number of bins for the power spectra (default: 32)",
        type=int,
        default=32,
    )
    parser.add_argument(
        "-p",
        "--power",
        help="Power for the powspace binning (default: 0.5)",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "-cc",
        "--cosmo-config",
        help="Path to the cosmology configuration file (default: None, use Planck18 cosmology by default)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-f",
        "--force",
        help="Force overwrite of existing output files (default: False)",
        action="store_true",
    )
    parser.add_argument(
        "-seb",
        "--save-eb",
        help="Save the EB covariance matrix (default: False)",
        action="store_true",
    )
    parser.add_argument(
        "-sbb",
        "--save-bb",
        help="Save the BB covariance matrix (default: False)",
        action="store_true",
    )
    return parser


# --- Helper function for reading configuration and extracting useful data ---
def load_version_config(config, version, tomography):
    if not os.path.exists(config):
        raise FileNotFoundError(f"Config file {config} does not exist.")
    else:
        # Load useful information from the config file
        with open(config, "r") as file:
            cc = yaml.load(file, Loader=yaml.FullLoader)

            # Check that the version exists in the config file
            if version not in cc:
                raise ValueError(f"Version {version} not found in config file.")
            else:
                subdir = Path(cc[version]["subdir"])
                cat_path = str(subdir / Path(cc[version]["shear"]["path"]))

                ra_col = cc[version]["shear"]["cols"].split(",")[0]
                dec_col = cc[version]["shear"]["cols"].split(",")[1]
                e1_col = cc[version]["shear"]["e1_col"]
                e2_col = cc[version]["shear"]["e2_col"]
                w_col = cc[version]["shear"]["w_col"]
                tomo_bin_col = None
                if tomography:
                    tomo_bin_col = cc[version]["shear"]["tomo_bin_col"]

                redshift_path = Path(cc[version]["shear"]["redshift_path"])
                return (
                    cat_path,
                    ra_col,
                    dec_col,
                    e1_col,
                    e2_col,
                    w_col,
                    tomo_bin_col,
                    redshift_path,
                )


def load_catalog_slices(cat_path, columns, tomography, tomo_bin_col):
    cat_gal = fits.getdata(cat_path)

    ra, dec, e1, e2, w = {}, {}, {}, {}, {}

    ra_col, dec_col, e1_col, e2_col, w_col = columns

    if tomography:
        tomo_bin_ids = np.unique(cat_gal[tomo_bin_col])
    else:
        tomo_bin_ids = ["all"]

    for bin_id in tomo_bin_ids:
        if bin_id == "all":
            mask_bin = np.ones(len(cat_gal), dtype=bool)
        else:
            mask_bin = cat_gal[tomo_bin_col] == bin_id
        ra[bin_id] = cat_gal[ra_col][mask_bin]
        dec[bin_id] = cat_gal[dec_col][mask_bin]
        e1[bin_id] = cat_gal[e1_col][mask_bin]
        e2[bin_id] = cat_gal[e2_col][mask_bin]
        w[bin_id] = cat_gal[w_col][mask_bin]

    del cat_gal  # Free memory

    return ra, dec, e1, e2, w, tomo_bin_ids


def build_pre_computed_maps(ra, dec, w, tomo_bin_ids, nside):
    n_gal, unique_pix, idx, idx_rep = {}, {}, {}, {}

    for bin_id in tomo_bin_ids:
        unique_pix[bin_id], idx[bin_id], idx_rep[bin_id] = spv_pseudo_cl.get_pixels(
            ra[bin_id], dec[bin_id], nside
        )
        n_gal[bin_id] = spv_pseudo_cl.get_n_gal_map(
            nside,
            ra[bin_id],
            dec[bin_id],
            weights=w[bin_id],
            unique_pix=unique_pix[bin_id],
            idx=idx[bin_id],
            idx_rep=idx_rep[bin_id],
        )

    return n_gal, unique_pix, idx, idx_rep


def build_fiducial_cl(cosmo_params, redshift_path, lmax, tomography, tomo_bin_ids):
    cosmo = get_cosmo(**cosmo_params)

    # --- Load the redshift distribution ---
    redshift_distr = np.loadtxt(redshift_path)
    z = redshift_distr[:, 0]
    dndz = redshift_distr[:, 1:]

    if not tomography and dndz.shape[1] > 1:
        # Sum the redshift distributions across all bins to get the total distribution
        dndz = np.sum(dndz, axis=1, keepdims=True)

        # Normalize the summed distribution
        dndz /= np.trapezoid(dndz[:, 0], z)

    if tomography and dndz.shape[1] != len(tomo_bin_ids):
        raise ValueError(
            "Mismatch between the number of tomographic bins and the redshift distributions provided."
        )

    fiducial_cl = spv_pseudo_cl.get_fiducial_cl(z, dndz, lmax, cosmo)

    # healpy.synalm expects spectra arrays with an explicit ell=0 slot.
    # The theory helper returns ell=1..lmax, so prepend a zero monopole here
    # to keep the Gaussian map synthesis numerically stable.
    fiducial_cl = {
        key: np.concatenate(([0.0], np.asarray(value, dtype=float)))
        for key, value in fiducial_cl.items()
    }

    if not tomography:
        fiducial_cl = {"WallxWall": fiducial_cl["W1xW1"]}

    return cosmo, fiducial_cl


def prepare_workspace(n_gal, tomo_bin_ids, nside, b, b_lmax, out_dir, force_run):
    """Precomputes the workspace objects for each pair of tomographic bins and saves them to disk."""
    f = {}
    for bin_id in tqdm(tomo_bin_ids, desc="Computing fields"):
        f[bin_id] = nmt.NmtField(
            mask=n_gal[bin_id],
            maps=[np.zeros(hp.nside2npix(nside)), np.zeros(hp.nside2npix(nside))],
            lmax=b_lmax,
        )

    # Compute wsp object for each pair of tomographic bins
    tomo_bin_pairs = list(itertools.combinations_with_replacement(tomo_bin_ids, 2))
    for tomo_bin_a, tomo_bin_b in tqdm(tomo_bin_pairs, desc="Computing workspaces"):
        wsp_file = os.path.join(out_dir, f"wsp_{tomo_bin_a}_{tomo_bin_b}.npz")
        if os.path.exists(wsp_file) and not force_run:
            print(
                f"Workspace for bins {tomo_bin_a} and {tomo_bin_b} already exists. Skipping."
            )
            continue
        wsp = nmt.NmtWorkspace(f[tomo_bin_a], f[tomo_bin_b], b)
        wsp.write_to(wsp_file)


# --- Function to generate Gaussian simulation, add noise and extract the spectra ---
def get_gaussian_simulation(
    nside,
    lmax,
    fiducial_cl,
    n_gal,
    unique_pix,
    idx,
    idx_rep,
    ra,
    dec,
    e1,
    e2,
    w,
    tomo_bin_ids,
):
    final_maps = {}
    for bin_id in tomo_bin_ids:
        noise_map_e1, noise_map_e2 = spv_pseudo_cl.get_noise_realisation(
            ra[bin_id],
            dec[bin_id],
            e1[bin_id],
            e2[bin_id],
            w[bin_id],
            nside,
            unique_pix[bin_id],
            idx[bin_id],
            idx_rep[bin_id],
            n_gal_map=n_gal[bin_id],
        )
        final_maps[bin_id] = noise_map_e1 + 1j * noise_map_e2

    # Generate a Gaussian field with the fiducial power spectrum
    tomo_bin_pairs = list(itertools.combinations_with_replacement(tomo_bin_ids, 2))
    # Organise the cls in a list to feed the helpy function
    cl_list = [
        fiducial_cl[f"W{tomo_bin_a}xW{tomo_bin_b}"]
        for tomo_bin_a, tomo_bin_b in tomo_bin_pairs
    ]
    # Generate alms with the correct cross-bin correlation
    alm_shear = hp.synalm(cl_list, lmax=lmax)

    for bin_id in tomo_bin_ids:
        if bin_id != "all":
            gauss_map = hp.alm2map(
                [
                    np.zeros_like(alm_shear[bin_id - 1]),
                    alm_shear[bin_id - 1],
                    np.zeros_like(alm_shear[bin_id - 1]),
                ],
                nside=nside,
                verbose=False,
            )
        else:
            gauss_map = hp.alm2map(
                [
                    np.zeros_like(alm_shear[0]),
                    alm_shear[0],
                    np.zeros_like(alm_shear[0]),
                ],
                nside=nside,
                verbose=False,
            )

        mask = n_gal[bin_id] > 0
        final_maps[bin_id][mask] += gauss_map[1][mask] + 1j * gauss_map[2][mask]

    return final_maps


def extract_spectra(noisy_gaussian_maps, n_gal, tomo_bin_ids, lmax, out_dir):
    # Create the NmtField for each tomo bin
    f = {}
    for bin_id in tomo_bin_ids:
        # Compute the power spectrum
        f[bin_id] = nmt.NmtField(
            mask=n_gal[bin_id],
            maps=[noisy_gaussian_maps[bin_id].real, noisy_gaussian_maps[bin_id].imag],
            lmax=lmax,
        )

    # Compute the power spectrum for each bin pair
    cl_final = {}
    tomo_bin_pairs = list(itertools.combinations_with_replacement(tomo_bin_ids, 2))
    for tomo_bin_a, tomo_bin_b in tomo_bin_pairs:
        cl_coupled = nmt.compute_coupled_cell(f[tomo_bin_a], f[tomo_bin_b])

        wsp = nmt.NmtWorkspace()
        wsp_file = os.path.join(out_dir, f"wsp_{tomo_bin_a}_{tomo_bin_b}.npz")
        wsp.read_from(wsp_file)
        cl_decoupled = wsp.decouple_cell(cl_coupled)
        cl_final[f"W{tomo_bin_a}xW{tomo_bin_b}"] = cl_decoupled

    return cl_final


# --- single unit of work distributed to the MPI processes ---
def run_one_simulation(sim_id, nside, version, tomography, out_dir, force_run):
    """Worker executes this simulation and saves results"""
    try:
        out_file = f"{out_dir}/cl_sample_{sim_id}_{version}_tomography_{tomography}.npz"
        if os.path.exists(out_file) and not force_run:
            print(f"Rank {rank} skipping {sim_id} (already exists)")
            return

        print(f"Rank {rank} starting {sim_id}")
        setup_file = os.path.join(
            out_dir, f"precomputed_setup_{version}_tomography_{tomography}.npz"
        )
        data = np.load(setup_file, allow_pickle=True)
        n_gal = data["n_gal"].item()
        unique_pix = data["unique_pix"].item()
        idx = data["idx"].item()
        idx_rep = data["idx_rep"].item()
        ra_col = data["ra_col"].item()
        dec_col = data["dec_col"].item()
        e1_col = data["e1_col"].item()
        e2_col = data["e2_col"].item()
        w = data["w"].item()
        fiducial_cl = data["fiducial_cl"].item()
        tomo_bin_ids = data["tomo_bin_ids"]

        lmin, lmax, b_lmax = spv_pseudo_cl.pseudo_cl_geometry(nside)

        # --- call the simulation ---
        noisy_gaussian_maps = get_gaussian_simulation(
            nside,
            lmax,
            fiducial_cl,
            n_gal,
            unique_pix,
            idx,
            idx_rep,
            ra_col,
            dec_col,
            e1_col,
            e2_col,
            w,
            tomo_bin_ids,
        )

        # --- Compute the power spectra ---
        cl_final = extract_spectra(
            noisy_gaussian_maps, n_gal, tomo_bin_ids, b_lmax, out_dir
        )

        # Save the results
        np.savez(out_file, cl_decoupled=cl_final)

        print(f"Rank {rank} finished simulation {sim_id} for version {version}")

    except Exception as e:
        print(f"Rank {rank} failed simulation {sim_id} with error {e}")


# --- Distribution of the MPI processes in a master-worker scheme ---
def distribute_work(comm, n_sims, worker_fn, worker_args=()):
    rank = comm.Get_rank()
    size = comm.Get_size()

    work_tag = 1
    done_tag = 2
    stop_tag = 0

    if rank == 0:
        next_sim = 0
        closed_workers = 0

        for worker in range(1, size):
            if next_sim < n_sims:
                comm.send(next_sim, dest=worker, tag=work_tag)
                next_sim += 1
            else:
                comm.send("STOP", dest=worker, tag=stop_tag)
                closed_workers += 1

        while closed_workers < size - 1:
            status = MPI.Status()
            _ = comm.recv(source=MPI.ANY_SOURCE, tag=done_tag, status=status)
            worker = status.Get_source()

            if next_sim < n_sims:
                comm.send(next_sim, dest=worker, tag=work_tag)
                next_sim += 1
            else:
                comm.send("STOP", dest=worker, tag=stop_tag)
                closed_workers += 1

    else:
        while True:
            sim_id = comm.recv(source=0, tag=MPI.ANY_TAG, status=MPI.Status())
            if sim_id == "STOP":
                break
            worker_fn(sim_id, *worker_args)
            comm.send(sim_id, dest=0, tag=done_tag)


# --- Final function to read the computed spectra and extract a covariance matrix ---
def concatenate_spectra(cl_sample, tomo_bin_ids, pol_index):
    """Concatenate the spectra from different tomographic bins into a single array."""
    concatenated_cl = []
    tomo_bin_pairs = itertools.combinations_with_replacement(tomo_bin_ids, 2)
    for tomo_bin_a, tomo_bin_b in tomo_bin_pairs:
        concatenated_cl.append(cl_sample[f"W{tomo_bin_a}xW{tomo_bin_b}"][pol_index])
    return np.concatenate(concatenated_cl)


def get_covariance_from_simulated_spectra(
    n_sims, version, tomography, tomo_bin_ids, pol, out_dir
):
    """Compute the covariance of the EE signal from the simulated spectra."""
    cl_samples = []
    pol_index = pol_to_pol_index_dict[pol]
    for sim_id in range(n_sims):
        out_file = f"{out_dir}/cl_sample_{sim_id}_{version}_tomography_{tomography}.npz"
        if not os.path.exists(out_file):
            raise FileNotFoundError(f"Simulation output file {out_file} not found.")
        data = np.load(out_file, allow_pickle=True)
        cl_samples.append(
            concatenate_spectra(data["cl_decoupled"].item(), tomo_bin_ids, pol_index)
        )

    cl_samples = np.array(cl_samples)

    covariance_matrix = np.cov(cl_samples, rowvar=False)

    return covariance_matrix


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- Get the parser and extract arguments ---
    parser = get_parser()
    args = parser.parse_args()

    out_dir = args.output
    version = args.version
    out_dir = str(Path(out_dir) / Path(f"gaussian_simulations_{version}"))

    config = args.config
    tomography = args.tomo
    ignore_warnings = args.ignore_warnings

    if ignore_warnings:
        warnings.filterwarnings("ignore")
        print("Ignoring warnings during the simulation.")

    nside = args.nside
    binning = args.binning
    ell_step = args.ell_step
    n_ell_bins = args.ell_bins
    power = args.power

    force_run = args.force

    save_eb = args.save_eb
    save_bb = args.save_bb

    if rank == 0:
        # Check that the config file exists (same than cosmo_val)
        (
            cat_path,
            ra_col,
            dec_col,
            e1_col,
            e2_col,
            w_col,
            tomo_bin_col,
            redshift_path,
        ) = load_version_config(config, version, tomography)
        columns = (ra_col, dec_col, e1_col, e2_col, w_col)

        os.makedirs(out_dir, exist_ok=True)
        print(f"Output directory: {out_dir}")

        print("Pre-compute setup for the simulations and workspace...")

        print("Reading catalog...")
        ra, dec, e1, e2, w, tomo_bin_ids = load_catalog_slices(
            cat_path, columns, tomography, tomo_bin_col
        )

        # Setup the binning
        nside = args.nside
        lmin, lmax, b_lmax = spv_pseudo_cl.pseudo_cl_geometry(nside)
        ells = np.arange(lmin, lmax + 1)
        b = spv_pseudo_cl.make_namaster_bin(
            lmin,
            lmax,
            b_lmax,
            binning=binning,
            ell_step=ell_step,
            n_ell_bins=n_ell_bins,
            power=power,
        )
        ell_eff = b.get_effective_ells()

        print("Getting n_gal map...")
        n_gal, unique_pix, idx, idx_rep = build_pre_computed_maps(
            ra, dec, w, tomo_bin_ids, nside
        )

        print("Getting fiducial Cl...")
        # --- Get the cosmology object to compute the fiducial ---
        if args.cosmo_config is None:
            cosmo_params = {}
        else:
            with open(args.cosmo_config, "r") as f:
                cosmo_params = yaml.safe_load(f)

        cosmo, fiducial_cl = build_fiducial_cl(
            cosmo_params, redshift_path, lmax, tomography, tomo_bin_ids
        )

        print("Saving precomputed setup...")
        setup_file = os.path.join(
            out_dir, f"precomputed_setup_{version}_tomography_{tomography}.npz"
        )

        np.savez(
            setup_file,
            n_gal=n_gal,
            unique_pix=unique_pix,
            idx=idx,
            idx_rep=idx_rep,
            ra_col=ra,
            dec_col=dec,
            e1_col=e1,
            e2_col=e2,
            w=w,
            fiducial_cl=fiducial_cl,
            tomo_bin_ids=tomo_bin_ids,
        )

        print("Setting up workspace...")
        prepare_workspace(n_gal, tomo_bin_ids, nside, b, b_lmax, out_dir, force_run)

        print(
            f"Running Gaussian simulations for version {version} with the following parameters:\n"
            f"  - Number of simulations: {args.n_sims}\n"
            f"  - Nside: {nside}\n"
            f"  - Binning: {binning}\n"
            f"  - Ell step: {ell_step}\n"
            f"  - Number of ell bins: {n_ell_bins}\n"
            f"  - Power for powspace: {power}\n"
            f" \n"
            f" --- Cosmological parameters ---\n"
            f"  - H0: {cosmo['H0']}\n"
            f"  - Omega_m: {cosmo['Omega_m']}\n"
            f"  - Omega_b: {cosmo['Omega_b']}\n"
            f"  - Omega_c: {cosmo['Omega_c']}\n"
            f"  - Omega_nu: {cosmo['Omega_nu_mass'] + cosmo['Omega_nu_rel']}\n"
            f"  - n_s: {cosmo['n_s']}\n"
            f"  - sigma_8: {cosmo.sigma8()}\n"
            f"  - w0: {cosmo['w0']}\n"
            f"  - wa: {cosmo['wa']}\n"
        )

    comm.Barrier()

    n_sims = args.n_sims
    distribute_work(
        comm,
        n_sims,
        run_one_simulation,
        worker_args=(nside, version, tomography, out_dir, force_run),
    )

    comm.Barrier()
    if rank == 0:
        print("All simulations completed ✅")

        print(f"Merging into a EE covariance matrix for version {version}...")
        outpath_cov = os.path.join(
            out_dir, f"covariance_matrix_EE_{version}_tomography_{tomography}.npy"
        )

        covariance_matrix = get_covariance_from_simulated_spectra(
            n_sims, version, tomography, tomo_bin_ids, "EE", out_dir
        )

        np.save(outpath_cov, covariance_matrix)
        print(f"EE covariance matrix saved to {outpath_cov}")

        if save_eb:
            print(f"Merging into a EB covariance matrix for version {version}...")
            outpath_cov = os.path.join(
                out_dir, f"covariance_matrix_EB_{version}_tomography_{tomography}.npy"
            )

            covariance_matrix = get_covariance_from_simulated_spectra(
                n_sims, version, tomography, tomo_bin_ids, "EB", out_dir
            )

            np.save(outpath_cov, covariance_matrix)
            print(f"EB covariance matrix saved to {outpath_cov}")

        if save_bb:
            print(f"Merging into a BB covariance matrix for version {version}...")
            outpath_cov = os.path.join(
                out_dir, f"covariance_matrix_BB_{version}_tomography_{tomography}.npy"
            )

            covariance_matrix = get_covariance_from_simulated_spectra(
                n_sims, version, tomography, tomo_bin_ids, "BB", out_dir
            )

            np.save(outpath_cov, covariance_matrix)
            print(f"BB covariance matrix saved to {outpath_cov}")
