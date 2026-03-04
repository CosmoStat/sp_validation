import numpy as np
import healpy as hp
import pymaster as nmt
from astropy.io import fits
from astropy.cosmology import Planck18
import camb
import matplotlib.pyplot as plt
from mpi4py import MPI

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=Warning)

sys.path.append(os.path.abspath('/home/guerrini/UNIONS_forward_model'))

out_dir = '/n17data/sguerrini/sp_validation/notebooks/cosmo_val/harmonic_covariance_gaussian_sims_v1463_C'
setup_file = out_dir +'/precomputed_setup.npz'
wsp_file = out_dir+'/wsp_harmony_covariance.fits'
path_redshift_distr = '/n17data/sguerrini/UNIONS/WL/nz/v1.4.6.3/nz_SP_v1.4.6.3_C.txt'

def get_fiducial(nside, lmax, redshift_distr):

    planck = Planck18

    h = planck.H0.value/100
    Om = planck.Om0
    Ob = planck.Ob0
    Oc = Om - Ob
    ns = 0.965
    As = 2.1e-9
    sigma8 = 0.8102
    m_nu = 0.06
    w = -1
    log_T_AGN = 7.8
    
    pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, ns=ns, mnu=m_nu, w=w, As=As, WantTransfer=True, NonLinear=camb.model.NonLinear_both)
    Onu = pars.omeganu
    Oc = Om - Ob - Onu
    pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, ns=ns, mnu=m_nu, w=w, As=As, WantTransfer=True, NonLinear=camb.model.NonLinear_both)

    pars.NonLinearModel.set_params(
        halofit_version='mead2020_feedback',
        HMCode_logT_AGN=log_T_AGN,
    )

    pars.set_matter_power(
        nonlinear=True,
        kmax=20
    )

    sigma8_temp = camb.get_results(pars).get_sigma8_0()
    As *= (sigma8 / sigma8_temp)**2

    init_power_scaled = camb.InitialPowerLaw()
    init_power_scaled.set_params(As=As, ns=ns)
    pars.InitPower = init_power_scaled

    z, dndz = np.loadtxt(redshift_distr, unpack=True)

    #getthe expected cl's from CAMB
    pars.min_l = 1
    pars.set_for_lmax(lmax)
    pars.SourceWindows = [
        camb.sources.SplinedSourceWindow(z=z, W=dndz, source_type='lensing')
    ]
    theory_cls = camb.get_results(pars).get_source_cls_dict(lmax=lmax, raw_cl=True)
    return theory_cls['W1xW1']

def get_n_gal_map(nside, cat_gal):
    ra = cat_gal['RA']
    dec = cat_gal['DEC']
    w = cat_gal['w_des']

    theta = (90. - dec) * np.pi / 180.
    phi = ra * np.pi / 180.
    pix = hp.ang2pix(nside, theta, phi)

    unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
    n_gal = np.zeros(hp.nside2npix(nside))
    n_gal[unique_pix] = np.bincount(idx_rep, weights=w)
    return n_gal, unique_pix, idx, idx_rep

def apply_random_rotation(e1, e2):
    rot_angle = np.random.rand(len(e1))*2*np.pi
    e1_out = e1*np.cos(rot_angle) + e2*np.sin(rot_angle)
    e2_out = -e1*np.sin(rot_angle) + e2*np.cos(rot_angle)
    return e1_out, e2_out

def get_gaussian_real(nside, lmax, cl_ee, n_gal, mask, unique_pix, idx_rep, w, e1_col, e2_col):

    alm_shear = hp.synalm(cl_ee, lmax=lmax)

    gauss_map = hp.alm2map([np.zeros_like(alm_shear), alm_shear, np.zeros_like(alm_shear)], nside=nside, verbose=False)

    e1_rot, e2_rot = apply_random_rotation(e1_col, e2_col)

    noise_map_e1 = np.zeros(hp.nside2npix(nside))
    noise_map_e2 = np.zeros(hp.nside2npix(nside))

    noise_map_e1[unique_pix] += np.bincount(idx_rep, weights=e1_rot*w)
    noise_map_e2[unique_pix] += np.bincount(idx_rep, weights=e2_rot*w)
    noise_map_e1[mask] /= n_gal[mask]
    noise_map_e2[mask] /= n_gal[mask]

    final_map_e1 = np.copy(noise_map_e1)
    final_map_e2 = np.copy(noise_map_e2)
    final_map_e1[mask] += gauss_map[1][mask]
    final_map_e2[mask] += gauss_map[2][mask]

    return final_map_e1 + 1j*final_map_e2, noise_map_e1 + 1j*noise_map_e2

def get_sample(nside, lmax, cl_ee, n_gal, mask, unique_pix, idx_rep, wsp, w, e1_col, e2_col):
    gauss_map, noise_map = get_gaussian_real(nside, lmax, cl_ee, n_gal, mask, unique_pix, idx_rep, w, e1_col, e2_col)

    f_all = nmt.NmtField(mask=n_gal, maps=[gauss_map.real, gauss_map.imag], lmax=lmax)
    cl_coupled = nmt.compute_coupled_cell(f_all, f_all)
    cl_all = wsp.decouple_cell(cl_coupled)

    f_noise = nmt.NmtField(mask=n_gal, maps=[noise_map.real, noise_map.imag], lmax=lmax)
    cl_noise_coupled = nmt.compute_coupled_cell(f_noise, f_noise)
    cl_noise = wsp.decouple_cell(cl_noise_coupled)

    return cl_all, cl_noise

def run_simulation(sim_id):
    """Worker executes this simulation and saves results"""
    try:
        out_file = f"{out_dir}/sample_{sim_id}.npz"
        if os.path.exists(out_file):
            print(f"Rank {rank} skipping {sim_id} (already exists)")
            return

        print(f"Rank {rank} starting {sim_id}")
        data = np.load(setup_file, allow_pickle=True)
        n_gal = data['n_gal']; mask = data['mask']
        unique_pix = data['unique_pix']; idx_rep = data['idx_rep']
        e1_col = data['e1_col']; e2_col = data['e2_col']
        w = data['w']; cl_ee = data['cl_ee']

        nside = 1024
        lmax = 2 * nside
        b_lmax = lmax - 1

        wsp = nmt.NmtWorkspace()
        wsp.read_from(wsp_file)

        # --- call your custom simulation ---
        sample_all, sample_noise = get_sample(
            nside, b_lmax, cl_ee,
            n_gal, mask, unique_pix, idx_rep,
            wsp=wsp, w=w, e1_col=e1_col, e2_col=e2_col
        )

        np.savez_compressed(out_file, cl_all=sample_all, cl_noise=sample_noise)
        print(f"Rank {rank} finished {sim_id}")

    except Exception as e:
        print(f"Rank {rank} failed {sim_id} with error {e}")

if __name__ == "__main__":

    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    path_cat = '/n17data/UNIONS/WL/v1.4.x/v1.4.6/unions_shapepipe_cut_struc_2024_v1.4.6.fits'
    e1_col = 'e1_leak_corrected'
    e2_col = 'e2_leak_corrected'
    w_col = 'w_des'

    os.makedirs(out_dir, exist_ok=True)
    if rank == 0:
        if os.path.exists(setup_file) and os.path.exists(wsp_file):
            print("Precomputed setup and workspace already exist. Skipping setup.")
        else:
            print("Reading catalog...")
            cat_gal = fits.getdata(path_cat)

            nside = 1024
            lmax = 2 * nside

            lmin = 8
            b_lmax = lmax - 1

            n_ell_bins = 32

            ells = np.arange(lmin, lmax + 1)

            print("Setting up ell bins...")
            power = 1/2
            start = np.power(lmin, power)
            end = np.power(lmax, power)
            bins_ell = np.power(np.linspace(start, end, n_ell_bins+1), 1/power).astype(float)

            bpws = np.digitize(ells.astype(float), bins_ell) - 1
            bpws[0] = 0
            bpws[-1] = n_ell_bins - 1

            b = nmt.NmtBin(ells=ells, bpws=bpws, lmax=b_lmax)
            ell_eff = b.get_effective_ells()

            print("Getting n_gal map...")
            n_gal, unique_pix, idx, idx_rep = get_n_gal_map(nside, cat_gal)
            mask = n_gal != 0

            print("Getting fiducial Cl...")
            pw = hp.pixwin(nside, lmax=lmax)
            cl_ee = get_fiducial(nside, lmax, path_redshift_distr) * pw**2

            print("Saving precomputed setup...")
            np.savez(setup_file, n_gal=n_gal, unique_pix=unique_pix, idx=idx, idx_rep=idx_rep, mask=mask, e1_col=cat_gal[e1_col], e2_col=cat_gal[e2_col], w=cat_gal[w_col], cl_ee=cl_ee)

            print("Setting up workspace...")
            f = nmt.NmtField(mask=n_gal,
                            maps=[np.zeros(hp.nside2npix(nside)),
                                np.zeros(hp.nside2npix(nside))],
                            lmax=b_lmax)
            wsp = nmt.NmtWorkspace(f, f, b)
            wsp.write_to(wsp_file)

        print("Creating output directory...")
        os.makedirs(out_dir, exist_ok=True)


    comm.Barrier()

    n_sims = 10_000  # Total number of simulations to run
    if rank == 0:
        # --- Master process ---
        next_sim = 0
        closed_workers = 0

        print("Master distributing work...")

        # Initial distribution: send one task to each worker
        for worker in range(1, size):
            if next_sim < n_sims:
                comm.send(next_sim, dest=worker, tag=1)
                next_sim += 1

        # Listen for workers finishing and assign new work
        while closed_workers < (size - 1):
            status = MPI.Status()
            finished_sim = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
            worker = status.Get_source()

            if next_sim < n_sims:
                comm.send(next_sim, dest=worker, tag=1)
                next_sim += 1
            else:
                comm.send("STOP", dest=worker, tag=0)
                closed_workers += 1

    else:
        # --- Worker processes ---
        while True:
            sim_id = comm.recv(source=0, tag=MPI.ANY_TAG, status=MPI.Status())
            if sim_id == "STOP":
                break
            run_simulation(sim_id)
            comm.send(sim_id, dest=0, tag=2)

    comm.Barrier()
    if rank == 0:
        print("All simulations completed ✅")