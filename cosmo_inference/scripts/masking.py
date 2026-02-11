import h5py
import healpy as hp
import numpy as np
from multiprocessing import Pool, cpu_count
import os
# -------------------------
# Masking logic
def apply_masks(data, data_ext):
    """
    Returns a boolean mask where True = galaxy passes all cuts (unmasked).
    """
    
    gal_rel_size = np.divide(
        data["NGMIX_T_NOSHEAR"],
        data["NGMIX_Tpsf_NOSHEAR"],
        out=np.zeros_like(data["NGMIX_T_NOSHEAR"]),  # what to fill when denom == 0
        where=(data["NGMIX_Tpsf_NOSHEAR"] > 0)       # only divide where valid
    )

    mask = (
        (data["FLAGS"] <= 3) & # v1.4.11.2
        # (data["FLAGS"] == 0) & # v1.4.5/6/7/8
        (data["overlap"] == True) &
        (data["IMAFLAGS_ISO"] == 0) &
        (data["N_EPOCH"] >= 2) &
        (data["mag"] >= 15) & (data["mag"] <= 30) &
        (data["NGMIX_MOM_FAIL"] == 0) &
        (data["NGMIX_ELL_PSFo_NOSHEAR_0"] != -10) &
        (data["NGMIX_ELL_PSFo_NOSHEAR_1"] != -10) &
        # (data_ext["1_Faint_star_halos"] == False) &  # v1.4.8
        # (data_ext["2_Bright_star_halos"] == False) & # v1.4.7
        (data_ext["4_Stars"] == False) &
        (data_ext["8_Manual"] == False) &
        (data_ext["64_r"] == False) &
        (data_ext["1024_Maximask"] == False) &
        (data_ext["npoint3"] >= 3) &
        (gal_rel_size < 3.0) &
        # (gal_rel_size > 0.707) # v1.4.6/7/8/11
        (gal_rel_size > 0.5) # v1.4.5 
        
    )
    return mask

# -------------------------
# Process one chunk
def process_chunk(args):
    start, stop, filename, nside = args
    with h5py.File(filename, "r") as f:
        data = f["data"][start:stop]
        data_ext = f["data_ext"][start:stop]

    mask = apply_masks(data, data_ext)

    ra = data["RA"][mask]
    dec = data["Dec"][mask]

    theta = np.radians(90.0 - dec)   # colatitude
    phi   = np.radians(ra)           # longitude

    pix = hp.ang2pix(nside, theta, phi)
    
    return np.unique(pix)

# -------------------------
# Build mask map in parallel
def build_mask_map_hdf5(filename, nside=512, chunk_size=1_000_000):
    with h5py.File(filename, "r") as f:
        nrows = f["data"].shape[0]

    chunks = [(i, min(i+chunk_size, nrows), filename, nside)
              for i in range(0, nrows, chunk_size)]

    mask_map = np.zeros(hp.nside2npix(nside), dtype=np.uint8)

    with Pool(cpu_count()) as pool:
        for pix_indices in pool.imap_unordered(process_chunk, chunks):
            mask_map[pix_indices] = 1

    return mask_map

# Example usage
if __name__ == "__main__":
    filename = "/n17data/UNIONS/WL/v1.4.x/v1.4.5/unions_shapepipe_comprehensive_struc_2024_v1.X.c.hdf5"
    nside = 8192
    out_dir = os.getcwd()
    mask_map = build_mask_map_hdf5(filename, nside=nside, chunk_size=500_000)
    
    npix = hp.nside2npix(nside)
    pix_area_sr   = 4 * np.pi / npix
    pix_area_deg2 = (180/np.pi)**2 * pix_area_sr
    n_obs = mask_map.sum()
    f_sky_obs = n_obs / npix
    area_obs_deg2 = n_obs * pix_area_deg2
    print('kept area = ', area_obs_deg2)
    
    cl_mask = hp.anafast(mask_map, lmax=3*nside-1)
    hp.write_map(os.path.join(out_dir, "mask_map_v1.4.5_nside_8192.fits"), mask_map, overwrite=True)
    np.savez(os.path.join(out_dir, "mask_cls_from_flags_v1.4.5_nside_8192.npz"), ells=np.arange(len(cl_mask)), cl_mask=cl_mask)