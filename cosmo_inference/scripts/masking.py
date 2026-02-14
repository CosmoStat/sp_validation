import argparse
import h5py
import healpy as hp
import numpy as np
from multiprocessing import Pool, cpu_count
import os
import sys
from pathlib import Path
import yaml

# -------------------------
# Spatially-structured cuts: these define the survey footprint.
# All other cuts (FLAGS, mag, SNR, shape measurement, PSF ellipticity,
# relative size) are per-galaxy quality cuts that should NOT affect
# the footprint definition.
SPATIAL_CUTS = {
    "overlap", "IMAFLAGS_ISO", "N_EPOCH",
    "4_Stars", "8_Manual", "64_r", "1024_Maximask", "npoint3",
    "1_Faint_star_halos", "2_Bright_star_halos",
}

# -------------------------
# Masking logic

def apply_condition(array, kind, value):
    """
    Apply a logical condition to a NumPy array and return a boolean mask, based 
    on the "kind" key in the mask config YAML file.
    """
    if kind == "equal":
        return array == value
    elif kind == "not_equal":
        return array != value
    elif kind == "greater_equal":
        return array >= value
    elif kind == "greater":
        return array > value
    elif kind == "less_equal":
        return array <= value
    elif kind == "less":
        return array < value
    elif kind == "range":
        return (array >= value[0]) & (array <= value[1])
    else:
        raise ValueError(f"Unknown kind: {kind}")

def apply_masks(data, data_ext, mask_config, footprint_only=False):
    """
    Construct a boolean mask selecting galaxies that satisfy all
    masking criteria defined in the YAML configuration file.

    Parameters
    ----------
    data : numpy.ndarray or structured array
        Slice of the HDF5 "data" group containing per-object
        measurements (e.g. FLAGS, mag, NGMIX quantities).

    data_ext : numpy.ndarray or structured array
        Slice of the HDF5 "data_ext" group containing external or
        post-processing flags (e.g. star masks, footprint flags).

    mask_config : dict
        Dictionary parsed from the YAML mask configuration file.
        Expected structure:
            - mask_config["dat"]     : list of cuts applied to `data`
            - mask_config["dat_ext"] : list of cuts applied to `data_ext`
            - mask_config["metacal"] : derived-quantity parameters
              (e.g. relative size limits)

    footprint_only : bool, optional
        If True, only apply spatially-structured cuts (those in
        SPATIAL_CUTS). Skips per-galaxy quality cuts (FLAGS, mag,
        SNR, shape measurement, PSF ellipticity, relative size).
        Used to define a consistent footprint from the comprehensive
        catalog. Default is False.

    Returns
    -------
    numpy.ndarray (bool)
        Boolean array of length equal to the input data slice.
        True indicates the object passes all cuts (kept),
        False indicates the object is masked (removed).
    """

    # Initialize mask
    mask = np.ones(len(data), dtype=bool)

    # --- dat group ---
    for cut in mask_config.get("dat", []):
        col = cut["col_name"]
        if footprint_only and col not in SPATIAL_CUTS:
            continue
        kind = cut["kind"]
        value = cut["value"]

        mask &= apply_condition(data[col], kind, value)

    # --- dat_ext group ---
    for cut in mask_config.get("dat_ext", []):
        col = cut["col_name"]
        if footprint_only and col not in SPATIAL_CUTS:
            continue
        kind = cut["kind"]
        value = cut["value"]

        mask &= apply_condition(data_ext[col], kind, value)

    # --- metacal relative size (skip for footprint-only) ---
    if not footprint_only:
        rel_size = np.divide(
            data["NGMIX_T_NOSHEAR"],
            data["NGMIX_Tpsf_NOSHEAR"],
            out=np.zeros_like(data["NGMIX_T_NOSHEAR"]),
            where=(data["NGMIX_Tpsf_NOSHEAR"] > 0)
        )

        rel_min = mask_config["metacal"]["gal_rel_size_min"]
        rel_max = mask_config["metacal"]["gal_rel_size_max"]

        mask &= (rel_size >= rel_min) & (rel_size <= rel_max)

    return mask

# -------------------------
# Process one chunk
def process_chunk(args):
    """
    Process a chunk of the HDF5 catalogue and return the unique
    HEALPix pixels containing unmasked galaxies,to be executed in
    parallel. It reads a slice of the catalogue, applies
    the defined masking criteria, converts the sky positions
    (RA, Dec) of retained galaxies into HEALPix pixel indices,
    and returns the unique pixel indices for that chunk.

    Parameters
    ----------
    args : tuple
        Tuple containing:
            - start : int
                Starting row index of the chunk (inclusive).
            - stop : int
                Ending row index of the chunk (exclusive).
            - filename : str
                Path to the input HDF5 catalogue.
            - nside : int
                HEALPix NSIDE parameter defining map resolution.
            - mask_config : dict
                Parsed YAML mask configuration.

    Returns
    -------
    numpy.ndarray
        Array of unique HEALPix pixel indices (int) corresponding
        to sky locations of galaxies that pass all mask cuts in
        this chunk.
    """

    start, stop, filename, nside, mask_config, footprint_only = args
    with h5py.File(filename, "r") as f:
        data = f["data"][start:stop]
        data_ext = f["data_ext"][start:stop]

    mask = apply_masks(data, data_ext, mask_config, footprint_only=footprint_only)

    ra = data["RA"][mask]
    dec = data["Dec"][mask]

    theta = np.radians(90.0 - dec)   # colatitude
    phi   = np.radians(ra)           # longitude

    pix = hp.ang2pix(nside, theta, phi)
    
    return np.unique(pix)

# -------------------------
# Build mask map in parallel
def build_mask_map_hdf5(filename, mask_config, nside, chunk_size=1_000_000,
                        footprint_only=False):
    """
    Build a binary HEALPix mask map from an HDF5 galaxy catalogue.

    The catalogue is processed in chunks to limit memory usage.

    Parameters
    ----------
    filename : str
        Path to the input HDF5 catalogue containing "data" and
        "data_ext" groups
    mask_config : dict
        Dictionary parsed from the YAML mask configuration file
    nside : int
        HEALPix NSIDE parameter defining the resolution of the
        output map.
    chunk_size : int, optional
        Number of catalogue rows to process per chunk.
        Default is 1,000,000.
    footprint_only : bool, optional
        If True, only apply spatially-structured cuts.

    Returns
    -------
    numpy.ndarray
        One-dimensional HEALPix map (dtype uint8) of length
        hp.nside2npix(nside), where:
            - 1 indicates at least one unmasked galaxy falls
              in that pixel,
            - 0 indicates no retained galaxies.
    """
    with h5py.File(filename, "r") as f:
        nrows = f["data"].shape[0]

    chunks = [(i, min(i+chunk_size, nrows), filename, nside, mask_config,
               footprint_only)
              for i in range(0, nrows, chunk_size)]

    mask_map = np.zeros(hp.nside2npix(nside), dtype=np.uint8)

    with Pool(cpu_count()) as pool:
        for pix_indices in pool.imap_unordered(process_chunk, chunks):
            mask_map[pix_indices] = 1

    return mask_map

############################################################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build HEALPix mask from HDF5 catalog")
    parser.add_argument("nside", type=int, help="HEALPix NSIDE parameter")
    parser.add_argument("--config", required=True, help="Path to mask config YAML")
    parser.add_argument("--output-prefix", required=True,
                        help="Output file prefix (e.g. 'footprint' or 'footprint_starhalo')")
    parser.add_argument("--footprint-only", action="store_true",
                        help="Only apply spatially-structured cuts (for footprint definition)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: data/mask/ relative to script)")
    args = parser.parse_args()

    nside = args.nside
    curr_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = curr_dir.parent / "data" / "mask"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        mask_config = yaml.safe_load(f)

    filename = f"/n17data/UNIONS/WL/v1.4.x/v1.4.5/{mask_config['params']['input_path']}"
    prefix = args.output_prefix

    if args.footprint_only:
        print(f"Footprint-only mode: applying only spatial cuts {SPATIAL_CUTS}")

    # Build mask map from comprehensive catalogue
    mask_map = build_mask_map_hdf5(filename, mask_config, nside, chunk_size=500_000,
                                   footprint_only=args.footprint_only)

    # Get survey area after masking
    npix = hp.nside2npix(nside)
    pix_area_sr   = 4 * np.pi / npix
    pix_area_deg2 = (180/np.pi)**2 * pix_area_sr
    n_obs = mask_map.sum()
    f_sky_obs = n_obs / npix
    area_obs_deg2 = n_obs * pix_area_deg2
    print(f"Kept area = {area_obs_deg2:.2f} deg^2\n")

    # Compute Cls of the mask map
    cl_mask = hp.anafast(mask_map, lmax=3*nside-1)
    ells = np.arange(len(cl_mask))

    # Save mask map and Cls
    map_path = out_dir / f"mask_map_{prefix}_nside_{nside}.fits"
    cls_path = out_dir / f"mask_cls_{prefix}_nside_{nside}.npz"
    hp.write_map(map_path, mask_map, overwrite=True)
    np.savez(cls_path, ells=ells, cl_mask=cl_mask)

    print(f"Mask map saved to {map_path}")
    print(f"Mask Cls saved to {cls_path}\n")

    # Compute normalising factor for the mask Cls
    integral_w = np.sum((2*ells + 1) / (4 * np.pi) * cl_mask) / (np.pi/180)**2
    norm_factor = area_obs_deg2 / integral_w
    norm_cls = cl_mask * norm_factor

    # Save normalised Cls to text file
    norm_path = out_dir / f"mask_cls_{prefix}_nside_{nside}_norm.txt"
    idx = np.arange(len(cl_mask))
    data_to_save = np.column_stack((idx, norm_cls))
    np.savetxt(norm_path, data_to_save, fmt=["%d", "%.10e"])
    print(f"Normalised mask Cls saved to {norm_path}")
