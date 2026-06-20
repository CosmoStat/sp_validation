#!/usr/bin/env python
"""
High-resolution ξ± measurement for COSEBIS integration.

Computes TreeCorr GGCorrelation with fine angular binning (10,000+ bins)
required for accurate COSEBIS mode integration. Uses MPI for patch-pair
distribution across nodes when available; falls back to multi-threaded
single-process otherwise.

Reference: Asgari et al. 2017 — minimum 10,000 bins for E_7 at 0.5% accuracy.

Usage:
    # MPI (via Slurm submission script):
    mpiexec --map-by ppr:1:node python run_2pcf_highres.py

    # Single-node fallback:
    python run_2pcf_highres.py
"""

import os
import sys
import time

import numpy as np
import treecorr
from astropy.io import fits

# ---------------------------------------------------------------------------
# MPI setup (graceful fallback)
# ---------------------------------------------------------------------------
try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    USE_MPI = size > 1
except ImportError:
    comm = None
    rank = 0
    size = 1
    USE_MPI = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAT_PATH = (
    "/n17data/UNIONS/WL/v1.4.x/v1.4.6.3/unions_shapepipe_cut_struc_2024_v1.4.6.3.fits"
)
VERSION = "SP_v1.4.6.3_leak_corr"

# Ellipticity columns (leak-corrected)
E1_COL = "e1_leak_corrected"
E2_COL = "e2_leak_corrected"
W_COL = "w_des"

# Shear response (R=1 for all SP catalogs)
R = 1.0

# TreeCorr parameters
TMIN = 0.5  # arcmin
TMAX = 300.0  # arcmin
NBINS = 10_000
NPATCH = 50
# Detect threads from Slurm or fall back to OS count
NUM_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 24))

# Output
OUTPUT_DIR = "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_val/output"
PATCH_FILE = os.path.join(
    OUTPUT_DIR,
    f"patch_centers_{VERSION}_{NPATCH}_{TMIN}_{TMAX}.dat",
)


def log(msg):
    """Print with timestamp on rank 0 only."""
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_catalog():
    """Load shear catalog and apply mean subtraction."""
    log(f"Loading catalog: {CAT_PATH}")
    hdul = fits.open(CAT_PATH, memmap=True)
    data = hdul[1].data

    ra = np.array(data["ra"], dtype=np.float64)
    dec = np.array(data["dec"], dtype=np.float64)
    e1 = np.array(data[E1_COL], dtype=np.float64)
    e2 = np.array(data[E2_COL], dtype=np.float64)
    w = np.array(data[W_COL], dtype=np.float64)
    hdul.close()

    log(f"  {len(ra):,} galaxies loaded")

    # Additive bias: c = <e/R>_w  (R=1 for SP catalogs)
    c1 = np.average(e1 / R, weights=w)
    c2 = np.average(e2 / R, weights=w)
    log(f"  Additive bias: c1={c1:.6e}, c2={c2:.6e}")

    # Calibrated shear: g = (e - c) / R
    g1 = (e1 - c1) / R
    g2 = (e2 - c2) / R

    return ra, dec, g1, g2, w


def compute_patch_centers(ra, dec):
    """Compute patch centers from subsampled catalog (rank 0 only)."""
    if os.path.exists(PATCH_FILE):
        log(f"Using existing patch centers: {PATCH_FILE}")
        return

    if rank != 0:
        return

    log(f"Computing patch centers (npatch={NPATCH}) from 1% subsample...")
    rng = np.random.default_rng(42)
    n_sub = max(len(ra) // 100, NPATCH * 100)
    idx = rng.choice(len(ra), size=n_sub, replace=False)

    cat_sub = treecorr.Catalog(
        ra=ra[idx],
        dec=dec[idx],
        ra_units="degrees",
        dec_units="degrees",
        npatch=NPATCH,
    )
    cat_sub.write_patch_centers(PATCH_FILE)
    log(f"  Wrote patch centers to {PATCH_FILE}")
    del cat_sub


def write_xi_fits(gg, prefix, xi_data):
    """Write ξ+ or ξ- to FITS matching CosmologyValidation format."""
    out_path = os.path.join(
        OUTPUT_DIR,
        f"{prefix}_{VERSION}_minsep={TMIN}_maxsep={TMAX}_nbins={NBINS}_npatch=1.fits",
    )
    n = len(xi_data)
    cols = [
        fits.Column(name="BIN1", format="K", array=np.ones(n, dtype=int)),
        fits.Column(name="BIN2", format="K", array=np.ones(n, dtype=int)),
        fits.Column(name="ANGBIN", format="K", array=np.arange(1, n + 1)),
        fits.Column(name="VALUE", format="D", array=xi_data),
        fits.Column(name="ANG", format="D", unit="arcmin", array=gg.meanr),
    ]
    ext_name = "XI_PLUS" if "plus" in prefix else "XI_MINUS"
    hdu = fits.BinTableHDU.from_columns(cols, name=ext_name)
    for key, val in {
        "2PTDATA": "T",
        "QUANT1": "G+R",
        "QUANT2": "G+R",
        "KERNEL_1": "NZ_SOURCE",
        "KERNEL_2": "NZ_SOURCE",
        "WINDOWS": "SAMPLE",
    }.items():
        hdu.header[key] = val
    hdu.writeto(out_path, overwrite=True)
    log(f"  Wrote {out_path}")


def main():
    t0 = time.time()

    log("=" * 60)
    log(f"High-resolution ξ± measurement")
    log(f"  MPI: {'yes' if USE_MPI else 'no'} (ranks={size})")
    log(f"  Config: {NBINS:,} bins, [{TMIN}, {TMAX}] arcmin")
    log(f"  Patches: {NPATCH}, Threads/rank: {NUM_THREADS}")
    log(f"  Version: {VERSION}")
    log("=" * 60)

    # All ranks load catalog (needed for TreeCorr patch assignment)
    ra, dec, g1, g2, w = load_catalog()

    # Compute patch centers (rank 0 only; others wait)
    compute_patch_centers(ra, dec)
    if USE_MPI:
        comm.Barrier()

    # Create TreeCorr catalog with patch centers
    log("Creating TreeCorr catalog with patches...")
    cat = treecorr.Catalog(
        ra=ra,
        dec=dec,
        g1=g1,
        g2=g2,
        w=w,
        ra_units="degrees",
        dec_units="degrees",
        patch_centers=PATCH_FILE,
    )
    cat.load()
    cat.get_patches()
    log(f"  Catalog ready ({cat.nobj:,} objects, {cat.npatch} patches)")

    # Free raw arrays (TreeCorr holds its own copy)
    del ra, dec, g1, g2, w

    # Compute GG correlation
    log("Computing GGCorrelation...")
    gg = treecorr.GGCorrelation(
        min_sep=TMIN,
        max_sep=TMAX,
        nbins=NBINS,
        sep_units="arcminutes",
        verbose=2,
    )

    process_kwargs = {"num_threads": NUM_THREADS}
    if USE_MPI:
        process_kwargs["comm"] = comm

    gg.process(cat, **process_kwargs)
    log(f"  Correlation complete ({time.time() - t0:.0f}s elapsed)")

    # Write output (rank 0 only)
    if rank == 0:
        out_txt = os.path.join(
            OUTPUT_DIR,
            f"{VERSION}_xi_minsep={TMIN}_maxsep={TMAX}_nbins={NBINS}_npatch=1.txt",
        )
        gg.write(out_txt, write_patch_results=True, write_cov=True)
        log(f"  Wrote {out_txt}")

        write_xi_fits(gg, "xi_plus", gg.xip)
        write_xi_fits(gg, "xi_minus", gg.xim)

        elapsed = time.time() - t0
        log(f"Done! Total time: {elapsed / 3600:.1f}h ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
