#!/usr/bin/env python
"""
High-resolution ξ± measurement for COSEBIS/pure-EB integration.

Computes TreeCorr GGCorrelation on the fine integration angular grid (default
1000 log bins, config-driven) required for accurate COSEBIS/pure-EB mode
integration. Uses MPI for patch-pair distribution across nodes when available;
falls back to multi-threaded single-process otherwise (the default at 1000 bins).

Reference: Asgari et al. 2017 motivates a fine integration grid; the B-modes
paper found no substantial 1k-vs-10k difference, so the operational default is
1000 bins (see config nbins_int).

Usage:
    # MPI (via Slurm submission script):
    mpiexec --map-by ppr:1:node python run_2pcf_highres.py \
        --cat-config /path/to/cosmo_val/cat_config.yaml --out <output_dir>

    # Single-process fallback:
    python run_2pcf_highres.py \
        --cat-config /path/to/cosmo_val/cat_config.yaml --out <output_dir>
"""

import argparse
import os
import time

import numpy as np
import treecorr
from astropy.io import fits

# sacc_io depends only on numpy + sacc (no healpy/cs_util), so the born-as-SACC
# integration ξ± write works on the bare-host MPI path too, where the full cosmo_val
# stack is unavailable.
from sp_validation import sacc_io
from sp_validation.cosmo_val.sacc_writers import xi_to_sacc

try:
    # In-container path: full sp_validation stack available.
    from sp_validation.cosmo_val import CosmologyValidation

    _HAVE_COSMO_VAL = True
except ImportError:
    # Bare-host path (host OpenMPI + host python for an optional MPI run): the
    # full sp_validation stack (cs_util.plots -> healpy/healsparse) is not
    # installed. This measurement only needs the shear catalog path + column
    # names, which are a pure cat_config.yaml lookup — resolve them standalone.
    CosmologyValidation = None
    _HAVE_COSMO_VAL = False

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
# Shear response (R=1 for all SP catalogs)
R = 1.0

# Detect threads from Slurm or fall back to OS count
NUM_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 24))

# The catalog path, ellipticity/weight columns, TreeCorr grid, patch count and
# output directory are resolved from the CLI in main() (defaults reproduce the
# historical hardcoded values for a no-arg run). They are declared here as
# module globals so the rank-aware helpers below resolve them at call time; the
# catalog path + columns come from cat_config + version exactly as run_2pcf.py
# resolves them (via CosmologyValidation).
CAT_PATH = None
VERSION = None
E1_COL = None
E2_COL = None
W_COL = None
REDSHIFT_PATH = None  # n(z) file for the SACC tracer
TMIN = None  # arcmin
TMAX = None  # arcmin
NBINS = None
NPATCH = None
OUTPUT_DIR = None
PATCH_FILE = None
# Campaign run type, stamped as the part's SACC `type`. Custody state, not
# decoration (see blinding.assert_consistent_blind).
RUN_TYPE = "data"


def parse_args(argv=None):
    """CLI mirroring run_xi_sweep's signature; defaults reproduce prior behavior."""
    ap = argparse.ArgumentParser(
        description="High-resolution TreeCorr ξ± measurement for COSEBIS integration."
    )
    ap.add_argument(
        "--config",
        default=None,
        help="Path to bmodes config.yaml (accepted for signature parity with "
        "run_xi_sweep; not read by this measurement).",
    )
    ap.add_argument(
        "--cat-config", required=True, help="Absolute path to cat_config.yaml"
    )
    ap.add_argument(
        "--version",
        default="SP_v1.4.6.3_leak_corr",
        help="Catalog version key in cat_config",
    )
    ap.add_argument("--nbins", type=int, default=1000, help="Number of log bins")
    ap.add_argument("--npatch", type=int, default=50, help="TreeCorr patch count")
    ap.add_argument(
        "--min-sep", type=float, default=0.5, help="Min separation [arcmin]"
    )
    ap.add_argument(
        "--max-sep", type=float, default=300.0, help="Max separation [arcmin]"
    )
    ap.add_argument("--out", required=True, help="Output directory (lc {output})")
    ap.add_argument(
        "--run-type",
        default="data",
        choices=("data", "mock"),
        help="Campaign run type stamped as the part's SACC `type`",
    )
    return ap.parse_args(argv)


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


def _wait_for_file(path, timeout=300, interval=1.0):
    """Block until `path` is visible to this node, defeating NFS dir caching.

    On a multi-node run the rank that wrote `path` sees it immediately, but
    peer nodes can carry a stale negative directory-cache entry past an MPI
    Barrier. Re-listing the parent directory forces an NFS attribute refresh;
    poll that until the entry appears (or raise after `timeout` seconds).
    """
    parent = os.path.dirname(path) or "."
    name = os.path.basename(path)
    waited = 0.0
    while waited < timeout:
        try:
            if name in os.listdir(parent):
                return
        except FileNotFoundError:
            pass
        time.sleep(interval)
        waited += interval
    raise TimeoutError(f"patch-center file not visible after {timeout}s: {path}")


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


def write_xi_integration_sacc(gg):
    """Write the integration-grid ξ± SACC part (``{version}_xi_integration.sacc``).

    This is an intermediate per-statistic part — COSEBIs and pure-E/B consume it.
    It stays a standalone per-part file and does not join the terminal
    ``{version}.sacc`` (see #247 ruling). It carries a ``DiagonalCovariance`` from
    TreeCorr ``varxip``/``varxim``
    (npatch=1 leaves shot-noise variance as the only covariance estimate).
    Both run paths land here: in-container this uses the full SACC stack; on the
    bare-host MPI run only ``sacc_io`` + the n(z) file are needed (no healpy).
    """
    z, nz = np.loadtxt(REDSHIFT_PATH, unpack=True)
    metadata = {
        "catalogue_version": VERSION,
        "sp_validation_version": _sp_validation_version(),
        "npatch": 1,
    }
    s = xi_to_sacc(
        {0: (z, nz)},
        metadata,
        gg.meanr,
        gg.xip,
        gg.xim,
        grid="integration",
        theta_nom=gg.rnom,
        variances=np.concatenate([gg.varxip, gg.varxim]),
    )
    out_path = os.path.join(OUTPUT_DIR, f"{VERSION}_xi_integration.sacc")
    sacc_io.save(s, out_path, type=RUN_TYPE)
    log(f"  Wrote {out_path}")


def _sp_validation_version():
    """Best-effort package version for the SACC metadata (empty if unavailable)."""
    try:
        from sp_validation import __version__

        return __version__
    except Exception:
        return ""


def resolve_shear_config(cat_config_path, version):
    """Standalone shear-config resolver (bare-host fallback for CosmologyValidation).

    Reproduces exactly the ``cc[version]["shear"]`` fields this measurement reads
    (path, e1_col, e2_col, w_col), replicating CosmologyValidation's two
    transforms: (1) subdir-relative path resolution, and (2) the ``_leak_corr``
    virtual version — deep-copy the base version and swap
    e1_col/e2_col -> e1_col_corrected/e2_col_corrected. See
    sp_validation/cosmo_val/core.py.
    """
    import copy

    import yaml

    with open(cat_config_path) as fh:
        cc = yaml.load(fh, Loader=yaml.FullLoader)

    def resolve_paths(ver):
        subdir = os.fspath(cc[ver]["subdir"])
        for section in cc[ver].values():
            if isinstance(section, dict) and "path" in section:
                p = section["path"]
                if not os.path.isabs(p):
                    section["path"] = os.path.join(subdir, p)

    leak_suffix = "_leak_corr"
    if version in cc:
        resolve_paths(version)
    elif version.endswith(leak_suffix):
        base = version[: -len(leak_suffix)]
        if base not in cc:
            raise ValueError(f"Base version '{base}' not in cat_config for '{version}'")
        resolve_paths(base)
        base_shear = cc[base]["shear"]
        if "e1_col_corrected" not in base_shear or "e2_col_corrected" not in base_shear:
            raise ValueError(
                f"{base} lacks e1_col_corrected/e2_col_corrected; cannot form {version}"
            )
        cc[version] = copy.deepcopy(cc[base])
        cc[version]["shear"]["e1_col"] = base_shear["e1_col_corrected"]
        cc[version]["shear"]["e2_col"] = base_shear["e2_col_corrected"]
        resolve_paths(version)
    else:
        raise ValueError(f"Version '{version}' not found in cat_config")

    return cc[version]["shear"]


def main():
    global CAT_PATH, VERSION, E1_COL, E2_COL, W_COL, REDSHIFT_PATH
    global TMIN, TMAX, NBINS, NPATCH, OUTPUT_DIR, PATCH_FILE, RUN_TYPE

    args = parse_args()
    VERSION = args.version
    RUN_TYPE = args.run_type
    NBINS = args.nbins
    NPATCH = args.npatch
    TMIN = args.min_sep
    TMAX = args.max_sep
    OUTPUT_DIR = args.out

    # Resolve catalog path + ellipticity/weight columns from cat_config + version
    # exactly as run_2pcf.py does (applies the _leak_corr column swap and the
    # subdir path resolution). In-container this uses CosmologyValidation; on the
    # bare-host MPI fallback it uses the standalone cat_config resolver, which is
    # byte-identical for the shear-config fields this measurement reads.
    if _HAVE_COSMO_VAL:
        cv = CosmologyValidation(
            versions=[VERSION], catalog_config=args.cat_config, output_dir=OUTPUT_DIR
        )
        shear_cfg = cv.cc[VERSION]["shear"]
    else:
        shear_cfg = resolve_shear_config(args.cat_config, VERSION)
    CAT_PATH = shear_cfg["path"]
    E1_COL = shear_cfg["e1_col"]
    E2_COL = shear_cfg["e2_col"]
    W_COL = shear_cfg["w_col"]
    REDSHIFT_PATH = shear_cfg["redshift_path"]

    PATCH_FILE = os.path.join(
        OUTPUT_DIR,
        f"patch_centers_{VERSION}_{NPATCH}_{TMIN}_{TMAX}.dat",
    )

    t0 = time.time()

    log("=" * 60)
    log("High-resolution ξ± measurement")
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
        # Cross-node visibility: rank 0 wrote PATCH_FILE on its node, but on a
        # multi-node allocation the other ranks' nodes may not see it yet (NFS
        # close-to-open + negative-dir caching persists past the Barrier). Poll
        # with a forced directory refresh until it appears before reading it.
        _wait_for_file(PATCH_FILE)

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
        # Write only the main per-bin correlation. The convergence consumer
        # (cosebis_binning_comparison.py) reads just the per-bin columns
        # (np.loadtxt max_rows=nbins); the fine-grid jackknife cov is used
        # nowhere. write_patch_results/write_cov=True serialised a full
        # (2*nbins)^2 cov + patch blocks that nothing reads and also cost the
        # estimate_cov compute; drop both. The patches
        # still parallelise gg.process; gg.xip/gg.xim (values, FITS) are
        # unaffected.
        gg.write(out_txt, write_patch_results=False, write_cov=False)
        log(f"  Wrote {out_txt}")

        write_xi_integration_sacc(gg)

        elapsed = time.time() - t0
        log(f"Done! Total time: {elapsed / 3600:.1f}h ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
