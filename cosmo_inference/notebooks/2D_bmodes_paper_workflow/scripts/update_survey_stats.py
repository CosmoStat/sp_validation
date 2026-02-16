"""Recompute survey statistics (A, n_e, sigma_e) for all active versions.

Uses the new footprint masks to define the survey area, then computes
n_eff and sigma_e from each version's shear catalog in chunks to limit
memory usage.

Usage:
    python workflow/scripts/update_survey_stats.py \
        --mask-standard <path> --mask-starhalo <path>
"""
import argparse
import gc
import sys
import os

import healpy as hp
import numpy as np
from astropy.io import fits
from pathlib import Path
import yaml

os.chdir("/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val")


# Versions that use the standard footprint mask
STANDARD_VERSIONS = [
    "SP_v1.4.5",
    "SP_v1.4.6",
    "SP_v1.4.6_ecut07",
    "SP_v1.4.11.3",
    "SP_v1.4.11.3_ecut07",
]

# Versions that use the star-halo footprint mask
STARHALO_VERSIONS = [
    "SP_v1.4.8",
]


def area_from_mask(mask_path):
    """Compute survey area from a binary HEALPix mask."""
    mask = hp.read_map(mask_path, dtype=np.float64)
    return float(mask.sum() * hp.nside2pixarea(hp.get_nside(mask), degrees=True))


def compute_stats_chunked(catalog_path, w_col, e1_col, e2_col, chunk_size=1_000_000):
    """Compute sum_w, sum_w2, sum_w2_e2 in chunks to limit memory."""
    with fits.open(catalog_path, memmap=True) as hdul:
        data = hdul[1].data
        nrows = len(data)

        sum_w = 0.0
        sum_w2 = 0.0
        sum_w2_e2 = 0.0

        for start in range(0, nrows, chunk_size):
            stop = min(start + chunk_size, nrows)
            w = np.asarray(data[w_col][start:stop], dtype=np.float64)
            e1 = np.asarray(data[e1_col][start:stop], dtype=np.float64)
            e2 = np.asarray(data[e2_col][start:stop], dtype=np.float64)

            sum_w += np.sum(w)
            w2 = w ** 2
            sum_w2 += np.sum(w2)
            sum_w2_e2 += np.sum(w2 * (e1**2 + e2**2))

            del w, e1, e2, w2

    return float(sum_w), float(sum_w2), float(sum_w2_e2), nrows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-standard", required=True)
    parser.add_argument("--mask-starhalo", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load cat_config
    config_path = Path("cat_config.yaml")
    with open(config_path) as f:
        cc = yaml.safe_load(f)

    # Compute areas from masks (only two masks to load)
    area_standard = area_from_mask(args.mask_standard)
    area_starhalo = area_from_mask(args.mask_starhalo)
    print(f"Standard footprint area: {area_standard:.2f} deg²")
    print(f"Star-halo footprint area: {area_starhalo:.2f} deg²\n")

    all_versions = [(v, area_standard) for v in STANDARD_VERSIONS] + \
                   [(v, area_starhalo) for v in STARHALO_VERSIONS]

    print(f"{'Version':<30} {'Area (deg²)':>12} {'n_eff':>10} {'sigma_e':>10}  {'Old A':>10} {'Old n_e':>10} {'Old σ_e':>10}")
    print("-" * 105)

    for ver, area_deg2 in all_versions:
        if ver not in cc:
            print(f"{ver:<30} SKIPPED (not in config)")
            continue

        shear_cfg = cc[ver].get("shear", {})
        if "path" not in shear_cfg:
            print(f"{ver:<30} SKIPPED (no shear path)")
            continue

        subdir = cc[ver].get("subdir", "")
        catalog_path = shear_cfg["path"]
        if not os.path.isabs(catalog_path):
            catalog_path = os.path.join(subdir, catalog_path)

        w_col = shear_cfg.get("w_col", "w")
        e1_col = shear_cfg.get("e1_col", "e1")
        e2_col = shear_cfg.get("e2_col", "e2")

        # Get old values for comparison
        old_cov = cc[ver].get("cov_th", {})
        old_A = old_cov.get("A", 0)
        old_ne = old_cov.get("n_e", 0)
        old_se = old_cov.get("sigma_e", 0)

        sum_w, sum_w2, sum_w2_e2, nrows = compute_stats_chunked(
            catalog_path, w_col, e1_col, e2_col
        )

        area_arcmin2 = area_deg2 * 3600.0
        n_eff = (sum_w**2) / (area_arcmin2 * sum_w2) if sum_w2 > 0 else 0.0
        sigma_e = np.sqrt(sum_w2_e2 / sum_w2) if sum_w2 > 0 else 0.0

        print(f"{ver:<30} {area_deg2:>12.2f} {n_eff:>10.6f} {sigma_e:>10.6f}  {old_A:>10.2f} {old_ne:>10.6f} {old_se:>10.6f}")

        if not args.dry_run:
            if "cov_th" not in cc[ver]:
                cc[ver]["cov_th"] = {}
            cc[ver]["cov_th"]["A"] = float(area_deg2)
            cc[ver]["cov_th"]["n_e"] = float(n_eff)
            cc[ver]["cov_th"]["sigma_e"] = float(sigma_e)

        gc.collect()

    if not args.dry_run:
        with open(config_path, "w") as f:
            yaml.dump(cc, f, sort_keys=False)
        print("\ncat_config.yaml updated.")
    else:
        print("\nDry run — no changes written.")


if __name__ == "__main__":
    main()
