"""Normalised mask power spectrum for CosmoCov from an existing binary HEALPix mask.

Same normalisation as scripts/masking.py (the producer of
mask_cls_footprint_nside_4096_norm.txt): C_ell of the mask from anafast,
rescaled so that sum_ell (2 ell + 1) / (4 pi) C_ell equals the unmasked area
in deg^2. Output: two columns (ell, normalised C_ell), ell = 0 .. 3 nside - 1.

Usage (script or snakemake):
    python mask_cls_from_map.py <mask_map.fits> <output_norm.txt>
"""

import sys

import healpy as hp
import numpy as np


def mask_cls_norm(mask_path, out_path):
    mask_map = hp.read_map(mask_path, dtype=np.float64)
    nside = hp.get_nside(mask_map)
    area_obs_deg2 = mask_map.sum() * hp.nside2pixarea(nside, degrees=True)
    print(f"{mask_path}: nside={nside}, area={area_obs_deg2:.2f} deg^2")

    cl_mask = hp.anafast(mask_map, lmax=3 * nside - 1)
    ells = np.arange(len(cl_mask))

    integral_w = np.sum((2 * ells + 1) / (4 * np.pi) * cl_mask) / (np.pi / 180) ** 2
    norm_cls = cl_mask * area_obs_deg2 / integral_w

    np.savetxt(out_path, np.column_stack((ells, norm_cls)), fmt=["%d", "%.10e"])
    print(f"Normalised mask Cls saved to {out_path}")


if __name__ == "__main__":
    try:
        smk = snakemake  # noqa: F821
        mask_cls_norm(smk.input.mask, smk.output[0])
    except NameError:
        mask_cls_norm(sys.argv[1], sys.argv[2])
