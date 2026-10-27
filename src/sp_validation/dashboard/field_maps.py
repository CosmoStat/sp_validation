"""Catalogue-level products behind the dashboard sky maps and weight figure.

Two cached products, each computed once from the calibrated catalogue and
read by the renderers in :mod:`sp_validation.dashboard`:

``field_maps_<version>.npz``
    Per-pixel sums on a nested HEALPix grid: galaxy counts and weight sums,
    weighted ellipticity and ellipticity-squared sums, and the PSF model
    (ellipticity, FWHM) at the galaxy positions; plus star counts and the
    star-minus-PSF-model residual sums. Means, dispersions and effective
    densities are ratios of these sums, so a renderer can coarsen the grid
    by summing children before dividing. Only occupied pixels are stored.

``weight_cells_<version>.npz``
    The galaxy weight on the SNR x size-ratio grid it is built on
    (:func:`sp_validation.calibration.get_w_des`): per-cell counts and the
    weight range within each cell, the grid edges, and the distinct weight
    values with their multiplicities.

Usage (inside the container)::

    python -m sp_validation.dashboard.field_maps \\
        --galaxies shape_catalog_cut_ngmix.fits --stars stars.hdf5 \\
        --version SP_v2.0 --out-dir cosmo_val/
"""

import argparse
import json
from pathlib import Path

import fitsio
import healpy as hp
import numpy as np

from sp_validation.io import read_catalogue

GALAXY_COLUMNS = [
    "RA",
    "Dec",
    "w_des",
    "e1",
    "e2",
    "e1_PSF",
    "e2_PSF",
    "fwhm_PSF",
    "snr",
    "NGMIX_T_NOSHEAR",
    "NGMIX_T_PSF_RECONV_NOSHEAR",
]
STAR_COLUMNS = [
    "RA",
    "DEC",
    "HSM_G1_PSF",
    "HSM_G2_PSF",
    "HSM_T_PSF",
    "HSM_G1_STAR",
    "HSM_G2_STAR",
    "HSM_T_STAR",
    "HSM_FLAG_PSF",
    "HSM_FLAG_STAR",
]
GALAXY_SUMS = [
    "n_gal",
    "sum_w",
    "sum_w2",
    "sum_w_e1",
    "sum_w_e2",
    "sum_w2_ee",
    "sum_w_e1_psf",
    "sum_w_e2_psf",
    "sum_w_fwhm_psf",
]
STAR_SUMS = [
    "n_star",
    "sum_de1",
    "sum_de2",
    "sum_dT_over_T_psf",
    "sum_T_psf",
]


def weight_grid_edges(n_bins, snr_range, size_ratio_range):
    """Log-spaced SNR and size-ratio edges of the weight grid."""
    snr_edges = np.logspace(*np.log10(snr_range), n_bins + 1)
    size_edges = np.logspace(*np.log10(size_ratio_range), n_bins + 1)
    return snr_edges, size_edges


def weight_cell_index(values, edges):
    """Cell index of each value, matching ``calibration.cut_to_bins``.

    Bins are right-closed, ``(edge[i], edge[i+1]]``; a value outside the
    edges belongs to the last cell, as ``cut_to_bins`` assigns it.
    """
    n_bins = len(edges) - 1
    index = np.searchsorted(edges, values, side="left") - 1
    outside = (index < 0) | (index >= n_bins) | ~np.isfinite(values)
    index[outside] = n_bins - 1
    return index


def _accumulate(total, pixels, weights, npix):
    total += np.bincount(pixels, weights=weights, minlength=npix)


def galaxy_products(path, nside, n_bins, snr_range, size_ratio_range, chunk):
    """Stream the galaxy catalogue once into map sums and weight-cell tallies."""
    npix = hp.nside2npix(nside)
    sums = {key: np.zeros(npix) for key in GALAXY_SUMS}
    snr_edges, size_edges = weight_grid_edges(n_bins, snr_range, size_ratio_range)
    cell_count = np.zeros((n_bins, n_bins), dtype=np.int64)
    cell_w_min = np.full((n_bins, n_bins), np.inf)
    cell_w_max = np.full((n_bins, n_bins), -np.inf)
    out_of_range = {"snr_low": 0, "snr_high": 0, "size_low": 0, "size_high": 0}
    weights = []

    with fitsio.FITS(path) as fits:
        table = fits[1]
        n_rows = table.get_nrows()
        for start in range(0, n_rows, chunk):
            rows = table[GALAXY_COLUMNS][start : start + chunk]
            pix = hp.ang2pix(nside, rows["RA"], rows["Dec"], lonlat=True, nest=True)
            w = rows["w_des"].astype(np.float64)
            e1, e2 = rows["e1"], rows["e2"]
            for key, value in (
                ("n_gal", None),
                ("sum_w", w),
                ("sum_w2", w**2),
                ("sum_w_e1", w * e1),
                ("sum_w_e2", w * e2),
                ("sum_w2_ee", w**2 * (e1**2 + e2**2)),
                ("sum_w_e1_psf", w * rows["e1_PSF"]),
                ("sum_w_e2_psf", w * rows["e2_PSF"]),
                ("sum_w_fwhm_psf", w * rows["fwhm_PSF"]),
            ):
                _accumulate(sums[key], pix, value, npix)

            snr = rows["snr"]
            size_ratio = rows["NGMIX_T_NOSHEAR"] / rows["NGMIX_T_PSF_RECONV_NOSHEAR"]
            out_of_range["snr_low"] += int(np.sum(snr <= snr_range[0]))
            out_of_range["snr_high"] += int(np.sum(snr > snr_range[1]))
            out_of_range["size_low"] += int(np.sum(size_ratio <= size_ratio_range[0]))
            out_of_range["size_high"] += int(np.sum(size_ratio > size_ratio_range[1]))
            i = weight_cell_index(snr, snr_edges)
            j = weight_cell_index(size_ratio, size_edges)
            flat = i * n_bins + j
            cell_count += np.bincount(flat, minlength=n_bins**2).reshape(n_bins, -1)
            np.minimum.at(cell_w_min.ravel(), flat, w)
            np.maximum.at(cell_w_max.ravel(), flat, w)
            weights.append(w)

    values, multiplicity = np.unique(np.concatenate(weights), return_counts=True)
    cells = {
        "snr_edges": snr_edges,
        "size_ratio_edges": size_edges,
        "count": cell_count,
        "w_min": np.where(cell_count > 0, cell_w_min, np.nan),
        "w_max": np.where(cell_count > 0, cell_w_max, np.nan),
        "w_values": values,
        "w_multiplicity": multiplicity,
        "n_rows": n_rows,
        "out_of_range": json.dumps(out_of_range),
    }
    return sums, cells


def star_sums(path, nside):
    """Star counts and star-minus-PSF-model residual sums per pixel."""
    npix = hp.nside2npix(nside)
    stars = read_catalogue(path, columns=STAR_COLUMNS)
    keep = (stars["HSM_FLAG_PSF"] == 0) & (stars["HSM_FLAG_STAR"] == 0)
    stars = stars[keep]
    pix = hp.ang2pix(nside, stars["RA"], stars["DEC"], lonlat=True, nest=True)
    t_star, t_psf = stars["HSM_T_STAR"], stars["HSM_T_PSF"]
    sums = {key: np.zeros(npix) for key in STAR_SUMS}
    for key, value in (
        ("n_star", None),
        ("sum_de1", stars["HSM_G1_STAR"] - stars["HSM_G1_PSF"]),
        ("sum_de2", stars["HSM_G2_STAR"] - stars["HSM_G2_PSF"]),
        ("sum_dT_over_T_psf", (t_star - t_psf) / t_psf),
        ("sum_T_psf", t_psf),
    ):
        _accumulate(sums[key], pix, value, npix)
    return sums, int(keep.sum())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--galaxies", required=True, help="calibrated galaxy FITS")
    parser.add_argument("--stars", required=True, help="star/PSF catalogue")
    parser.add_argument("--version", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--weight-bins", type=int, default=20)
    parser.add_argument("--snr-range", type=float, nargs=2, default=(10.0, 500.0))
    parser.add_argument("--size-ratio-range", type=float, nargs=2, default=(0.707, 3.0))
    parser.add_argument("--chunk", type=int, default=4_000_000)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    gal, cells = galaxy_products(
        args.galaxies,
        args.nside,
        args.weight_bins,
        args.snr_range,
        args.size_ratio_range,
        args.chunk,
    )
    stars, n_stars = star_sums(args.stars, args.nside)
    occupied = np.flatnonzero((gal["n_gal"] > 0) | (stars["n_star"] > 0))
    np.savez_compressed(
        out_dir / f"field_maps_{args.version}.npz",
        nside=args.nside,
        nest=True,
        pixels=occupied,
        n_stars_used=n_stars,
        **{key: value[occupied] for key, value in {**gal, **stars}.items()},
    )
    np.savez_compressed(
        out_dir / f"weight_cells_{args.version}.npz",
        version=args.version,
        **cells,
    )


if __name__ == "__main__":
    main()
